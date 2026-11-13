"""Dual ASR engine.

* ``StreamingZipformer`` -- sherpa-onnx streaming Zipformer transducer
  (int8) running in a background thread, producing low-latency live
  partial hypotheses for the HUD subtitle strip.
* ``WhisperFinalizer``  -- faster-whisper (CTranslate2, int8) used for the
  authoritative final text of each VAD speech segment, and for live
  translation (``task="translate"`` -> X -> English).

Both engines are fully offline: all weights live in ``./models`` and the
process runs with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (set by
``backend/__init__.py`` before anything imports these libraries).
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import sherpa_onnx
except Exception:  # pragma: no cover
    sherpa_onnx = None


def _m(obj, snake: str, camel: str):
    fn = getattr(obj, snake, None)
    if fn is None:
        fn = getattr(obj, camel)
    return fn


def _pick_model(model_dir: Path, kind: str) -> Path:
    """Find ``*kind*.onnx`` preferring int8 variants."""
    cands = sorted(model_dir.glob(f"*{kind}*.onnx"))
    if not cands:
        raise FileNotFoundError(f"missing {kind} model in {model_dir} (run setup_models.py)")
    int8 = [c for c in cands if "int8" in c.name.lower()]
    return (int8 or cands)[0]


# Whisper output hallucinations that appear on near-silent input.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "bye", "bye.", "you", "thank you very much", "okay", "ok",
}


class StreamingZipformer:
    """Streaming transducer recognizer with a background decode thread."""

    def __init__(self, model_dir, num_threads: int = 2,
                 decoding_method: str = "greedy_search"):
        if sherpa_onnx is None:
            raise RuntimeError("sherpa-onnx is not installed")
        model_dir = Path(model_dir)
        tokens = model_dir / "tokens.txt"
        if not tokens.exists():
            raise FileNotFoundError(f"missing tokens.txt in {model_dir} (run setup_models.py)")
        encoder = _pick_model(model_dir, "encoder")
        decoder = _pick_model(model_dir, "decoder")
        joiner = _pick_model(model_dir, "joiner")
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens), encoder=str(encoder), decoder=str(decoder),
                joiner=str(joiner), num_threads=int(num_threads),
                sample_rate=16000, feature_dim=80,
                decoding_method=decoding_method, provider="cpu")
        except (TypeError, AttributeError):
            # Fallback for sherpa-onnx versions where decoding_method lives
            # on OnlineRecognizerConfig rather than OnlineModelConfig.
            try:
                mc = sherpa_onnx.OnlineModelConfig()
                mc.transducer = sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=str(encoder), decoder=str(decoder), joiner=str(joiner),
                    tokens=str(tokens))
                mc.tokens = str(tokens)
                mc.num_threads = int(num_threads)
                mc.sample_rate = 16000
                mc.feature_dim = 80
                mc.provider = "cpu"
                rc = sherpa_onnx.OnlineRecognizerConfig(model_config=mc)
                rc.decoding_method = decoding_method
                self.recognizer = sherpa_onnx.OnlineRecognizer(rc)
            except Exception:
                mc = sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder=str(encoder), decoder=str(decoder),
                        joiner=str(joiner), tokens=str(tokens)),
                    tokens=str(tokens), num_threads=int(num_threads),
                    sample_rate=16000, feature_dim=80, provider="cpu",
                    decoding_method=decoding_method)
                self.recognizer = sherpa_onnx.OnlineRecognizer(
                    sherpa_onnx.OnlineRecognizerConfig(model_config=mc))

        self._q: queue.Queue = queue.Queue()
        self._partial = ""
        self._on_partial: Optional[Callable] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._RESET = object()
        self._STOP = object()

    # ------------------------------------------------------------------ #
    def start(self, on_partial: Optional[Callable] = None) -> None:
        self._on_partial = on_partial
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="voxstream-zipformer",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._q.put_nowait(self._STOP)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def feed(self, chunk: np.ndarray) -> None:
        if self._running:
            try:
                self._q.put_nowait(np.ascontiguousarray(chunk, dtype=np.float32))
            except Exception:
                pass

    def reset(self) -> None:
        """Clear decoder state at utterance boundaries (VAD endpoints)."""
        if self._running:
            try:
                self._q.put_nowait(self._RESET)
            except Exception:
                pass

    def current_partial(self) -> str:
        return self._partial

    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        stream = self.recognizer.create_stream()
        last_emit = 0.0
        while self._running:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is self._STOP:
                break
            if item is self._RESET:
                self._partial = ""
                self._notify("")
                stream = self.recognizer.create_stream()  # fresh state
                continue
            try:
                _m(stream, "accept_waveform", "AcceptWaveform")(16000, item)
                while _m(self.recognizer, "is_ready", "IsReady")(stream):
                    _m(self.recognizer, "decode_stream", "DecodeStream")(stream)
                now = time.time()
                if now - last_emit >= 0.1:
                    last_emit = now
                    result = _m(self.recognizer, "get_result", "GetResult")(stream)
                    if isinstance(result, (list, tuple)):
                        result = " ".join(result)
                    text = (result or "").strip()
                    if text != self._partial:
                        self._partial = text
                        self._notify(text)
            except Exception as exc:
                print(f"[zipformer] decode error: {exc}")
                stream = self.recognizer.create_stream()

    def _notify(self, text: str) -> None:
        if self._on_partial:
            try:
                self._on_partial(text)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def decode_offline(self, samples: np.ndarray) -> str:
        """Synchronous full-utterance decode (used by the verification suite)."""
        stream = self.recognizer.create_stream()
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        step = 480
        for i in range(0, samples.size, step):
            _m(stream, "accept_waveform", "AcceptWaveform")(16000, samples[i:i + step])
        _m(stream, "input_finished", "InputFinished")()
        while _m(self.recognizer, "is_ready", "IsReady")(stream):
            _m(self.recognizer, "decode_stream", "DecodeStream")(stream)
        result = _m(self.recognizer, "get_result", "GetResult")(stream)
        if isinstance(result, (list, tuple)):
            result = " ".join(result)
        return (result or "").strip()


class WhisperFinalizer:
    """faster-whisper INT8 finalizer / translator (lazy-loaded, thread-safe)."""

    def __init__(self, model_dir, compute_type: str = "int8", cpu_threads: int = 3):
        self.model_dir = Path(model_dir)
        self.compute_type = compute_type
        self.cpu_threads = int(cpu_threads)
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not (self.model_dir / "model.bin").exists():
                raise FileNotFoundError(
                    f"whisper model missing: {self.model_dir} (run setup_models.py)")
            from faster_whisper import WhisperModel  # offline env already enforced
            self._model = WhisperModel(
                str(self.model_dir), device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads, num_workers=1)

    def transcribe(self, samples: np.ndarray, task: str = "transcribe",
                   language: Optional[str] = None) -> dict:
        """Return ``{"text", "language", "probability", "elapsed"}`` for a segment."""
        t0 = time.time()
        if not self.is_loaded:
            self.load()
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        segments, info = self._model.transcribe(
            audio, task=task, language=language,
            beam_size=1, best_of=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True)
        text = "".join(s.text for s in segments).strip()
        elapsed = time.time() - t0
        # Suppress the classic whisper hallucinations on near-silent input.
        if text and audio.size / 16000.0 < 1.2 and text.lower().strip(".!? ") in _HALLUCINATIONS:
            text = ""
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "probability": getattr(info, "language_probability", None),
            "elapsed": elapsed,
        }
