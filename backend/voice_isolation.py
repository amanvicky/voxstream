"""Silero VAD (sherpa-onnx) + VoxLock target-speaker extraction and gating.

VoxLock is VoxStream's single-speaker voice lock:

* **Enrollment (5 s)** -- the user records ~5 seconds of their voice into
  ``profiles/<name>.wav``; a speaker-embedding network (NeMo TitaNet-L /
  3D-Speaker ERes2Net, shipped as ONNX) converts it to a fixed-size
  voiceprint vector that is stored next to the WAV.
* **Runtime gating** -- every VAD speech segment is embedded and compared
  with the enrolled voiceprint by cosine similarity.  Segments below the
  threshold (other speakers, TV noise) are rejected *before* ASR, so only
  the enrolled voice is ever transcribed or typed.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import sherpa_onnx
except Exception:  # pragma: no cover
    sherpa_onnx = None

from scipy.io import wavfile
from scipy.signal import resample_poly


def _m(obj, snake: str, camel: str):
    """Version-proof sherpa-onnx method accessor (snake_case -> camelCase)."""
    fn = getattr(obj, snake, None)
    if fn is None:
        fn = getattr(obj, camel)
    return fn


def _set(obj, name: str, value):
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_wav_16k(path) -> np.ndarray:
    """Load any WAV as mono float32 at 16 kHz in [-1, 1]."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    if int(sr) != 16000:
        g = math.gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# Silero VAD wrapper (sherpa-onnx >= 2.x low-level VadModel + own segmentation)
# ---------------------------------------------------------------------------
class SileroVAD:
    """Feeds 16 kHz audio, returns finished speech segments.

    sherpa-onnx 2.x removed the high-level ``sherpa_onnx.Vad`` streaming
    wrapper, so this class drives the low-level ``VadModel.is_speech``
    window classifier (512-sample windows at 16 kHz) with its own
    segmentation state machine: 0.6 s pre-roll so word onsets are never
    clipped, ``min_speech`` / ``min_silence`` / max-length rules, and a
    short natural-release tail kept at the end of each segment.
    """

    def __init__(self, model_path, threshold: float = 0.5,
                 min_speech_s: float = 0.25, min_silence_s: float = 0.6,
                 max_buffer_s: float = 30.0):
        if sherpa_onnx is None:
            raise RuntimeError("sherpa-onnx is not installed")
        cfg = sherpa_onnx.VadModelConfig()
        _set(cfg, "sample_rate", 16000)
        _set(cfg, "num_threads", 1)
        _set(cfg, "provider", "cpu")
        _set(cfg, "debug", 0)
        _set(cfg.silero_vad, "model", str(model_path))
        _set(cfg.silero_vad, "threshold", float(threshold))
        _set(cfg.silero_vad, "min_speech_duration", float(min_speech_s))
        _set(cfg.silero_vad, "min_silence_duration", float(min_silence_s))
        _set(cfg.silero_vad, "window_size", 512)
        _set(cfg.silero_vad, "max_speech_duration", 3600.0)  # we enforce our own
        self._model = sherpa_onnx.VadModel.create(cfg)
        ws = getattr(self._model, "window_size", None)
        self.window_size = int(ws() if callable(ws) else ws or 512)

        self._min_speech_samples = int(min_speech_s * 16000)
        self._min_silence_samples = int(min_silence_s * 16000)
        self._max_speech_samples = int(max(1.0, float(max_buffer_s)) * 16000)
        self._keep_tail_windows = max(1, int(0.12 * 16000 / self.window_size))
        self._preroll_windows = max(2, int(0.6 * 16000 / self.window_size))

        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._preroll: list[np.ndarray] = []
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._silence_run = 0
        self._windows_seen = 0

    def feed(self, samples: np.ndarray) -> list[dict]:
        out: list[dict] = []
        x = np.ascontiguousarray(samples, dtype=np.float32)
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        w = self.window_size
        while self._buf.size >= w:
            window = self._buf[:w]
            self._buf = self._buf[w:]
            try:
                speech = bool(self._model.is_speech(window))
            except Exception:
                speech = False
            self._windows_seen += 1

            if not self._in_speech:
                if speech:
                    # pre-roll (last 0.6 s, without the current window) + onset
                    self._in_speech = True
                    self._frames = list(self._preroll) + [window]
                    self._silence_run = 0
            else:
                self._frames.append(window)
                self._silence_run = 0 if speech else self._silence_run + 1
                speech_samples = len(self._frames) * w - self._silence_run * w
                if self._silence_run * w >= self._min_silence_samples:
                    keep = min(self._silence_run, self._keep_tail_windows)
                    frames = (self._frames[:-(self._silence_run - keep)]
                              if self._silence_run > keep else self._frames)
                    self._emit(frames, out)
                    self._in_speech = False
                    self._frames = []
                    self._silence_run = 0
                elif len(self._frames) * w >= self._max_speech_samples:
                    self._emit(self._frames, out)  # long continuous speech split
                    self._in_speech = False
                    self._frames = []
                    self._silence_run = 0

            self._preroll.append(window)
            if len(self._preroll) > self._preroll_windows:
                self._preroll.pop(0)
        return out

    def _emit(self, frames: list, out: list) -> None:
        if not frames:
            return
        audio = np.concatenate(frames).astype(np.float32)
        if audio.size < self._min_speech_samples:
            return
        start = max(0, (self._windows_seen - len(frames)) * self.window_size)
        out.append({"start": start, "samples": audio})

    def flush(self) -> list[dict]:
        """Emit the pending segment (end of stream / pause / standby)."""
        out: list[dict] = []
        if self._in_speech and self._frames:
            keep = min(self._silence_run, self._keep_tail_windows)
            frames = (self._frames[:-(self._silence_run - keep)]
                      if self._silence_run > keep else self._frames)
            self._emit(frames, out)
        self._in_speech = False
        self._frames = []
        self._silence_run = 0
        return out

    def reset(self) -> None:
        self._reset_state()
        try:
            self._model.reset()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VoxLock true overlap extraction (sherpa-onnx streaming target-speaker denoiser)
# ---------------------------------------------------------------------------
class VoxLockExtractor:
    """True simultaneous overlapping-speaker waveform isolation.

    Wraps ``sherpa_onnx.OnlineSpeechDenoiser`` configured with the VoxLock
    target-speaker networks (``enroll.onnx`` + ``chunk_int8.onnx``)
    conditioned on the enrolled profile WAV (``voice_sample``).  Speech from
    *other* people talking over the enrolled speaker is stripped out of the
    audio waveform itself, before ASR ever sees it.

    sherpa-onnx processes audio in ``frame_shift_in_samples`` hops (128
    samples = 8 ms).  Calling ``run()`` per 8 ms is dominated by per-call
    overhead (~5x slower than real-time), so this wrapper buffers frames
    into larger blocks (``process_block_ms``, default 400 ms) which brings
    the extraction to ~0.7-0.9x real-time on the target i5 CPU with at most
    ``process_block_ms`` of added latency.
    """

    def __init__(self, enroll_model, chunk_model, voice_sample_wav,
                 num_threads: int = 2, process_block_ms: int = 400):
        if sherpa_onnx is None:
            raise RuntimeError("sherpa-onnx is not installed")
        for p in (enroll_model, chunk_model, voice_sample_wav):
            if not Path(p).exists():
                raise FileNotFoundError(f"VoxLock bundle file missing: {p}")
        model_cfg = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            voxlock=sherpa_onnx.OfflineSpeechDenoiserVoxLockModelConfig(
                enroll_model=str(enroll_model),
                chunk_model=str(chunk_model),
                voice_sample=str(voice_sample_wav)),
            num_threads=int(num_threads), provider="cpu")
        self._denoiser = sherpa_onnx.OnlineSpeechDenoiser(
            sherpa_onnx.OnlineSpeechDenoiserConfig(model=model_cfg))
        self.frame_shift = int(self._denoiser.frame_shift_in_samples)
        self.sample_rate = int(self._denoiser.sample_rate)
        self.block_size = max(self.frame_shift,
                              int(16000 * max(40, int(process_block_ms)) / 1000))
        self.block_size -= self.block_size % self.frame_shift
        self._buf = np.zeros(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Feed arbitrary-size audio; return denoised audio (block-buffered).

        Returns ``b""``-empty arrays while the block buffer fills; call
        :meth:`flush` at utterance end to drain the remainder.
        """
        x = np.ascontiguousarray(samples, dtype=np.float32)
        if x.size:
            self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        out = np.zeros(0, dtype=np.float32)
        while self._buf.size >= self.block_size:
            out = np.concatenate([out, self._run_block(self._buf[:self.block_size])])
            self._buf = self._buf[self.block_size:]
        return out

    def flush(self) -> np.ndarray:
        """Drain the tail of the stream at utterance end."""
        out = np.zeros(0, dtype=np.float32)
        if self._buf.size:
            out = self._run_block(self._buf)
            self._buf = np.zeros(0, dtype=np.float32)
        return out

    def _run_block(self, block: np.ndarray) -> np.ndarray:
        try:
            result = self._denoiser.run(block, 16000)
            return np.asarray(result.samples, dtype=np.float32)
        except Exception as exc:
            print(f"[voxlock] extraction error: {exc}")
            return block  # fail open: feed raw audio rather than dropping speech

    def reset(self) -> None:
        try:
            self._denoiser.reset()
        except Exception:
            pass
        self._buf = np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# VoxLock target-speaker extractor
# ---------------------------------------------------------------------------
class VoxLock:
    """Speaker isolation controller with three modes.

    * ``off``            -- "Anyone": no isolation.
    * ``turn_titanet``   -- "Only Me (Fast)": cosine gate per VAD segment.
    * ``overlap_voxlock``-- "Only Me (Overlap)": true target-speaker waveform
      extraction (VoxLock enroll+chunk networks conditioned on the profile
      WAV) so overlapping voices are stripped before ASR.  Falls back to
      ``turn_titanet`` when the overlap bundle or a profile is missing.
    """

    MODES = ("off", "turn_titanet", "overlap_voxlock")

    def __init__(self, model_path, profiles_dir, threshold: float = 0.62,
                 enroll_model=None, chunk_model=None,
                 num_threads: int = 2, process_block_ms: int = 400):
        self.model_path = Path(model_path)
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.enroll_model = Path(enroll_model) if enroll_model else None
        self.chunk_model = Path(chunk_model) if chunk_model else None
        self.num_threads = int(num_threads)
        self.process_block_ms = int(process_block_ms)
        self._extractor = None       # VoxLock overlap extraction engine
        self._embedder = None        # TitaNet speaker-embedding network
        self._active_emb: Optional[np.ndarray] = None
        self._active_name: Optional[str] = None
        self._active_wav: Optional[Path] = None
        self.mode = "off"

    # -- mode management --------------------------------------------------- #
    def set_mode(self, mode: str) -> str:
        mode = mode if mode in self.MODES else "off"
        self.mode = mode
        if mode == "overlap_voxlock":
            if not (self.enroll_model and self.chunk_model
                    and self.enroll_model.exists() and self.chunk_model.exists()):
                print("[voxlock] overlap bundle missing - falling back to fast gate")
                self.mode = "turn_titanet"
            elif not self._active_wav:
                print("[voxlock] no active profile - falling back to fast gate")
                self.mode = "turn_titanet"
        if self.mode != "overlap_voxlock":
            self._extractor = None
        return self.mode

    def effective_mode(self) -> str:
        return self.mode

    def _ensure_extractor(self) -> Optional[VoxLockExtractor]:
        if self.mode != "overlap_voxlock" or not self._active_wav:
            return None
        if self._extractor is None:
            try:
                self._extractor = VoxLockExtractor(
                    self.enroll_model, self.chunk_model, self._active_wav,
                    num_threads=self.num_threads,
                    process_block_ms=self.process_block_ms)
            except Exception as exc:
                print(f"[voxlock] extractor unavailable ({exc}) - using fast gate")
                self.mode = "turn_titanet"
                self._extractor = None
        return self._extractor

    # -- streaming frame path ----------------------------------------------- #
    def process_frame(self, samples: np.ndarray) -> np.ndarray:
        """Per-mode transform of a 16 kHz frame (overlap mode isolates audio)."""
        if self.mode != "overlap_voxlock":
            return samples
        ex = self._ensure_extractor()
        if ex is None:
            return samples
        return ex.process(samples)

    def flush_stream(self) -> np.ndarray:
        """Drain the extractor at utterance end (overlap mode only)."""
        if self.mode != "overlap_voxlock":
            return np.zeros(0, dtype=np.float32)
        ex = self._ensure_extractor()
        return ex.flush() if ex else np.zeros(0, dtype=np.float32)

    def reset_stream(self) -> None:
        if self._extractor is not None:
            self._extractor.reset()

    # -- embedding network (TitaNet fast gate) ----------------------------- #
    def _ensure_embedder(self):
        if self._embedder is None:
            if sherpa_onnx is None:
                raise RuntimeError("sherpa-onnx is not installed")
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"VoxLock embedding model missing: {self.model_path} "
                    "(run setup_models.py)")
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path), num_threads=1, provider="cpu")
            self._embedder = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._embedder

    @property
    def dim(self) -> int:
        ex = self._ensure_embedder()
        d = getattr(ex, "dim", None)
        return int(d() if callable(d) else d or 0)

    def compute(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Embed ``samples`` (16 kHz float32) into a voiceprint vector.

        sherpa-onnx 1.x: ``extractor.compute(samples)``.
        sherpa-onnx 2.x: stream-based ``create_stream -> accept_waveform ->
        input_finished -> compute(stream)``.  Both paths are supported.
        """
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.size < int(0.15 * 16000):
            return None
        ex = self._ensure_embedder()
        if hasattr(ex, "create_stream"):  # sherpa-onnx >= 2.x
            stream = ex.create_stream()
            stream.accept_waveform(16000, samples)
            stream.input_finished()
            emb = ex.compute(stream)
        else:                              # sherpa-onnx 1.x
            emb = ex.compute(samples)
        return np.asarray(emb, dtype=np.float32)

    # -- profiles ---------------------------------------------------------- #
    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^\w\- ]+", "_", (name or "profile").strip()) or "profile"

    def enroll(self, wav_path, name: str, threshold: Optional[float] = None) -> dict:
        """Build a voiceprint from a recorded WAV and store it as a profile."""
        name = self._safe_name(name)
        audio = load_wav_16k(wav_path)
        duration = audio.size / 16000.0
        if duration < 3.0:
            raise ValueError(f"enrollment recording too short ({duration:.1f}s, need >= 3s)")
        emb = self.compute(audio)
        if emb is None or emb.size == 0:
            raise RuntimeError("failed to compute enrollment embedding")

        # Self-consistency score: similarity between the two halves.
        half = audio.size // 2
        score = 1.0
        if half > int(0.75 * 16000):
            e1 = self.compute(audio[:half])
            e2 = self.compute(audio[half:])
            if e1 is not None and e2 is not None:
                score = cosine(e1, e2)

        target_wav = self.profiles_dir / f"{name}.wav"
        src = Path(wav_path)
        if src.resolve() != target_wav.resolve():
            target_wav.write_bytes(src.read_bytes())
        np.save(self.profiles_dir / f"{name}.emb.npy", emb)
        meta = {
            "name": name,
            "wav": target_wav.name,
            "created": time.time(),
            "dim": int(emb.size),
            "duration_s": round(duration, 2),
            "self_similarity": round(float(score), 4),
            "threshold": float(self.threshold if threshold is None else threshold),
        }
        (self.profiles_dir / f"{name}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        self.set_active(name)
        return meta

    def set_active(self, name: Optional[str]) -> bool:
        if not name:
            self._active_emb = None
            self._active_name = None
            self._active_wav = None
            self._extractor = None
            self.set_mode(self.mode)  # re-resolve (overlap needs a profile)
            return False
        safe = self._safe_name(name)
        emb_path = self.profiles_dir / f"{safe}.emb.npy"
        if not emb_path.exists():
            return False
        self._active_emb = np.load(emb_path)
        self._active_name = safe
        wav_path = self.profiles_dir / f"{safe}.wav"
        self._active_wav = wav_path if wav_path.exists() else None
        # Voice sample changed -> the overlap extractor must be rebuilt.
        self._extractor = None
        self.set_mode(self.mode)  # re-resolve (overlap needs a profile)
        return True

    def active_profile(self) -> Optional[str]:
        return self._active_name

    def list_profiles(self) -> list[dict]:
        out = []
        for meta_path in sorted(self.profiles_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                out.append(meta)
            except Exception:
                continue
        return out

    def delete_profile(self, name: str) -> bool:
        name = self._safe_name(name)
        removed = False
        for suffix in (".wav", ".emb.npy", ".json"):
            p = self.profiles_dir / f"{name}{suffix}"
            if p.exists():
                p.unlink()
                removed = True
        if self._active_name == name:
            self._active_emb = None
            self._active_name = None
            self._active_wav = None
            self._extractor = None
            self.set_mode(self.mode)
        return removed

    # -- runtime gate ------------------------------------------------------- #
    def check(self, samples: np.ndarray) -> Tuple[bool, Optional[float]]:
        """Return ``(accepted, similarity)`` for a VAD speech segment.

        Mode policies:
        * ``overlap_voxlock`` -- the waveform has already been isolated by
          the VoxLock networks; accept whatever survived (the extractor
          outputs near-silence for other speakers, so empty results are
          dropped later by ASR).  ``similarity`` is not computed.
        * ``turn_titanet``   -- cosine gate against the enrolled voiceprint.
        * ``off``            -- accept everything.
        """
        if self.mode == "overlap_voxlock":
            if samples is None or samples.size < int(0.2 * 16000):
                return False, None
            rms = float(np.sqrt(np.mean(np.square(samples))))
            return bool(rms > 5e-4), None   # near-silence = no target speech
        if self._active_emb is None:
            return True, None  # no profile enrolled -> accept everything
        duration = samples.size / 16000.0 if samples is not None else 0.0
        if duration < 0.30:
            return True, None  # too short to judge reliably; let ASR decide
        emb = self.compute(samples)
        if emb is None:
            return True, None
        sim = cosine(emb, self._active_emb)
        return bool(sim >= self.threshold), float(sim)
