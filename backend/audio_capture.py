"""WASAPI native-rate microphone capture, polyphase resampling to 16 kHz,
30 ms frame emission, and the S0 standby zero-signal watchdog.

Design:
* The input stream is opened at the device's *native* shared-mode rate via
  WASAPI (48 kHz on most machines) instead of forcing a resampled 16 kHz
  device open, which avoids double resampling and driver quirks.
* A processing thread converts blocks to mono, tracks the level meter,
  and runs the standby watchdog: while the system is awake (S0) but the
  mic produces pure digital zero (muted/suspended device) for
  ``audio.standby_after_s`` seconds, ASR/VAD feeding is suspended to save
  CPU; any real signal immediately wakes the pipeline.
* Audio is resampled to 16 kHz with ``scipy.signal.resample_poly`` using an
  overlap tail to eliminate block-edge filter artifacts, then emitted as
  fixed 480-sample (30 ms) frames.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio host
    sd = None

from scipy.signal import resample_poly


def list_input_devices() -> list[dict]:
    """Enumerate input-capable devices with host API info."""
    if sd is None:
        return []
    out = []
    apis = list(sd.query_hostapis())
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        api_idx = int(dev.get("hostapi", 0))
        out.append({
            "index": idx,
            "name": str(dev.get("name", f"device {idx}")),
            "hostapi": str(apis[api_idx]["name"]) if api_idx < len(apis) else "?",
            "default_samplerate": float(dev.get("default_samplerate", 48000.0)),
            "channels": int(dev.get("max_input_channels", 1)),
        })
    return out


def resolve_input_device(config_value=None) -> Optional[dict]:
    """Resolve a configured device (None | index | name substring).

    Prefers the WASAPI default input when no explicit device is configured,
    falling back to any host API's default.
    """
    devices = list_input_devices()
    if not devices:
        return None
    if isinstance(config_value, int):
        for d in devices:
            if d["index"] == config_value:
                return d
    if isinstance(config_value, str) and config_value.strip():
        needle = config_value.strip().lower()
        for d in devices:
            if needle in d["name"].lower():
                return d
    # Default preference chain: WASAPI default -> any WASAPI -> first device.
    if sd is not None:
        try:
            default_in = int(sd.default.device[0])
            for d in devices:
                if d["index"] == default_in:
                    if d["hostapi"].lower().startswith("wasapi"):
                        return d
                    default_dev = d
        except Exception:
            default_dev = None
        for d in devices:
            if d["hostapi"].lower().startswith("wasapi"):
                return d
        if default_dev is not None:
            return default_dev
    return devices[0]


class Resampler:
    """Stateful block polyphase resampler with overlap tail (no edge clicks)."""

    def __init__(self, in_rate: int, out_rate: int = 16000, tail: int = 1024):
        in_rate = int(in_rate)
        g = math.gcd(in_rate, out_rate)
        self.up = out_rate // g
        self.down = in_rate // g
        self.tail_size = max(8, int(tail))
        self._tail = np.zeros(self.tail_size, dtype=np.float32)
        # Samples at the head of each output that depend on the previous
        # block's context -- drop them to avoid re-emitting filtered overlap.
        self.drop = int(round(self.tail_size * self.up / self.down))

    def process(self, block: np.ndarray) -> np.ndarray:
        x = block.astype(np.float32, copy=False)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate([self._tail, x])
        y = resample_poly(joined, self.up, self.down).astype(np.float32)
        self._tail = x[-self.tail_size:].copy()
        if self.drop and y.size > self.drop:
            return y[self.drop:]
        return y[:0]


class AudioCapture:
    """Microphone capture producing 16 kHz float32 frames via callbacks."""

    def __init__(self, config, on_frame: Optional[Callable] = None,
                 on_meter: Optional[Callable] = None,
                 on_standby: Optional[Callable] = None,
                 on_error: Optional[Callable] = None):
        audio_cfg = (config.get("audio", {}) or {}) if config else {}
        self.config = config
        self.frame_ms = int(audio_cfg.get("frame_ms", 30))
        self.zero_floor = float(audio_cfg.get("zero_floor", 1e-5))
        self.standby_after_s = float(audio_cfg.get("standby_after_s", 45.0))
        self.frame_len = int(round(16000 * self.frame_ms / 1000))  # 480 @ 30ms

        self.on_frame = on_frame
        self.on_meter = on_meter
        self.on_standby = on_standby
        self.on_error = on_error

        self._raw_q: queue.Queue = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._running = False
        self._resampler: Optional[Resampler] = None
        self._leftover = np.zeros(0, dtype=np.float32)
        self._device: Optional[dict] = None
        self.device_name: Optional[str] = None
        self.sample_rate: Optional[int] = None
        self._last_signal_t: Optional[float] = None
        self._last_meter_t = 0.0
        self._standby = False
        self._status_warned = False

    # ------------------------------------------------------------------ #
    @property
    def standby(self) -> bool:
        return self._standby

    def start(self) -> bool:
        if sd is None:
            self._error("sounddevice is not available")
            return False
        if self._running:
            return True
        dev = resolve_input_device(
            (self._config_device() if self.config else None))
        if dev is None:
            self._error("no input audio device found")
            return False
        self._device = dev
        self.device_name = f"{dev['name']} ({dev['hostapi']})"
        rate = int(round(dev["default_samplerate"])) or 48000
        self.sample_rate = rate
        channels = max(1, min(2, int(dev["channels"])))
        blocksize = max(1, int(round(rate * self.frame_ms / 1000)))
        try:
            self._stream = sd.InputStream(
                samplerate=rate,
                blocksize=blocksize,
                channels=channels,
                dtype="float32",
                device=int(dev["index"]),
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            self._error(f"failed to open input stream: {exc}")
            self._stream = None
            return False
        self._resampler = Resampler(rate, 16000)
        self._leftover = np.zeros(0, dtype=np.float32)
        self._last_signal_t = time.time()
        self._standby = False
        self._running = True
        self._thread = threading.Thread(target=self._process_loop,
                                        name="voxstream-capture", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------ #
    def _config_device(self):
        return self.config.get("audio.input_device") if self.config else None

    def _error(self, message: str) -> None:
        if self.on_error:
            try:
                self.on_error(message)
            except Exception:
                pass

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status and not self._status_warned:
            self._status_warned = True
            print(f"[audio] stream status: {status}")
        try:
            self._raw_q.put_nowait(indata.copy())
        except queue.Full:
            try:
                self._raw_q.get_nowait()
                self._raw_q.put_nowait(indata.copy())
            except Exception:
                pass

    def _set_standby(self, value: bool) -> None:
        if value != self._standby:
            self._standby = value
            if self.on_standby:
                try:
                    self.on_standby(value)
                except Exception:
                    pass

    def _process_loop(self) -> None:
        meter_interval = 0.1
        while self._running:
            try:
                block = self._raw_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                mono = block.mean(axis=1, dtype=np.float32) if block.ndim > 1 else block
                mono = np.ascontiguousarray(mono, dtype=np.float32)
                peak = float(np.max(np.abs(mono))) if mono.size else 0.0
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                now = time.time()

                # --- S0 standby zero-signal watchdog -----------------------
                if peak > self.zero_floor:
                    self._last_signal_t = now
                    if self._standby:
                        self._set_standby(False)
                elif (not self._standby and self._last_signal_t is not None
                      and now - self._last_signal_t >= self.standby_after_s):
                    self._set_standby(True)

                # --- meter (throttled) --------------------------------------
                if self.on_meter and now - self._last_meter_t >= meter_interval:
                    self._last_meter_t = now
                    try:
                        self.on_meter(rms)
                    except Exception:
                        pass

                if self._standby:
                    continue  # suspend heavy processing while digitally silent

                # --- resample + frame emission ------------------------------
                out = self._resampler.process(mono)
                if out.size:
                    joined = np.concatenate([self._leftover, out])
                    usable = (joined.size // self.frame_len) * self.frame_len
                    if usable:
                        frames = joined[:usable]
                        for i in range(0, usable, self.frame_len):
                            if self.on_frame:
                                self.on_frame(frames[i:i + self.frame_len])
                        self._leftover = joined[usable:]
                    else:
                        self._leftover = joined
            except Exception as exc:
                self._error(f"capture processing error: {exc}")
                time.sleep(0.05)
