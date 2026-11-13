"""VoxStream backend: FastAPI control/telemetry server on 127.0.0.1:8765.

Runs the realtime pipeline as a background sidecar process (spawned by
``app.py``):

    mic -> WASAPI native-rate capture -> 16 kHz resample (30 ms frames)
        -> streaming Zipformer (live partials)  --> WS "partial"
        -> Silero VAD segments -> VoxLock speaker gate
        -> faster-whisper INT8 (final / X->English translate)
        -> voice commands -> history store -> Win32 injection --> WS "final"

Wire protocol (JSON over ``/ws``) -- client -> server:
    ping, get_state, start, stop, set_target{hwnd,title}, clear_target,
    set_inject{enabled}, set_translate{enabled}, set_voxlock{enabled,threshold?},
    set_profile{name}, enroll{name,wav_path}, delete_profile{name},
    get_history, clear_history, shutdown

server -> client:
    state, meter, partial, final, rejected, command_executed,
    request_pick_window, history, history_cleared, enroll_result,
    telemetry, info, error, pong, bye
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend import APP_VERSION, BASE_DIR, CONFIG, LOGS_DIR, resource_path
from backend.asr_engine import StreamingZipformer, WhisperFinalizer
from backend.audio_capture import AudioCapture
from backend.commands import CommandProcessor
from backend.history_store import HistoryStore
from backend.llm_polish import LLMPolisher
from backend.voice_isolation import SileroVAD, VoxLock
from backend.win32_injector import TextInjector

log = logging.getLogger("voxstream.server")


# ---------------------------------------------------------------------------
# WebSocket hub
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.clients: set = set()
        self.queue: Optional[asyncio.Queue] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def attach(self, loop) -> None:
        self.loop = loop
        self.queue = asyncio.Queue()

    def emit_threadsafe(self, msg: dict) -> None:
        """Called from pipeline threads."""
        if self.loop is not None and self.queue is not None:
            try:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)
            except RuntimeError:
                pass  # loop shutting down

    async def direct(self, ws, msg: dict) -> None:
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            self.clients.discard(ws)

    async def broadcaster(self) -> None:
        while True:
            msg = await self.queue.get()
            data = json.dumps(msg, ensure_ascii=False)
            for ws in list(self.clients):
                try:
                    await ws.send_text(data)
                except Exception:
                    self.clients.discard(ws)


# ---------------------------------------------------------------------------
# Realtime pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    """Owns all engines and glues the audio->ASR->command->injection flow."""

    def __init__(self, config, load_models: bool = True):
        self.cfg = config
        self.emit = lambda msg: None  # replaced by Hub.emit_threadsafe at startup
        self.stop_flag = False

        self.models_loaded = False
        self.load_error: Optional[str] = None
        self.listening = False
        self.standby = False
        self.device_name: Optional[str] = None
        self._started_at = time.time()

        self.commands = CommandProcessor(config)
        self.history = HistoryStore(
            BASE_DIR / (config.get("history.db") or "data/history.db"),
            int(config.get("history.max_items", 50) or 50))
        self.injector = TextInjector(config)

        self.capture: Optional[AudioCapture] = None
        self.vad: Optional[SileroVAD] = None
        self.zipformer: Optional[StreamingZipformer] = None
        self.voxlock: Optional[VoxLock] = None
        self.whisper: Optional[WhisperFinalizer] = None
        self.polisher: Optional[LLMPolisher] = None

        self._lock = threading.RLock()
        self._seg_q: "queue.Queue" = queue.Queue(maxsize=8)
        self._whisper_thread: Optional[threading.Thread] = None
        self._load_thread: Optional[threading.Thread] = None
        self.load_models_enabled = bool(load_models)
        self._whisper_rtf: Optional[float] = None

    # -- model loading ------------------------------------------------------ #
    def begin_load(self) -> None:
        if not self.load_models_enabled:
            self.models_loaded = False
            return
        self._load_thread = threading.Thread(target=self._load_models,
                                             name="voxstream-modelload", daemon=True)
        self._load_thread.start()

    def _load_models(self) -> None:
        try:
            t0 = time.time()
            self.zipformer = StreamingZipformer(
                resource_path(self.cfg.get("asr.zipformer_dir")),
                num_threads=int(self.cfg.get("asr.num_threads", 2)),
                decoding_method=self.cfg.get("asr.decoding_method") or "greedy_search")
            self.zipformer.start(self._on_partial)

            self.vad = SileroVAD(
                resource_path(self.cfg.get("vad.model")),
                threshold=float(self.cfg.get("vad.threshold", 0.5)),
                min_speech_s=float(self.cfg.get("vad.min_speech_s", 0.25)),
                min_silence_s=float(self.cfg.get("vad.min_silence_s", 0.6)),
                max_buffer_s=float(self.cfg.get("vad.max_buffer_s", 30.0)))

            self.voxlock = VoxLock(
                resource_path(self.cfg.get("voxlock.model")),
                BASE_DIR / "profiles",
                threshold=float(self.cfg.get("voxlock.threshold", 0.62)),
                enroll_model=resource_path(self.cfg.get("voxlock.enroll_model",
                                                        "models/voxlock/enroll.onnx")),
                chunk_model=resource_path(self.cfg.get("voxlock.chunk_model",
                                                       "models/voxlock/chunk_int8.onnx")),
                num_threads=int(self.cfg.get("voxlock.num_threads", 2)),
                process_block_ms=int(self.cfg.get("voxlock.process_block_ms", 400)))
            active = self.cfg.get("voxlock.active_profile")
            if active:
                self.voxlock.set_active(active)
            self.voxlock.set_mode(self.cfg.get("voxlock.isolation_mode")
                                  or "overlap_voxlock")

            self.whisper = WhisperFinalizer(
                resource_path(self.cfg.get("whisper.dir")),
                compute_type=self.cfg.get("whisper.compute_type") or "int8",
                cpu_threads=int(self.cfg.get("whisper.cpu_threads", 3)))
            self.whisper.load()

            self.polisher = LLMPolisher(self.cfg)

            self._whisper_thread = threading.Thread(
                target=self._whisper_loop, name="voxstream-whisper", daemon=True)
            self._whisper_thread.start()

            self.models_loaded = True
            log.info("models ready in %.1fs (voxlock profile: %s, isolation: %s)",
                     time.time() - t0, self.voxlock.active_profile(),
                     self.voxlock.effective_mode())
            self.emit({"type": "info", "message":
                       f"models ready - isolation: {self.voxlock.effective_mode()}"})
            self.emit_state()
        except Exception as exc:
            self.load_error = f"{exc}"
            log.error("model load failed:\n%s", traceback.format_exc())
            self.emit({"type": "error", "message": f"model load failed: {exc}"})

    # -- listening control -------------------------------------------------- #
    def start_listening(self) -> None:
        with self._lock:
            if not self.models_loaded:
                self.emit({"type": "error",
                           "message": "models not ready" + (f": {self.load_error}" if self.load_error else "")})
                return
            if self.capture is None:
                self.capture = AudioCapture(
                    self.cfg, on_frame=self._on_frame, on_meter=self._on_meter,
                    on_standby=self._on_standby, on_error=self._on_capture_error)
                if not self.capture.start():
                    self.capture = None
                    return
                self.device_name = self.capture.device_name
            self.listening = True
            self.standby = False
        log.info("listening (device: %s)", self.device_name)
        self.emit_state()

    def stop_listening(self) -> None:
        with self._lock:
            self.listening = False
        self._flush_voxlock()
        self._flush_vad()
        if self.zipformer:
            self.zipformer.reset()
        self.emit({"type": "partial", "text": ""})
        log.info("stopped listening")
        self.emit_state()

    def _flush_voxlock(self) -> None:
        """Drain the VoxLock extractor tail into VAD/ASR at utterance end."""
        if self.voxlock is None:
            return
        try:
            tail = self.voxlock.flush_stream()
            if tail.size:
                if self.zipformer:
                    self.zipformer.feed(tail)
                if self.vad:
                    for seg in self.vad.feed(tail):
                        self._on_segment(seg["samples"])
            self.voxlock.reset_stream()
        except Exception as exc:
            log.error("voxlock flush error: %s", exc)

    def _flush_vad(self) -> None:
        """Finalize a segment that was still open (end of speech at stop/standby)."""
        if self.vad is None:
            return
        try:
            for seg in self.vad.flush():
                self._on_segment(seg["samples"])
        except Exception as exc:
            log.error("vad flush error: %s", exc)

    def toggle_listening(self) -> None:
        if self.listening:
            self.stop_listening()
        else:
            self.start_listening()

    # -- audio callbacks ----------------------------------------------------- #
    def _on_capture_error(self, message: str) -> None:
        self.emit({"type": "error", "message": message})

    def _on_standby(self, value: bool) -> None:
        if value:
            self._flush_voxlock()  # drain the extractor tail before suspending
            self._flush_vad()      # finalize any open utterance before suspending
        self.standby = value
        log.info("S0 standby watchdog: %s", "engaged (zero signal)" if value else "woken")
        self.emit_state()

    def _on_meter(self, rms: float) -> None:
        self.emit({"type": "meter", "rms": float(rms)})

    def _on_partial(self, text: str) -> None:
        if self.listening and not self.standby:
            self.emit({"type": "partial", "text": text})

    def _on_frame(self, chunk: np.ndarray) -> None:
        if not self.listening or self.standby:
            return
        # Overlap mode: strip other speakers out of the waveform itself
        # before Zipformer/VAD ever see the audio.
        if self.voxlock is not None:
            try:
                chunk = self.voxlock.process_frame(chunk)
            except Exception as exc:
                log.error("voxlock frame error: %s", exc)
        if self.zipformer:
            self.zipformer.feed(chunk)
        if self.vad:
            try:
                for seg in self.vad.feed(chunk):
                    self._on_segment(seg["samples"])
            except Exception as exc:
                log.error("vad feed error: %s", exc)

    # -- segmentation + speaker gate ------------------------------------------ #
    def _on_segment(self, samples: np.ndarray) -> None:
        if samples.size < int(0.2 * 16000):
            return
        streaming_text = self.zipformer.current_partial() if self.zipformer else ""
        if self.zipformer:
            self.zipformer.reset()
        voxlock_on = bool(self.cfg.get("voxlock.enabled", True))
        if voxlock_on and self.voxlock:
            try:
                accepted, sim = self.voxlock.check(samples)
            except Exception as exc:
                log.error("voxlock check error: %s", exc)
                accepted, sim = True, None
        else:
            accepted, sim = True, None
        if not accepted:
            self.emit({"type": "rejected", "reason": "speaker", "similarity": sim})
            return
        try:
            self._seg_q.put_nowait((samples, sim, streaming_text))
        except queue.Full:
            try:
                self._seg_q.get_nowait()
                self._seg_q.put_nowait((samples, sim, streaming_text))
            except Exception:
                pass

    # -- whisper finalization worker ------------------------------------------- #
    def _whisper_loop(self) -> None:
        while not self.stop_flag:
            try:
                samples, sim, streaming_text = self._seg_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._finalize(samples, sim, streaming_text)
            except Exception as exc:
                log.error("finalize error:\n%s", traceback.format_exc())
                self.emit({"type": "error", "message": f"finalize error: {exc}"})

    def _finalize(self, samples: np.ndarray, sim: Optional[float],
                  streaming_text: str) -> None:
        translate = bool(self.cfg.get("translation.enabled", False))
        task = "translate" if translate else "transcribe"
        language_cfg = self.cfg.get("asr.source_language")
        res = self.whisper.transcribe(samples, task=task, language=language_cfg)
        text = res["text"] or streaming_text  # whisper result, streaming fallback
        if not text:
            return
        proc = self.commands.process(text)

        if proc["command"]:
            self._execute_command(proc)
            return

        final_text = proc["text"].strip()
        if not final_text:
            return

        # Optional local-LLM polish (hard 1.5s timeout -> raw ASR fallback).
        polished = False
        if self.polisher is not None and len(final_text) > 25:
            try:
                new_text, polished = self.polisher.polish(final_text)
                if polished and new_text:
                    final_text = new_text
            except Exception as exc:
                log.error("llm polish error: %s", exc)

        injected_tier: Optional[str] = None
        injected_chars = 0
        if bool(self.cfg.get("injection.enabled", True)):
            if self.injector.is_target_valid():
                tier = self.injector.inject(final_text + " ")
                if tier:
                    injected_tier = tier
                    injected_chars = len(final_text) + 1
            else:
                # "Type Anywhere": type into whatever the user focused on;
                # if they're on the desktop/taskbar there is no text box -
                # tell the UI to show the friendly "Copy Last Text" banner.
                tier = self.injector.inject_anywhere(final_text + " ")
                if tier == "no_text_box":
                    self.emit({"type": "no_text_box", "text": final_text})
                elif tier:
                    injected_tier = tier
                    injected_chars = len(final_text) + 1

        row = self.history.add(
            final_text, lang=res.get("language"), translated=translate,
            engine="whisper", confidence=res.get("probability"),
            injected=injected_tier is not None, injected_chars=injected_chars)
        if samples.size:
            self._whisper_rtf = res["elapsed"] / (samples.size / 16000.0)
        self.emit({
            "type": "final", "text": final_text, "id": row["id"],
            "language": res.get("language"), "translated": translate,
            "similarity": sim, "elapsed": round(res["elapsed"], 3),
            "injected": injected_tier, "polished": polished,
        })

    # -- voice commands ------------------------------------------------------- #
    def _execute_command(self, proc: dict) -> None:
        name = proc["command"]
        argument = proc["argument"]
        log.info("voice command: %s (arg=%r)", name, argument)
        if name == "stop_listening":
            self.stop_listening()
        elif name == "start_listening":
            self.start_listening()
        elif name in ("new_line", "new_paragraph"):
            payload = "\r" if name == "new_line" else "\r\r"
            if bool(self.cfg.get("injection.enabled", True)) and self.injector.is_target_valid():
                if argument:
                    self.injector.inject(payload + argument + " ")
                else:
                    self.injector.inject(payload)
        elif name == "scratch_that":
            rows = self.history.last(1)
            n = int(rows[-1].get("injected_chars") or 0) if rows else 0
            if n and self.injector.is_target_valid():
                self.injector.send_backspaces(n)
            self.history.delete_last()
        elif name == "clear_all":
            self.history.clear()
            self.emit({"type": "history_cleared"})
        elif name == "toggle_translate":
            self.set_translation(not bool(self.cfg.get("translation.enabled", False)))
        elif name == "pick_window":
            self.emit({"type": "request_pick_window"})
        self.emit({"type": "command_executed", "name": name, "arg": argument})

    # -- settings --------------------------------------------------------------- #
    def set_target(self, hwnd: int, title: Optional[str] = None) -> None:
        self.injector.set_target(hwnd, title)
        self.cfg.set("injection.target_title", title)
        log.info("injection target locked: hwnd=%s title=%r", hwnd, title)
        self.emit_state()

    def clear_target(self) -> None:
        self.injector.clear_target()
        self.cfg.set("injection.target_title", None)
        self.emit_state()

    def set_injection(self, enabled: bool) -> None:
        self.cfg.set("injection.enabled", bool(enabled))
        self.emit_state()

    def set_translation(self, enabled: bool) -> None:
        self.cfg.set("translation.enabled", bool(enabled))
        self.emit_state()

    def set_voxlock(self, enabled: Optional[bool] = None,
                    threshold: Optional[float] = None) -> None:
        if enabled is not None:
            self.cfg.set("voxlock.enabled", bool(enabled))
        if threshold is not None:
            self.cfg.set("voxlock.threshold", float(threshold))
            if self.voxlock:
                self.voxlock.threshold = float(threshold)
        self.emit_state()

    def set_isolation(self, mode: str) -> None:
        """Switch speaker isolation: off | turn_titanet | overlap_voxlock."""
        mode = str(mode or "off")
        self.cfg.set("voxlock.isolation_mode", mode)
        if self.voxlock is None:
            self.emit_state()
            return
        effective = self.voxlock.set_mode(mode)
        if effective != mode:
            log.info("isolation mode %r unavailable -> %s", mode, effective)
        else:
            log.info("isolation mode -> %s", effective)
        self.emit({"type": "info", "message": f"isolation mode: {effective}"})
        self.emit_state()

    def set_llm_polish(self, enabled: bool) -> None:
        self.cfg.set("llm_polish_enabled", bool(enabled))
        if self.polisher:
            self.polisher.set_enabled(bool(enabled))
        log.info("AI polish -> %s", bool(enabled))
        self.emit_state()

    def apply_settings_runtime(self) -> None:
        """Re-read config.json (saved by the UI settings dialog) and re-apply
        every runtime-relevant piece: jargon table, LLM polish params,
        VoxLock threshold/mode."""
        self.cfg.reload()
        self.commands = CommandProcessor(self.cfg)
        if self.polisher is None:
            self.polisher = LLMPolisher(self.cfg)  # lightweight, model-free
        else:
            self.polisher._reload()
        if self.voxlock:
            self.voxlock.threshold = float(self.cfg.get("voxlock.threshold", 0.62))
            self.voxlock.set_mode(self.cfg.get("voxlock.isolation_mode")
                                  or "overlap_voxlock")
        log.info("settings re-applied from config.json")
        self.emit({"type": "info", "message": "settings applied"})
        self.emit_state()

    def set_profile(self, name: Optional[str]) -> bool:
        if name and self.voxlock and self.voxlock.set_active(name):
            self.cfg.set("voxlock.active_profile", name)
            self.cfg.set("voxlock.enabled", True)
            self.voxlock.set_mode(self.cfg.get("voxlock.isolation_mode")
                                  or "overlap_voxlock")
            self.emit_state()
            return True
        self.emit({"type": "error", "message": f"profile not found: {name}"})
        return False

    def enroll(self, name: str, wav_path: str) -> dict:
        if self.voxlock is None:
            self.voxlock = VoxLock(
                resource_path(self.cfg.get("voxlock.model")), BASE_DIR / "profiles",
                threshold=float(self.cfg.get("voxlock.threshold", 0.62)),
                enroll_model=resource_path(self.cfg.get("voxlock.enroll_model",
                                                        "models/voxlock/enroll.onnx")),
                chunk_model=resource_path(self.cfg.get("voxlock.chunk_model",
                                                       "models/voxlock/chunk_int8.onnx")),
                num_threads=int(self.cfg.get("voxlock.num_threads", 2)),
                process_block_ms=int(self.cfg.get("voxlock.process_block_ms", 400)))
        meta = self.voxlock.enroll(wav_path, name)
        self.cfg.set("voxlock.active_profile", meta["name"])
        self.cfg.set("voxlock.enabled", True)
        # A profile is now active: (re)apply the configured isolation mode.
        self.voxlock.set_mode(self.cfg.get("voxlock.isolation_mode")
                              or "overlap_voxlock")
        log.info("enrolled profile %r (self-sim %.3f, isolation: %s)", meta["name"],
                 meta.get("self_similarity", 0), self.voxlock.effective_mode())
        self.emit_state()
        return meta

    def delete_profile(self, name: str) -> bool:
        if self.voxlock is None:
            return False
        removed = self.voxlock.delete_profile(name)
        if self.cfg.get("voxlock.active_profile") == name:
            self.cfg.set("voxlock.active_profile", None)
        self.emit_state()
        return removed

    # -- queries ------------------------------------------------------------------ #
    def history_items(self) -> list:
        return self.history.last(int(self.cfg.get("history.max_items", 50) or 50))

    def clear_history(self) -> None:
        self.history.clear()
        self.emit({"type": "history_cleared"})

    def status(self) -> dict:
        return {
            "listening": self.listening,
            "standby": self.standby,
            "models_loaded": self.models_loaded,
            "load_error": self.load_error,
            "device": self.device_name,
            "target_title": self.injector.target_title,
            "target_valid": self.injector.is_target_valid(),
            "inject_enabled": bool(self.cfg.get("injection.enabled", True)),
            "translate": bool(self.cfg.get("translation.enabled", False)),
            "voxlock": bool(self.cfg.get("voxlock.enabled", True)),
            "voxlock_threshold": float(self.cfg.get("voxlock.threshold", 0.62)),
            "isolation_mode": self.cfg.get("voxlock.isolation_mode") or "off",
            "isolation_effective": self.voxlock.effective_mode() if self.voxlock else None,
            "profile": self.voxlock.active_profile() if self.voxlock else self.cfg.get("voxlock.active_profile"),
            "llm_polish": bool(self.cfg.get("llm_polish_enabled", False)),
            "llm_endpoint": self.cfg.get("llm_endpoint"),
            "version": APP_VERSION,
        }

    def emit_state(self) -> None:
        self.emit({"type": "state", **self.status()})

    # -- lifecycle ----------------------------------------------------------------- #
    def shutdown(self) -> None:
        self.stop_flag = True
        self.stop_listening()
        self.emit({"type": "bye"})

    def cleanup(self) -> None:
        self.stop_flag = True
        if self.capture:
            self.capture.stop()
        if self.zipformer:
            self.zipformer.stop()
        self.history.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def build_app(pipeline: Pipeline) -> FastAPI:
    hub = Hub()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()
        hub.attach(loop)
        pipeline.emit = hub.emit_threadsafe
        tasks = [asyncio.create_task(hub.broadcaster()),
                 asyncio.create_task(_telemetry_loop(hub, pipeline))]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()

    app = FastAPI(title="VoxStream Backend", version=APP_VERSION,
                  docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "service": "voxstream-backend",
            "version": APP_VERSION,
            "models_loaded": pipeline.models_loaded,
            "listening": pipeline.listening,
            "standby": pipeline.standby,
            "device": pipeline.device_name,
            "load_error": pipeline.load_error,
        })

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        await hub.direct(ws, {"type": "state", **pipeline.status()})
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                await _handle(ws, hub, pipeline, msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.debug("ws handler exited", exc_info=True)
        finally:
            hub.clients.discard(ws)

    return app


async def _telemetry_loop(hub: Hub, pipeline: Pipeline) -> None:
    while True:
        await asyncio.sleep(2.0)
        try:
            proc = psutil.Process()
            pending = pipeline._seg_q.qsize()
            hub.emit_threadsafe({
                "type": "telemetry",
                "cpu": psutil.cpu_percent(interval=None),
                "ram": proc.memory_info().rss / (1024 * 1024),
                "pending_segments": pending,
                "whisper_rtf": pipeline._whisper_rtf,
                "llm_timeouts": pipeline.polisher.timeouts if pipeline.polisher else 0,
                "llm_errors": pipeline.polisher.errors if pipeline.polisher else 0,
                "isolation_mode": pipeline.voxlock.effective_mode() if pipeline.voxlock else None,
                "uptime_s": round(time.time() - pipeline._started_at, 1),
            })
        except Exception:
            continue


async def _handle(ws: WebSocket, hub: Hub, pipeline: Pipeline, msg: dict) -> None:
    mtype = msg.get("type")

    if mtype == "ping":
        await hub.direct(ws, {"type": "pong"})
    elif mtype == "get_state":
        await hub.direct(ws, {"type": "state", **pipeline.status()})
    elif mtype == "start":
        pipeline.start_listening()
    elif mtype == "stop":
        pipeline.stop_listening()
    elif mtype == "set_target":
        try:
            hwnd = int(msg.get("hwnd"))
        except (TypeError, ValueError):
            await hub.direct(ws, {"type": "error", "message": "set_target: invalid hwnd"})
            return
        pipeline.set_target(hwnd, msg.get("title"))
    elif mtype == "clear_target":
        pipeline.clear_target()
    elif mtype == "set_inject":
        pipeline.set_injection(bool(msg.get("enabled", True)))
    elif mtype == "set_translate":
        pipeline.set_translation(bool(msg.get("enabled", False)))
    elif mtype == "set_voxlock":
        pipeline.set_voxlock(msg.get("enabled"), msg.get("threshold"))
    elif mtype == "set_isolation":
        pipeline.set_isolation(str(msg.get("mode") or "off"))
    elif mtype == "set_llm":
        pipeline.set_llm_polish(bool(msg.get("enabled", False)))
    elif mtype == "refresh_settings":
        pipeline.apply_settings_runtime()
    elif mtype == "set_profile":
        pipeline.set_profile(msg.get("name"))
    elif mtype == "enroll":
        loop = asyncio.get_running_loop()
        try:
            meta = await loop.run_in_executor(
                None, lambda: pipeline.enroll(str(msg.get("name") or "me"),
                                              str(msg.get("wav_path"))))
            hub.emit_threadsafe({"type": "enroll_result", "ok": True, "profile": meta})
        except Exception as exc:
            hub.emit_threadsafe({"type": "enroll_result", "ok": False,
                                 "message": f"enrollment failed: {exc}"})
    elif mtype == "delete_profile":
        pipeline.delete_profile(str(msg.get("name") or ""))
    elif mtype == "get_history":
        await hub.direct(ws, {"type": "history", "items": pipeline.history_items()})
    elif mtype == "clear_history":
        pipeline.clear_history()
    elif mtype == "shutdown":
        pipeline.shutdown()
    else:
        await hub.direct(ws, {"type": "error", "message": f"unknown message type: {mtype!r}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VoxStream backend server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-models", action="store_true",
                        help="light mode: skip model loading (used by tests)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr),
                  logging.FileHandler(LOGS_DIR / "backend.log", encoding="utf-8")])

    host = args.host or CONFIG.get("server.host") or "127.0.0.1"
    port = int(args.port or CONFIG.get("server.port") or 8765)

    pipeline = Pipeline(CONFIG, load_models=not args.no_models)
    pipeline.begin_load()
    app = build_app(pipeline)

    server = uvicorn.Server(uvicorn.Config(
        app, host=host, port=port, log_level=args.log_level,
        lifespan="on", access_log=False))
    thread = threading.Thread(target=server.run, name="voxstream-uvicorn", daemon=True)
    thread.start()

    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        log.error("uvicorn failed to start on %s:%s", host, port)
        return 1
    log.info("VoxStream backend %s listening on http://%s:%d", APP_VERSION, host, port)
    sys.stdout.flush()

    try:
        while not pipeline.stop_flag:
            time.sleep(0.3)
    finally:
        log.info("shutting down")
        server.should_exit = True
        thread.join(timeout=10)
        pipeline.cleanup()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
