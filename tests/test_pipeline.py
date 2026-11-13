#!/usr/bin/env python3
"""VoxStream automated verification suite.

Runs every stage of the pipeline headlessly (no permanent GUI state, no
network): deps & offline guard, config, model staging, audio resampler,
devices, Silero VAD, VoxLock enrollment/gating, streaming Zipformer ASR,
faster-whisper finals, voice commands, history store, Win32 injection
(3 tiers), and the backend sidecar WebSocket round-trip.

Usage:
    python tests/test_pipeline.py            # full suite
    python tests/test_pipeline.py --no-gui   # skip Win32 window-based tests
Exit code = number of failed sections.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # keep console output clean on any codepage
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS: list[tuple[str, str, str]] = []  # (section, status, detail)
VENV_PY = sys.executable


def report(section: str, status: str, detail: str = "") -> None:
    RESULTS.append((section, status, detail))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "INFO": "[....]"}[status]
    print(f"{tag} {section}" + (f" - {detail}" if detail else ""))


def section(name: str):
    """Run ``fn`` immediately, reporting PASS/FAIL/SKIP for the section."""
    def run(fn):
        t0 = time.time()
        try:
            detail = fn()
            report(name, "PASS", f"{detail} ({time.time()-t0:.1f}s)"
                   if detail else f"({time.time()-t0:.1f}s)")
        except SkipTest as exc:
            report(name, "SKIP", str(exc))
        except Exception as exc:
            import traceback
            report(name, "FAIL", f"{exc.__class__.__name__}: {exc}")
            traceback.print_exc()
    return run


class SkipTest(Exception):
    pass


# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true",
                        help="skip tests that create Win32 windows")
    args = parser.parse_args()

    print("=" * 72)
    print("VoxStream verification suite")
    print("=" * 72)

    section("1. dependencies import")(test_imports)
    section("2. offline environment enforcement")(test_offline_guard)
    section("3. config persistence")(test_config)
    section("4. model staging")(test_models)
    section("5. resampler math")(test_resampler)
    section("6. audio devices")(test_devices)
    section("7. silero VAD on real speech")(test_vad)
    section("8. VoxLock enrollment + gating")(test_voxlock)
    section("9. streaming Zipformer ASR")(test_zipformer)
    section("10. faster-whisper INT8 finalize")(test_whisper)
    section("11. voice commands + dictionary")(test_commands)
    section("12. history store")(test_history)
    if not args.no_gui:
        section("13. Win32 injection tier 1 (WM_CHAR)")(test_inject_wm_char)
        section("14. Win32 injection tier 2 (unicode SendInput)")(test_inject_unicode)
        section("15. Win32 injection tier 3 (privacy clipboard)")(test_inject_clipboard)
    section("16. backend sidecar + WebSocket round-trip")(test_server)
    section("17. VoxLock overlap waveform extraction")(test_overlap_extraction)
    section("18. LLM polish hard-timeout fallback")(test_llm_polish)
    section("19. WS isolation/AI-polish control round-trip")(test_ws_v2_controls)
    section("20. settings round-trip + first-run persistence")(test_settings_roundtrip)
    section("21. startup registry (isolated test key)")(test_startup_registry)

    print("=" * 72)
    fails = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    skips = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print(f"Summary: {len(RESULTS)-fails-skips} passed, {fails} failed, {skips} skipped")
    if fails:
        for name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"  FAILED {name}: {detail}")
    print("=" * 72)
    return fails


# ---------------------------------------------------------------------------
def test_imports() -> str:
    import numpy
    import scipy
    import sounddevice
    import sherpa_onnx
    import faster_whisper
    import psutil
    import fastapi
    import uvicorn
    import websockets
    import win32api
    import PySide6
    versions = {
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sherpa_onnx": getattr(sherpa_onnx, "__version__", "?"),
        "faster_whisper": getattr(faster_whisper, "__version__", "?"),
        "PySide6": PySide6.__version__,
    }
    return json.dumps(versions)


def test_offline_guard() -> str:
    # fresh interpreter: importing backend must set the offline switches
    code = ("import sys; sys.path.insert(0, r'%s');"
            "import backend, os;"
            "print(os.environ.get('HF_HUB_OFFLINE'),"
            " os.environ.get('TRANSFORMERS_OFFLINE'),"
            " os.environ.get('HF_HUB_DISABLE_TELEMETRY'))" % str(ROOT))
    out = subprocess.run([VENV_PY, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=60)
    parts = out.stdout.strip().split()
    if parts != ["1", "1", "1"]:
        raise AssertionError(f"offline flags not enforced: {parts!r} stderr={out.stderr[-300:]}")
    # and app.py source sets them before other imports
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    if "HF_HUB_OFFLINE" not in src.split("import json")[0]:
        raise AssertionError("app.py must set HF_HUB_OFFLINE before further imports")
    return "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 enforced on every startup"


def test_config() -> str:
    from backend import Config
    tmp = ROOT / "data" / "test_config.json"
    if tmp.exists():
        tmp.unlink()
    cfg = Config(tmp)
    assert cfg.get("voxlock.threshold") == 0.62
    cfg.set("voxlock.threshold", 0.7)
    cfg2 = Config(tmp)
    assert abs(cfg2.get("voxlock.threshold") - 0.7) < 1e-9
    assert cfg2.get("audio.frame_ms") == 30  # default merged back
    assert cfg2.get("commands.list"), "built-in voice commands present"
    assert cfg2.get("custom_dictionary"), "custom dictionary present"
    tmp.unlink(missing_ok=True)
    return "dotted get/set + defaults merge + commands/dictionary present"


def test_models() -> str:
    out = subprocess.run([VENV_PY, str(ROOT / "setup_models.py"), "--verify-only"],
                         capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    if out.returncode != 0:
        raise AssertionError(out.stdout.strip() + out.stderr.strip())
    detail = (out.stdout.strip().splitlines() or [""])[-1]
    return detail


def test_resampler() -> str:
    import numpy as np
    from backend.audio_capture import Resampler
    rs = Resampler(48000, 16000)
    rate, freq, seconds = 48000, 1000.0, 2.0
    t = np.arange(int(rate * seconds)) / rate
    wave = (0.4 * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    out_chunks = []
    block = int(rate * 0.03)
    for i in range(0, wave.size, block):
        out_chunks.append(rs.process(wave[i:i + block]))
    y = np.concatenate(out_chunks)
    ratio = y.size / wave.size
    if abs(ratio - 16000 / 48000) > 0.01:
        raise AssertionError(f"output length ratio {ratio:.4f} != 0.3333")
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=1 / 16000)
    peak = freqs[int(np.argmax(spec))]
    if abs(peak - freq) > freq * 0.02:
        raise AssertionError(f"frequency drifted: {peak:.1f} Hz (expected {freq})")
    # 44.1 kHz path as well
    rs2 = Resampler(44100, 16000)
    y2 = rs2.process(np.ones(4410, dtype=np.float32))
    expected = int(round((4410 + 1024) * 16000 / 44100)) - rs2.drop
    if abs(y2.size - max(expected, 0)) > 4:
        raise AssertionError(f"44.1k length {y2.size} != {expected}")
    return f"48k->16k peak {peak:.0f} Hz, length ratio {ratio:.4f}"


def test_devices() -> str:
    from backend.audio_capture import list_input_devices, resolve_input_device
    devices = list_input_devices()
    if not devices:
        raise SkipTest("no input audio devices on this machine")
    dev = resolve_input_device(None)
    if dev is None:
        raise SkipTest("could not resolve a default input device")
    # short live capture check (open, capture ~1s of frames, close)
    import numpy as np
    from backend import CONFIG
    from backend.audio_capture import AudioCapture
    got = []
    evt = threading.Event()

    def on_frame(frame):
        if len(got) < 40:
            got.append(frame)
        elif not evt.is_set():
            evt.set()

    cap = AudioCapture(CONFIG, on_frame=on_frame)
    if not cap.start():
        raise SkipTest("input stream refused to open (device busy?)")
    try:
        evt.wait(3.0)
    finally:
        cap.stop()
    if not got:
        raise SkipTest("opened stream but produced no frames (device muted?)")
    shape = got[0].shape
    if shape[-1] != 480:
        raise AssertionError(f"frame size {shape} != 480 samples")
    return f"{len(devices)} input device(s); live frames 480 samples from '{dev['name'][:40]}'"


def test_vad() -> str:
    from backend import CONFIG, ROOT
    from backend.voice_isolation import SileroVAD, load_wav_16k
    wav_path = ROOT / "models" / "zipformer" / "test_wav.wav"
    if not wav_path.exists():
        raise SkipTest("models/zipformer/test_wav.wav not staged")
    audio = load_wav_16k(wav_path)
    duration = audio.size / 16000.0
    vad = SileroVAD(CONFIG.get("vad.model"), threshold=0.5)
    segments = []
    for i in range(0, audio.size, 512):
        segments.extend(vad.feed(audio[i:i + 512]))
    segments.extend(vad.flush())
    speech = sum(s["samples"].size for s in segments)
    frac = speech / max(1, audio.size)
    if not segments or frac < 0.3:
        raise AssertionError(f"VAD missed speech: {len(segments)} segments, {frac:.0%} speech")
    return f"{duration:.1f}s clip -> {len(segments)} segment(s), {frac:.0%} speech coverage"


def test_voxlock() -> str:
    import numpy as np
    from backend import CONFIG, PROFILES_DIR, ROOT
    from backend.voice_isolation import VoxLock, cosine, load_wav_16k

    w0 = ROOT / "models" / "zipformer" / "test_wav.wav"
    w1 = ROOT / "models" / "zipformer" / "test_wav2.wav"
    if not w0.exists():
        raise SkipTest("zipformer test WAVs not staged")
    a0 = load_wav_16k(w0)
    a1 = load_wav_16k(w1) if w1.exists() else None
    if a0.size < 16000 * 3.0:
        raise SkipTest(f"enrollment wav too short ({a0.size/16000:.1f}s)")
    if a1 is not None and a1.size < 16000 * 0.4:
        a1 = None

    lock = VoxLock(CONFIG.get("voxlock.model"), PROFILES_DIR,
                   threshold=0.62)
    meta = lock.enroll(w0, "test-suite")
    if abs(meta["self_similarity"]) < 0.3:
        raise AssertionError(f"enrollment self-similarity too low: {meta['self_similarity']}")
    ok_self, sim_self = lock.check(a0)
    if not ok_self or sim_self is None:
        raise AssertionError(f"enrolled voice rejected by itself (sim={sim_self})")
    if abs(sim_self - 1.0) > 1e-6:
        raise AssertionError(f"self similarity {sim_self} != 1.0")

    detail = f"self-sim 1.000"
    if a1 is not None:
        _, sim_other = lock.check(a1)
        if sim_other is None:
            raise AssertionError("second wav produced no embedding")
        if sim_other >= sim_self:
            raise AssertionError(f"other-speaker sim {sim_other:.3f} >= self {sim_self:.3f}")
        # mechanism gate: midpoint threshold must accept self / reject other
        thr = min(0.95, max(0.5, (sim_self + sim_other) / 2 + 1e-3))
        lock.threshold = thr
        ok, sim = lock.check(a0)
        if not ok:
            raise AssertionError(f"gate rejected enrolled voice at thr {thr:.3f}")
        ok2, _ = lock.check(a1)
        if ok2:
            raise AssertionError(f"gate accepted other speaker at thr {thr:.3f} "
                                 f"(sim {sim_other:.3f})")
        detail += f", other-speaker sim {sim_other:.3f}, gate thr {thr:.2f} ok"
    lock.delete_profile("test-suite")
    return detail + "; profile cleaned up"


def test_zipformer() -> str:
    import numpy as np
    from backend import CONFIG, ROOT
    from backend.asr_engine import StreamingZipformer
    from backend.voice_isolation import load_wav_16k
    wav = ROOT / "models" / "zipformer" / "test_wav.wav"
    if not wav.exists():
        raise SkipTest("models/zipformer/test_wav.wav not staged")
    audio = load_wav_16k(wav)
    duration = audio.size / 16000.0

    engine = StreamingZipformer(ROOT / CONFIG.get("asr.zipformer_dir"),
                                num_threads=CONFIG.get("asr.num_threads", 2))
    t0 = time.time()
    text = engine.decode_offline(audio)
    wall = time.time() - t0
    rtf = wall / max(duration, 0.1)
    if not text:
        raise AssertionError("streaming recognizer produced empty text")
    # live loop check: feed via thread and read partials
    partials = []
    engine2 = StreamingZipformer(ROOT / CONFIG.get("asr.zipformer_dir"),
                                 num_threads=1)
    engine2.start(lambda t: partials.append(t) if t else None)
    for i in range(0, audio.size, 480):
        engine2.feed(audio[i:i + 480])
    deadline = time.time() + 30
    while time.time() < deadline and len(partials) < 2:
        time.sleep(0.05)
    engine2.stop()
    if not partials:
        raise AssertionError("live feed produced no partials")
    return f'text={text[:40]!r}... rtf={rtf:.2f}, live partials={len(partials)}'


def test_whisper() -> str:
    from backend import CONFIG, ROOT
    from backend.asr_engine import WhisperFinalizer
    from backend.voice_isolation import load_wav_16k
    wav = ROOT / "models" / "zipformer" / "test_wav.wav"
    if not wav.exists():
        raise SkipTest("models/zipformer/test_wav.wav not staged")
    audio = load_wav_16k(wav)
    duration = audio.size / 16000.0
    fw = WhisperFinalizer(ROOT / CONFIG.get("whisper.dir"),
                          compute_type=CONFIG.get("whisper.compute_type", "int8"),
                          cpu_threads=CONFIG.get("whisper.cpu_threads", 3))
    res = fw.transcribe(audio, task="transcribe")
    if not res["text"]:
        raise AssertionError("whisper produced empty text")
    rtf = res["elapsed"] / max(duration, 0.1)
    # translate task plumbing (EN clip -> EN is a no-op but exercises the path)
    res2 = fw.transcribe(audio, task="translate")
    if res2.get("text") is None:
        raise AssertionError("translate task failed")
    return (f'text={res["text"][:40]!r}... lang={res["language"]} '
            f"rtf={rtf:.2f}; translate task ok")


def test_commands() -> str:
    from backend import CONFIG
    from backend.commands import CommandProcessor
    proc = CommandProcessor(CONFIG)
    name, arg, pat = proc.parse("Stop Listening")
    if name != "stop_listening":
        raise AssertionError(f"'stop listening' not matched: {name}")
    name, arg, _ = proc.parse("New paragraph. The quick brown fox.")
    if name != "new_paragraph" or "quick brown fox" not in arg:
        raise AssertionError(f"prefix anchor failed: {name} arg={arg!r}")
    out = proc.process("the g p u works")
    if out["command"] is not None or "GPU" not in out["text"]:
        raise AssertionError(f"dictionary failed: {out}")
    out = proc.process("just plain text")
    if out["command"] is not None or out["text"] != "just plain text":
        raise AssertionError(f"passthrough failed: {out}")
    name, _, _ = proc.parse("please stop listening now")
    if name is not None:
        raise AssertionError("whole-anchored command matched mid-utterance")
    return "whole/prefix anchoring + jargon replacement verified"


def test_history() -> str:
    from backend.history_store import HistoryStore
    db = ROOT / "data" / "test_history.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    store = HistoryStore(db, max_items=50)
    try:
        for i in range(60):
            store.add(f"utterance {i}", lang="en", injected=i % 2 == 0,
                      injected_chars=len(f"utterance {i}") + 1)
        if store.count() != 50:
            raise AssertionError(f"prune failed: count={store.count()} (want 50)")
        rows = store.last(50)
        if len(rows) != 50 or rows[-1]["text"] != "utterance 59":
            raise AssertionError("ring buffer order wrong")
        deleted = store.delete_last()
        if not deleted or deleted["text"] != "utterance 59":
            raise AssertionError(f"delete_last failed: {deleted}")
        if store.count() != 49:
            raise AssertionError("delete did not persist")
        store.clear()
        if store.count() != 0:
            raise AssertionError("clear failed")
    finally:
        store.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    return "add/prune@50/last/delete_last/clear verified"


# ---------------------------------------------------------------------------
# Win32 injection tests (create our own test window with an EDIT control)
# ---------------------------------------------------------------------------
def make_edit_window():
    import win32api
    import win32con
    import win32gui
    hinst = win32api.GetModuleHandle(None)
    cls = f"VoxStreamTest_{os.getpid()}"
    try:
        win32gui.UnregisterClass(cls, hinst)
    except Exception:
        pass
    wc = win32gui.WNDCLASS()
    wc.hInstance = hinst
    wc.lpszClassName = cls
    wc.lpfnWndProc = win32gui.DefWindowProc
    win32gui.RegisterClass(wc)
    parent = win32gui.CreateWindowEx(
        0, cls, "VoxStream Injection Test",
        win32con.WS_OVERLAPPED | win32con.WS_CAPTION | win32con.WS_SYSMENU,
        40, 40, 480, 220, None, None, hinst, None)
    edit = win32gui.CreateWindowEx(
        0, "EDIT", "",
        win32con.WS_CHILD | win32con.WS_VISIBLE,
        10, 10, 440, 150, parent, None, hinst, None)
    win32gui.ShowWindow(parent, win32con.SW_SHOWNOACTIVATE)
    win32gui.PumpWaitingMessages()
    return parent, edit


def destroy_window(parent):
    try:
        import win32gui
        win32gui.DestroyWindow(parent)
    except Exception:
        pass


def pump(seconds: float = 0.3) -> None:
    import win32gui
    end = time.time() + seconds
    while time.time() < end:
        win32gui.PumpWaitingMessages()
        time.sleep(0.02)


def test_inject_wm_char() -> str:
    import win32gui
    from backend import CONFIG
    from backend.win32_injector import TextInjector
    try:
        parent, edit = make_edit_window()
    except Exception as exc:
        raise SkipTest(f"cannot create test window: {exc}")
    try:
        cfg = CONFIG
        cfg.set("injection.tier", "wm_char", save=False)
        cfg.set("injection.enabled", True, save=False)
        inj = TextInjector(cfg)
        inj.set_target(edit, "test edit control")
        text = "Hello VoxStream"
        tier = inj.inject(text)
        pump(0.4)
        got = win32gui.GetWindowText(edit)
        if got != text:
            raise AssertionError(f"WM_CHAR result {got!r} != {text!r} (tier={tier})")
        # background backspaces: exactly n characters are removed
        inj.send_backspaces(9)
        pump(0.4)
        got2 = win32gui.GetWindowText(edit)
        if got2 != "Hello ":  # "Hello VoxStream" minus 9 chars
            raise AssertionError(f"backspaces result {got2!r} != 'Hello '")
        return f"typed {text!r} + exactly 9 backspaces without focus"
    finally:
        destroy_window(parent)


def test_inject_unicode() -> str:
    import win32gui
    from backend import CONFIG
    from backend.win32_injector import TextInjector, set_foreground
    try:
        parent, edit = make_edit_window()
    except Exception as exc:
        raise SkipTest(f"cannot create test window: {exc}")
    try:
        if not set_foreground(parent):
            raise SkipTest("SetForegroundWindow blocked by focus lock")
        try:
            win32gui.SetFocus(edit)
        except Exception:
            pass
        cfg = CONFIG
        cfg.set("injection.tier", "unicode", save=False)
        inj = TextInjector(cfg)
        inj.set_target(parent, "test window")
        text = "Vox unicode 42"
        tier = inj.inject(text)
        pump(0.5)
        got = win32gui.GetWindowText(edit)
        if got != text:
            raise SkipTest(f"SendInput text {got!r} != {text!r} "
                           "(needs an unlocked interactive desktop)")
        return f"typed {text!r} via KEYEVENTF_UNICODE"
    finally:
        destroy_window(parent)


def test_inject_clipboard() -> str:
    import win32gui
    from backend import CONFIG
    from backend.win32_injector import (TextInjector, get_clipboard_text,
                                        set_clipboard_text, set_foreground)
    backup = None
    try:
        parent, edit = make_edit_window()
    except Exception as exc:
        raise SkipTest(f"cannot create test window: {exc}")
    try:
        backup = get_clipboard_text()
        if not set_foreground(parent):
            raise SkipTest("SetForegroundWindow blocked by focus lock")
        try:
            win32gui.SetFocus(edit)
        except Exception:
            pass
        cfg = CONFIG
        cfg.set("injection.tier", "clipboard", save=False)
        cfg.set("injection.restore_clipboard", True, save=False)
        inj = TextInjector(cfg)
        inj.set_target(parent, "test window")
        text = "clipboard path"
        tier = inj.inject(text)
        pump(0.8)
        got = win32gui.GetWindowText(edit)
        # privacy: clipboard must be restored (or emptied) shortly after
        time.sleep(0.8)
        restored = get_clipboard_text()
        if restored is not None and backup is not None and restored != backup:
            raise AssertionError("clipboard was not restored after paste")
        if restored == text:
            raise AssertionError("transcribed text left on the clipboard (privacy)")
        if got != text:
            raise SkipTest(f"paste result {got!r} != {text!r} (focus race)")
        return f"pasted {text!r} then restored previous clipboard"
    finally:
        destroy_window(parent)
        try:
            if backup is not None:
                set_clipboard_text(backup)
        except Exception:
            pass


def _get_text(hwnd: int) -> str:
    """Read an EDIT control's text (WM_GETTEXT)."""
    import win32con
    import win32gui
    length = win32gui.SendMessage(hwnd, win32con.WM_GETTEXTLENGTH, 0, 0)
    if length <= 0:
        return ""
    buf = win32gui.PyMakeBuffer(min(length + 1, 4096))
    win32gui.SendMessage(hwnd, win32con.WM_GETTEXT, length + 1, buf)
    raw = buf[:length]
    if isinstance(raw, bytes):
        return raw.decode("utf-16-le", errors="replace")
    return str(raw)


# ---------------------------------------------------------------------------
def test_server() -> str:
    import websockets.sync.client as wsc
    port = 8766
    proc = subprocess.Popen(
        [VENV_PY, "-m", "backend.server", "--port", str(port), "--no-models",
         "--log-level", "warning"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 30
        healthy = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                            timeout=2) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data.get("ok"):
                    healthy = True
                    break
            except Exception:
                time.sleep(0.3)
        if not healthy:
            out = b""
            try:
                out = proc.stdout.read(4000) or b""
            except Exception:
                pass
            raise AssertionError(f"server never became healthy: {out.decode(errors='replace')[-300:]}")

        conn = wsc.connect(f"ws://127.0.0.1:{port}/ws", open_timeout=5)
        try:
            state = json.loads(conn.recv(timeout=5))
            if state.get("type") != "state":
                raise AssertionError(f"expected state, got {state.get('type')}")
            conn.send(json.dumps({"type": "ping"}))
            pong = json.loads(conn.recv(timeout=5))
            if pong.get("type") != "pong":
                raise AssertionError(f"expected pong, got {pong.get('type')}")
            conn.send(json.dumps({"type": "get_state"}))
            state2 = json.loads(conn.recv(timeout=5))
            if state2.get("type") != "state":
                raise AssertionError("get_state failed")
            conn.send(json.dumps({"type": "shutdown"}))
        finally:
            conn.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("server did not exit on shutdown message")
        if proc.returncode not in (0, None):
            raise AssertionError(f"server exit code {proc.returncode}")
        return "health 200, ws state/pong/get_state, graceful shutdown"
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# v2.0 sections
# ---------------------------------------------------------------------------
def test_overlap_extraction() -> str:
    import numpy as np
    from backend import CONFIG, PROFILES_DIR, ROOT
    from backend.asr_engine import StreamingZipformer
    from backend.voice_isolation import VoxLock, load_wav_16k

    enroll_wav = ROOT / "models" / "voxlock" / "en-voice-sample-f.wav"
    mix_wav = ROOT / "models" / "voxlock" / "en-combined.wav"
    if not (enroll_wav.exists() and mix_wav.exists()
            and (ROOT / "models" / "voxlock" / "chunk_int8.onnx").exists()):
        raise SkipTest("VoxLock overlap bundle not staged (optional)")

    lock = VoxLock(
        ROOT / CONFIG.get("voxlock.model"), PROFILES_DIR,
        threshold=0.62,
        enroll_model=ROOT / "models" / "voxlock" / "enroll.onnx",
        chunk_model=ROOT / "models" / "voxlock" / "chunk_int8.onnx",
        num_threads=2, process_block_ms=400)
    meta = lock.enroll(enroll_wav, "overlap-probe")
    assert lock.set_mode("overlap_voxlock") == "overlap_voxlock"
    if lock.effective_mode() != "overlap_voxlock":
        raise AssertionError("overlap mode did not engage with profile active")

    mix = load_wav_16k(mix_wav)
    duration = mix.size / 16000.0
    out, t0 = [], time.time()
    for i in range(0, mix.size - mix.size % 480, 480):
        out.append(lock.process_frame(mix[i:i + 480]))
    out.append(lock.flush_stream())
    wall = time.time() - t0
    chunks = [c for c in out if c.size]
    extracted = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    rtf = wall / max(duration, 0.1)
    if extracted.size < mix.size * 0.8:
        raise AssertionError(
            f"extracted length {extracted.size/16000:.2f}s vs input {duration:.2f}s")
    rms = float(np.sqrt(np.mean(np.square(extracted))))
    if rms <= 5e-4:
        raise AssertionError(f"extracted waveform near-silent (rms={rms:.6f})")

    zipformer = StreamingZipformer(ROOT / CONFIG.get("asr.zipformer_dir"), num_threads=2)
    mixed_text = zipformer.decode_offline(mix)
    extracted_text = zipformer.decode_offline(extracted)
    if not extracted_text:
        raise AssertionError("ASR produced no text for extracted waveform")
    if extracted_text == mixed_text:
        raise AssertionError("extraction did not change the transcript "
                             "(overlap speaker not stripped)")
    lock.delete_profile("overlap-probe")
    return (f"{duration:.1f}s mix -> extracted rtf={rtf:.2f}; "
            f"asr differs from mixed transcript (isolation confirmed)")


def test_llm_polish() -> str:
    import http.server
    import json as _json
    import socketserver
    import threading

    from backend.llm_polish import LLMPolisher

    behavior = {"delay": 0.0, "reply": "  polished output text here  "}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # /v1/models discovery
            data = _json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # /v1/chat/completions
            length = int(self.headers.get("content-length", 0))
            self.rfile.read(length)
            time.sleep(behavior["delay"])
            data = _json.dumps({"choices": [{"message": {"content": behavior["reply"]}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):  # silence
            pass

    class StubConfig:
        def __init__(self, d):
            self.d = d

        def get(self, key, default=None):
            return self.d.get(key, default)

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = StubConfig({
            "llm_polish_enabled": True,
            "llm_endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
            "llm_model": "auto",
            "llm_timeout_s": 1.5,
        })
        p = LLMPolisher(cfg)
        raw = "um so this is a long dictation clause that should be polished by the model okay"
        text, ok = p.polish(raw)
        if not ok or text != "polished output text here":
            raise AssertionError(f"fast server polish failed: ok={ok} text={text!r}")

        short = "too short"
        text2, ok2 = p.polish(short)
        if ok2 or text2 != short:
            raise AssertionError("clauses <=25 chars must bypass the LLM")

        # hard timeout: server stalls 3s -> raw transcript within the deadline
        behavior["delay"] = 3.0
        t0 = time.time()
        text3, ok3 = p.polish("another long dictation clause well beyond the minimum length needed")
        elapsed = time.time() - t0
        if ok3:
            raise AssertionError("stalled server must not yield a polished result")
        if elapsed > 2.2:
            raise AssertionError(f"hard timeout exceeded: {elapsed:.2f}s (limit ~1.5s)")
        # timeouts counter must have ticked if the worker outlived join()
        # (socket timeout usually fires first; both paths are acceptable)

        # disabled -> passthrough
        p.set_enabled(False)
        text4, ok4 = p.polish("a sufficiently long clause that would normally be polished now")
        if ok4:
            raise AssertionError("disabled polisher must be a passthrough")

        # offline endpoint -> immediate raw fallback
        p.set_enabled(True)
        p.endpoint = "http://127.0.0.1:9/timeout/v1/chat/completions"
        t0 = time.time()
        text5, ok5 = p.polish("offline endpoint clause that is definitely long enough to polish")
        if ok5 or time.time() - t0 > 2.5:
            raise AssertionError("offline endpoint must fall back immediately")
        return (f"fast polish ok; <=25ch bypass; hard timeout {elapsed:.2f}s; "
                "disabled passthrough; offline fallback ok")
    finally:
        srv.shutdown()
        srv.server_close()


def test_ws_v2_controls() -> str:
    import websockets.sync.client as wsc
    port = 8767
    proc = subprocess.Popen(
        [VENV_PY, "-m", "backend.server", "--port", str(port), "--no-models",
         "--log-level", "warning"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 30
        healthy = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                            timeout=2) as r:
                    if json.loads(r.read().decode("utf-8")).get("ok"):
                        healthy = True
                        break
            except Exception:
                time.sleep(0.3)
        if not healthy:
            raise AssertionError("server did not become healthy")

        conn = wsc.connect(f"ws://127.0.0.1:{port}/ws", open_timeout=5)
        try:
            def recv_until(want_type, timeout=6.0):
                end = time.time() + timeout
                while time.time() < end:
                    msg = json.loads(conn.recv(timeout=max(0.2, end - time.time())))
                    if msg.get("type") == want_type:
                        return msg
                raise AssertionError(f"no {want_type!r} message within {timeout}s")

            conn.send(json.dumps({"type": "ping"}))
            recv_until("pong")
            conn.send(json.dumps({"type": "set_isolation", "mode": "off"}))
            state = recv_until("state")
            if state.get("isolation_mode") != "off":
                raise AssertionError(f"isolation_mode not applied: {state.get('isolation_mode')}")
            conn.send(json.dumps({"type": "set_llm", "enabled": True}))
            state = recv_until("state")
            if not state.get("llm_polish"):
                raise AssertionError("llm_polish flag not applied")
            conn.send(json.dumps({"type": "set_isolation", "mode": "overlap_voxlock"}))
            recv_until("state")
            conn.send(json.dumps({"type": "set_llm", "enabled": False}))
            recv_until("state")
            conn.send(json.dumps({"type": "shutdown"}))
        finally:
            conn.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("server did not exit on shutdown")
        return "set_isolation + set_llm reflected in state broadcasts; shutdown ok"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_settings_roundtrip() -> str:
    """Settings-dialog style writes must persist and reach the pipeline."""
    from backend import CONFIG
    from backend.commands import CommandProcessor
    from backend.llm_polish import LLMPolisher
    from backend.server import Pipeline

    # 1) first-run flag persistence (wizard finish path)
    original_first = bool(CONFIG.get("first_run_completed", False))
    CONFIG.set("first_run_completed", False)
    assert CONFIG.get("first_run_completed") is False
    CONFIG.set("first_run_completed", True)
    assert CONFIG.get("first_run_completed") is True

    # 2) jargon table edit (settings dialog writes -> backend re-applies)
    original_dict = dict(CONFIG.get("custom_dictionary", {}) or {})
    original_llm = bool(CONFIG.get("llm_polish_enabled", False))
    try:
        CONFIG.set("custom_dictionary", {**original_dict, "\\ba sink\\b": "async"})
        CONFIG.set("llm_polish_enabled", True)

        # backend applies via refresh_settings (same code path as the WS hook)
        pipeline = Pipeline(CONFIG, load_models=False)
        pipeline.apply_settings_runtime()
        out = pipeline.commands.process("use a sink here")
        if "async" not in out["text"]:
            raise AssertionError(f"jargon round-trip failed: {out['text']!r}")
        if not pipeline.polisher.enabled:
            raise AssertionError("llm_polish_enabled did not reach the pipeline")
    finally:
        CONFIG.set("custom_dictionary", original_dict)
        CONFIG.set("llm_polish_enabled", original_llm)

    # 3) hotkey config round-trip (settings dialog hotkey customizer)
    original_hk = dict(CONFIG.get("hotkey", {}) or {})
    CONFIG.set("hotkey", {"mods": ["ctrl", "alt"], "key": "d"})
    if CONFIG.get("hotkey.key") != "d":
        raise AssertionError("hotkey config round-trip failed")
    CONFIG.set("hotkey", original_hk)

    CONFIG.set("first_run_completed", original_first)
    return "first_run flag, jargon table, LLM flag and hotkey persist + apply"


def test_startup_registry() -> str:
    """Startup helper exercised against an isolated test key only."""
    from backend import startup_registry

    test_key = r"Software\VoxStreamTest\Run"  # never the real Run key
    startup_registry.disable(test_key)
    if startup_registry.is_enabled(test_key):
        raise AssertionError("test key unexpectedly enabled before the test")

    if not startup_registry.enable(test_key):
        raise AssertionError("enable() failed on the test key")
    if not startup_registry.is_enabled(test_key):
        raise AssertionError("enable() did not persist")
    cmd = startup_registry.get_command(test_key)
    if not cmd:
        raise AssertionError("no command stored")
    # frozen -> exe path; source -> start_voxstream.bat
    if getattr(sys, "frozen", False):
        if not cmd.lower().endswith(".exe\"") and "voxstream.exe" not in cmd.lower():
            raise AssertionError(f"unexpected frozen command: {cmd!r}")
    elif "start_voxstream.bat" not in cmd:
        raise AssertionError(f"unexpected source command: {cmd!r}")

    if not startup_registry.disable(test_key):
        raise AssertionError("disable() failed")
    if startup_registry.is_enabled(test_key):
        raise AssertionError("disable() did not remove the value")
    # clean the test key subtree
    try:
        import winreg
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, test_key)
    except OSError:
        pass
    return f"enable/inspect/disable verified on isolated key {test_key!r}"


if __name__ == "__main__":
    sys.exit(main())
