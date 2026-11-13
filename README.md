# VoxStream

<p align="center"><b>Free · Unlimited · 100% offline voice-to-text for Windows 10/11</b><br/>
Always-on dictation that survives tab &amp; app switches — typed straight into
any window, isolated to <i>your</i> voice, transcribed and translated entirely on your CPU.</p>

## What problem does VoxStream solve?

Cloud dictation dies the moment you switch tabs, uploads your voice, and
locks you into quotas. VoxStream is a **local, always-on dictation engine**:

- **Dictate across app switches** — the floating HUD never steals focus
  (`WS_EX_NOACTIVATE`), and a locked target window is typed into *in the
  background* via Win32 (`WM_CHAR` / `KEYEVENTF_UNICODE` / privacy clipboard),
  so you can keep reading docs while talking into Notepad.
- **Only your voice** — VoxLock isolates a single enrolled speaker:
  - `Only Me (Fast)`: TitaNet voiceprint gate rejects other speakers per utterance.
  - `Only Me (Overlap)`: VoxLock ONNX target-speaker extraction strips
    **simultaneous overlapping voices out of the audio waveform itself**
    before ASR ever sees them.
- **Live translation** — any language → English finals via Whisper
  (`task="translate"`), all on-device.
- **Private by construction** — after the one-time model download the app
  runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`: zero network calls,
  no telemetry, no accounts. Optional AI polish talks only to your own local
  LLM (Ollama / LM Studio) with a hard 1.5 s fallback timeout.

## Quick start

### End users (no Python needed)
1. Download `VoxStream-v2.0-win64.zip` from
   [Releases](../../releases) and extract it anywhere.
2. Run **`VoxStream.exe`** → the onboarding wizard walks you through the mic
   check, the 5-second "Only Me" voiceprint, and the hotkey demo.
3. Click into any text field and talk. `Ctrl+Shift+Space` toggles the mic
   globally; the tray icon has everything (target locking, isolation mode,
   settings, translation, history).

### Developers
```bat
py -3.11 -m venv .venv            (or: uv venv --python 3.11 .venv)
.venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py            :: one-time download of all offline models
python app.py                     :: dev launch (or start_voxstream.bat, silent)
```

### Build the standalone release
```bat
build_windows.bat                 :: venv + deps + PyInstaller -> dist\VoxStream\
release_package.bat               :: -> release\VoxStream-v2.0-win64.zip
```

## Architecture

```
mic → WASAPI native-rate capture (48 kHz) → polyphase resample 16 kHz (30 ms frames)
   └─(Only Me · Overlap)→ VoxLock OnlineSpeechDenoiser ─ strips other speakers from the waveform
       ├─→ streaming Zipformer thread (int8) ──────────────────────► HUD live partials
       └─→ Silero VAD segments ─► isolation policy (fast gate / overlap / anyone)
             └─► faster-whisper INT8 (final text / X → English translate)
                   └─► optional local-LLM polish (hard 1.5 s timeout fallback)
                         ├─► boundary-anchored voice commands + regex jargon table
                         ├─► SQLite history (last 50 utterances)
                         └─► Win32 injector → locked HWND or active cursor
backend.server (FastAPI :8765)  ⇄  WebSocket  ⇄  PySide6 UI (HUD · tray · wizard · settings)
```

| Module | Role |
|---|---|
| `app.py` | entry point: offline env, sidecar spawn (incl. frozen `--run-backend`), HUD, tray, hotkey, first-run wizard |
| `backend/server.py` | FastAPI + WS control/telemetry + realtime pipeline glue |
| `backend/audio_capture.py` | WASAPI native-rate capture, polyphase resampler, S0 standby watchdog |
| `backend/voice_isolation.py` | Silero VAD + VoxLock (fast gate, overlap extractor, 3 modes) |
| `backend/llm_polish.py` | optional local-LLM polish with mandatory 1.5 s hard timeout |
| `backend/asr_engine.py` | StreamingZipformer + WhisperFinalizer |
| `backend/win32_injector.py` | 3-tier injection (background `WM_CHAR` → unicode → privacy clipboard) |
| `backend/commands.py` | boundary-anchored commands + custom dictionary |
| `backend/history_store.py` | SQLite + ring buffer (last 50 utterances) |
| `backend/startup_registry.py` | "Start with Windows" HKCU Run-key helper (test-safe) |
| `ui/` | floating HUD, window picker, enrollment dialog, onboarding wizard, settings dialog, tray/hotkey |

## Hotkeys & voice commands

| Input | Action |
|---|---|
| `Ctrl+Shift+Space` (customizable) | toggle listening globally |
| "stop listening" / "mic off" | stop the mic |
| "start listening" / "mic on" | resume the mic |
| "new paragraph <dictation>" | paragraph break + dictation |
| "new line <dictation>" | line break + dictation |
| "scratch that" | delete the last injected utterance (backspaces) |
| "pick window" / "choose window" | open the crosshair target picker |
| "translate mode" | toggle live X → English translation |
| tray → Start with Windows | launch at logon (HKCU Run key) |

## Performance (Intel Core i5-11th-gen, 4C/8T, CPU-only, INT8)

| Stage | Real-time factor (RTF) |
|---|---|
| Streaming Zipformer partials (int8) | **0.06 – 0.10** (≈16× faster than realtime) |
| faster-whisper small finals (int8) | 0.75 – 1.12 per utterance |
| VoxLock overlap extraction (400 ms blocks) | 0.64 – 0.90 |
| VoxLock embedding gate (TitaNet) | ~0.02 per segment |
| LLM polish hard timeout | ≤ 1.5 s (then raw ASR text) |
| RAM (backend + UI, models loaded) | ≈ 1.5 GB |

Measured on: Windows 11 Pro, Python 3.11, sherpa-onnx 2.0.1, faster-whisper 1.2.1.

## Verification

```bat
.venv\Scripts\python -m pytest tests\test_pipeline.py -v   :: 21-section suite
python setup_models.py --verify-only                       :: model staging check
```

## Repository layout

```
voxstream/
├── app.py                  # main entry point
├── config.json             # persistent settings (created on first run)
├── requirements.txt
├── setup_models.py         # one-time offline model staging
├── start_voxstream.bat     # silent one-click launcher
├── build_windows.bat       # standalone PyInstaller build
├── release_package.bat     # GitHub release zip packager
├── VoxStream.spec
├── backend/                # pipeline + FastAPI sidecar
├── ui/                     # PySide6 HUD, wizard, settings, tray
└── tests/test_pipeline.py  # automated verification suite
```

`models/`, `profiles/`, `logs/`, `data/`, `dist/`, `build/`, `release/` are
gitignored — the repository stays pure source (< 2 MB).

## License & privacy

Runs 100% locally (sherpa-onnx, faster-whisper, silero-vad — all free,
permissive licenses). Your voice never leaves the machine; the optional
polish hook talks only to the endpoint you configure (default: your own
Ollama on `127.0.0.1`).
