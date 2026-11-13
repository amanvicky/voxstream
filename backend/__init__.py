"""VoxStream backend package.

Importing this package enforces the 100% offline runtime guarantee: every
Hugging Face / Transformers network switch is turned off *before* any model
library is imported, so the application can never issue an internet request
at runtime.  ``setup_models.py`` is the only place that intentionally
re-enables network access, and it deliberately does not import this package
while downloading.

It also hosts the shared ``Config`` singleton (config.json) so both the
backend process and the UI process read the exact same defaults, and the
frozen-executable (PyInstaller) resource resolution.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Offline enforcement -- MUST run before faster_whisper / transformers import
# ---------------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

APP_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Path resolution (dev tree vs PyInstaller frozen bundle)
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):  # running as a PyInstaller bundle
    EXE_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", EXE_DIR))
    BASE_DIR = EXE_DIR                    # writable location (logs/data/profiles)
else:
    EXE_DIR = Path(__file__).resolve().parent.parent
    BUNDLE_DIR = EXE_DIR
    BASE_DIR = EXE_DIR

ROOT = BASE_DIR  # alias used by tests/tools


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


MODELS_DIR = _first_existing(EXE_DIR / "models", BUNDLE_DIR / "models")
PROFILES_DIR = BASE_DIR / "profiles"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

for _d in (MODELS_DIR, PROFILES_DIR, DATA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def resource_path(rel: str | Path) -> Path:
    """Resolve a bundled read-only resource path (models etc.).

    Prefers a sibling of the executable (``<exe>/models/...``) so a release
    folder can ship models outside the bundle, then falls back to the
    PyInstaller payload (``_internal/models/...``), then to the dev-tree
    location for clear error messages.
    """
    rel = Path(rel)
    for base in (EXE_DIR, BUNDLE_DIR):
        candidate = base / rel
        if candidate.exists():
            return candidate
    return EXE_DIR / rel


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "server": {"host": "127.0.0.1", "port": 8765},
    "audio": {
        "input_device": None,        # None = system default, int index, or name substring
        "frame_ms": 30,
        "standby_after_s": 45.0,     # S0 standby watchdog: pure zero-signal for N seconds
        "zero_floor": 1e-5,          # peak amplitude below this counts as "digital zero"
    },
    "asr": {
        "zipformer_dir": "models/zipformer",
        "num_threads": 2,
        "decoding_method": "greedy_search",
        "source_language": None,     # whisper source language: None = auto-detect
    },
    "whisper": {
        "dir": "models/whisper/small",
        "compute_type": "int8",
        "cpu_threads": 3,
    },
    "vad": {
        "model": "models/silero_vad.onnx",
        "threshold": 0.5,
        "min_speech_s": 0.25,
        "min_silence_s": 0.6,
        "max_buffer_s": 30.0,
    },
    "voxlock": {
        "model": "models/voxlock/embedding.onnx",   # TitaNet embedding (fast gate)
        "enroll_model": "models/voxlock/enroll.onnx",   # VoxLock overlap bundle
        "chunk_model": "models/voxlock/chunk_int8.onnx",
        "enabled": True,
        "threshold": 0.62,           # cosine similarity gate for the enrolled speaker
        # off            -> "Anyone": transcribe every speaker
        # turn_titanet   -> "Only Me (Fast)": cosine gate per VAD segment
        # overlap_voxlock-> "Only Me (Overlap)": true waveform isolation before ASR
        "isolation_mode": "overlap_voxlock",
        "process_block_ms": 400,     # VoxLock streaming block size (amortizes ORT overhead)
        "num_threads": 2,
        "active_profile": None,
    },
    "injection": {
        "enabled": True,
        "tier": "auto",              # auto | wm_char | unicode | clipboard
        "max_chars": 5000,
        "restore_clipboard": True,   # privacy: restore prior clipboard after paste
        "char_delay_s": 0.002,
        "target_title": None,
    },
    "translation": {"enabled": False, "target": "en"},
    # Optional local LLM polish (Ollama / LM Studio, OpenAI-compatible).
    # A hard timeout guarantees live dictation never freezes on the LLM.
    "llm_polish_enabled": False,
    "llm_endpoint": "http://127.0.0.1:11434/v1/chat/completions",
    "llm_model": "auto",
    "llm_timeout_s": 1.5,
    "commands": {
        "enabled": True,
        "list": [
            {"name": "stop_listening", "anchoring": "whole",
             "patterns": ["stop listening", "stop recording", "pause listening", "mic off"]},
            {"name": "start_listening", "anchoring": "whole",
             "patterns": ["start listening", "start recording", "resume listening", "mic on"]},
            {"name": "new_line", "anchoring": "prefix", "patterns": ["new line"]},
            {"name": "new_paragraph", "anchoring": "prefix", "patterns": ["new paragraph"]},
            {"name": "scratch_that", "anchoring": "whole",
             "patterns": ["scratch that", "delete that", "undo that"]},
            {"name": "clear_all", "anchoring": "whole",
             "patterns": ["clear all", "clear everything"]},
            {"name": "toggle_translate", "anchoring": "whole",
             "patterns": ["translate mode", "toggle translation", "switch translation"]},
            {"name": "pick_window", "anchoring": "whole",
             "patterns": ["pick window", "choose window", "select window"]},
        ],
    },
    "custom_dictionary": {
        "\\bg p u\\b": "GPU",
        "\\bh d m i\\b": "HDMI",
        "\\bsherpa onnx\\b": "sherpa-onnx",
        "\\bv o x\\b": "VOX",
    },
    "hotkey": {"mods": ["ctrl", "shift"], "key": "space"},
    "history": {"max_items": 50, "db": "data/history.db"},
    "first_run_completed": False,   # onboarding wizard shown on first launch
}


def deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` (dst is modified in place)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


class Config:
    """Thread-safe, dotted-path settings store backed by config.json."""

    def __init__(self, path: Path | None = None):
        if path is not None:
            self.path = Path(path)
        else:
            self.path = BASE_DIR / "config.json"
            # Frozen release: seed a writable config next to the executable.
            try:
                if not self.path.exists():
                    bundled = BUNDLE_DIR / "config.json"
                    if bundled.exists() and bundled.resolve() != self.path.resolve():
                        shutil.copyfile(bundled, self.path)
            except Exception:
                pass
        self._lock = threading.RLock()
        self.data: dict = copy.deepcopy(DEFAULT_CONFIG)
        self.reload()

    # -- loading -----------------------------------------------------------
    def reload(self) -> None:
        with self._lock:
            self.data = copy.deepcopy(DEFAULT_CONFIG)
            if self.path.exists():
                try:
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        deep_merge(self.data, loaded)
                except (json.JSONDecodeError, OSError):
                    # Corrupt config -> keep defaults; never crash the app.
                    pass
            if not self.path.exists():
                self.save()

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    # -- dotted access ------------------------------------------------------
    def get(self, dotted: str, default=None):
        with self._lock:
            node = self.data
            for part in dotted.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

    def set(self, dotted: str, value, save: bool = True) -> None:
        with self._lock:
            parts = dotted.split(".")
            node = self.data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            if save:
                self.save()

    def as_dict(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.data)


CONFIG = Config()
