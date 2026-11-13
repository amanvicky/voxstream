# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the VoxStream standalone Windows release (v2.0).

Produces a --onedir bundle at dist/VoxStream/ with:
  VoxStream.exe           UI process; spawns the backend via "--run-backend"
  _internal/              Python runtime + all libraries (PySide6, sherpa-onnx,
                          faster-whisper/CTranslate2, FastAPI/uvicorn, ...)
  _internal/models/       every offline model (ASR, VAD, VoxLock, whisper)
  _internal/config.json   default settings (copied next to the exe on first run)

Build:  .venv\\Scripts\\python.exe -m PyInstaller VoxStream.spec --noconfirm
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

PROJECT = Path(SPECPATH)  # noqa: F821  (PyInstaller injects SPECPATH)

datas = [
    (str(PROJECT / "config.json"), "."),
    (str(PROJECT / "README.md"), "."),
]

# Bundle every offline model (the release is 100% standalone / offline).
_models_root = PROJECT / "models"
if _models_root.exists():
    for path in sorted(_models_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(_models_root)
        parts = rel.parts
        # Skip caches / temp artifacts (HF metadata, probe files).
        if any(p in (".cache", "__pycache__") for p in parts):
            continue
        if path.suffix in (".npy", ".tmp", ".part0", ".part1", ".part2", ".part3"):
            continue
        datas.append((str(path), str(_models_root.name / rel.parent)))

hiddenimports = [
    # uvicorn imports these lazily
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # explicit third-party entry points
    "websockets",
    "websockets.sync",
    "websockets.sync.client",
    "sounddevice",
    "sherpa_onnx",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "psutil",
    "scipy.signal",
    "scipy.io",
]

binaries = collect_dynamic_libs("sherpa_onnx")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("sounddevice")

a = Analysis(
    ["app.py"],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas", "torch", "PyQt5", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoxStream",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,               # windowed: no terminal on launch
    disable_windowed_traceback=False,
    icon=str(PROJECT / "installer" / "app_icon.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoxStream",            # -> dist/VoxStream/
)
