@echo off
rem ================================================================
rem  VoxStream v2.0 - standalone Windows release builder
rem  Output: dist\VoxStream\VoxStream.exe  (one-click, fully offline)
rem
rem  Requires: Python 3.10-3.12 (py launcher) or uv on PATH.
rem  Internet is needed once (deps + models); the built app never phones home.
rem ================================================================
setlocal
cd /d "%~dp0"

rem --- 1) ensure virtualenv ---------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [build] creating .venv with Python 3.11 ...
    py -3.11 -m venv .venv 2>nul || uv venv --python 3.11 .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [build] ERROR: could not create .venv. Install Python 3.11 ^(py launcher^) or uv.
    pause
    exit /b 1
)

rem --- 2) ensure pip inside the venv (uv-created venvs lack it) ----
".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [build] bootstrapping pip ...
    ".venv\Scripts\python.exe" -m ensurepip --upgrade
)

rem --- 3) runtime dependencies -------------------------------------
echo [build] installing runtime dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    where uv >nul 2>nul && uv pip install -r requirements.txt --python ".venv\Scripts\python.exe"
)
".venv\Scripts\python.exe" -c "import sherpa_onnx, faster_whisper, PySide6" 2>nul
if errorlevel 1 (
    echo [build] ERROR: dependencies missing.
    pause
    exit /b 1
)

rem --- 4) build-time dependency ------------------------------------
echo [build] installing PyInstaller ...
".venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 (
    where uv >nul 2>nul && uv pip install pyinstaller --python ".venv\Scripts\python.exe"
)
".venv\Scripts\python.exe" -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo [build] ERROR: PyInstaller unavailable.
    pause
    exit /b 1
)

rem --- 5) models (only if not staged yet) ---------------------------
if not exist "models\zipformer\tokens.txt" (
    echo [build] staging offline models ^(one-time download^) ...
    ".venv\Scripts\python.exe" setup_models.py
    if errorlevel 1 (
        echo [build] ERROR: model staging failed.
        pause
        exit /b 1
    )
)

rem --- 6) build the standalone bundle -------------------------------
echo [build] running PyInstaller ...
".venv\Scripts\python.exe" -m PyInstaller VoxStream.spec --noconfirm --clean
if errorlevel 1 (
    echo [build] ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo [build] SUCCESS: dist\VoxStream\VoxStream.exe
echo [build] The dist\VoxStream folder is fully portable - copy it anywhere.
pause
exit /b 0
