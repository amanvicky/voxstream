@echo off
rem ---------------------------------------------------------------
rem  VoxStream silent one-click launcher (v2.0)
rem  - dev tree : .venv\Scripts\pythonw.exe app.py   (no console)
rem  - release  : dist\VoxStream\VoxStream.exe       (if built)
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

if exist "dist\VoxStream\VoxStream.exe" (
    start "" "%~dp0dist\VoxStream\VoxStream.exe"
    exit /b 0
)

if not exist ".venv\Scripts\pythonw.exe" (
    rem No venv yet -> show a hint once, silently try nothing else.
    if exist "build_windows.bat" (
        echo VoxStream venv not found. Run build_windows.bat first. > "%TEMP%\VoxStream-setup-hint.txt"
        start notepad "%TEMP%\VoxStream-setup-hint.txt"
    )
    exit /b 1
)

start "" "%~dp0.venv\Scripts\pythonw.exe" app.py
exit /b 0
