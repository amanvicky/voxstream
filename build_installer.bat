@echo off
rem ================================================================
rem  VoxStream v2.0 - one-click Windows installer builder
rem  Output: release\VoxStream-Setup-v2.0.exe  (Inno Setup)
rem  Keeps release\VoxStream-v2.0-win64.zip as the portable alternative.
rem ================================================================
setlocal
cd /d "%~dp0"

rem --- 1) locate Inno Setup compiler -------------------------------
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ISCC%" goto have_iscc

echo [installer] Inno Setup 6 not found - installing via winget ...
winget install --id JRSoftware.InnoSetup -e --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo [installer] ERROR: winget install failed. Install Inno Setup 6 from https://jrsoftware.org/isdl.php and re-run.
    pause
    exit /b 1
)
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo [installer] ERROR: ISCC.exe still missing after install.
    pause
    exit /b 1
)

:have_iscc
echo [installer] using compiler: %ISCC%

rem --- 2) ensure the standalone bundle is built ----------------------
if exist "dist\VoxStream\VoxStream.exe" goto compile

echo [installer] dist\VoxStream missing - running build_windows.bat ...
call build_windows.bat
if errorlevel 1 goto fail_build
if not exist "dist\VoxStream\VoxStream.exe" goto fail_missing

:compile
rem --- 3) optional icon check (missing icon is non-fatal) ------------
if not exist "installer\app_icon.ico" (
    echo [installer] NOTE: installer\app_icon.ico missing - building without a custom icon.
    copy /y "installer\VoxStream.iss" "installer\VoxStream_noicon.iss" >nul
    powershell -NoProfile -Command "(Get-Content 'installer\VoxStream_noicon.iss') -replace 'SetupIconFile=app_icon.ico', '' | Set-Content 'installer\VoxStream_noicon.iss'"
    set "ISS=installer\VoxStream_noicon.iss"
) else (
    set "ISS=installer\VoxStream.iss"
)

echo [installer] compiling installer ...
"%ISCC%" "%ISS%"
if errorlevel 1 goto fail_compile

echo.
echo [installer] SUCCESS: release\VoxStream-Setup-v2.0.exe
echo [installer] Portable alternative kept: release\VoxStream-v2.0-win64.zip
pause
exit /b 0

:fail_build
echo [installer] ERROR: build_windows.bat failed.
pause
exit /b 1

:fail_missing
echo [installer] ERROR: dist\VoxStream\VoxStream.exe still missing.
pause
exit /b 1

:fail_compile
echo [installer] ERROR: Inno Setup compilation failed.
pause
exit /b 1
