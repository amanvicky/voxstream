@echo off
rem ================================================================
rem  VoxStream v2.0 - GitHub release packager
rem  1) ensures dist\VoxStream\VoxStream.exe exists (builds if needed)
rem  2) compresses dist\VoxStream\ into release\VoxStream-v2.0-win64.zip
rem  The zip is ready to upload directly to GitHub Releases.
rem ================================================================
setlocal
cd /d "%~dp0"

if exist "dist\VoxStream\VoxStream.exe" goto compress

echo [release] standalone build missing - running build_windows.bat ...
call build_windows.bat
if errorlevel 1 goto fail_build
if not exist "dist\VoxStream\VoxStream.exe" goto fail_missing

:compress
if not exist "release" mkdir "release"
if exist "release\VoxStream-v2.0-win64.zip" del "release\VoxStream-v2.0-win64.zip"

echo [release] compressing dist\VoxStream to release\VoxStream-v2.0-win64.zip ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'dist\VoxStream' -DestinationPath 'release\VoxStream-v2.0-win64.zip' -CompressionLevel Optimal -Force; $z = Get-Item 'release\VoxStream-v2.0-win64.zip'; Write-Host ('[release] created ' + $z.FullName + ' - ' + [math]::Round($z.Length/1MB,1) + ' MB')"
if errorlevel 1 goto fail_zip

echo.
echo [release] DONE. Upload release\VoxStream-v2.0-win64.zip to GitHub Releases.
pause
exit /b 0

:fail_build
echo [release] ERROR: build_windows.bat failed.
pause
exit /b 1

:fail_missing
echo [release] ERROR: dist\VoxStream\VoxStream.exe still missing.
pause
exit /b 1

:fail_zip
echo [release] ERROR: compression failed.
pause
exit /b 1
