@echo off
rem Runs the zoro-gravity app on AMD ROCm/HIP. Needs the HIP SDK and MSVC 14.44 (scripts\rocm-env.bat).
rem Set ZOROGRAVITY_ENGINE=rules to run without models. Scanner images are read from
rem ZOROGRAVITY_SCANNER_DIR (default %%LOCALAPPDATA%%\zorogravity\scanner).
cd /d "%~dp0..\.."
call scripts\rocm-env.bat
if errorlevel 1 goto :failed
set ZOROGRAVITY_DEVICE=rocm
set CARGO_TARGET_DIR=%CD%\target-rocm
cd app
call npm install
if errorlevel 1 goto :failed
call npx tauri dev --release --no-watch --features rocm -- --no-default-features
if errorlevel 1 goto :failed
exit /b 0
:failed
pause
exit /b 1
