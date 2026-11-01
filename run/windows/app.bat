@echo off
rem Runs the zoro-gravity app. Missing models download on start, with progress in the settings panel.
cd /d "%~dp0..\..\app"
call npm install
if errorlevel 1 goto :failed
call npm run tauri dev
if errorlevel 1 goto :failed
exit /b 0
:failed
pause
exit /b 1
