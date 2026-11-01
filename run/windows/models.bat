@echo off
rem Opens the model settings page: download with a progress bar, delete. No GPU SDK is needed.
cd /d "%~dp0..\.."
cargo run --release -p zorogravity-models
if errorlevel 1 pause
