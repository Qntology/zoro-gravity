@echo off
rem ASCII-only on purpose: cmd.exe mis-parses non-ASCII bytes in .bat files on CP949 consoles.
rem
rem Builds one zorogravity-core example for a backend and runs it. run\windows\<backend>.bat calls this:
rem     call scripts\run-windows.bat <cuda|rocm|vulkan> [example] [example arguments]
rem
rem The default example is decide_probe, the whole decision path. It needs the zoro model in the
rem models folder, %LOCALAPPDATA%\zorogravity\models. run\windows\models.bat downloads it.
rem Each backend keeps its own build folder so the caches do not mix (vulkan is the default
rem feature, so it shares target). With no example the window stays open at the end, so a
rem double click can still read the output.

setlocal
cd /d "%~dp0.."

set "BACKEND=%~1"
set "EXAMPLE=%~2"
set "KEEP_WINDOW="
if "%EXAMPLE%"=="" (
    set "EXAMPLE=decide_probe"
    set "KEEP_WINDOW=1"
)
shift
shift
set "ARGS="
:collect
if "%~1"=="" goto :collected
set ARGS=%ARGS% "%~1"
shift
goto :collect
:collected

where cargo > nul 2>&1
if errorlevel 1 (
    echo [run] cargo not found. Install Rust from https://rustup.rs
    goto :failed
)

set "TARGET=target-%BACKEND%"
if /i "%BACKEND%"=="vulkan" set "TARGET=target"
if /i "%BACKEND%"=="rocm" call scripts\rocm-env.bat
if errorlevel 1 goto :failed
if /i "%BACKEND%"=="cuda" call scripts\cuda-env.bat
if errorlevel 1 goto :failed

set "CARGO_TARGET_DIR=%TARGET%"
set "ZOROGRAVITY_DEVICE=%BACKEND%"
echo [run] backend %BACKEND% ^| example %EXAMPLE% ^| build folder %TARGET%
cargo build -p zorogravity-core --release --no-default-features --features %BACKEND% --example %EXAMPLE%
if errorlevel 1 goto :failed

"%TARGET%\release\examples\%EXAMPLE%.exe"%ARGS%
set "CODE=%errorlevel%"
goto :done

:failed
set "CODE=1"

:done
if defined KEEP_WINDOW pause
endlocal & exit /b %CODE%
