@echo off
rem ASCII-only on purpose: cmd.exe mis-parses non-ASCII bytes in .bat files on CP949 consoles.
rem
rem Sets up the NVIDIA CUDA build environment for zorogravity-core. Use it before cargo:
rem     call scripts\cuda-env.bat
rem     cargo build -p zorogravity-core --no-default-features --features cuda
rem
rem 1) MSVC: nvcc compiles the candle kernels during the build and calls cl.exe.
rem 2) CUDA Toolkit: nvcc must be on PATH or under CUDA_PATH (the installer sets CUDA_PATH).
rem 3) CUDA 13 headers need the standard preprocessor on MSVC, hence NVCC_PREPEND_FLAGS.

set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS goto :no_vcvars
if not exist "%VCVARS%" goto :no_vcvars

call "%VCVARS%" > nul 2>&1
echo [CUDA] MSVC toolset %VCToolsVersion%

if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
where nvcc > nul 2>&1
if errorlevel 1 goto :no_nvcc

if not defined NVCC_PREPEND_FLAGS set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
echo [CUDA] nvcc found ^| CUDA_PATH %CUDA_PATH%
exit /b 0

:no_vcvars
echo [CUDA] vcvars64.bat not found. Install Visual Studio C++ Build Tools.
exit /b 1

:no_nvcc
echo [CUDA] nvcc not found. Install the NVIDIA CUDA Toolkit or set CUDA_PATH.
exit /b 1
