@echo off
rem ASCII-only on purpose: cmd.exe mis-parses non-ASCII bytes in .bat files on CP949 consoles.
rem
rem Sets up the AMD ROCm (HIP) build environment for zorogravity-core. Use it before cargo:
rem     call scripts\rocm-env.bat
rem     cargo build -p zorogravity-core --no-default-features --features rocm
rem
rem 1) MSVC toolset 14.44. The default 14.51 STL conflicts with the clang 21 HIP headers
rem    ("__device__ function 'isgreater' cannot overload ...") and the kernel build fails.
rem    14.44 is installed with the Visual Studio component "MSVC v143 ... (14.44)".
rem 2) HIP SDK: defaults to 7.2 (legacy track). Set HIP_PATH before the call to use another one.
rem 3) GPU target: defaults to gfx1032 (Radeon RX 6600). Set HIP_ARCH before the call to change it.

if not defined VCVARS_VER set "VCVARS_VER=14.44"
set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto :no_vcvars

call "%VCVARS%" -vcvars_ver=%VCVARS_VER% > nul
echo [ROCm] MSVC toolset %VCToolsVersion%

if not defined HIP_PATH set "HIP_PATH=%ProgramFiles%\AMD\ROCm\7.2"
if not exist "%HIP_PATH%\bin\hipcc.exe" goto :no_hipcc

set "ROCM_PATH=%HIP_PATH%"
set "PATH=%HIP_PATH%\bin;%PATH%"
if not defined CANDLE_ROCM_TRACK set "CANDLE_ROCM_TRACK=legacy"
if not defined HIP_ARCH set "HIP_ARCH=gfx1032"
echo [ROCm] HIP SDK %HIP_PATH% ^| track %CANDLE_ROCM_TRACK% ^| GPU target %HIP_ARCH%
exit /b 0

:no_vcvars
echo [ROCm] vcvars64.bat not found. Install Visual Studio C++ Build Tools with the 14.44 toolset.
exit /b 1

:no_hipcc
echo [ROCm] hipcc.exe not found under %HIP_PATH%. Install the AMD HIP SDK or set HIP_PATH.
exit /b 1
