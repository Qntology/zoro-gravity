@echo off
rem Builds and runs zorogravity-core on Vulkan. Needs only the GPU driver (vulkan-1.dll).
rem     vulkan.bat [example] [example arguments]
call "%~dp0..\..\scripts\run-windows.bat" vulkan %*
