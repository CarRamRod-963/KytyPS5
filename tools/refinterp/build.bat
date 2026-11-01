@echo off
rem Builds tools\refinterp\refinterp.exe (CPU only; no GPU, no emulator code).
rem   VCVARS64  optional: path to vcvars64.bat of a Visual Studio / Build Tools install (not needed when run from an
rem             "x64 Native Tools" prompt).
rem   CLANG_CL  optional: path to clang-cl.exe (default: clang-cl from PATH; the emulator is built with LLVM 20.1.8).
if defined VCVARS64 call "%VCVARS64%" >nul
if not defined CLANG_CL set "CLANG_CL=clang-cl"
cd /d "%~dp0"
"%CLANG_CL%" /nologo /std:c++20 /O2 /EHsc /W3 /Fe:refinterp.exe refinterp.cpp || exit /b 1
echo BUILD_OK
