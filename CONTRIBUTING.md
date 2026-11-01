# Contributing to the KytyPS5 Yōtei fork

Thank you for helping. This fork targets **one game only: Ghost of Yōtei (PS5)**. Every change is made, measured and
tested against that game; changes for other titles belong in upstream [KytyPS5](https://github.com/KytyPS5/KytyPS5).
Please read this page before opening a pull request. Bug reports go through the issue forms (see
[Reporting problems](#reporting-problems)).

## Never include game material

A pull request, issue or comment must **never** contain:

- game code in any form: disassembly, decompiler output, pseudo-code rewritten from it, or "clean-room" code that
  follows it line by line;
- game files or parts of them: executables, libraries, data archives, textures, audio, video, save files;
- shader binaries or shader dumps taken from the game (including their disassembly or SPIR-V translations);
- addresses, offsets, symbol names or function names from the game's executable, or reverse-engineering databases;
- firmware, system software, keys or anything else from a console.

What you may describe is **behaviour**: what the game asks the GPU to do (the commands, formats and resources it
uses), what the emulator does with it, and what appears on screen. Contributions that break this rule are closed and
removed from the history.

## What "correct" means

A change is correct when the emulator produces **the same result as the PS5 on the data the game really uses**.
Accuracy is checked on recordings of the game's own work (for example with `KYTY_TRACE_REF_CAPTURE` and the CPU
reference interpreter in `tools/refinterp`), not built in advance for cases the game never hits. Among ways that
give the same result, the cheapest one wins.

Improvements beyond the PS5 (a fixed higher resolution, an upscaler, and similar) are welcome, but only as a
**separate switch that the user turns on**, with its cost stated, and only once the PS5-accurate path works.

## Rules for emulation changes

1. **No permanent workarounds.** A workaround (a limit, skipped work, a substitute result) may be added only for a
   short time to move the game forward or to find a bug: off by default and described as a diagnostic. Once the cause
   is known it is replaced by a proper implementation and removed. A proper implementation contains no limits, no
   skipped work and no substitute results.
2. **Every change has a switch.** A new code path is selected by an environment variable read where it is used
   (`std::getenv("KYTY_...")`), with its values and default documented in the comment above it:
   - `0` – the old path, `1` – the new path;
   - `2` – **verify mode**: both paths run, their results are compared and every difference is logged. A change is
     accepted only with **0 differences** in the game.

   Use `KYTY_FIX_*` for correctness fixes, `KYTY_PERF_*` for performance changes, `KYTY_DIAG_*` for temporary
   diagnostics and `KYTY_TRACE_*` / `KYTY_CAPTURE_*` for diagnostic output (off by default).
3. **Default only after a test.** A switch becomes the default in code (its read site or `src/common/envDefaults.h`)
   only after verify mode shows 0 differences and a test in the game shows a gain or a better image. After that the old
   path and any workaround it replaced are removed in a follow-up change.
4. **Every change states its cost**, in milliseconds per frame:
   - GPU work: the time of the affected pass with `KYTY_TRACE_GPU_TIME` (and `KYTY_TRACE_GPU_TIME_MARKERS=1` for the
     per-pass breakdown);
   - CPU work and frame rate: a **live A/B in the same run** with `KYTY_LIVE_AB=<SWITCH>` (the emulator alternates the
     switch at frame boundaries and prints per-phase fps and frame times; only switches that are exact to toggle
     there are accepted, see `src/common/liveAb.h`). Comparisons of separate runs vary by about ±3% (±6% in the open
     world) and are not accepted as proof of a speed-up.

   An expensive fix should come with a plan for a faster version of the same fix.
5. **Image changes come with screenshots**: before and after, same place in the game, same settings and resolution.
6. **One change per pull request.** Small independent fixes may be grouped only if each has its own switch and its
   own verify-mode result.

## Pull requests

The pull request template asks for the points above. In short, the description should give:

- what was wrong (symptom) and the cause on the emulator side;
- the switch name, its default and what `2` compares;
- the verify-mode result (number of differences, where in the game it ran);
- the cost or gain (ms per frame or live A/B numbers, GPU model and driver);
- screenshots for visual changes;
- confirmation that no game material is included.

Code style: format C++ with the repository's clang-format settings (`pre-commit install` sets up the hook from
`.pre-commit-config.yaml`), and write code that reads like the code around it. Comments explain why the code does
what it does, in terms of the hardware or the emulator, never in terms of the game's code.

Continuous integration builds the launcher and the emulator on Windows for every pull request (documentation-only
changes are skipped). It cannot run the game, so the verify-mode results and measurements in the description are what
reviewers rely on.

## Building (Windows)

Requirements:

- Windows 10/11, 64-bit, and Git;
- Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload (Windows SDK);
- CMake 3.22 or newer and Ninja (the copies installed with Visual Studio work);
- LLVM **20.1.8** for Windows (`clang-cl`, `lld-link`, `llvm-mt`); the MSVC compiler `cl.exe` is not supported;
- [glslang](https://github.com/KhronosGroup/glslang/releases) (`glslangValidator` or `glslang` on `PATH`, or
  `-DKYTY_GLSLANG_VALIDATOR=<path>`);
- for the launcher only: Qt 6 for MSVC, 64-bit, with Concurrent, Network and Widgets (6.7.2 is used for this fork);
- FFmpeg is downloaded by CMake on the first configure (or set `-DFFMPEG_PREBUILT_DIR=<dir>` to an unpacked package).

Open an **x64 Native Tools Command Prompt for VS 2022**, set `LLVM` to your LLVM 20.1.8 folder and `QT` to your Qt
folder (for example `<Qt>/6.7.2/msvc2019_64`), then in the repository root:

```bat
git submodule update --init --recursive

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_C_COMPILER=%LLVM%/bin/clang-cl.exe" "-DCMAKE_CXX_COMPILER=%LLVM%/bin/clang-cl.exe" ^
  "-DCMAKE_LINKER=%LLVM%/bin/lld-link.exe" "-DCMAKE_MT=%LLVM%/bin/llvm-mt.exe" ^
  "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2 -flto=thin" ^
  "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2 -flto=thin" ^
  "-DCMAKE_PREFIX_PATH=%QT%"

cmake --build build --target launcher
```

Building `launcher` also builds `kyty_emulator` and copies the Qt runtime next to it. Other configurations:

| Goal | Configure option | Build target |
|---|---|---|
| Emulator only, no Qt needed | `-DKYTY_BUILD_LAUNCHER=OFF` (omit `CMAKE_PREFIX_PATH`) | `kyty_emulator` |
| Launcher only, without compiling the emulator | `-DKYTY_LAUNCHER_STANDALONE=ON` | `launcher` |

`KYTY_LAUNCHER_STANDALONE` still configures the whole tree (glslang and FFmpeg are needed), but builds only the
launcher and the libraries it uses; place the launcher next to an existing `kyty_emulator.exe` to run it. The release
flags above match the maintainer's builds; for a quicker local build you can leave out `-flto=thin`.

The CPU reference interpreter (`tools/refinterp`) builds with `tools\refinterp\build.bat`; set `VCVARS64` (path to
`vcvars64.bat`) and `CLANG_CL` (path to `clang-cl.exe`) first.

## Reporting problems

Use the issue forms on GitHub. For a problem in the game, the launcher's **Export bug report** button
(Advanced → Support) saves a zip (`kyty-bug-report-<date>.zip`) on your computer with the logs, settings and system
information; the Windows user name, the computer name and the PS5 user names from the settings are replaced with
placeholders. Nothing is sent automatically. **Issues are public**: check the file list and preview the launcher shows
before you attach the zip, and never attach game files or save files.

## License

This fork, like KytyPS5, is licensed under the **GNU General Public License version 2 only** ([LICENSE](LICENSE)).
By submitting a contribution you agree that it is licensed under GPL-2.0-only. The original Kyty code is MIT-licensed;
keep its notice in [LICENSES/Kyty-MIT.txt](LICENSES/Kyty-MIT.txt) and in the files that carry it. Code adapted from
another fork must name its origin in the file ("Based on ..."), and its license must be compatible with GPL-2.0-only.
