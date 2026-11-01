# KytyPS5 Yōtei fork

An experimental fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5) (itself based on
[Kyty](https://github.com/InoriRus/Kyty)) that targets **one game only: Ghost of Yōtei (PS5)**. Every change in this
tree is made, measured and tested against that game. Other titles are not tested and may run worse than on upstream
KytyPS5; use upstream for them. The upstream README ([README.md](README.md)) is kept unchanged.

This repository contains **no game files, no firmware, no system software and no material taken from the game's
code**. You need your own legally obtained dump of the game. Where the emulator relies on how the game behaves (for
example the layout of a table the game fills in memory), that knowledge is described in comments as behaviour; no game
code is copied.

## Status

- Work in progress. The game boots, loads saves and is playable, but runs well below the PS5's frame rate and some
  effects still differ from the PS5 (see [Known workarounds](#known-workarounds)).
- Correctness is checked against reference captures of the game's own GPU work: a fix counts as correct when the
  emulator computes the same results as the PS5 on the data the game really uses.
- Performance work is done behind switches that have a verify mode (old and new path compared, 0 differences) before
  they become the default.

## Requirements

- Windows 10/11, 64-bit; a CPU with AVX2 (the release build uses `/arch:AVX2`).
- A Vulkan 1.3 GPU with current drivers. Development and all testing are done on an **NVIDIA GeForce RTX** card
  (RTX 5070 Ti, 16 GB). AMD and Intel GPUs are untested; a few exact half-precision paths are enabled only on devices
  that were measured (NVIDIA), other vendors take slower but exact fallbacks.
- 32 GB of RAM recommended (the game needs about 20 GB of free commit) and several GB of free disk space for the shader
  and pipeline caches.

## Building (Windows)

Toolchain used for this fork:

- Git (with submodules);
- CMake 3.22.1 or newer and Ninja (the copies shipped with Visual Studio work);
- Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload (Windows SDK, `vcvars64.bat`);
- LLVM **20.1.8** for Windows (`clang-cl`, `lld-link`, `llvm-mt`); the MSVC compiler `cl.exe` is not supported;
- [glslang](https://github.com/KhronosGroup/glslang/releases) (`glslangValidator` or `glslang` on `PATH`, or
  `-DKYTY_GLSLANG_VALIDATOR=<path>`);
- FFmpeg is downloaded by CMake on the first configure (or point `-DFFMPEG_PREBUILT_DIR=<dir>` at an unpacked package);
- for the launcher only: Qt 6 for MSVC, 64-bit, with Concurrent, Network and Widgets (6.7.2 is used for this fork).

The fork has its own Qt launcher (`src/launcher`): game selection, display and graphics settings, shader
precompilation with a progress bar, the shader cache, log folder and a local bug-report export. Its design is
described in [docs/launcher/DESIGN-SPEC.md](docs/launcher/DESIGN-SPEC.md); how to build it (with or without the
emulator) is in [CONTRIBUTING.md](CONTRIBUTING.md). The emulator also runs on its own from the command line. To build
the emulator only, without Qt, from an **x64 Native Tools Command Prompt for VS 2022** (or after calling
`vcvars64.bat`), in the repository root, with `LLVM` set to your LLVM 20.1.8 folder:

```bat
git submodule update --init --recursive

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DKYTY_BUILD_LAUNCHER=OFF ^
  "-DCMAKE_C_COMPILER=%LLVM%/bin/clang-cl.exe" "-DCMAKE_CXX_COMPILER=%LLVM%/bin/clang-cl.exe" ^
  "-DCMAKE_LINKER=%LLVM%/bin/lld-link.exe" "-DCMAKE_MT=%LLVM%/bin/llvm-mt.exe" ^
  "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2 -flto=thin" ^
  "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2 -flto=thin"

cmake --build build --target kyty_emulator -j 4
```

The reference interpreter used for correctness checks (`tools/refinterp`, CPU only) builds with
`tools\refinterp\build.bat`; set `VCVARS64` (path to `vcvars64.bat`) and `CLANG_CL` (path to `clang-cl.exe`) unless
both are already on hand in your prompt.

## Running

Start `launcher.exe` (next to `kyty_emulator.exe`), choose the game folder and press **Play**, or start the emulator
directly:

```bat
kyty_emulator.exe --game "D:\Games\<your Ghost of Yotei dump>" --screen-width 1280 --screen-height 720
```

Run `kyty_emulator.exe --help` for the logging, validation and debugging options. Tessellation is on by default
(`--no-tessellation` turns it off). The first start compiles shaders and pipelines in the background, so the first
minutes stutter; later starts reuse the caches. `--precompile-only` (the launcher's "Precompile shaders") compiles the
shaders recorded in earlier runs and exits without starting the game.

## Documentation

- [docs/FRAME.md](docs/FRAME.md) – a Ghost of Yōtei frame pass by pass (shadows, G-buffer, terrain, cloth, particles,
  probes and RTGI, fog, water, FSR3, post-processing) and how the emulator handles each pass;
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) – the GPU thread, preparation workers, command recording, shader
  translation and caches, memory tracking, readbacks, bindless textures, `envDefaults.h`;
- [docs/TESTING.md](docs/TESTING.md) – verify modes (`=2`), live A/B in one run, test places, pictures, workaround
  counters, warming the shader cache before measuring;
- [docs/launcher/DESIGN-SPEC.md](docs/launcher/DESIGN-SPEC.md) – the launcher's design;
- [CONTRIBUTING.md](CONTRIBUTING.md) – rules for changes, building, reporting problems.

## Settings are defaults in code

No launch script or settings file is needed: a plain `kyty_emulator.exe --game ...` uses the same settings as the
maintainer's test runs. All settings this fork depends on are defaults in the source:

- `src/common/envDefaults.h` holds the table of settings that used to come from the launch script, each marked as
  `config` (ordinary setting or measured performance switch), `correct` (hardware behaviour still carrying an old
  `DIAG_` name) or `workaround` (see below), with the work item that will replace it;
- every other switch has its default where it is read (`std::getenv("KYTY_...")`), documented in the comment above it.

An environment variable still overrides any default (`KYTY_X=0` turns a switch off), and `KYTY_ENV_DEFAULTS=0` ignores
the whole table. At start the emulator prints one `ENV_DEFAULT KYTY_X=value source=env|code` line per table entry.
`KYTY_TRACE_*` and `KYTY_CAPTURE_*` switches are diagnostics, off by default.

Which settings keep the shader cache warm: only the settings listed in `Common::ShaderEnv::Names`
(`src/common/shaderEnv.h`) change the translated shaders or pipelines, and only their effective values (the
environment, else the code default) decide whether the pipelines of earlier runs count as compiled. Changing any other
`KYTY_` variable - reports (`KYTY_TRACE_*`), the overlay, live A/B, pad scripts, dispatch skip lists such as the "No
particles" mod (`KYTY_DIAG_SKIP_CS_HASHES`), precompiler threads, VRAM budget - keeps the cache; changing a listed one
makes the next start compile in the background again. The start log has one `ENV_FINGERPRINT hash=...` line: two runs
with the same hash share the cache.

## Known workarounds

The goal of the fork is a build with **no workarounds**: every approximation is replaced by a proper implementation
(checked with a verify mode, 0 differences) and then removed. Until then the following are on by default. They are
the `workaround` entries of `src/common/envDefaults.h` plus a few defaults in code ("(code)"); names are given without
the `KYTY_` prefix. The environment can turn them off (`=0`), but the game then fails or shows errors: they stay on
until their replacement lands.

| Area (work item) | Switches | What may differ from the PS5 |
|---|---|---|
| GPU-written memory (D6) | `DIAG_BDA_STORE_TRACK`, `DIAG_BDA_STORE_UNTRACKED`, `DIAG_BDA_PIN_FAULTED`, `DIAG_FAULT_WINDOW_KB`, `DIAG_SCALAR_READS_IGNORE_ADDRESS_WRITES` | pages the GPU writes through pointers are tracked late and kept generously; rare stale reads |
| Image memory shared by several images (D4/D5) | `DIAG_ALIAS_SYNC` | transient targets copied between aliases; some post-processing (screen-space GI filter) may be off |
| Stencil and depth metadata (D8) | `DIAG_STENCIL_PLANE_SYNC`, `DIAG_STENCIL_STORAGE_SHADOW`, `DIAG_PRESERVE_STENCIL`, `DIAG_STENCIL_ALIAS_COPY`, `DIAG_HTILE_FULL_RANGE`, `DIAG_HTILE_RMW_CLEAR` (code), `DIAG_DEPTH_ENCODING_LENIENT` | stencil kept as copies, HTILE emulated as "full range"; occasional missing masks or shadow tiles |
| Colour compression (A5) | `DIAG_DCC_FAST_CLEAR`, `DIAG_DCC_KEEP_BUFFER_WRITES`, `DIAG_COLOR_META_UNALIGNED_SKIP` | simplified DCC clear handling |
| GPU-selected (bindless) textures (D1/D2) | `DIAG_BINDLESS_EAGER`, `DIAG_NULL_TEXTURE_GRAY`, `DIAG_BINDLESS_IMMEDIATE_BUDGET` (code: 64 new textures per frame) | a new texture can show a placeholder for 1-3 frames; unresolved descriptors read as empty (e.g. snow and mud tracks). Fixed since 2026-10-29 (C9): texture tables over 1 MiB (particles, mist) are translated like every other table instead of showing placeholders past 1 MiB, and a table the table path cannot take stops with `BINDLESS_TABLE_REFUSED` instead of a silent placeholder |
| Shaders capped (D7) | `SHADER_LOOP_LIMIT_HASHES` (2 shaders), `DIAG_SKIP_UNSUPPORTED_CS`, global loop watchdog `KYTY_SHADER_LOOP_LIMIT` (code; a safety net, not a workaround: every shader loop ends after 65,536 iterations so a runaway loop cannot hang the GPU - it never fired in our tests, 0 of ~1,950 checks at the gate and in the village, 29.10). The particle-emit shader 142383a7 is no longer skipped (29.10; it runs rarely - if a crash follows the log line `W9_PARTICLE_EMIT`, report it) | rare shaders with a loop cap |
| Terrain shaders (M42) | `DIAG_WATERFALL_POINTER_RECORD0` | approximate material table in some terrain shaders |
| Texture edge cases | `SDK_FIX_MIP_CHAIN_CLAMP`, `SDK_FIX_NULL_COMPARE_DEPTH`, `SDK_FIX_INTEGER_POINT` (code) | rare sampling edge cases approximated (logged) |
| Game-specific table (D6) | `RECORD_KEYED_GAME_COUNTS` (code) | one shader's record count is read from the game's data layout instead of a generic bound |

Seven more table entries are marked `correct` (believed to be the hardware's behaviour; three still need an exactness
check). The game's GPU timer reads host GPU busy time (`FIX_VIRTUAL_GPU_CLOCK`); because the emulated frame takes
longer than on a PS5, the game stays at a lower internal resolution than on the console (1080p at the test scene of
most measurements). Reporting the time a PS5 would take is open.

## Licenses

- This fork, like KytyPS5, is licensed under the **GNU General Public License version 2 only** ([LICENSE](LICENSE)).
  If you distribute a binary, you must also offer the complete corresponding source code.
- The original Kyty is MIT-licensed; its copyright and license notice are kept in
  [LICENSES/Kyty-MIT.txt](LICENSES/Kyty-MIT.txt).
- `tools/refinterp/generated/isa_tables.inc` is generated from AMD's machine-readable GPU ISA (AMD GPUOpen, MIT):
  [LICENSES/AMD-GPU-MR-ISA-MIT.txt](LICENSES/AMD-GPU-MR-ISA-MIT.txt).
- Code adapted from other KytyPS5 forks (chenxiao07/KytyPS5, TheCruZ/KytyPS5-GTA, both GPL-2.0) carries a "Based on ..." note with its origin
  in the file; ideas taken from other forks name their source in the comment.
- Third-party components in `3rdparty/` and the submodules remain under their own licenses, included with them.
- Ghost of Yōtei, PlayStation and PS5 are trademarks of their respective owners. This project is not affiliated with
  or endorsed by Sony Interactive Entertainment or Sucker Punch Productions.
