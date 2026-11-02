# Testing changes in the Yōtei fork

How a change is checked before it becomes a default: exactness first, then cost, then the picture. The switches named
here are built into the emulator; their code is described in [ARCHITECTURE.md](ARCHITECTURE.md) and the passes they
touch in [FRAME.md](FRAME.md). The rules for contributions are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## 1. What "correct" means

A change is correct when the emulator computes **the same results as the PS5 on the data the game really uses**.
Exactness is checked on what the game actually runs (captured dispatches, counters from real play), not built in
advance for cases the game never reaches. Among ways that give the same result, the cheapest one wins.

Improvements beyond the PS5 (for example a fixed higher resolution or a different upscaler) are allowed only as
separate switches the player turns on, with their cost stated, after the exact path works.

## 2. Every change has a switch and a verify mode

- A new behaviour goes behind a `KYTY_*` switch, **off** until it is tested.
- The switch has a **verify mode, `=2`**: the old and the new way both run, the old result is used, and every
  difference is counted. The condition is **0 differences** over a whole run. The counts appear in the log (usually a
  `PERF <name> verify checks=… mismatches=…` line with `KYTY_TRACE_PERF=1`, or a line of the switch's own).
- Only after `=2` shows 0 differences and a measurement shows a gain (performance) or a better picture (fix) does the
  switch become the default in the code, with the measurement in the comment above the place it is read. The old code
  and any workaround it replaces are then removed.
- A workaround may be added only for a while, to get further in the game or to learn about a bug: off by default,
  described as a diagnostic, logged when it acts.

## 3. Where to test

Five places in the game cover the passes that break most often. They are described by what is on screen; any save
that reaches them works (do not share saves or other game data in issues or pull requests).

| Place | What it exercises |
|---|---|
| **Gate**: a torii gate in daylight, standing still | the most draws per frame (G-buffer, foliage, cloth); the main fps measurement |
| **Open world**: riding through fields and forest | streaming of textures and meshes, bindless terrain, far shadow pages, readbacks |
| **Night**: an outdoor night scene | night lighting, fog, particles |
| **Night RT**: the same kind of scene with the game's ray-traced lighting mode | the RTGI chain, BVH builds, the emulated ray intersection |
| **House by the lake**: around the protagonist's home, inside and outside | water lighting, the second stencil plane, interiors, skin in close-ups |

Plus a **run from the start of the game**: boot, menus, loading a save, the first minutes of play. This catches what
only happens once (first shader compiles, loading, cutscene transitions).

Which places a change needs (the rule since October 2026):

- **Trial variants** (several candidates, one of which will be kept): `=2` only where the change acts, that is the gate
  plus the place where the counter of the affected workaround or path is above 0.
- **The version that becomes the default**: `=2` in all five places plus a run from the start of the game.

## 4. Measuring speed: live A/B in the same run

Frame rate differs by about ±3% between two separate runs at the gate and ±6% in the open world, more than most
single improvements. So only one comparison counts: **old and new in the same run, switched live**.

```bat
set KYTY_LIVE_AB=KYTY_PERF_SOMETHING
set KYTY_PERF_SOMETHING=1
set KYTY_LIVE_AB_START_FRAME=1250
kyty_emulator.exe --game "<your game folder>"
```

- From the start frame the GPU thread rotates phases at frame boundaries: "off" (every listed switch off), then one
  phase per listed switch at its value. Each phase lasts `KYTY_LIVE_AB_PERIOD_MS` (default 5000 ms); its first
  `KYTY_LIVE_AB_WARMUP_FRAMES` frames are not measured. Every second cycle runs backwards
  (`KYTY_LIVE_AB_ORDER`, default alternate), so a slow drift of the scene cancels out.
- Each phase prints `PERF live_ab phase=… fps=… gpu_thread_busy_ms=… fence_wait_ms=… ft_mean_ms=… ft_median_ms=…`, and
  every few cycles a summary: mean fps per phase, its spread, and the per-cycle difference to "off" with its standard
  error.
- Only switches whose effect is a per-call choice can toggle live; they are listed in `src/common/liveAb.cpp` with the
  reason each one is exact at any frame boundary. Any other name is refused with a warning (and the run should be
  stopped: it would measure nothing).
- At the gate the frame rate is stable from about frame 1,000; the measurement starts at frame **1250** and runs for a
  fixed long window. Do not shorten the window.
- **One pair is enough** when the direction is clear (mean and median of the 5-second phases move the same way). A
  second pair only when the first is unclear; never a third.
- Several switches may share a run (each in its own phases), but their gains are **not added up**. A phase with all of
  them on (`KYTY_LIVE_AB_ALL=1`) is not used unless asked for.
- Intermediate steps are measured at the gate only; the open world is measured once at the end of a piece of work, as
  a safety check (no slowdown, no new errors), before the change becomes the default.
- Work is measured in closed pieces: every change inside a piece gets its `=2` check, the fps pair is taken once at the
  end of the piece. If the piece is slower, one extra run with parts switched off finds the culprit.

## 5. Measuring cost per pass

Every image fix states its cost in **ms per frame**: the GPU time of the pass it changes, or the live A/B result.

```bat
set KYTY_TRACE_GPU_TIME=1
set KYTY_TRACE_GPU_TIME_MARKERS=1
set KYTY_TRACE_GPU_TIME_TOP=40
```

- `GPUTIME` (every ~5 s): host GPU busy time, gaps, commands, time attributed to emulator operations and to nothing;
- `GPUTIME_TOP`: the costliest shaders by hash, with dispatch counts;
- `GPUTIME_MARKERS`: GPU time by the game's pass names;
- `KYTY_TRACE_GPU_TIMELINE=<frame>`: per command buffer, in order, for three frames (what runs while the CPU waits).

Per-command timestamps themselves cost about 0.3 µs per command, so compare timed runs only with timed runs. A shader
change is measured by the time of its pass, not by whole-frame fps alone. Performance work starts with the largest
items: GPU time of a pass and busy time of the emulator's GPU thread (`gpu_thread_busy_ms` in the live A/B lines).

## 6. Warm the shader cache before measuring

A changed emulator build often changes the SPIR-V of many shaders, and the host driver needs up to tens of seconds for
a large compute pipeline. A run with a cold cache measures compilation, not the change.

- Run `kyty_emulator.exe --game "<your game folder>" --precompile-only` (or the launcher's "Precompile shaders")
  after building, or play one warm-up run first. Shader journals are kept per game in `_PipelineCache/`.
- Check that background compiles are finished before the measured window: `--precompile-only` prints
  `PRECOMPILE_PROGRESS` / `PRECOMPILE_DONE` lines.
- Keep one pipeline cache per build you compare. A run of another build rewrites the warm list and the driver cache,
  and the next run of the first build is cold again.
- Changes that alter SPIR-V cannot be measured live in one module; they get a shader variant bit so both programs exist
  side by side (as the pair-mode and half-forward switches do), or a separate warmed run per variant.

## 7. Pictures

- Dump presented frames: `KYTY_TRACE_FRAME_RANGE=<first>-<last>` with `KYTY_TRACE_FRAME_DIR=<folder>` writes each
  presented frame of the range as raw 3840×2160 data (10-bit RGB, A2B10G10R10), which can be compared pixel by pixel
  or turned into PNG.
- Fixed views that show most regressions: the tree at the gate, a kimono (cloth and skin), the ground (terrain,
  decals, grass), water near the house by the lake.
- The game animates (wind, particles, time of day), so one pair of frames proves little: a claim about the picture
  needs **3 or more runs**.
- When a fix is visible, add before/after pictures to the pull request (your own play, no game files).

## 8. Workaround counters and settings in the log

- `KYTY_TRACE_PERF=1` prints `PERF workaround_hits …` every few seconds: for each approximation still in the code,
  how often it changed what a draw or dispatch got (null descriptors, skipped shaders, loop caps, mip-chain clamps,
  depth-encoding leniency, …). **0 means the approximation did nothing in that run**: in that place the game never
  reaches it, and an exact version can be tested there by turning the approximation off.
- The first lines of every log list the settings the run used: `ENV_DEFAULT KYTY_X=value source=env|code kind=…`.
  Check them when comparing two runs.
- Errors to look for in every run: device lost, emulator exits that name a shader or resource, and
  `PREP_WORKER_COMMIT_WAIT_STUCK` (a preparation worker stuck waiting).

## 9. Exactness tools

- **Reference capture + interpreter.** `KYTY_TRACE_REF_CAPTURE=<shader hash>:<first frame>` (folder from
  `KYTY_TRACE_REF_DIR`) captures one dispatch of that compute shader: registers, user data, shader code, reachable
  memory and every bound buffer and image before and after. `tools/refinterp` (`refinterp run <capture folder>`) runs
  it on the CPU from AMD's machine-readable ISA and compares with what the translated shader wrote.
- **Twin dispatch.** `KYTY_DIAG_TWIN_DISPATCH=<shader hash>:<every n>:<variant bits>` runs a dispatch with two variants
  of the translated shader on the same inputs and compares the outputs on the GPU (`differ=` must be 0). Run a
  control (`variant bits` 0, the same module twice) first: shaders that read pixels they write, or compact results
  with atomics, can differ even against themselves.
- **Validation.** `--vulkan-validation true` (slow) for the Vulkan layers; SPIR-V modules can be dumped
  (`KYTY_TRACE_SHADER_CODE_DIR`, `KYTY_TRACE_SPIRV_TEXT=<hashes>`) and checked with `spirv-val --target-env vulkan1.3`.
- **Unit tests** live in `tests/` and build as separate targets (not part of the default build). They run with
  `KYTY_ENV_DEFAULTS=0`, so they check the strict paths without the defaults table.

## 10. Checklist for a change

1. Switch, off by default; `=2` verify mode with a counter.
2. `=2` with 0 differences: gate + the place where it acts (trial variant) or all five places + a run from the start
   (version for the default).
3. Warmed shader cache; one live A/B pair at the gate (open world once at the end of the piece of work).
4. Cost in ms per frame of the pass it touches.
5. Pictures (3+ runs) for anything visible.
6. Default in code with the numbers in the comment; old path and replaced workaround removed.
