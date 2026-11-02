# Architecture of the emulator (Yōtei fork)

This is a map of the parts of the emulator that matter for Ghost of Yōtei: where the game's GPU work goes, which
threads do what, and where the caches and memory tracking sit. It follows the code in this repository; file names are
relative to `src/`. For the game's side of the frame see [FRAME.md](FRAME.md); for how changes are checked see
[TESTING.md](TESTING.md).

## 1. The big picture

```
 game threads (native x86-64)                 emulator threads
 ───────────────────────────                  ──────────────────────────────────────────────────────────
 game code ── HLE libraries (libs/, kernel/)
              └ libAgcDriver: command buffers ──► GPU thread (guest_gpu/graphicsRun.cpp)
                                                   ├ PM4 packets → registers   (guest_gpu/command_processor/)
                                                   ├ draws / dispatches → Vulkan (host_gpu/renderer/)
                                                   │    ├ scout thread (reads ahead, one per graphics queue)
                                                   │    ├ preparation worker(s) (draw bindings)
                                                   │    └ shader / pipeline compile workers
                                                   └ command recording worker → vkQueueSubmit
                                                  presentation thread (presentation/) → window swapchain
```

- The PS5 CPU is x86-64, so the game's own code runs natively. The loader (`loader/`) maps the game's modules into a
  guest address space, and system libraries are replaced by high-level emulation (`libs/`, `kernel/`).
- Everything the game sends to the GPU arrives as AMD PM4 command buffers through the graphics driver library
  (`libs/libAgcDriver.cpp`, `libs/agc.cpp`). The emulator executes them on one **GPU thread** and turns them into
  Vulkan work on the host GPU.
- The host side is Vulkan 1.3. Game shaders (AMD RDNA2 machine code) are translated to SPIR-V by the emulator's own
  recompiler (`graphics/shader/recompiler/`).

## 2. The GPU thread

`graphics/guest_gpu/graphicsRun.cpp` owns the guest GPU queues: the graphics queue and the async compute queues. The
GPU thread runs one queue's command stream at a time in work slices. A stream that reaches a wait (for a memory value
another queue or the CPU will write) is suspended and resumed later, so queues interleave as on the console, but all of
them execute on one host thread and one host Vulkan queue.

For each packet the command processor (`guest_gpu/command_processor/pm4Handlers.cpp`) either

- **writes registers**: `SET_CONTEXT_REG`, `SET_SH_REG`, `SET_UCONFIG_REG` are decoded into typed state
  (`guest_gpu/hardwareContext.h`: render targets, viewports, blend, shader stage registers and user data), or
- **does work**: draws (`DrawIndex`, `DrawAuto`, indirect draws), dispatches, waits (`WAIT_REG_MEM`), memory writes,
  end-of-pipe events and labels, DMA and copies.

Draws go to `RenderExecutor` (`host_gpu/renderer/renderDraw.cpp`), dispatches to `renderCompute.cpp`. For one draw the
renderer must find or translate its shader programs, resolve its render targets, its pipeline, every texture, sampler
and buffer it reads (by walking the descriptors and tables the game left in guest memory), then record Vulkan
commands. At the gate scene that is several thousand draws per frame, and this CPU work, not the host GPU, limits the
frame rate today. The rest of this section is about spreading it over more threads without changing results.

### 2.1 Scout: reading ahead

`guest_gpu/drawSpeculator.*`. One scout thread per graphics queue walks the
command stream ahead of the GPU thread with a shadow copy of the shader registers. For draws and dispatches whose
program already exists it reads their resources from guest memory in advance and logs every read. When the GPU thread
reaches the draw it re-checks the log (the same bytes are still there, no pending GPU write or readback covers them)
and only then adopts the result. A changed byte means the result is thrown away and the work is done again in order.

### 2.2 Preparation workers and the run queue

`host_gpu/renderer/prepWorker.*`, `renderDraw.cpp`.

- **Run queue** (`KYTY_PERF_PREP_WORKER=3`, default). Consecutive draws separated only by register writes form a
  *run*. Each draw is queued at its packet with a copy of the registers and translated later, in order, at the point
  where something must follow it (another kind of packet, a submission, a wait, a guest memory read).
- **Binder** (`KYTY_PERF_RUN_QUEUE_BINDER=3`, default). Inside a run the binding preparation of draw *i* (textures,
  samplers, buffers, uploads) runs on a preparation worker while the GPU thread commits draw *i−1* and does the front
  of draw *i+1* (programs, render targets, pipeline). Whatever the preparation may not share with that commit waits
  for it first; anything that must happen in GPU order (recording a command, allocating from a GPU ring, a submission)
  is taken over by the GPU thread. Mode `=4` adds a separate commit worker.
- `KYTY_PERF_PREP_WORKERS=2` (off) adds a second preparation worker for pairs of draws; parts that change shared
  caches are kept in draw order between the two workers ("turns"), with a race checker
  (`KYTY_PERF_PREP_WORKERS_CHECK=1`).
- `KYTY_PERF_TABLE_ASYNC` (default on): a draw whose graphics pipeline does not exist yet is skipped while background
  workers create it; later draws of the same key take it. (The name is project D's, the table-draw path of 21.10,
  archived and removed on 31.10.2026 together with `KYTY_PERF_SELF_SERVE` and `KYTY_PERF_TABLE_BUFFERS`.)

Each mode has a verify mode and was made the default only after a measured gain (see [TESTING.md](TESTING.md)).

### 2.3 Command recording and submission

`host_gpu/renderer/commandRecorder.*`, `commandScheduler.*`.

- The `vkCmd*` calls of the render scheduler's command buffer are copied
  into a ring and a **recording worker** makes the driver calls in the same order, then ends and submits the buffer.
  The GPU thread never waits for the driver at a command buffer boundary. Other queue users (presentation, overlay,
  readback helpers) first wait until every render submission enqueued so far is made.
- The open command buffer is submitted (without waiting) after 128 draws or
  dispatches, so the host GPU starts while the CPU records the rest. At the gate that is ~90 command buffers per frame.
- Every command buffer completes a tick on a timeline semaphore (`masterSemaphore.*`); deferred work (freeing
  resources, applying readbacks, writing guest labels) is attached to ticks and runs once the host GPU passed them.

### 2.4 GPU clock

`host_gpu/renderer/virtualGpuClock.*` (`KYTY_FIX_VIRTUAL_GPU_CLOCK`, on). The GPU clock values the game asks for with
end-of-pipe events count the host GPU's busy time on the renderer's command buffers (100 MHz, idle gaps not counted),
and are written to guest memory when the host GPU reaches that point. The game's dynamic resolution reads these
timers. Because the emulated frame takes longer than on a PS5, the game stays at a lower internal resolution than on
the console; reporting the time a PS5 would take is open.

## 3. Shaders

### 3.1 Translation

`graphics/shader/recompiler/`:

- `frontend/` decodes RDNA2 machine code (`decode/`), builds the control flow graph (`cfg/`) and translates
  instructions into an SSA intermediate representation (`translate/`);
- `ir/passes/` simplifies it (constant propagation, dead code, read-lane elimination) and analyses resources: how the
  shader reaches its descriptors through user data and tables (`ResourceTracking`, `SrtWalker`, `SrtTape`) and what
  must be read from guest memory per draw (`ResourceMaterialization`);
- `backend/spirv/` emits SPIR-V. Notable parts: wave64 on 32-wide host subgroups (two guest lanes per invocation,
  or "pair mode" with exchanges between the two halves for listed shaders), exact 16-bit float paths, the AMD ray
  intersection instruction emulated in code (`spirvEmitterBvh.cpp`), tessellation and mesh stages, GPU pointer
  accesses through the page table (section 4.3).

A translated program is keyed by stage, code hash, user-data count, code size and a static key of the state that
changes the translation; a permutation adds the resource specialization (`host_gpu/renderer/pipeline/pipelineCache.*`).

### 3.2 Pipeline cache and precompilation

`host_gpu/renderer/pipeline/pipelineCache.*`, `shaderPrecompile.*`. Files live in `_PipelineCache/` next to the
emulator, per game title:

- the Vulkan **driver pipeline cache**, saved during play and at exit;
- **journals** (`<title>.cs.journal`, `<title>.gfx.journal`, `KYTY_SHADER_PRECOMPILE=1`, default): every compute
  permutation and every graphics stage and pipeline the GPU thread compiled, with all inputs needed to compile it
  again. The journal header names the GPU; records of another GPU are dropped;
- a **warm list** (`.warm`): pipelines already in the driver cache for this build, which need no background job. Its
  entries hash the effective values of the settings that change translated shaders or pipelines
  (`Common::ShaderEnv::Names` in `src/common/shaderEnv.h`, the `ENV_FINGERPRINT` log line) and no other `KYTY_`
  variable, so reports, the overlay or a dispatch skip list keep it. A setting read in the recompiler that is on
  neither that list nor its list of output-only settings does not compile (`Common::ShaderEnv::Read`).

At start, background workers (`KYTY_SHADER_PRECOMPILE_THREADS`, default 6, below normal priority) translate and
compile the journal's records and create their pipelines through the driver cache, so the driver has them before the
game asks. The GPU thread still translates its own programs; it takes a worker's result when the input is the same
word for word. A record that was compiling when a run ended without a
clean exit is skipped as a suspect for that build. `kyty_emulator --precompile-only` compiles the journals and exits
without starting the game (used by the launcher's "Precompile" button). A changed emulator build changes the SPIR-V of
many shaders, so the first start after an update recompiles them. A record made under other settings that this run
cannot translate (another resource layout or image format) is skipped and counted (`SHADER_PRECOMPILE done: ... records
rejected`), not compiled.

## 4. Memory

### 4.1 One guest memory, two copies

Guest memory is mapped in the emulator's address space (`kernel/memory.cpp`); the game reads and writes it directly.
The host GPU works on Vulkan buffers and images that mirror parts of it: the **buffer cache**
(`host_gpu/renderer/cache/bufferCache.*`) and the **texture cache** (`textureCache.*`, images found by guest address,
format and layout; tiled images are converted by compute shaders in `image/tiler.*`).

### 4.2 Page tracking

`host_gpu/memoryTracker.*`, `regionManager.h`, `pageManager.*`. Guest memory is tracked in 4 KiB pages grouped in
4 MB regions. A page can be:

- **CPU-modified**: the CPU wrote it since the GPU copy was made; it is uploaded before the GPU next reads it;
- **GPU-modified**: the GPU wrote it and guest memory is stale; a CPU access must first get the GPU's bytes.

The emulator learns about CPU accesses through page protection: write watchers catch CPU writes to pages the GPU
holds; access watchers catch CPU reads of GPU-modified pages. The faulting thread is stopped in the exception handler,
the page is resolved (download, or mark), and the instruction runs again. Uploads and protection changes of one draw
are applied together when it is committed.

### 4.3 GPU pointers (buffer device addresses)

Many Yōtei shaders read and write memory through 64-bit pointers instead of descriptors. The emulator keeps a page
table of host device addresses for 16 KiB guest pages (`BufferCache::BDA_PAGETABLE_SIZE`); translated shaders look up
each pointer there (`get_bda_pointer`). A lookup of a page without a host buffer sets a bit in a **fault buffer**
(`cache/faultManager.*`) that the GPU thread reads back to create the missing buffer for later draws (the missing
access itself reads zero). Creating the buffer for a whole 2 MiB window around the page (`KYTY_DIAG_FAULT_WINDOW_KB`)
is still a listed workaround.

Writes through pointers are the hardest case: the emulator must know which guest bytes the GPU now owns. The current
design lists the written ranges when the call is recorded ("record-keyed" ranges, `composed V#s` and
`KYTY_D6_C0`, on), so those bytes are GPU-owned exactly from the right point. A store the listing does not cover stops
the emulator with the shader's name instead of guessing. A few safety nets from before this design are still listed as
workarounds (item D6 in [README-YOTEI.md](../README-YOTEI.md#known-workarounds)).

### 4.4 Reading GPU results back

The game reads some GPU results on the CPU (hi-Z for culling, terrain images, small result buffers):

- **On demand**: a CPU access to a GPU-modified page faults and the GPU thread downloads that range (waiting for the
  host GPU if needed).
- **Eager readback**: 64 KiB windows the CPU keeps reading after GPU writes
  ("hot") are copied back asynchronously after each submission; once the copy is published and no newer GPU write
  happened, the pages stop being GPU-modified and the CPU reads them with no fault and no wait.
- **Linear images** (`the linear readback`, on): the pages of GPU-written linear images, and of images the texture
  cache replaced by a copy in another format, stay closed from the write on. The first CPU read faults, waits for the
  copy and reopens them, so the CPU sees the bytes of every write it has a label for. Pages nobody reads cost nothing.
- `KYTY_PERF_WRITE_PUBLISH` (off, verify mode `=2`): finished GPU writes to pages the CPU also writes are published
  from the host mapping instead of downloaded on the next fault.

## 5. Bindless textures

`host_gpu/renderer/pipeline/bindlessTable.*` (`KYTY_BINDLESS`, on). Terrain, foliage and other materials pick
textures at run time from descriptor heaps the game keeps in GPU memory. Every pipeline that samples such images gets
descriptor set 1:

- typed arrays of sampled images;
- a **translation buffer**: per guest heap, one entry per key pointing at an array slot (slots are shared by every
  heap entry holding the same descriptor words, so per-frame ring heaps re-resolve without new slots);
- a **feedback bitset** where shaders flag keys whose texture is not resident yet. The host reads a host-cached copy
  of it, resolves the flagged keys (creating images, up to a per-frame budget) and points their slots at them.

Until a key is resolved, it reads a placeholder; that is the "texture appears 1–3 frames late" workaround listed as
D1/D2 (`KYTY_DIAG_BINDLESS_EAGER`).

## 6. Settings: `common/envDefaults.h`

All settings are environment variables named `KYTY_*`, read once where they are used. Their defaults live in the code:

- `common/envDefaults.h` holds the table of settings that used to come from a launch script, each marked `config`
  (ordinary setting or measured performance switch), `correct` (hardware behaviour, confirmed; no entry now - such
  behaviour is no switch any more), `unproven` (probably
  hardware behaviour, exactness check open) or `workaround` (a logged approximation that stays until its replacement
  lands, with its work item). `Common::EnvOrDefault` returns the environment value or the table's default and prints
  every entry once at start (`ENV_DEFAULT KYTY_X=value source=env|code kind=…`), so each log lists the settings the run
  used.
- Every other switch has its default where it is read (`std::getenv("KYTY_...")`), with the reason and the measurement
  that made it the default in the comment above.
- The environment always wins (`KYTY_X=0` turns a switch off); `KYTY_ENV_DEFAULTS=0` ignores the whole table (the unit
  tests use it to check the strict paths).
- Naming: `KYTY_FIX_*` corrections, `KYTY_PERF_*` performance changes with an exact result, `KYTY_WORKAROUND_*`
  approximations that stay on until their replacement lands (`=0` turns one off), `KYTY_DIAG_*` diagnostic tools (off by
  default; a few older ones are still approximations), `KYTY_SDK_FIX_*` edge
  cases of texture sampling, `KYTY_TRACE_*` / `KYTY_CAPTURE_*` diagnostics only, off by default.

## 7. Measurement and checking built into the emulator

Short list; [TESTING.md](TESTING.md) explains how they are used.

| Part | Where | What |
|---|---|---|
| Live A/B | `common/liveAb.*` | switches toggled at frame boundaries in one run, fps per phase |
| GPU time | `host_gpu/renderer/gpuTimer.*` | per-command timestamps, per-shader and per-marker totals, timeline |
| Workaround counters | `common/workaroundHits.h` | how often each approximation changed a draw or dispatch (`PERF workaround_hits`) |
| Reference capture | `host_gpu/renderer/refCapture.*`, `tools/refinterp` | one dispatch captured and re-run on the CPU from the AMD ISA |
| Twin dispatch | `renderCompute.cpp` (`KYTY_DIAG_TWIN_DISPATCH`) | the same dispatch run with two shader variants, outputs compared on the GPU |
| Performance panel | `presentation/performanceOverlay.*` (F2, `--perf-overlay`) | fps, frame-time graph, CPU waiting on GPU, draw calls, compiles, transfers, memory in game |

## 8. Presentation

`graphics/presentation/` (`videoOut.cpp`, `presenter.h`). When the game flips, the presentation thread copies the
flipped image into the window's swapchain, scaled to the window, and draws the optional overlay on top. It records on
its own command buffers, separately from the renderer; the renderer's mutex keeps it from changing shared state in
the middle of a draw.
