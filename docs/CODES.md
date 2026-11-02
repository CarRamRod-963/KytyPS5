# Project codes in the comments

Comments in `src` often name a change by a short code. The codes come from this fork's working lists, which are not part
of the repository (the same goes for the `notes\...` files some comments point to). This page says what each code still
found in `src` stands for, so a comment can be read without those lists.

How to read them:

- The same code can mean different things in different places. Always read it with the word in front of it: "D6 step
  C0" and "project C, stage C0" have nothing in common.
- "Step" and "stage" are local to the project named next to them. `K1` alone says nothing: "D4 step K1" and "D6 step K1"
  are unrelated.
- `KYTY_*` names are the switches themselves. In `KYTY_D6_C0=2` the value 2 is the check mode, see
  [TESTING.md](TESTING.md).

## Fixes and workarounds

| Code | What it is |
|---|---|
| D1 | Textures the GPU selects while drawing (bindless): every texture a draw can pick is translated before the draw. Also the null textures that stand in for unresolved ones. |
| D4 | Several images over the same guest memory (transient memory, dynamic resolution). `KYTY_WORKAROUND_ALIAS_SYNC` is the workaround. Steps: K0 the page event log of `KYTY_D6_C0=2/3`, K1 `KYTY_FIX_SHARED_IMAGE_MEMORY` (a K2 view-layout match was tried and removed). |
| D4b | Tracking what the GPU really wrote. Step 1 is the append windows: the bytes of an append pool the GPU did not write are released. |
| D4c | One holder for every byte of guest memory under several images: S0 what the recompiler knows of a written binding, S1 `KYTY_FIX_IMAGE_MEMORY_OWNER`. |
| D5 | Keeping the contents of render targets across a resolution change; goes together with D4. |
| D6 | Never losing bytes the GPU wrote in memory the CPU also writes. Steps: A a buffer for every page written through a pointer; B record-keyed V#s (B1 to B6 are listed in `ShaderIR.h`); C0 `KYTY_D6_C0`, ownership by byte instead of by page; K0 census only; K1 composed V#s; K2 written record-keyed lists (`KYTY_RECORD_KEYED_VERIFY`); K3 record counts supplied by the game; K4 the BDA fallback; K5 the exact scalar-read overlap check; K6 the late store report removed. |
| D6b | The follow-up to D6: indirect draws run on the GPU, BDA fallback entries cleared by range. |
| D7 | Shaders that were skipped or had their loops limited. |
| D8 | Stencil planes as surfaces in memory, and HTILE. |
| A5 | DCC colour metadata: the emulator tracks clears instead of reading the keys back. E0 counters, E1 keys read where they lie. |
| C7 | Workaround: dynamic sample offsets. |
| C10 | Workaround: bindless placeholders for unusable table entries. |
| W9 | Workaround removed: a shader that used to be skipped. |
| W11 | Workaround: a scalar base address a waterfall loop reads per lane is taken from record 0. |

## Entries of the numbered fix list

| Code | What it is |
|---|---|
| M06 | The BDA fallback: a mapped direct-memory page without a buffer gets a BDA table entry (D6 step K4). |
| M14 | Images that hold only the mip levels in use (its step S2). |
| M21 | Smaller SPIR-V: step A is the branchless raw dword read. |
| M42 | Approximations for lane exchange and waterfall pointers. |
| M46 | DCC keys in general: keys per access, metadata belongs to the owner of the memory. |
| M48 | Switches that turned out to be correct hardware behaviour and so became defaults: the confirmed ones are always on without a switch now, the `unproven` ones are `KYTY_FIX_*` entries of `envDefaults.h`. |
| M109 | Crash report for a V# at an invalid address. |
| M110 | Atomics on the same dword return their old values in lane order, as the hardware does. |
| M115 | One shader translating differently between sessions (empty and filled formatted buffers). |
| M118 | Guest reads of GPU-written images get the current bytes (hi-Z readback, then read coherence). |
| M119 | `KYTY_FIX_STENCIL_PLANE_WRITES`. |
| M120 | Writes to an image table the GPU indexes go the way reads do; also the capture frame for it. |
| M122 | `KYTY_FIX_HTILE_SLICE_CLEAR`: clears of single layers inside the metadata. |
| M124 | `KYTY_FIX_CLAMP_NAN`: the `.clamp` output modifier with DX10_CLAMP set. |

## Projects

| Code | What it is |
|---|---|
| A1 | Two uses. "A1 stage 1, stage 2": shared guest memory (labels at completion, `KYTY_PERF_DIRECT_RANGES`). "Project A, stage A1": the texture cache's lock-free answers. |
| Project C | Draw preparation. C0 the clean-draw census (`KYTY_TRACE_CLEAN_DRAWS`), C1 the memo of a draw's render target search. |
| Project D | The new draw path and its table mode (`KYTY_PERF_TABLE_BUFFERS`, removed); the name also survives in `KYTY_PERF_PREP_WORKERS`. |
| S0, S1, S2 | Draw phases in parallel: S0 measurement, S1 splitting a draw into preparation and commit on one thread, S2 preparation on a worker (steps B.3 to B.6b). The same letters also number the steps of other projects (D4c, texture streaming). |
| O1 | Record packets: commits written into the record ring, the calls made by the record worker. |
| P2, P3 | Multi-threaded draw preparation: P2 `KYTY_PERF_PLAN_STAGES`, P3 reading ahead after the GPU thread parsed the packet. |
| Stage F | Helper threads running the GPU thread's lookups ahead: F0 the census (removed), F2 the scout's lookups, F3 the render target inputs. Not the same as the F1 below. |
| F1 | Grouped IndirectBuffer reads in the recompiler (one bounds check, several loads). |
| T1 | Table sets: a stage's image set kept by its program and the words its images and samplers read. |
| E1, E2c, E5 | Bindless steps: E1 tables over 1 MiB and refresh events, E2c the heap path gone, E5 `KYTY_PERF_BINDLESS_CLASSES`. |
| C9 | Bindless: the list path of tables over 1 MiB. |
| B1 | In the specialization census: compiling at `AgcCreateShader`, built and dropped. Not the D6 B1 above. |
| K1 | "Active project step K1": timing the parts of a draw's front. "The K1 black-particles hunt": the wave64 pair rule. |
