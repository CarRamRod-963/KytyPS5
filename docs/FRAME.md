# Anatomy of a Ghost of Yōtei frame

This document walks through one frame of Ghost of Yōtei as the emulator sees it, pass by pass, and says how the
emulator handles each pass today. It is written for contributors who want to know where time goes and where a visual
difference may come from.

**Where the information comes from.** Only from what anyone can observe while the game runs in the emulator:

- the game's own debug markers in its GPU command stream (pass names such as `Frame>Deferred` or
  `FSR3>FSR3-ACCUMULATE`), which the emulator can attribute GPU time to;
- the GPU work itself: which shaders run (identified by the hash of their code), draw and dispatch counts, render
  target sizes and formats, what each pass reads and writes;
- the emulator's logs and the picture on screen.

Nothing here is taken from the game's program code. Pass descriptions say what a pass *does on the GPU*, not how the
game decides to do it.

**How to see it yourself** (diagnostics, off by default; see [TESTING.md](TESTING.md)):

| Switch | Output |
|---|---|
| `KYTY_TRACE_GPU_TIME=1` | `GPUTIME` line every ~5 s: host GPU busy time, command count, time not attributed to a command |
| `+ KYTY_TRACE_GPU_TIME_TOP=<n>` | `GPUTIME_TOP`: the n costliest shaders (hash, ms, dispatches) |
| `+ KYTY_TRACE_GPU_TIME_MARKERS=1` | `GPUTIME_MARKERS`: GPU time summed by the game's marker path (top 40 per window) |
| `+ KYTY_TRACE_GPU_TIMELINE=<frame>` | `GPUTL`: for three frames from `<frame>`, one line per command buffer in submission order with its start, duration, idle gap and costliest markers |
| `KYTY_TRACE_VK_LABELS=1` | the game's markers (and the shaders of each draw) as Vulkan debug labels, for Nsight Graphics or RenderDoc |

## How to read the numbers

- Times are **host GPU time in the emulator** on the development machine (RTX 5070 Ti), October 2026 builds. They
  are not PS5 times: translated shaders cost differently from the original ones, and the emulator adds its own work
  (copies, barriers, format conversions) between the game's commands.
- Two scenes are used. **Gate**: daytime scene at a torii gate in Performance mode (no ray tracing), about 1,750 draws
  in the G-buffer pass alone and ~8,700 GPU commands per frame. **Night RT**: a night scene with the game's ray-traced
  lighting mode.
- Output is 3840×2160. The game picks its internal resolution itself (dynamic resolution from its own GPU timer); in
  the emulator it settles at 1920×1080 at the gate and 2560×1440 in the night RT scene, then FSR3 upscales. The
  game's GPU timer reads host GPU busy time (`KYTY_FIX_VIRTUAL_GPU_CLOCK`, on); because the emulated frame takes
  longer than on a PS5, the game stays at a lower resolution than on the console (at the gate, the lowest size it
  uses there). Reporting the time a PS5 would take is open.
- With per-command timing on, the gate frame has ~41 ms of host GPU work and the night RT frame ~56 ms. At the time of
  writing the frame rate is limited by the emulator's CPU side (its GPU thread, see [ARCHITECTURE.md](ARCHITECTURE.md)),
  not by the host GPU.
- About 3 ms per frame at the gate run with no marker (shown as `-`), and ~0.9–1.0 ms per frame is emulator work
  between the game's commands (barriers, image layout changes, uploads, tiling) in the last command buffers alone.

## The frame at a glance

The order below is the submission order of the gate frame's command buffers (`GPUTL`). Several async-compute passes
run beside the graphics work on the PS5; the emulator executes all queues on one host queue, so here they appear
interleaved. Within one command buffer the timeline lists passes by cost, not by order.

| # | Stage | Game markers (examples) | Gate ms/frame |
|---|---|---|---:|
| 1 | Particle simulation | `Particle CS Frame>Particle Update / Emit / Ribbon / Sort` | ~3.4 |
| 2 | Terrain generation | `Generate Terrain Index Buffers`, `Generate high-res terrain instances` | ~0.5 |
| 3 | Cloth | `CS Cloth` | 5.4 |
| 4 | Geometry streams, depth pre-pass, culling | `Frame>Proxy Set Fill Streams`, `Frame>CS Mesh`, `Frame>Terrain Clipmap Update`, `Frame>Terrain Deformation Update`, `Frame>Depth-only`, `Frame>Proxy Set Vcull` | ~1.5 |
| 5 | Async world updates | `Clear G-Buffer`, `Async Clouds Update`, `Ocean`, `Async Sh Grid Update`, `Async Sky LUT Update`, `Flicker Lights`, `Sky-Biased Lights Update` | ~1.3 |
| 6 | G-buffer | `Frame>Deferred`, `Grass>GrassCompute`, `Grass>GrassDraw`, `Deferred>Tiled Decals` | ~3.6 |
| 7 | Screen-space prep | `Ambient Spec Update`, `Ripple`, `Proxy Set Frame>Occlusion Depth Reproject`, `Frame>Water Ripple`, `Frame>Composite depth bias`, `Frame>Velocity` | ~0.5 |
| 8 | Particles without dependencies | `Particle CS Frame>Particle No Deps` | 3.6 |
| 9 | Sun shadow, screen space | `Screen Space Shadows`, `Dirshadow occlusion` | ~0.1 |
| 10 | Water surface | `Frame>Water Particles`, `Frame>Deferred wetness`, `Frame>Water Deferred` | ~0.4 |
| 11 | Shadow maps | `Frame>Cascaded shadow map`, `Cascaded shadow map>Finalize / Particle Finalize / Particle Shadow` | ~1.0 |
| 12 | Shadow resolve, light culling, SSR, SSDO, GI maps | `Enqueue Shadow Resolve`, `Resolve Shadow Map Slice to deferred shadow / to EVSM`, `Cull CS Lights`, `SSR`, `Async ambient sh`, `Proxy Set Frame>Proxy Set Cull`, `Frame>SSDO`, `Update Oct Maps`, `Directional Shadow Combine`, `Directional Shadow Copy` | ~2.9 |
| 13 | Deferred lighting | `Lighting>PS Lights`, `PS Lights>Finalize / Particle Finalize`, `Lighting>Water Lighting`, `Frame>Lighting` | ~0.7 |
| 14 | Volumetric fog | `Volfog` | 1.1 |
| 15 | Ambient, lights, skin, translucency | `Deferred Ambient>Eval non-RT Probes`, `Lighting>CS Lights`, `Frame>Separable Scattering`, `Postprocessing>Translucent Particles`, `Postprocessing>Hair` | ~4.2 |
| 16 | Upscaling and post-processing | `FSR3>FSR3-PREPARE-REACTIVITY`, `FSR3>FSR3-ACCUMULATE`, `Frame>Postprocessing`, `Postprocessing>Bloom` | ~2.0 |
| 17 | End of frame | `Post-frame` | <0.1 |

In ray-traced mode a `RTGI>…` chain and `Frame>Build BVH` are added (section 8).

## 1. Particles

**What the GPU does.** Particle systems are simulated in compute: `Particle Update`, `Particle Emit`,
`Particle Ribbon` (trails) and `Particle Sort` early in the frame, a large batch of `Particle No Deps` later (3.6 ms at
the gate, the second largest pass), and `Particle Finalize` steps inside the shadow and light passes that turn
particles into draw lists. Most of these are indirect dispatches whose size the GPU itself computed.
`Postprocessing>Translucent Particles` draws the soft mist volumes as a tessellated draw (LS `f8097866`, HS
`c2c7621a`, DS `3ff76523`, PS `c3c05f72`; the pixel shader marches through the volume). Its cost moves from frame to
frame (0.4–1.3 ms).

**In the emulator.** Indirect dispatches read their group counts from GPU memory when the GPU gets there. Coherent
(`GLC`) loads in shaders that write no buffers are routed through a coherent read-only alias. One particle compute shader (`142383a7de1e5111`) is still skipped
(`KYTY_DIAG_SKIP_CS_HASHES`, a listed workaround) until a place where the game runs it can be tested; the
`PERF workaround_hits skip_cs=` counter shows whether a run reached it.

## 2. Terrain

**What the GPU does.** `Generate Terrain Index Buffers` and `Generate high-res terrain instances` build the terrain
mesh for the frame in compute; `Depth-only>Generate high-res terrain verts` produces vertices for the depth pass.
`Frame>Terrain Clipmap Update` (not every frame) renders terrain material pages into large linear and tiled render
targets; `Frame>Terrain Deformation Update` keeps a small ping-pong height image for snow and mud tracks. Terrain
pixel shaders select their textures at run time from descriptor tables in GPU memory (bindless).

**In the emulator.** Bindless textures go through one descriptor set of typed arrays filled per guest descriptor heap
(`KYTY_BINDLESS`, on); see [ARCHITECTURE.md](ARCHITECTURE.md). Some of these images are also read back by the CPU (the
emulator sees CPU reads of their memory): linear images are written back to guest memory on the CPU's first read
(`the linear readback`, on). Workarounds still active here:
`KYTY_WORKAROUND_WATERFALL_POINTER_RECORD0` (the material table of some terrain shaders, item M42),
`KYTY_DIAG_BINDLESS_EAGER` and related bindless entries (a newly streamed texture may show a placeholder for 1–3
frames).

## 3. Cloth

**What the GPU does.** `CS Cloth` simulates clothing (kimono, banners) in compute: ~240 small dispatches per frame,
5.4 ms at the gate, the largest single marker there. The shaders read and write simulation state through 64-bit
pointers into GPU memory.

**In the emulator.** Pointer accesses are translated to Vulkan buffer device addresses through a page table of host
buffers. For the main cloth shader the page lookup is remembered between consecutive accesses to the same page
(`KYTY_PERF_BDA_ENTRY_SHARE`, on for that shader; `=2` checks every remembered lookup against a fresh one). An exact
variant that treats a single-workgroup dispatch's coherent accesses as workgroup-coherent exists behind
`KYTY_PERF_WG_COHERENCE` (off).

## 4. Geometry streams, depth pre-pass and occlusion culling

**What the GPU does.** The game culls on the GPU. `Proxy Set Fill Streams` and `CS Mesh` prepare per-object streams;
`Depth-only` renders a depth pre-pass; a hierarchical depth pyramid (hi-Z, level 0 R32F 960×540) is built from scene
depth and reprojected for the next frame (`Occlusion Depth Reproject`); `Proxy Set Vcull` and `Proxy Set Cull` test
object sets against it and write the draw lists that later passes draw indirectly. Two hi-Z copies alternate between
frames, and the CPU reads the hi-Z back (visible as CPU reads of that image's memory).

**In the emulator.** The hi-Z is written as an R32 colour image and sampled as a D32 depth image, so the texture
cache keeps a depth copy beside it. Until fix M118 that copy was never written back to guest memory, so the CPU read
stale hi-Z and distant houses, trees and fog vanished every other frame after cutscenes. The replacement image is now
read back too, and linear images reach guest memory on the CPU's first read (`the linear readback`, on). Side
effect: the culling lists became stable (~305–340 sets instead of alternating ~137/216), gate +20% fps,
open world +12%. GPU-generated (indirect) draws are queued with the surrounding direct draws instead of breaking the
batch. The GPU indirect path still does not reproduce state that
indirect draws leave behind.

## 5. Async world updates

**What the GPU does.** On async compute queues: `Async Clouds Update` (cloud shadow and coverage maps; section 11),
`Ocean`, `Async Sky LUT Update` (sky lookup tables), `Async Sh Grid Update` (a grid of spherical-harmonics ambient
light, ~0.9 ms at the gate, 1.8 ms in the night RT scene), `Flicker Lights` and `Sky-Biased Lights Update`.
`Clear G-Buffer` clears the targets of the next stage.

**In the emulator.** Async compute queues are emulated on the same GPU thread as the graphics queue: a queue's command
stream is suspended at a wait and resumed when the awaited value is written, while other queues run. Blocks made only
of conditional waits run their waits without reading the condition early (`the queue waits`, with a liveness
guard). Consecutive dispatches get no emulator barrier unless the guest synchronizes or the dispatch uses
the emulator's LDS buffer.

## 6. G-buffer (`Frame>Deferred`)

**What the GPU does.** The main geometry pass: ~1,750 draws per frame at the gate (the most draws of any pass), into
several colour targets plus depth and stencil. The stencil holds material classes that later passes test (for example
water writes `0x4C`, and lighting classifies tiles by `(stencil & 0x48) == 0x48`). Grass is generated and drawn here
(`GrassCompute`, `GrassDraw`); `Tiled Decals` applies decals in screen tiles; `Velocity` and
`Composite depth bias` follow.

**In the emulator.** This pass is where the CPU cost per draw dominates: every draw's registers, shader programs,
render targets, pipeline, textures and buffers are resolved before it is recorded. Most of that work runs on
preparation workers in parallel with the GPU thread (see [ARCHITECTURE.md](ARCHITECTURE.md)). Colour compression and
depth/stencil metadata are emulated with workarounds that are still on: `KYTY_WORKAROUND_DCC_FAST_CLEAR` and related A5
entries (simplified DCC clears), and the D8 stencil group (`KYTY_WORKAROUND_STENCIL_PLANE_SYNC`,
`KYTY_WORKAROUND_PRESERVE_STENCIL`, `KYTY_WORKAROUND_HTILE_FULL_RANGE`, …), which keep stencil as copies and treat HTILE as
"full range".

## 7. Shadows

**What the GPU does.**
- `Frame>Cascaded shadow map`: two D32 2048² cascades in the open world (several hundred draws each), plus a cache of
  far shadow pages: a D32 array of 256 pages of 128×128. After the first fill the game refreshes about one page per
  frame; for each refreshed page one compute shader (`b25b01ff1361e795`) clears that page's HTILE (depth metadata)
  and ~200 draws render into it. `Finalize`, `Particle Finalize` and `Particle Shadow` render particle and
  translucent shadows.
- `Screen Space Shadows` (contact shadows, `b90e2024732c6111` in the open world) and `Dirshadow occlusion`
  (`a7661ff4ea282325`, reads the hi-Z and writes a small result buffer).
- `Enqueue Shadow Resolve` (`d8959888`, builds tile lists), `Resolve Shadow Map Slice to deferred shadow / to EVSM`
  (`f166de52`, `4d845308`, `b7c3bc6f`), then `Directional Shadow Combine` (`3b9d96bf`), which merges cascades, far
  pages, particle shadows, cloud shadows and contact shadows into one screen-space shadow mask and blends it with the
  previous frame's mask, kept by `Directional Shadow Copy`.

**In the emulator.** `Directional Shadow Combine` was checked against the CPU reference interpreter on captured
dispatches (identical results). Fix M122: the emulator recognised an HTILE clear only at the image's base address and
tracked "cleared" only for the first 64 slices, so far pages were redrawn over stale depth and a dark patch with
straight page edges stayed on the distant landscape for hundreds of frames after cutscenes. Clears are now tracked per
slice (`KYTY_FIX_HTILE_SLICE_CLEAR`, on; `=2` found 0 differences on 2.7 M checks; cost below 0.01 ms). HTILE itself
is still emulated as "full range" (`KYTY_WORKAROUND_HTILE_FULL_RANGE`, workaround D8).

## 8. Ambient light, probes and RTGI

**Non-RT mode.**
- `Async ambient sh`: one compute shader computes depth min/max per 8×8 screen tile (`b7636df611835502`), another
  sorts the tiles into 9 lists (`7655afaf219f230f`).
- `Deferred Ambient>Eval non-RT Probes`: indirect variants per tile class (`85312e99`, `3ab9b653`, `8368b02a`,
  `2b018f33`, `40395313`, `5be61606`, `dad60b37`) write the ambient image that `CS Lights` reads. 2.7 ms at the gate.
- `Update Oct Maps` (`6cc64dee32dc7094`, two dispatches per frame, after the sort shaders `d7a83911` and `b5d1ae71`)
  updates octahedral irradiance and colour maps of the GI volumes around the camera (nine at the gate). Which volume's
  maps a workgroup writes is chosen on the GPU from a sorted job list and 32-entry descriptor tables.
- `Ambient Spec Update`: ambient specular data.

**Ray-traced mode** (night RT scene, ms/frame): `Frame>Build BVH` 2.4 (the game builds and refits its own bounding
volume hierarchy on the GPU every frame), `RTGI>Place Probes` 0.24 (`8b26ed44`), `RTGI>Choose Rays` 0.36,
`RTGI>Cast Rays` 6.6 (`759c666d`, 200–330 thousand rays per frame, closest hit), `RTGI>Cast Shadow Rays` 0.36
(`bbfb96d1`, any hit), `RTGI>Light Rays` 1.4 (`7ff41f74`), `RTGI>Eval RT Probes` 1.0, `RTGI>Filter Probes` 0.54.

**In the emulator.**
- These shaders use 64-thread workgroups (one guest wave of 64). The host GPU has 32-wide subgroups. By default a
  host invocation runs two guest lanes ("lane2"); listed shaders instead run one guest lane per invocation and
  exchange wave-wide values between the two host halves ("pair mode"; the probe shaders and
  `6cc64dee` are on the default list). For 64-thread groups the exchange goes through a workgroup barrier.
  Probe loads through GPU-built buffer descriptors of the common 16-bit-float format take an inline path (`KYTY_PERF_INDIRECT_LOAD_FAST`, on; `=2` checks every load, 0 mismatches on 4.7·10¹⁰).
  Together these took the probes from ~2.65 to ~1.65 ms and `6cc64dee` from 1.21 to 0.92 ms.
- Fix M120: `Update Oct Maps` reads used the GPU-selected table entry but its image stores always wrote entry 0, so
  one volume got everyone's data and the others stayed black. Stores now follow the selected entry
  (`indirect image writes`, on; +0.02 ms).
- The hardware ray-tracing instruction (`IMAGE_BVH64_INTERSECT_RAY`, one box or triangle node test) has no Vulkan
  equivalent that gives the same bits, so it is emulated in shader code with AMD's box and triangle arithmetic
  (`spirvEmitterBvh.cpp`). That is why `Cast Rays` is the costliest pass in RT mode. Faster exact variants (vector
  node fetches, remembered page lookups) are planned; using the host's RT cores would not match bit for bit and could
  only be an opt-in speed option.

## 9. Lighting

**What the GPU does.** `Cull CS Lights` sorts lights into 16×16 screen tiles; `Lighting>CS Lights` shades the
G-buffer in compute with the ambient image and the shadow mask; `Lighting>PS Lights` draws light volumes (with
`Finalize` and `Particle Finalize` steps); `Lighting>Water Lighting` classifies 8×8 sub-tiles by stencil and lights
water with indirect dispatches.

**In the emulator.** Fix M119: near buildings the game uses a second stencil plane, and the emulator backed both
planes with one host image; an upload of the second plane overwrote the first, the water classification found no
water tiles and the lake near a house by the shore stayed flat brown, with SSR and SSDO lost nearby. Each plane's
writes now reach its own image (`KYTY_FIX_STENCIL_PLANE_WRITES`, on; `=2` 0 differences; no measurable cost).

## 10. Water

**What the GPU does.** `Ocean` (async compute), `Ripple` and `Frame>Water Ripple` (ripple simulation),
`Frame>Water Deferred` (water surfaces into six targets; lakes are tessellated patch draws), `Frame>Water Particles`,
and `Frame>Deferred wetness` (wet surfaces; reads both stencil planes). An "Interior Volumes" step clears the second
stencil plane with a compute shader that treats it as a buffer.

**In the emulator.** See the stencil-plane fix in section 9. Tessellation is on by default (`--no-tessellation` turns
it off). Open: `Deferred wetness` still reads the first plane for both of its stencil bindings
(`KYTY_WORKAROUND_PRESERVE_STENCIL` is still a workaround), so some interiors may look wet.

## 11. Volumetric fog and clouds

**What the GPU does.**
- `Volfog`: two compute shaders (`c6b0a54e` 0.88 ms, `e94e0c58` ~0.23 ms) fill a froxel volume; a post-depth pass
  (`5ec70ce5`) applies it to the HDR light buffer.
- `Async Clouds Update`: tile classification (`ab8d16cc`, `3a2b5ad7`), an indirect ray-march (`f82e549a`) and a fill
  of empty tiles (`44e75231`) into a ring of 512² map pairs, a merge into cloud moment and coverage maps
  (`daa782b2`), and reductions read by the SH grid (`2cbedd74`). The 3D noise it marches through is generated once at
  load. `Directional Shadow Combine` reads the cloud maps for cloud shadows.

**In the emulator.** `Volfog` runs in pair mode with barrier exchanges (0.99 → 0.88 ms). The cloud chain was checked
against the reference interpreter (no worse results). Known small difference: 16-bit float denormals are flushed on
RG16F stores.

## 12. Screen-space reflections, occlusion and GI

**What the GPU does.** `SSR` (`61502c2f`, reads the first stencil plane); `Frame>SSDO` (screen-space directional
occlusion at half resolution, R8 960×540, with a horizontal and a vertical filter pass `1c26edd9` / `1cc26043`);
screen-space GI: input `8832ebbf` → ray pass `526db6dc` → horizontal filter `1608b938` → vertical and temporal filter
`0872689a` with history.

**In the emulator.** In our captures the SSGI horizontal filter leaves its input unchanged (its neighbour weights come
out 0 from the constants the game binds); the emulator runs exactly what the game provides, and the temporal step
still smooths the result. Whether the PS5 shows the same is not verified.

## 13. Skin and hair

**What the GPU does.** `Frame>Separable Scattering` (subsurface scattering for skin, `722412ab9e88b57e` /
`e2fc69dc66748b17`, wave64 with a per-pixel light-list loop) writes a B10G11R11 image composited into the HDR buffer;
`Postprocessing>Hair` shades hair.

**In the emulator.** Open: in one cutscene a dot grid appears on a face: NaN in red and green at one pixel per 8×4
tile, only where stencil bit 3 covers a tile partially. 8×4 pixels is exactly one 32-wide host subgroup, so this is an
emulator artefact; the cause is not found yet (several exact checks found no difference).

## 14. Upscaling: FSR3

**What the GPU does.** `FSR3>FSR3-PREPARE-REACTIVITY` (`00a9c156`) and a reactive-mask shader (`bfb173a9`) prepare
masks; `FSR3>FSR3-ACCUMULATE` (`d42889155b0a6b9f`, one 64-thread-group dispatch, 1.4 ms) upscales the frame to
3840×2160 with history; a copy (`d4d41b06`) brings it back to render size for the post chain. The RCAS sharpening
step never runs in our frames.

**In the emulator.** The accumulate shader is ALU bound in translation: it runs in pair mode with barrier exchanges,
native 16-bit float math and hardware conversions (4.87 → 1.40 ms since mid-October). An
exact pass that forwards 16-bit register halves instead of re-assembling 32-bit registers (`KYTY_PERF_HALF_FORWARD`,
off) is being measured.

## 15. Post-processing

**What the GPU does.** `Frame>Postprocessing` with `Postprocessing>Bloom`; a depth blur (2×2 downsample `6a2e2949` to
960×540, two blur passes `6136dd0e`, composite `cd9d0033` with at most 30% weight on far pixels); the final composite
(`ffc5cc51`). Depth of field appears only in cutscenes.

**In the emulator.** Open: distant scenery looks sharper than on the PS5. The blur itself reaches the screen with the
expected weight; one hypothesis is that sampler fields the emulator ignores (anisotropy threshold and bias, secondary
LOD bias, the "perf" mip and Z fields) make textures sharper. Not verified.

## 16. HUD and presentation

**What the GPU does.** The game draws its HUD and menus into the final image and flips it (3840×2160); `Post-frame`
is the last marker of the frame. *(Which marker the HUD draws belong to is not verified.)*

**In the emulator.** The presentation thread copies the flipped image to the window's swapchain, scaled to the window
size. The optional performance overlay (`KYTY_OVERLAY=1`, F2 / Shift+F2) is drawn on top of the presented frame and
never touches the game's image.

## Where the frame time goes, in short

| Pass (gate) | ms/frame | Pass (night RT) | ms/frame |
|---|---:|---|---:|
| `CS Cloth` | 5.4 | `Particle No Deps` | 7.3 |
| `Particle No Deps` | 3.6 | `RTGI>Cast Rays` | 6.6 |
| (no marker) | 3.1 | (no marker) | 4.6 |
| `Eval non-RT Probes` | 2.7 | `CS Cloth` | 3.1 |
| `Frame>Deferred` | 2.7 | `Particle Update` | 2.8 |
| `Particle Update` | 2.5 | `Frame>Build BVH` | 2.4 |
| `Update Oct Maps` | 1.6 | `Frame>Deferred` | 2.2 |
| `FSR3-ACCUMULATE` | 1.4 | `Translucent Particles` | 1.8 |
| `Volfog` | 1.1 | `Async Sh Grid Update` | 1.8 |
| `Async Sh Grid Update` | 0.9 | `FSR3-ACCUMULATE`, `RTGI>Light Rays` | 1.4 each |

Pass costs change with every optimisation; measure them again with the switches above before relying on a number.
