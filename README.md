# KytyPS5 — Saros Experimental Fork

An unofficial experimental fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5) focused on performance, stability, and rendering correctness for **Saros** (`PPSA07631`, game version `01.005.002`).

> **Note:** This is an independent fork and is not affiliated with the upstream `KytyPS5` project. Please report any issues, logs, or test results directly to this repository's [Issues](../../issues) tab rather than upstream.

---

## Key Improvements in This Fork

- **Path-finding & Red-Zone Stability (Intel CPUs without AVX-512):** Expanded Windows SysV red-zone exception protection (covering 679 game functions) to prevent guest crashes (`0x900e7c3d1`) during large fights and navigation mesh updates.
- **Render-Thread Synchronization Fixes:** Resolved race conditions (`0x90083b16d`) and GPU-to-CPU readback ordering issues so the ready signal is only raised after GPU results reach guest memory.
- **Texture Binding & Visual Artifacts:** Improved handling of previously rejected textures and dynamic texture reference updates to reduce colored fog, glow blobs, and rainbow artifacts.
- **Intel CPU Performance Tuning:** Dedicated options for higher-priority GPU threads, secondary command-recording threads, and selective readbacks to resolve 10–20 FPS bottlenecks on 12th–14th Gen Intel CPUs.

---

## System Requirements

- **OS:** Windows 10 / Windows 11 (x64)
- **CPU:** x86-64-v3 processor with **AVX2** support
- **GPU:** **Vulkan 1.3** capable graphics card (NVIDIA, AMD Radeon, or Intel) with up-to-date drivers
- **Runtime:** [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://microsoft.com)
- **Game Files:** Your own legally obtained backup copy of the game (no copyrighted files or keys are included).

---

## Quick Start & Upgrading

1. Download the latest portable archive from the [Releases](../../releases) page.
2. Extract the zip into a **new, empty folder** (settings `Kyty.ini`, saves, shader caches, and logs stay completely portable inside this folder).
3. Launch `kyty_launcher.exe`, select your game folder, and set your preferred display resolution.
4. Double-click the game row (or right-click -> **Run**).

### Upgrading from an Earlier `saros` Build
- Extract the new release into a fresh folder and copy over your `Kyty.ini` and `_SaveData` folder.
- You can also copy `_PipelineCache` (the emulator will automatically rebuild updated shaders in the background during the first boot, which takes ~1 minute).

> **What to expect:** The first session compiles shaders on first use and caches them to disk, so initial loading takes a few minutes with occasional stutters in new areas. Subsequent sessions run significantly smoother.

---

## Hardware-Specific Tips

### NVIDIA GeForce
- Run with the **default launcher settings** (one unified build supports both NVIDIA and AMD).

### AMD Radeon
- Keep **Maximal reconvergence** set to **Auto** (forcing `On` causes driver crashes on startup with current Adrenalin drivers).
- Leave **Emulate wave64 compute shaders** unticked.
- *Known visual glitches (black/blocky character, stretched triangles, dark specks):* If encountered, capture a screenshot on default settings, then tick **Skip shader simplification for the driver**, restart, take a second screenshot in the same spot, and attach both screenshots along with `_kyty.txt`, GPU model, and driver version.

### Intel CPUs (12th–14th Gen)
If you still experience low frame rates after the default improvements, test these options **one step at a time**:
1. Test with defaults and `Printf output: File`.
2. Set **Read GPU results in place** to **Off**.
3. Enable **Run the GPU thread at higher priority** (*Intel CPU options*).
4. Enable **Record GPU commands on a second thread (experimental)**.
5. Enable both **Read back only what the game touches** and **Let game threads copy GPU results themselves**.

---

## How to Submit a Useful Bug Report / Log

Before reproducing an issue to attach `_kyty.txt`, please ensure your **Global Settings** (`Edit global settings and game folders`) match the clean baseline:

- **Debug tab:**
  - `Printf output`: **File** (attach `_kyty.txt` from the emulator folder)
  - `Shader logging`: **Silent** (enabling it creates huge logs and disables background shader compilation)
  - Leave validation, RenderDoc, command buffer dumps, Tracy, and `Submit GPU work every N draws` (`0`) disabled.
- **General tab:**
  - Keep **Enable occlusion queries** and **Lazy GPU memory synchronization (faster)** ticked.
  - Keep **Maximal reconvergence** and **Read GPU results in place** on **Auto**.
  - Leave all experimental readback, tessellation, wave64, ray tracing, PS5 Pro, and Intel compatibility checkboxes **unticked** unless specifically testing a step above.
- **Per-Game Config:** Right-click the game and verify no custom overrides are left over (or click **Clear game config**).

---

## Credits

- Upstream emulator core by [KytyPS5](https://github.com/KytyPS5/KytyPS5) and its contributors.
- Saros compatibility, synchronization, and CPU/GPU fixes by [@CarRamRod-963](https://github.com/CarRamRod-963/KytyPS5).
