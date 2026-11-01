#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFOVERLAY_H_

#include "common/overlayStats.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

union SDL_Event;

// KYTY_OVERLAY=1 (user play only, default off; automated runs never set it): an in-game performance overlay drawn by
// the system overlay's ImGui pass on top of every presented frame. F2 hides/shows it, Shift+F2 switches between the
// full page and the compact one (FPS, frame-time graph, limit line). A low-priority sampler thread gathers the numbers
// twice a second (OS counters, per-thread CPU, guest memory, the emulator's counters); the present thread only draws
// the last page and, every 0.5 s, reads the Vulkan heap budgets. With the switch off nothing here runs and every hook
// elsewhere is one load of a constant-initialized flag.
namespace Libs::Graphics {

struct GraphicContext;

[[nodiscard]] inline bool PerfOverlayEnabled() noexcept {
	return Common::OverlayStats::Enabled();
}
// Enabled and not hidden with F2.
[[nodiscard]] bool PerfOverlayVisible() noexcept;
// Window event thread: F2 / Shift+F2 (true: the event was the overlay's key and is consumed).
bool ProcessPerfOverlayKey(const SDL_Event& event);
// Presenter: a new frame was handed to the swapchain (guest flip or blank frame) with this source size.
void PerfOverlayNoteFrame(uint32_t width, uint32_t height);
// Swapchain::Create: the window's swapchain.
void PerfOverlayNoteSwapchain(uint32_t width, uint32_t height, vk::PresentModeKHR mode, uint32_t images, bool hdr);
// Presenter, every frame: starts the sampler and reads the GPU heaps when KYTY_OVERLAY is 1 or 2 (=2 draws nothing).
void PerfOverlayTick(const GraphicContext& graphics);
// SystemOverlay::PrepareFrame, inside its ImGui frame (present thread).
void DrawPerfOverlay(const GraphicContext& graphics, vk::Extent2D extent);
// ShutdownSystemOverlayInput: stops and joins the sampler thread (idempotent).
void StopPerfOverlay();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFOVERLAY_H_
