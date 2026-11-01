#ifndef KYTY_COMMON_OVERLAYSTATS_H_
#define KYTY_COMMON_OVERLAYSTATS_H_

#include "common/onceValue.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

// KYTY_OVERLAY=1 (user play only, default off; automated runs never set it): the in-game performance overlay
// (graphics/presentation/perfOverlay.cpp, F2 hide/show, Shift+F2 compact). The counters below are the ones the
// emulator had no source for yet; each hook is one load of a constant-initialized flag and a not-taken branch while the
// switch is off, a relaxed atomic add while it is on. The overlay's sampler reads them twice a second and shows the
// difference per second (they only grow; nothing resets them). With the switch on, Common::StartupTrace's counters are
// collected too (shaders, pipelines, uploads, fence waits, guest file reads; startupTrace.h) without its reporter.
namespace Common::OverlayStats {

enum class Counter : uint32_t {
	Draw,             // host draws recorded for guest draw packets (DrawIndex, DrawIndexAuto, CPU-expanded indirect)
	Instance,         // their instances (bytes slot: unused)
	IndirectDraw,     // indirect draw packets (DrawIndirect / DrawIndirectMulti), whichever path runs them
	Dispatch,         // compute dispatches (direct and indirect)
	QueueSubmit,      // vkQueueSubmit of render command buffers (CommandScheduler)
	FlipRequest,      // sceVideoOutSubmitFlip / EOP flip requests reserved in the flip queue
	FlipQueueFull,    // flip requests refused with VIDEO_OUT_ERROR_FLIP_QUEUE_FULL
	LateVblank,       // vblank periods the present thread ended after the next vblank was due
	SwapchainRebuild, // swapchain recreations (resize, HDR switch, out of date, surface lost)
	GpuMemoryFault,   // host access violations on tracked guest memory handed to the GPU-memory fault handler
	GuestFileWrite,   // sceKernelWrite/Pwrite/Pwritev to files (bytes slot: bytes written)
	AudioStarved,     // audio out: a primed port's SDL queue held less than one buffer when the next one came
	ImageEvict,       // texture cache collector freed an image (bytes slot: its size)
	BufferEvict,      // buffer cache collector retired a buffer (bytes slot: its size)
	RecordSpin,       // Thread_VkRecord busy-waiting for commands (bytes slot: ns); the overlay takes it off its CPU time
	PrepSpin,         // Thread_PrepWorker busy-waiting for a job (bytes slot: ns); likewise
	Count
};

struct Slot {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> bytes {0};
};

// Constant-initialized: no guard, no registration for destruction.
inline constinit Slot g_slots[static_cast<uint32_t>(Counter::Count)] {};

// KYTY_OVERLAY=1: counters + the in-game overlay; =2: counters + a text copy of the overlay in stderr every 10 s
// (PERF overlay lines, for automated runs), nothing drawn.
inline uint32_t Mode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	return mode_once.Get([] {
		const char* value = std::getenv("KYTY_OVERLAY");
		return value != nullptr && (value[0] == '1' || value[0] == '2') ? static_cast<uint32_t>(value[0] - '0') : 0u;
	});
}
inline bool Enabled() {
	return Mode() != 0u;
}
inline bool LogOnly() {
	return Mode() == 2u;
}

inline void Add(Counter counter, uint64_t count = 1, uint64_t bytes = 0) noexcept {
	auto& slot = g_slots[static_cast<uint32_t>(counter)];
	slot.count.fetch_add(count, std::memory_order_relaxed);
	if (bytes != 0) slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Count(Counter counter) noexcept {
	return g_slots[static_cast<uint32_t>(counter)].count.load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Bytes(Counter counter) noexcept {
	return g_slots[static_cast<uint32_t>(counter)].bytes.load(std::memory_order_relaxed);
}

} // namespace Common::OverlayStats

#endif // KYTY_COMMON_OVERLAYSTATS_H_
