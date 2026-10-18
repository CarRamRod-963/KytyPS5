#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PENDINGDRAWPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PENDINGDRAWPOLICY_H_

#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

// With --skip-pending-draws, a draw whose graphics pipeline is still being built in the
// background may be left out instead of waiting for the driver. That is only safe when the
// picture is all the draw changes: nothing else can observe that it did not run. This is the
// reason a draw has to wait for its pipeline instead, or None when it may be skipped.
enum class PendingDrawBlocker : uint8_t {
	None,
	// The option is off.
	Disabled,
	// Dispatches always wait: a skipped dispatch is never replayed, and its writes are lost.
	Compute,
	// A stage stores to or atomically updates guest memory, GDS or a storage image.
	WritesMemory,
	// A guest query counts the draw's samples.
	QueryActive,
	// A color target is linear, or depth/stencil is written to a linear target. The CPU reads
	// such images back as data.
	LinearTarget,
	// The draw clears depth or stencil (DB_RENDER_CONTROL); the clear would be lost.
	DepthStencilClear,
};

struct PendingDrawDescription {
	bool skip_enabled        = false;
	bool compute             = false;
	bool writes_memory       = false;
	bool query_active        = false;
	bool linear_color_target = false;
	bool linear_depth_write  = false;
	bool depth_stencil_clear = false;
};

[[nodiscard]] constexpr PendingDrawBlocker
PendingDrawSkipBlocker(const PendingDrawDescription& draw) {
	if (!draw.skip_enabled) {
		return PendingDrawBlocker::Disabled;
	}
	if (draw.compute) {
		return PendingDrawBlocker::Compute;
	}
	if (draw.writes_memory) {
		return PendingDrawBlocker::WritesMemory;
	}
	if (draw.query_active) {
		return PendingDrawBlocker::QueryActive;
	}
	if (draw.linear_color_target || draw.linear_depth_write) {
		return PendingDrawBlocker::LinearTarget;
	}
	if (draw.depth_stencil_clear) {
		return PendingDrawBlocker::DepthStencilClear;
	}
	return PendingDrawBlocker::None;
}

[[nodiscard]] constexpr const char* PendingDrawBlockerName(PendingDrawBlocker blocker) {
	switch (blocker) {
		case PendingDrawBlocker::None: return "none";
		case PendingDrawBlocker::Disabled: return "skipping disabled";
		case PendingDrawBlocker::Compute: return "dispatch";
		case PendingDrawBlocker::WritesMemory: return "a stage writes memory";
		case PendingDrawBlocker::QueryActive: return "inside a query";
		case PendingDrawBlocker::LinearTarget: return "linear render target";
		case PendingDrawBlocker::DepthStencilClear: return "depth/stencil clear";
	}
	return "unknown";
}

// How long the draws of one pipeline may be skipped after its build was queued. Afterwards they
// wait for it, so a pipeline that takes very long to build cannot hide its geometry for good.
inline constexpr std::chrono::milliseconds PendingDrawSkipWindow {2000};

[[nodiscard]] constexpr bool PendingDrawMaySkip(PendingDrawBlocker                  blocker,
                                                std::chrono::steady_clock::duration since_queued,
                                                std::chrono::steady_clock::duration window) {
	return blocker == PendingDrawBlocker::None && since_queued < window;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PENDINGDRAWPOLICY_H_
