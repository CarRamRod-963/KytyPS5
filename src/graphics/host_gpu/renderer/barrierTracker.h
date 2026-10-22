#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERTRACKER_H_

#include <cstdint>

namespace Libs::Graphics {

// Chooses the pipeline barriers that order guest dispatches against the commands recorded
// around them. A dispatch used to be followed by a compute -> all-commands barrier and, when it
// writes memory, preceded by an all-commands -> compute barrier; a guest flush recorded an
// all-commands -> all-commands barrier. The tracker keeps every ordering those barriers gave
// and drops the parts nothing recorded since the last barrier can need:
//   - the barrier after a dispatch is deferred until a command that is not a guest dispatch
//     is recorded (or a barrier, or the end of the command buffer);
//   - between dispatches it narrows to compute -> compute, and to nothing between two
//     dispatches that only read;
//   - the barrier before a writing dispatch covers all earlier commands only when a command
//     other than a guest dispatch was recorded since an earlier barrier covered them;
//   - a guest flush is recorded only when a command other than a guest dispatch was recorded
//     since the last flush; pending dispatches are ordered by the deferred barrier instead.
// Commands other than guest dispatches stay ordered by their own barriers, as before.
class BarrierTracker {
public:
	enum class Barrier : uint8_t {
		None,
		// Compute-shader accesses -> compute-shader accesses.
		ComputeToCompute,
		// All earlier commands -> compute-shader accesses.
		AllToCompute,
		// Compute-shader accesses -> all later commands.
		ComputeToAll,
		// All earlier commands -> all later commands.
		AllToAll,
	};

	// A guest dispatch is about to be recorded. `writes` must be true whenever the dispatch may
	// write memory through any path (storage buffers and images, atomics, addresses, scratch).
	[[nodiscard]] Barrier BeforeDispatch(bool writes) noexcept {
		Barrier barrier = Barrier::None;
		if (writes && m_other_since_compute_order) {
			// Other commands order themselves before reads with their own barriers; a write
			// also waits for their reads.
			barrier                     = Barrier::AllToCompute;
			m_other_since_compute_order = false;
		} else if (m_compute_writes_since_compute_order ||
		           (writes && m_compute_since_compute_order)) {
			barrier = Barrier::ComputeToCompute;
		}
		if (barrier != Barrier::None) {
			m_compute_since_compute_order        = false;
			m_compute_writes_since_compute_order = false;
		}
		m_compute_since_compute_order = true;
		m_compute_writes_since_compute_order |= writes;
		m_compute_pending = true;
		return barrier;
	}

	// A command of any other kind is about to be recorded: a copy, a draw, a clear, emulator
	// compute work. Its accesses are unknown, so a later writing dispatch waits for everything.
	[[nodiscard]] Barrier BeforeCommand() noexcept {
		const auto barrier          = OrderDispatches();
		m_other_since_compute_order = true;
		m_other_since_flush         = true;
		return barrier;
	}

	// A pipeline barrier recorded by other code, the start of a render pass instance or the end
	// of the command buffer: pending dispatches are ordered first, as they were before.
	[[nodiscard]] Barrier BeforeBarrier() noexcept { return OrderDispatches(); }

	// A guest flush: all earlier commands -> all later commands.
	[[nodiscard]] Barrier Flush() noexcept {
		if (!m_other_since_flush) {
			// Only guest dispatches since the last flush; they are ordered against every later
			// command by the barriers above.
			return Barrier::None;
		}
		m_compute_pending                    = false;
		m_compute_since_compute_order        = false;
		m_compute_writes_since_compute_order = false;
		m_other_since_compute_order          = false;
		m_other_since_flush                  = false;
		return Barrier::AllToAll;
	}

	[[nodiscard]] bool HasPendingDispatches() const noexcept { return m_compute_pending; }

private:
	Barrier OrderDispatches() noexcept {
		if (!m_compute_pending) {
			return Barrier::None;
		}
		m_compute_pending                    = false;
		m_compute_since_compute_order        = false;
		m_compute_writes_since_compute_order = false;
		return Barrier::ComputeToAll;
	}

	// Dispatches recorded since the last barrier from compute to all later commands.
	bool m_compute_pending = false;
	// Dispatches (any, and writing ones) recorded since the last barrier from compute to compute.
	bool m_compute_since_compute_order        = false;
	bool m_compute_writes_since_compute_order = false;
	// Other commands recorded since the last barrier from all commands to compute, and since the
	// last all-commands barrier. Unknown at the start.
	bool m_other_since_compute_order = true;
	bool m_other_since_flush         = true;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERTRACKER_H_
