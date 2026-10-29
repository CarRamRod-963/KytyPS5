#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_

#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <utility>

namespace Libs::Graphics {

// The guest ranges whose CPU-modified pages buffer device address (BDA) accesses must see,
// collected for the next synchronization (Take, on the GPU thread). The owner serializes Queue and
// Take; NoteOrderingPoint and Take run on one thread.
//
// A range queued for the next access (a new buffer, a mapping, a store of the emulated command
// processor) is taken by the next Take. A store of a guest CPU thread is taken by the first Take
// after the next ordering point of the command stream instead (NoteOrderingPoint: a submission
// starts or resumes, or the command processor waits for a memory value). On the console a CPU
// store reaches GPU work only through such a point: it precedes the submission, or the store of
// the value a wait observes; a store made while the current work runs races with that work. A
// page the guest rewrites between two BDA accesses of the same work is then taken once per
// ordering point instead of once per access: one write protection and one copy instead of
// several.
class BdaSyncQueue final {
public:
	void Queue(uint64_t address, uint64_t size, bool at_ordering_point) {
		(at_ordering_point ? m_at_ordering_point : m_next_access).Add(address, size);
	}

	void NoteOrderingPoint() { m_ordering_point = true; }

	// The ranges to synchronize now. Ranges queued later stay for a later Take.
	[[nodiscard]] RangeSet Take() {
		auto taken = std::exchange(m_next_access, {});
		if (std::exchange(m_ordering_point, false)) {
			if (taken.Empty()) {
				taken = std::exchange(m_at_ordering_point, {});
			} else {
				m_at_ordering_point.ForEach(
				    [&](uint64_t begin, uint64_t end) { taken.Add(begin, end - begin); });
				m_at_ordering_point.Clear();
			}
		}
		return taken;
	}

private:
	RangeSet m_next_access;
	RangeSet m_at_ordering_point;
	// The first Take synchronizes everything.
	bool m_ordering_point = true;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_
