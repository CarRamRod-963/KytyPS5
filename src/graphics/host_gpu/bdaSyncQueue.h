#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_

#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <utility>

namespace Libs::Graphics {

class BdaSyncDue;

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
//
// Take only exchanges sets: the owner's serialization (guest threads queue from their write
// faults) is held for constant time, and the two sets are merged by the taking thread, outside
// it, while it walks them (BdaSyncDue::ForEach).
class BdaSyncQueue final {
public:
	void Queue(uint64_t address, uint64_t size, bool at_ordering_point) {
		(at_ordering_point ? m_at_ordering_point : m_next_access).Add(address, size);
	}

	void NoteOrderingPoint() { m_ordering_point = true; }

	// Moves the ranges to synchronize now into `due`, which must be empty. Ranges queued later
	// stay for a later Take. Pass the same object every time and Clear it after use: the
	// exchange then neither allocates nor frees.
	void Take(BdaSyncDue& due);

private:
	RangeSet m_next_access;
	RangeSet m_at_ordering_point;
	// The first Take synchronizes everything.
	bool m_ordering_point = true;
};

// The ranges one Take moved out of the queue: their union is due.
class BdaSyncDue final {
public:
	// The due ranges in address order, overlapping and adjacent ones merged.
	template <typename Func>
	void ForEach(Func&& func) const {
		m_next_access.ForEachUnion(m_at_ordering_point, std::forward<Func>(func));
	}

	[[nodiscard]] bool Empty() const {
		return m_next_access.Empty() && m_at_ordering_point.Empty();
	}

	void Clear() {
		m_next_access.Clear();
		m_at_ordering_point.Clear();
	}

private:
	friend class BdaSyncQueue;
	RangeSet m_next_access;
	RangeSet m_at_ordering_point;
};

inline void BdaSyncQueue::Take(BdaSyncDue& due) {
	EXIT_IF(!due.Empty());
	m_next_access.Swap(due.m_next_access);
	if (std::exchange(m_ordering_point, false)) {
		m_at_ordering_point.Swap(due.m_at_ordering_point);
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_
