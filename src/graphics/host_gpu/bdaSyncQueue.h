#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_

#include "graphics/host_gpu/rangeSet.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class BdaSyncDue;

// The guest ranges whose CPU-modified pages buffer device address (BDA) accesses must see,
// collected for the next synchronization (Take, on the GPU thread). The owner serializes Queue,
// BeginStoreEpoch and Take; NoteOrderingPoint and Take run on one thread.
//
// A range queued for the next access (a new buffer, a mapping, a store of the emulated command
// processor) is taken by the next Take. A store of a guest CPU thread is taken by the first Take
// after an ordering point of the command stream that orders it instead (NoteOrderingPoint). On
// the console a CPU store reaches GPU work only through such a point: it precedes the submission
// of that work, or the store of the value a wait of the work observes; a store made while the
// work runs races with it. A page the guest rewrites between two BDA accesses of the same work is
// then taken once per ordering point instead of once per access: one write protection and one
// copy instead of several.
//
// Stores are kept by store epoch: each guest submission begins a new one (BeginStoreEpoch), and
// the first command of the submission orders only the stores of earlier epochs. The guest writes
// the data of its next submissions while the GPU thread runs the earlier ones; those stores wait
// for the submission they precede instead of being taken, and the page faulted and taken again,
// in the middle of the guest's writes.
//
// Take only exchanges sets: the owner's serialization (guest threads queue from their write
// faults) is held for constant time, and the sets are merged by the taking thread, outside it,
// while it walks them (BdaSyncDue::ForEach).
class BdaSyncQueue final {
public:
	// Orders every store queued so far (all epochs).
	static constexpr uint64_t AllEpochs = UINT64_MAX;

	void Queue(uint64_t address, uint64_t size, bool at_ordering_point) {
		if (!at_ordering_point) {
			m_next_access.Add(address, size);
			return;
		}
		if (m_stores.empty() || m_stores.back().epoch != m_epoch) {
			m_stores.push_back({m_epoch, {}});
		}
		m_stores.back().ranges.Add(address, size);
	}

	// A guest submission: returns its epoch, which orders the stores queued before this call.
	uint64_t BeginStoreEpoch() { return ++m_epoch; }

	// The stores of the epochs before `epoch` (AllEpochs: every store queued until the next Take)
	// are due at the next Take.
	void NoteOrderingPoint(uint64_t epoch = AllEpochs) { m_ordered = std::max(m_ordered, epoch); }

	// Moves the ranges to synchronize now into `due`, which must be empty. Ranges queued later,
	// and stores of epochs no ordering point has ordered yet, stay for a later Take. Pass the same
	// object every time and Clear it after use: the exchange then allocates little.
	void Take(BdaSyncDue& due);

private:
	struct EpochStores {
		uint64_t epoch = 0;
		RangeSet ranges;
	};

	RangeSet m_next_access;
	// Ascending epochs.
	std::deque<EpochStores> m_stores;
	uint64_t                m_epoch = 0;
	// The first Take synchronizes everything.
	uint64_t m_ordered = AllEpochs;
};

// The ranges one Take moved out of the queue: their union is due.
class BdaSyncDue final {
public:
	// The due ranges in address order, overlapping and adjacent ones merged.
	template <typename Func>
	void ForEach(Func&& func) {
		if (m_used <= 1) {
			m_next_access.ForEachUnion(m_used == 0 ? m_empty : m_stores[0],
			                           std::forward<Func>(func));
			return;
		}
		// Stores of several epochs: one sorted list of every range, merged while walking it.
		m_merge.clear();
		m_next_access.ForEach(
		    [&](uint64_t begin, uint64_t end) { m_merge.emplace_back(begin, end); });
		for (size_t i = 0; i < m_used; i++) {
			m_stores[i].ForEach(
			    [&](uint64_t begin, uint64_t end) { m_merge.emplace_back(begin, end); });
		}
		std::sort(m_merge.begin(), m_merge.end());
		for (size_t i = 0; i < m_merge.size();) {
			const auto begin = m_merge[i].first;
			auto       end   = m_merge[i].second;
			for (i++; i < m_merge.size() && m_merge[i].first <= end; i++) {
				end = std::max(end, m_merge[i].second);
			}
			func(begin, end);
		}
	}

	[[nodiscard]] bool Empty() const { return m_next_access.Empty() && m_used == 0; }

	void Clear() {
		m_next_access.Clear();
		for (size_t i = 0; i < m_used; i++) {
			m_stores[i].Clear();
		}
		m_used = 0;
	}

private:
	friend class BdaSyncQueue;
	RangeSet m_next_access;
	// The first m_used sets hold stores; the rest are kept for later Takes.
	std::vector<RangeSet>                      m_stores;
	size_t                                     m_used = 0;
	RangeSet                                   m_empty;
	std::vector<std::pair<uint64_t, uint64_t>> m_merge;
};

inline void BdaSyncQueue::Take(BdaSyncDue& due) {
	EXIT_IF(!due.Empty());
	m_next_access.Swap(due.m_next_access);
	while (!m_stores.empty() && m_stores.front().epoch < m_ordered) {
		if (due.m_used == due.m_stores.size()) {
			due.m_stores.emplace_back();
		}
		m_stores.front().ranges.Swap(due.m_stores[due.m_used++]);
		m_stores.pop_front();
	}
	m_ordered = 0;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_BDASYNCQUEUE_H_
