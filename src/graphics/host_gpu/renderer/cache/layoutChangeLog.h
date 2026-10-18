#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_LAYOUTCHANGELOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_LAYOUTCHANGELOG_H_

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// The guest memory ranges whose image layout changed (an image inserted, registered,
// unregistered, deleted, expanded or given a stencil plane), in order. A stamp taken at a lookup
// can later be asked whether a change touched a range since; memos of image lookups stay valid
// while it did not.
//
// Recent changes are kept exactly. Past MaxRecent the older half is folded into granules that
// remember the last change touching them, so a query about an old stamp sees every change it
// saw before, only rounded out to granules: an answer can turn from "unchanged" into "changed"
// (one more lookup), never the other way. Stamps never expire as a whole, so memos of unrelated
// memory survive however many images come and go elsewhere.
class LayoutChangeLog {
public:
	struct Stamp {
		uint64_t sequence = 0;
	};

	static constexpr size_t   MaxRecent    = 4096;
	static constexpr uint64_t GranuleBits  = 21;
	static constexpr uint64_t GranuleBytes = uint64_t {1} << GranuleBits;

	[[nodiscard]] Stamp Now() const noexcept { return {m_sequence}; }

	void Note(uint64_t address, uint64_t size) {
		if (size == 0) return;
		if (m_recent.size() >= MaxRecent) Fold(MaxRecent / 2u);
		m_recent.push_back({++m_sequence, address, size});
	}

	[[nodiscard]] bool ChangedSince(Stamp stamp, uint64_t address, uint64_t size) const {
		if (stamp.sequence >= m_sequence || size == 0) return false;
		for (auto it = m_recent.rbegin(); it != m_recent.rend(); ++it) {
			if (it->sequence <= stamp.sequence) return false;
			if (address < it->address + it->size && it->address < address + size) return true;
		}
		// Every change after the stamp that is not in the recent list is in the granules.
		if (stamp.sequence >= m_folded_through) return false;
		const auto first = address >> GranuleBits;
		const auto last  = (address + size - 1u) >> GranuleBits;
		if (last - first >= m_granules.size()) {
			return std::ranges::any_of(m_granules, [&](const auto& granule) {
				return granule.first >= first && granule.first <= last &&
				       granule.second > stamp.sequence;
			});
		}
		for (auto granule = first; granule <= last; ++granule) {
			const auto found = m_granules.find(granule);
			if (found != m_granules.end() && found->second > stamp.sequence) return true;
		}
		return false;
	}

	[[nodiscard]] size_t RecentCount() const noexcept { return m_recent.size(); }
	[[nodiscard]] size_t GranuleCount() const noexcept { return m_granules.size(); }

private:
	struct Change {
		uint64_t sequence;
		uint64_t address;
		uint64_t size;
	};

	void Fold(size_t count) {
		for (size_t index = 0; index < count; ++index) {
			const auto& change = m_recent[index];
			const auto  first  = change.address >> GranuleBits;
			const auto  last   = (change.address + change.size - 1u) >> GranuleBits;
			for (auto granule = first; granule <= last; ++granule) {
				auto& sequence = m_granules[granule];
				sequence       = std::max(sequence, change.sequence);
			}
			m_folded_through = change.sequence;
		}
		m_recent.erase(m_recent.begin(), m_recent.begin() + static_cast<std::ptrdiff_t>(count));
	}

	std::vector<Change>                    m_recent;
	std::unordered_map<uint64_t, uint64_t> m_granules;
	uint64_t                               m_sequence       = 0;
	uint64_t                               m_folded_through = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_LAYOUTCHANGELOG_H_ */
