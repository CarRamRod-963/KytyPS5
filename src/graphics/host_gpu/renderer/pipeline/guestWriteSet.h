#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTWRITESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTWRITESET_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// The guest ranges a draw or dispatch writes (storage images, attachments and their metadata,
// written buffers), collected once per commit and tested against each guest read its resource
// specialization depends on. The ranges' common bound rejects a read outside all of them with
// one comparison. A read inside the bound is answered by a binary search over the ranges sorted
// by address, built at the first such query after the last Add: a dispatch may write thousands
// of candidate buffers and read tens of thousands of table ranges between them.
class GuestWriteSet {
public:
	enum class Kind : uint8_t { Image, Buffer };

	struct Range {
		uint64_t address = 0;
		uint64_t size    = 0;
		Kind     kind    = Kind::Image;
	};

	// A range is valid when it is not empty and does not wrap the address space.
	[[nodiscard]] static constexpr bool ValidRange(uint64_t address, uint64_t size) noexcept {
		return size != 0 && address <= UINT64_MAX - size;
	}

	void Clear() noexcept {
		m_ranges.clear();
		m_sorted.starts_and_ends.clear();
		m_begin = UINT64_MAX;
		m_end   = 0;
	}

	// Empty ranges write nothing and are ignored; a wrapping range is refused (false).
	[[nodiscard]] bool Add(uint64_t address, uint64_t size, Kind kind) {
		if (size == 0) return true;
		if (!ValidRange(address, size)) return false;
		m_ranges.push_back({address, size, kind});
		m_sorted.starts_and_ends.clear();
		m_begin = std::min(m_begin, address);
		m_end   = std::max(m_end, address + size);
		return true;
	}

	[[nodiscard]] bool Empty() const noexcept { return m_ranges.empty(); }

	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }

	// The first added range that overlaps a valid range [address, address + size), or nullptr.
	[[nodiscard]] const Range* FindOverlap(uint64_t address, uint64_t size) const {
		if (m_ranges.empty() || address >= m_end || m_begin >= address + size) {
			return nullptr;
		}
		if (m_ranges.size() > LinearRanges && !Indexed().Overlaps(address, size)) {
			return nullptr;
		}
		for (const auto& range: m_ranges) {
			if (address < range.address + range.size && range.address < address + size) {
				return &range;
			}
		}
		return nullptr;
	}

private:
	// Up to this many ranges a scan is cheaper than building the index.
	static constexpr size_t LinearRanges = 16;

	// The ranges by start address, each with the furthest end among it and every range before it.
	struct SortedRanges {
		std::vector<std::pair<uint64_t, uint64_t>> starts_and_ends;

		[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size) const {
			// The ranges starting before the read's end overlap it when one of them ends after
			// its start, i.e. when the furthest end among them does.
			const auto end = address + size;
			const auto after =
			    std::partition_point(starts_and_ends.begin(), starts_and_ends.end(),
			                         [end](const auto& entry) { return entry.first < end; });
			return after != starts_and_ends.begin() && std::prev(after)->second > address;
		}
	};

	const SortedRanges& Indexed() const {
		if (m_sorted.starts_and_ends.empty()) {
			auto& entries = m_sorted.starts_and_ends;
			entries.reserve(m_ranges.size());
			for (const auto& range: m_ranges) {
				entries.emplace_back(range.address, range.address + range.size);
			}
			std::ranges::sort(entries);
			for (size_t i = 1; i < entries.size(); ++i) {
				entries[i].second = std::max(entries[i].second, entries[i - 1].second);
			}
		}
		return m_sorted;
	}

	std::vector<Range>   m_ranges;
	mutable SortedRanges m_sorted;
	uint64_t           m_begin = UINT64_MAX;
	uint64_t           m_end   = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTWRITESET_H_
