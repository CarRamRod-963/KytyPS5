#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_FILTEREDRANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_FILTEREDRANGESET_H_

#include "graphics/host_gpu/rangeSet.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

// A RangeSet that answers most Intersects queries without its tree. Every granule a range of the
// set touches has its bit set in a hashed filter, so a query whose granules have no bit set is
// disjoint from the set. Small ranges mark 64 KiB granules, large ones 32 MiB granules of a
// second filter, and a query consults both. Subtract leaves bits set (a stale bit costs one tree
// query); the filters are rebuilt from the ranges every RebuildSubtracts subtractions and cleared
// when the set empties.
//
// The renderer asks whether a few guest bytes hold unpublished GPU writes for every chunk an SRT
// walk reads and every strict read, hundreds of thousands of times a second, and the answer is
// almost always no.
class FilteredRangeSet final {
public:
	static constexpr uint64_t FineBits          = 16;
	static constexpr uint64_t CoarseBits        = 25;
	static constexpr size_t   FilterBits        = size_t {1} << 16u;
	static constexpr uint64_t MaxFineGranules   = 64;
	static constexpr uint64_t MaxCoarseGranules = 4096;
	static constexpr uint32_t RebuildSubtracts  = 1024;

	void Add(uint64_t address, uint64_t size) {
		// A covered range's granules are marked already.
		if (m_ranges.Add(address, size)) {
			Mark(address, size);
		}
	}

	void Subtract(uint64_t address, uint64_t size) {
		m_ranges.Subtract(address, size);
		if (m_ranges.Empty()) {
			Reset();
		} else if (++m_subtracts >= RebuildSubtracts) {
			Rebuild();
		}
	}

	void Clear() {
		m_ranges.Clear();
		Reset();
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		return MayIntersect(address, size) && m_ranges.Intersects(address, size);
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		return m_ranges.Contains(address, size);
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		m_ranges.ForEach(func);
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		if (MayIntersect(address, size)) {
			m_ranges.ForEachInRange(address, size, func);
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.Empty(); }

	[[nodiscard]] const RangeSet& Ranges() const { return m_ranges; }

private:
	using Filter = std::array<uint64_t, FilterBits / 64u>;

	static size_t Index(uint64_t granule) {
		return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ull) >> 48u);
	}
	static bool Test(const Filter& filter, uint64_t granule) {
		const auto index = Index(granule);
		return ((filter[index / 64u] >> (index % 64u)) & 1u) != 0;
	}
	static void Set(Filter& filter, uint64_t granule) {
		const auto index = Index(granule);
		filter[index / 64u] |= uint64_t {1} << (index % 64u);
	}

	void Mark(uint64_t address, uint64_t size) {
		if (size == 0 || m_saturated) {
			return;
		}
		const auto last = address + (size - 1u);
		if ((last >> FineBits) - (address >> FineBits) < MaxFineGranules) {
			for (auto granule = address >> FineBits; granule <= last >> FineBits; ++granule) {
				Set(m_fine, granule);
			}
			return;
		}
		if ((last >> CoarseBits) - (address >> CoarseBits) >= MaxCoarseGranules) {
			m_saturated = true;
			return;
		}
		for (auto granule = address >> CoarseBits; granule <= last >> CoarseBits; ++granule) {
			Set(m_coarse, granule);
		}
	}

	[[nodiscard]] bool MayIntersect(uint64_t address, uint64_t size) const {
		if (m_saturated || size == 0) {
			return true;
		}
		const auto last = address + (size - 1u);
		if ((last >> FineBits) - (address >> FineBits) >= MaxFineGranules) {
			return true;
		}
		for (auto granule = address >> CoarseBits; granule <= last >> CoarseBits; ++granule) {
			if (Test(m_coarse, granule)) return true;
		}
		for (auto granule = address >> FineBits; granule <= last >> FineBits; ++granule) {
			if (Test(m_fine, granule)) return true;
		}
		return false;
	}

	void Reset() {
		m_fine.fill(0);
		m_coarse.fill(0);
		m_saturated = false;
		m_subtracts = 0;
	}

	void Rebuild() {
		Reset();
		m_ranges.ForEach([this](uint64_t begin, uint64_t end) { Mark(begin, end - begin); });
	}

	RangeSet m_ranges;
	Filter   m_fine {};
	Filter   m_coarse {};
	bool     m_saturated = false;
	uint32_t m_subtracts = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_FILTEREDRANGESET_H_
