#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>

namespace Libs::Graphics {

class RangeSet final {
public:
	// The range a ForEachInRange call with a cursor ended in, for the next call of a sweep.
	class Cursor final {
	private:
		friend class RangeSet;
		std::map<uint64_t, uint64_t>::const_iterator m_range;
		bool                                         m_valid = false;
	};

	// Returns whether the set changed: a range already covered (a buffer written again) leaves
	// the set as it is without erasing and re-inserting its covering range.
	bool Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		if (it == m_ranges.end() || it->first > address) {
			it = m_ranges.emplace_hint(it, address, end);
		} else {
			if (it->second >= end) {
				return false;
			}
			it->second = end;
		}
		auto next = std::next(it);
		while (next != m_ranges.end() && next->first <= it->second) {
			it->second = std::max(it->second, next->second);
			next       = m_ranges.erase(next);
		}
		return true;
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	// ForEachInRange that first checks the range the cursor's previous call ended in: a query
	// inside that range needs no lookup (a sweep in address order: one lookup per range of the
	// set it enters). The set must not change while the cursor is in use.
	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Cursor& cursor, Func&& func) const {
		const auto end = End(address, size);
		if (cursor.m_valid && cursor.m_range->first <= address && cursor.m_range->second >= end) {
			func(address, end);
			return;
		}
		auto it = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			cursor.m_range   = it;
			cursor.m_valid   = true;
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	// The ranges of the union with `other` in address order, overlapping and adjacent ranges
	// merged as Add merges them, without building the union.
	template <typename Func>
	void ForEachUnion(const RangeSet& other, Func&& func) const {
		auto       a     = m_ranges.begin();
		auto       b     = other.m_ranges.begin();
		const auto a_end = m_ranges.end();
		const auto b_end = other.m_ranges.end();
		while (a != a_end || b != b_end) {
			auto&    first = b == b_end || (a != a_end && a->first <= b->first) ? a : b;
			uint64_t begin = first->first;
			uint64_t end   = first->second;
			++first;
			for (;;) {
				if (a != a_end && a->first <= end) {
					end = std::max(end, a->second);
					++a;
				} else if (b != b_end && b->first <= end) {
					end = std::max(end, b->second);
					++b;
				} else {
					break;
				}
			}
			func(begin, end);
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }
	void               Clear() { m_ranges.clear(); }
	void               Swap(RangeSet& other) noexcept { m_ranges.swap(other.m_ranges); }

private:
	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
