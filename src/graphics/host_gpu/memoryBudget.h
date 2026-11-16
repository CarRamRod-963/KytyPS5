#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

// Device-memory budget arithmetic for the caches. Vulkan-free so it can be unit tested; the
// numbers come from VK_EXT_memory_budget through VMA.
namespace Libs::Graphics::MemoryBudget {

inline constexpr uint64_t MiB = 1024ull * 1024;
inline constexpr uint64_t GiB = 1024ull * MiB;

// A heap (or the sum of the heaps of one kind) as VK_EXT_memory_budget reports it.
struct HeapState {
	uint64_t budget = 0; // What the process may use before the OS moves allocations out.
	uint64_t usage  = 0; // Whole process: VMA's blocks plus allocations the driver makes itself.
	uint64_t blocks = 0; // Device memory that VMA allocated for the emulator.
};

// Memory of the heap that VMA does not own: the driver's local-memory surface for spills and
// private arrays, pipelines, descriptor heaps, swapchain images.
[[nodiscard]] constexpr uint64_t DriverUsage(const HeapState& heap) {
	return heap.usage > heap.blocks ? heap.usage - heap.blocks : 0;
}

// Room kept free for driver allocations that appear later. When the heap is full, the OS
// demotes whole allocations to system memory, and it may pick a driver surface that every
// shader reads (a demoted local-memory surface made each spill cross the bus).
[[nodiscard]] constexpr uint64_t DriverHeadroom(uint64_t budget) {
	return std::clamp<uint64_t>(budget / 16, 256 * MiB, 1 * GiB);
}

// What VMA's blocks may occupy while the headroom stays free: the budget minus the driver's
// memory minus the headroom.
[[nodiscard]] constexpr uint64_t CacheLimit(const HeapState& heap) {
	const auto reserved = DriverUsage(heap) + DriverHeadroom(heap.budget);
	return heap.budget > reserved ? heap.budget - reserved : 0;
}

// Usage of the device heaps (VMA's blocks plus the driver's memory) above which the caches give
// memory back at once: past it the driver headroom is used up, and past the budget the OS pages
// allocations over the bus around each submission.
[[nodiscard]] constexpr uint64_t UsageLimit(uint64_t budget) {
	const auto headroom = DriverHeadroom(budget);
	return budget > headroom ? budget - headroom : 0;
}

// Device-memory usage marks of a cache's garbage collector, derived from the budget the caches
// work with (GraphicContext::GetTotalMemoryBudget: the device-local budget minus a reserve).
// Above `trigger` the collector deletes entries nobody used for a while, above `pressure` it
// also writes GPU-written contents back to guest memory to delete them, above `critical` it
// collects more per pass.
struct CollectionMarks {
	uint64_t trigger  = 0;
	uint64_t pressure = 0;
	uint64_t critical = 0;

	friend constexpr bool operator==(const CollectionMarks&, const CollectionMarks&) = default;
};

namespace Detail {

// Up to 8 GiB of the budget are spread between the marks; larger budgets keep the marks at the
// same distance below the budget.
[[nodiscard]] constexpr int64_t MarkSpan(int64_t budget) {
	return std::min<int64_t>(budget, 8 * static_cast<int64_t>(GiB));
}
[[nodiscard]] constexpr int64_t PressureMark(int64_t budget) {
	return std::min(budget - 6 * MarkSpan(budget) / 10, budget - static_cast<int64_t>(GiB));
}
[[nodiscard]] constexpr int64_t CriticalMark(int64_t budget) {
	return std::min(budget - 2 * MarkSpan(budget) / 10, budget - static_cast<int64_t>(GiB / 2));
}
[[nodiscard]] constexpr uint64_t AtLeast(int64_t mark, uint64_t floor) {
	return std::max(mark, static_cast<int64_t>(floor)) > 0
	           ? static_cast<uint64_t>(std::max(mark, static_cast<int64_t>(floor)))
	           : 0;
}

} // namespace Detail

[[nodiscard]] constexpr CollectionMarks TextureMarks(uint64_t cache_budget) {
	const auto budget = static_cast<int64_t>(std::min<uint64_t>(cache_budget, INT64_MAX / 2));
	return {.trigger  = Detail::AtLeast((budget - Detail::MarkSpan(budget)) / 2, 0),
	        .pressure = Detail::AtLeast(Detail::PressureMark(budget), GiB + GiB / 2),
	        .critical = Detail::AtLeast(Detail::CriticalMark(budget), 3 * GiB)};
}

// The buffer collector has no write-back stage of its own: it starts at the texture pressure mark.
[[nodiscard]] constexpr CollectionMarks BufferMarks(uint64_t cache_budget) {
	const auto budget  = static_cast<int64_t>(std::min<uint64_t>(cache_budget, INT64_MAX / 2));
	const auto trigger = Detail::AtLeast(Detail::PressureMark(budget), GiB);
	return {.trigger  = trigger,
	        .pressure = trigger,
	        .critical = Detail::AtLeast(Detail::CriticalMark(budget), 2 * GiB)};
}

// Follows the budget the OS gives the process while the title runs: other programs' device
// memory and the OS change it (a 12 GB card's budget fell from 10.9 to 10.0 GiB during one
// session), and marks read once at start then let the caches grow past it. Budget changes below
// Hysteresis are ignored: the budget of a busy session moves by up to ~300 MiB from one second
// to the next. A drop is followed at once; a rise only once the budget stayed Hysteresis above
// the marks' budget for RiseHoldMs, and then to the lowest budget seen while it held: a 16 GB
// card's budget switched between 14,185 and ~13,300 MiB about every minute in one session, and
// following both directions moved every mark by ~900 MiB each time. O(1).
class LiveMarks {
public:
	static constexpr uint64_t Hysteresis = 512 * MiB;
	static constexpr uint64_t RiseHoldMs = 30'000;

	// Returns true when the marks changed.
	bool Update(uint64_t cache_budget, uint64_t now_ms) {
		if (!m_valid || cache_budget + Hysteresis <= m_budget) {
			Set(cache_budget);
			return true;
		}
		if (cache_budget < m_budget + Hysteresis) {
			m_rising = false;
			return false;
		}
		if (!m_rising) {
			m_rising     = true;
			m_rise_since = now_ms;
			m_rise_low   = cache_budget;
			return false;
		}
		m_rise_low = std::min(m_rise_low, cache_budget);
		if (now_ms - m_rise_since < RiseHoldMs) {
			return false;
		}
		Set(m_rise_low);
		return true;
	}

	[[nodiscard]] uint64_t        Budget() const { return m_budget; }
	[[nodiscard]] CollectionMarks Texture() const { return m_texture; }
	[[nodiscard]] CollectionMarks Buffer() const { return m_buffer; }
	// Changes with every update that moved the marks (0 before the first).
	[[nodiscard]] uint64_t Generation() const { return m_generation; }

private:
	void Set(uint64_t cache_budget) {
		m_valid   = true;
		m_rising  = false;
		m_budget  = cache_budget;
		m_texture = TextureMarks(cache_budget);
		m_buffer  = BufferMarks(cache_budget);
		m_generation++;
	}

	uint64_t        m_budget     = 0;
	uint64_t        m_generation = 0;
	uint64_t        m_rise_since = 0;
	uint64_t        m_rise_low   = 0;
	CollectionMarks m_texture;
	CollectionMarks m_buffer;
	bool            m_valid  = false;
	bool            m_rising = false;
};

// Accesses through the BDA page table to memory no buffer caches are lost for their dispatch (a
// read sees zeros, a write is dropped). Over the device budget a collector can evict buffers that
// shaders still use, and one 12 GB session logged 108,527 of them, a line each. The first
// FirstReports are logged one by one; later ones are counted and summarized at most once per
// period. O(1) per access.
class LostAccessReport {
public:
	static constexpr uint64_t FirstReports = 32;
	static constexpr uint64_t PeriodMs     = 10'000;

	// Returns true when this access is to be logged on its own.
	bool Note() {
		m_total++;
		if (m_total <= FirstReports) {
			return true;
		}
		m_unreported++;
		return false;
	}

	// The fault report area overflowed: reports past its end were dropped, so the accesses they
	// held are neither logged nor counted (watched buffers count as accessed instead).
	void NoteOverflow() {
		m_overflows++;
		m_unreported_overflows++;
	}

	struct Due {
		uint64_t accesses  = 0;
		uint64_t overflows = 0;

		[[nodiscard]] explicit operator bool() const { return accesses != 0 || overflows != 0; }
	};

	// Returns what to summarize now (nothing when nothing is due): the accesses not logged on
	// their own and the overflows since the last summary, once a period has passed since it.
	[[nodiscard]] Due Summary(uint64_t now_ms) {
		if ((m_unreported == 0 && m_unreported_overflows == 0) ||
		    (m_summarized && now_ms - m_last_ms < PeriodMs)) {
			return {};
		}
		m_summarized = true;
		m_last_ms    = now_ms;
		return {std::exchange(m_unreported, 0), std::exchange(m_unreported_overflows, 0)};
	}

	[[nodiscard]] uint64_t Total() const { return m_total; }
	[[nodiscard]] uint64_t Overflows() const { return m_overflows; }

private:
	uint64_t m_total                = 0;
	uint64_t m_unreported           = 0;
	uint64_t m_overflows            = 0;
	uint64_t m_unreported_overflows = 0;
	uint64_t m_last_ms              = 0;
	bool     m_summarized           = false;
};

// Decides when to print the heap state: at most once per period, and only when the driver's
// device memory grew, the driver's memory in the host heap grew (allocations demoted out
// of device memory land there), or the memory at low residency priority (the first to be
// demoted) moved by PriorityChange in either direction. Shrinking driver memory moves the
// reference down silently. While the device heaps use more than their budget it reports every
// period (the OS then pages allocations in and out around each submission: a 12 GB card ran
// 60-128 ms of GPU time per frame instead of ~35), and once more when usage is back within the
// budget. O(1) per sample.
class Monitor {
public:
	static constexpr uint64_t PeriodMs       = 10'000;
	static constexpr uint64_t DeviceGrowth   = 64 * MiB;
	static constexpr uint64_t HostGrowth     = 64 * MiB;
	static constexpr uint64_t PriorityChange = 512 * MiB;

	enum class Report : uint8_t {
		None,
		First,
		DriverGrew,
		Demoted,
		Priorities,
		OverBudget,
		WithinBudget
	};

	[[nodiscard]] Report Sample(const HeapState& device, const HeapState& host, uint64_t now_ms,
	                            uint64_t low_priority = 0) {
		if (m_sampled && now_ms - m_last_ms < PeriodMs) {
			return Report::None;
		}
		m_last_ms          = now_ms;
		const auto driver  = DriverUsage(device);
		const auto demoted = DriverUsage(host);
		const bool first   = !m_sampled;
		const bool grew    = !first && driver >= m_driver + DeviceGrowth;
		const bool moved   = !first && demoted >= m_demoted + HostGrowth;
		const bool aged    = !first && (low_priority >= m_low + PriorityChange ||
		                                m_low >= low_priority + PriorityChange);
		const bool over    = device.usage > device.budget;
		const bool back    = m_over && !over;
		m_sampled          = true;
		m_over             = over;
		if (first || grew || moved || aged || over || back) {
			m_driver  = driver;
			m_demoted = demoted;
			m_low     = low_priority;
			return first   ? Report::First
			       : over  ? Report::OverBudget
			       : back  ? Report::WithinBudget
			       : moved ? Report::Demoted
			       : grew  ? Report::DriverGrew
			               : Report::Priorities;
		}
		m_driver  = std::min(m_driver, driver);
		m_demoted = std::min(m_demoted, demoted);
		return Report::None;
	}

private:
	uint64_t m_last_ms = 0;
	uint64_t m_driver  = 0;
	uint64_t m_demoted = 0;
	uint64_t m_low     = 0;
	bool     m_sampled = false;
	bool     m_over    = false;
};

} // namespace Libs::Graphics::MemoryBudget
