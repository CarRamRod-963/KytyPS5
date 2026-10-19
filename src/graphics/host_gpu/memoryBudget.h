#pragma once

#include <algorithm>
#include <cstdint>

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

// Decides when to print the heap state: at most once per period, and only when the driver's
// device memory grew or the driver's memory in the host heap grew (allocations demoted out
// of device memory land there). Shrinking moves the reference down silently. O(1) per sample.
class Monitor {
public:
	static constexpr uint64_t PeriodMs     = 10'000;
	static constexpr uint64_t DeviceGrowth = 64 * MiB;
	static constexpr uint64_t HostGrowth   = 64 * MiB;

	enum class Report : uint8_t { None, First, DriverGrew, Demoted };

	[[nodiscard]] Report Sample(const HeapState& device, const HeapState& host, uint64_t now_ms) {
		if (m_sampled && now_ms - m_last_ms < PeriodMs) {
			return Report::None;
		}
		m_last_ms          = now_ms;
		const auto driver  = DriverUsage(device);
		const auto demoted = DriverUsage(host);
		const bool first   = !m_sampled;
		const bool grew    = !first && driver >= m_driver + DeviceGrowth;
		const bool moved   = !first && demoted >= m_demoted + HostGrowth;
		m_sampled          = true;
		if (first || grew || moved) {
			m_driver  = driver;
			m_demoted = demoted;
			return first ? Report::First : moved ? Report::Demoted : Report::DriverGrew;
		}
		m_driver  = std::min(m_driver, driver);
		m_demoted = std::min(m_demoted, demoted);
		return Report::None;
	}

private:
	uint64_t m_last_ms = 0;
	uint64_t m_driver  = 0;
	uint64_t m_demoted = 0;
	bool     m_sampled = false;
};

} // namespace Libs::Graphics::MemoryBudget
