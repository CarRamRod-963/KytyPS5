#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
}

MemoryTracker::~MemoryTracker() = default;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

// Queries take no lock while no state change of the region runs (TryQueryUnlocked); the dirty
// words are read atomically. A query racing a change takes the lock, as before.
bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		const auto query = [&] { return manager->IsModified<DirtySource::Cpu>(offset, bytes); };
		if (bool modified = false; manager->TryQueryUnlocked(query, modified)) {
			return modified;
		}
		std::scoped_lock lock(manager->lock);
		return query();
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		const auto query = [&] { return manager->IsModified<DirtySource::Gpu>(offset, bytes); };
		if (bool modified = false; manager->TryQueryUnlocked(query, modified)) {
			return modified;
		}
		std::scoped_lock lock(manager->lock);
		return query();
	});
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(offset, bytes);
	});
}

void MemoryTracker::SetGpuModifiedReadable(uint64_t vaddr, uint64_t size, bool readable) {
	Iterate<false>(vaddr, size,
	               [readable](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		               std::scoped_lock lock(manager->lock);
		               manager->SetStaleReadable(offset, bytes, readable);
	               });
}

void MemoryTracker::SetGpuModifiedReadable(std::span<const uint64_t> pages, uint64_t page_size,
                                           bool readable) {
	for (size_t first = 0; first < pages.size();) {
		ValidateRange(pages[first], page_size);
		const auto index = pages[first] / TRACKER_REGION_SIZE;
		auto       last  = first + 1;
		while (last < pages.size() && pages[last] / TRACKER_REGION_SIZE == index) {
			ValidateRange(pages[last], page_size);
			last++;
		}
		if ((pages[last - 1] + page_size - 1) / TRACKER_REGION_SIZE != index) {
			EXIT("readable pages cross a tracking region\n");
		}
		if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
			std::scoped_lock lock(manager->lock);
			manager->SetStaleReadable(pages.subspan(first, last - first), page_size, readable);
		}
		first = last;
	}
}

bool MemoryTracker::IsGpuModifiedReadable(uint64_t vaddr, uint64_t size) {
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsStaleReadable(offset, bytes);
	});
}

} // namespace Libs::Graphics
