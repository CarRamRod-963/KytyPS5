#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute };

// Who holds a write watch. The page manager counts the watches of each owner per page, so that
// a page with too many watchers can be traced to its owner.
enum class PageWatcher : uint8_t { MemoryTracker, TextureImage, ImageHeap };

[[nodiscard]] const char* PageWatcherName(PageWatcher watcher) noexcept;

// The write watchers of one page.
struct PageWatcherCounts {
	uint32_t total         = 0;
	uint32_t texture_image = 0;
	uint32_t image_heap    = 0;
	// The rest of the total.
	[[nodiscard]] uint32_t MemoryTracker() const noexcept {
		return total - texture_image - image_heap;
	}
};

// The most write watchers any page had at once in this process, and the page.
struct PageWatcherPeak {
	uint32_t count = 0;
	uint64_t page  = 0;
};

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	// `watcher` names the owner of write watches (ignored for read watches).
	template <bool track, bool is_read = false>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size,
	                        PageWatcher watcher = PageWatcher::MemoryTracker);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// Writes the bytes through the guest backing when no page of the range has a watcher. The
	// region lock is held across the check and the write, so a watcher added concurrently
	// either sees the write (the caches protect a page before they copy it) or makes this
	// return false. False, with nothing written, when a page is watched, the range crosses a
	// tracking region or lies outside the tracked space, or it has no backing.
	[[nodiscard]] bool TryWriteUnwatched(uint64_t vaddr, const void* data, uint64_t size);
	// Whether TryWriteUnwatched would find no watcher on the range now.
	[[nodiscard]] bool IsUnwatched(uint64_t vaddr, uint64_t size);

	// A page holds at most 255 write watchers: one more ends the process. The peak is kept for
	// all page managers of the process (O(1) per watch) and printed with the GPU timing line;
	// the first page to reach each of WarnWriteWatchers is reported once.
	static constexpr uint32_t            WarnWriteWatchers[] = {128, 192, 224};
	[[nodiscard]] static PageWatcherPeak WriteWatcherPeak() noexcept;
	// Tests only: forgets the peak and the warnings given.
	static void ResetWriteWatcherPeak() noexcept;

	// The write watchers on the page of `vaddr` now.
	[[nodiscard]] PageWatcherCounts WriteWatchers(uint64_t vaddr);

private:
	template <typename Action>
	bool WithUnwatched(uint64_t vaddr, uint64_t size, Action&& action);

	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
