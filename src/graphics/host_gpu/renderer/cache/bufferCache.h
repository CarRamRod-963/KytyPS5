#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// Reads back the GPU-modified range containing vaddr, from its page to at most window
	// bytes past vaddr, or just the byte at vaddr when no GPU-modified range contains it.
	void                                       ReadMemoryAround(uint64_t vaddr, uint64_t window);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Reads back the GPU-dirty bytes inside the range, each dirty interval clipped to it, so the
	// backing holds every GPU write of the range recorded so far. Bytes outside the dirty
	// intervals are neither read back nor looked up as buffers.
	void ReadGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Advances whenever a GPU write of guest memory is recorded.
	[[nodiscard]] uint64_t GpuWriteSequence() const noexcept { return m_gpu_write_sequence; }
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	// Reports whether the CPU wrote the range since the last upload or consumption, and arms the
	// range so the next CPU write is reported again.
	[[nodiscard]] bool ConsumeCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// A counter that advances whenever the GPU is bound to write bytes of the watched range.
	// Unlike the GPU-modified state, a readback of the range does not consume it, so a consumer
	// can tell a fresh guest write (a re-applied metadata clear) from the bytes it already
	// honoured.
	[[nodiscard]] uint64_t WatchedWriteSequence(uint64_t vaddr, uint64_t size);
	// Whether a watched range was bound for GPU writing since the last call: the host reads
	// such a range soon (a clear code), and a readback of a pending writer flushes the whole
	// recorded command buffer, so the recorder submits early instead.
	[[nodiscard]] bool ConsumeWatchedWritePending() noexcept {
		return std::exchange(m_watched_write_pending, false);
	}
	void               ProcessFaultBuffer();
	void               SynchronizeBda(const RangeSet& mapped);
	void               QueueBdaSync(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.QueueBdaSync(vaddr, size);
	}
	void               RunGarbageCollector();
	// Called before every submission on the GPU thread: records downloads of the hot read-back
	// pages the GPU wrote since the last submission, published into guest memory when the
	// submission completes, so the guest finds them clean instead of faulting into a
	// synchronous readback that drains the queue.
	void RecordAsyncDownloads();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// Pages the guest recently read back (a read fault, a CPU write into GPU-dirty bytes or a
	// scalar read): the GPU tends to write them every frame and the guest to read them right
	// after, so their next GPU writes are downloaded ahead of the guest. The epoch advances on
	// every GPU write of the page; a publish is valid only at the epoch it was recorded at.
	struct HotPage {
		uint64_t                               touched;
		std::shared_ptr<std::atomic<uint64_t>> epoch;
		// The submission that carries the page's latest asynchronous download, and the epoch
		// it was recorded at; 0 when none was recorded.
		uint64_t download_tick  = 0;
		uint64_t download_epoch = 0;
	};
	void                        TouchHotPages(uint64_t vaddr, uint64_t size);
	// A synchronous readback of a range whose GPU-dirty bytes all sit on hot pages with a
	// submitted, still valid asynchronous download waits for those downloads to publish
	// instead of submitting a copy of its own and waiting for the whole GPU tail.
	void                        AwaitSubmittedDownloads(uint64_t vaddr, uint64_t size);
	std::vector<uint64_t>       m_keep_readable;
	std::vector<uint64_t>       m_make_unreadable;
	void                        NoteGpuWrite(uint64_t vaddr, uint64_t size);
	void                        DrainPublished();
	std::map<uint64_t, HotPage> m_hot_pages;
	uint64_t                    m_hot_touch_sequence = 0;
	RangeSet                    m_pending_async;
	// Ranges a publish cleaned, with the epoch it was valid at: the GPU thread drops their exact
	// dirty intervals unless the page was written again since (the new intervals overlap).
	struct Published {
		uint64_t                               address;
		uint64_t                               size;
		uint64_t                               epoch;
		std::shared_ptr<std::atomic<uint64_t>> epoch_ptr;
	};
	std::mutex             m_published_mutex;
	std::vector<Published> m_published;
	struct WatchedRange {
		uint64_t address;
		uint64_t size;
		uint64_t sequence;
	};
	std::vector<WatchedRange>                         m_watched_ranges;
	uint64_t                                          m_watched_write_sequence = 0;
	bool                                              m_watched_write_pending  = false;
	uint64_t                                          m_gpu_write_sequence     = 0;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
