#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/memoryBudget.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

class BufferCache;

class FaultManager {
	static constexpr size_t MaxPendingFaults = 8;

public:
	// Page records a fault-bitmap scan reports (one slot holds the count). The pages past them
	// stay set in the bitmap for the next scan, so with more reported pages a buffer that counts
	// as unread through the page table (lazy BDA synchronization) waits a frame longer for its
	// stores. Must match MAX_PAGE_FAULTS of fault_buffer_process.comp (CMakeLists.txt).
	static constexpr size_t MaxPageFaults = 8192;

	FaultManager(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~FaultManager();
	KYTY_CLASS_NO_COPY(FaultManager);

	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return &m_fault_buffer; }
	void                  ProcessFaultBuffer();
	// Reported accesses to memory without a cached buffer (each lost for its dispatch).
	[[nodiscard]] uint64_t LostAccesses() const noexcept { return m_lost_accesses.Total(); }

private:
	GraphicContext&                            m_graphics;
	CommandScheduler&                          m_scheduler;
	BufferCache&                               m_buffer_cache;
	Buffer                                     m_fault_buffer;
	Buffer                                     m_download_buffer;
	std::array<uint64_t, MaxPendingFaults>      m_fault_areas {};
	uint32_t                                   m_current_area = 0;
	// The scheduler tick of the previous fault-buffer scan: the accesses a scan reports ran
	// after it.
	uint64_t m_last_scan_tick = 0;
	// Accesses to memory without a cached buffer (each lost for its dispatch); GPU thread.
	MemoryBudget::LostAccessReport             m_lost_accesses;
	vk::DescriptorSetLayout                    m_fault_process_desc_layout = nullptr;
	vk::Pipeline                               m_fault_process_pipeline = nullptr;
	vk::PipelineLayout                         m_fault_process_pipeline_layout = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
