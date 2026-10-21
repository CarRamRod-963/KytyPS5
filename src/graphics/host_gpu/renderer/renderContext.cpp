#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/gpuTiming.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_command_scheduler), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_command_scheduler.GetTimestamps().SetPrimary(true);
	// After the bindless table: precompiled layouts reference its set layout.
	if (Config::PrecompilePipelinesEnabled() && !Config::PrecompileOnlyEnabled()) {
		// Like the background pipeline builds: driver compilers are multithreaded themselves,
		// and the guest needs the CPU.
		m_pipeline_cache.StartPrecompile(
		    std::clamp(std::thread::hardware_concurrency() / 8u, 1u, 2u));
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
	// Before the bindless table destroys the set layout that background builds reference.
	m_pipeline_cache.StopBackgroundCompiles();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
	m_notified_gpu.store(m_gpu.get());
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		// The completion thread may still store labels; wait until none uses the GPU object.
		m_notified_gpu.store(nullptr);
		while (m_gpu_notifiers.load() != 0) {
			std::this_thread::yield();
		}
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

void RenderContext::NotifyGpuMemoryWritten() {
	m_gpu_notifiers.fetch_add(1);
	if (auto* gpu = m_notified_gpu.load(); gpu != nullptr) {
		gpu->NotifyMemoryWritten();
	}
	m_gpu_notifiers.fetch_sub(1);
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		// A guest read of GPU-written memory tends to continue through the following pages, so
		// one readback fetches the GPU-modified extent from the faulting page instead of a GPU
		// round trip per page.
		constexpr uint64_t fault_window = uint64_t {1} << 20u;
		m_buffer_cache.ReadMemoryAround(fault_vaddr, fault_window);
	}
	return true;
}

void RenderContext::DeliverLabel(void* dst, uint64_t value, uint32_t size,
                                 Common::UniqueFunction<void>&& notify, uint64_t label_id) {
	EXIT_IF(dst == nullptr || (size != sizeof(uint32_t) && size != sizeof(uint64_t)));
	// The completion thread runs the publications of a submission's downloads before its
	// labels, so whatever the label announces is already in guest memory.
	if (m_gpu_thread_labels.load(std::memory_order_acquire) == 0 &&
	    m_page_manager.TryWriteUnwatched(reinterpret_cast<uint64_t>(dst), &value, size)) {
		Common::GpuTiming::Sync().AddLabel(true);
		if (label_id != 0) {
			m_pending_labels.Stored(label_id);
		}
		NotifyGpuMemoryWritten();
		if (notify) {
			notify();
		}
		return;
	}
	Common::GpuTiming::Sync().AddLabel(false);
	m_gpu_thread_labels.fetch_add(1, std::memory_order_relaxed);
	GetGpu().SendCommand([this, dst, value, size, label_id, notify = std::move(notify)]() mutable {
		// What a write fault would do, without relying on one: the caches drop or read back
		// what they hold of the label's pages and stop watching them.
		(void)InvalidateMemory(reinterpret_cast<uint64_t>(dst), size);
		// A settled access may have stored the label, and later ones, already.
		if (label_id == 0 || m_pending_labels.ShouldStore(label_id)) {
			if (size == sizeof(uint32_t)) {
				const auto word = static_cast<uint32_t>(value);
				std::memcpy(dst, &word, sizeof(word));
			} else {
				std::memcpy(dst, &value, sizeof(value));
			}
		}
		if (label_id != 0) {
			m_pending_labels.Stored(label_id);
		}
		NotifyGpuMemoryWritten();
		if (notify) {
			notify();
		}
		m_gpu_thread_labels.fetch_sub(1, std::memory_order_release);
	});
}

void RenderContext::SettleElidedLabels(uint64_t vaddr, uint64_t size) {
	static thread_local bool settling = false;
	if (settling || !GuestGpu::IsGpuThread()) {
		return;
	}
	const auto tick = m_pending_labels.ElidedTick(vaddr, size);
	if (tick == 0) {
		return;
	}
	settling = true;
	{
		const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Label);
		m_command_scheduler.Wait(tick);
		// The completion thread stores the labels of that submission now ...
		m_command_scheduler.WaitPriorityOperations(tick);
	}
	// ... unless it handed them to this thread, behind the current command. Those are stored
	// here, through the backing like the completion thread's own stores, and skipped there.
	m_pending_labels.StoreCompleted(vaddr, size, tick, [](const PendingLabels::Label& label) {
		(void)LibKernel::Memory::TryWriteBacking(label.address, &label.value, label.size);
	});
	settling = false;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Unmap);
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_graphics.MonitorMemoryBudget();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
