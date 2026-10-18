#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_

#include "common/common.h"
#include "common/gpuTiming.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

// One timestamp pair around every submitted command buffer. Results are read only for
// submissions the timeline semaphore already reports complete, so nothing ever waits on a
// query. With calibrated timestamps the device times map onto the host clock, which gives the
// latencies around each submission; without them only GPU busy time is reported.
class GpuTimestamps {
public:
	GpuTimestamps(GraphicContext& graphics, MasterSemaphore& master);
	~GpuTimestamps();
	KYTY_CLASS_NO_COPY(GpuTimestamps);

	// Right after vkBeginCommandBuffer / right before vkEndCommandBuffer of the buffer that
	// will be submitted with `tick`.
	void Begin(vk::CommandBuffer buffer, uint64_t tick);
	void End(vk::CommandBuffer buffer, uint64_t tick);
	// After the queue submission of `tick` returned; `submit_ns` is the steady-clock time
	// taken right before it.
	void Submitted(uint64_t tick, int64_t submit_ns);
	// A thread that blocked on `tick` woke up.
	void RecordWake(uint64_t tick);
	// The guest end-of-pipe label deferred to the completion of `tick` was written.
	void RecordLabel(uint64_t tick);

private:
	static constexpr uint32_t RING_SIZE = 256;

	struct Slot {
		uint64_t               tick      = 0;
		int64_t                submit_ns = 0;
		bool                   ended     = false;
		bool                   submitted = false;
		std::optional<int64_t> host_end_ns;
	};

	void                                 Collect();
	void                                 Calibrate();
	void                                 RecordEndLatency(uint64_t tick, bool label);
	void                                 MaybeRotate(int64_t now_ns);
	[[nodiscard]] uint64_t               DeviceNs(uint64_t ticks) const;
	[[nodiscard]] std::optional<int64_t> HostNs(uint64_t device_ticks) const;
	[[nodiscard]] Slot&                  SlotOf(uint64_t tick) { return m_slots[tick % RING_SIZE]; }

	GraphicContext&   m_graphics;
	MasterSemaphore&  m_master;
	vk::QueryPool     m_pool           = nullptr;
	double            m_period_ns      = 1.0;
	uint64_t          m_valid_mask     = 0;
	bool              m_calibration    = false;
	vk::TimeDomainKHR m_host_domain    = vk::TimeDomainKHR::eDevice;
	uint64_t          m_host_frequency = 0; // Ticks per second of the host domain, 0 = ns.
	uint64_t          m_cal_device     = 0;
	int64_t           m_cal_host_ns    = 0;
	bool              m_calibrated     = false;
	uint64_t          m_last_raw       = 0; // Last begin timestamp, to extend across wraps.
	uint64_t          m_wrap_ns        = 0;

	std::mutex                     m_mutex;
	std::array<Slot, RING_SIZE>    m_slots {};
	uint64_t                       m_collected = 0; // Every tick up to this one is accounted.
	Common::GpuTiming::Accumulator m_accumulator;
	Common::GpuTiming::Summary     m_log_window;
	int64_t                        m_window_start_ns = 0;
	uint64_t                       m_window_flips    = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
