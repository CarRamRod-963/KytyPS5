#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_QUEUESUBMITTER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_QUEUESUBMITTER_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

// Passes recorded command buffers to vkQueueSubmit on a thread of its own, so that the threads
// that record them do not wait for the driver. Buffers reach the queue in the order they were
// pushed, from every thread that pushes, so the queue sees the same submission order as with
// direct submission. Anything else that uses the queue (presentation, waiting for the queue to
// idle, external submissions) must Drain first: until then, earlier pushed buffers may not be
// on the queue yet.
//
// A tick pushed but not yet submitted is a valid target of a host wait on its timeline
// semaphore (timeline semaphores allow waits before the signal is submitted).
class QueueSubmitter {
public:
	explicit QueueSubmitter(GraphicContext& graphics);
	~QueueSubmitter();
	KYTY_CLASS_NO_COPY(QueueSubmitter);

	// Allocates the next tick of `master`, adds its signal to `submit` and queues the buffer.
	// Ticks of one semaphore reach the queue in increasing order.
	uint64_t Push(vk::CommandBuffer buffer, SubmitInfo submit, MasterSemaphore& master);
	// Returns once every buffer pushed before the call has been passed to vkQueueSubmit.
	void Drain();
	// Buffers pushed and buffers passed to vkQueueSubmit since the start.
	[[nodiscard]] uint64_t Pushed() const noexcept {
		return m_pushed.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t Submitted() const noexcept {
		return m_submitted.load(std::memory_order_acquire);
	}

private:
	struct Batch {
		vk::CommandBuffer buffer = nullptr;
		SubmitInfo        submit;
		uint64_t          tick = 0;
	};

	void Run(const std::stop_token& stop);
	void SubmitBatches(const std::vector<Batch>& batches);

	GraphicContext&             m_graphics;
	std::mutex                  m_mutex;
	std::condition_variable_any m_available;
	std::vector<Batch>          m_queued;
	std::atomic<uint64_t>       m_pushed {0};
	std::atomic<uint64_t>       m_submitted {0};
	std::jthread                m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_QUEUESUBMITTER_H_
