#include "graphics/host_gpu/renderer/queueSubmitter.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <cinttypes>

namespace Libs::Graphics {

void GraphicContext::DrainSubmissions() const {
	if (submitter != nullptr) {
		submitter->Drain();
	}
}

QueueSubmitter::QueueSubmitter(GraphicContext& graphics)
    : m_graphics(graphics), m_thread([this](std::stop_token stop) { Run(stop); }) {}

QueueSubmitter::~QueueSubmitter() {
	Drain();
	m_thread.request_stop();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

uint64_t QueueSubmitter::Push(vk::CommandBuffer buffer, SubmitInfo submit,
                              MasterSemaphore& master) {
	EXIT_IF(buffer == nullptr);
	uint64_t tick = 0;
	{
		std::lock_guard lock(m_mutex);
		tick = master.NextTick();
		submit.AddSignal(master.Handle(), tick);
		m_queued.push_back({buffer, submit, tick});
		m_pushed.fetch_add(1, std::memory_order_release);
	}
	m_available.notify_one();
	return tick;
}

void QueueSubmitter::Drain() {
	EXIT_IF(std::this_thread::get_id() == m_thread.get_id());
	const auto target = m_pushed.load(std::memory_order_acquire);
	auto       done   = m_submitted.load(std::memory_order_acquire);
	while (done < target) {
		m_submitted.wait(done, std::memory_order_acquire);
		done = m_submitted.load(std::memory_order_acquire);
	}
}

void QueueSubmitter::Run(const std::stop_token& stop) {
	std::vector<Batch> batches;
	for (;;) {
		{
			std::unique_lock lock(m_mutex);
			m_available.wait(lock, stop, [this] { return !m_queued.empty(); });
			if (m_queued.empty()) {
				return;
			}
			batches.swap(m_queued);
		}
		SubmitBatches(batches);
		m_submitted.fetch_add(batches.size(), std::memory_order_release);
		m_submitted.notify_all();
		batches.clear();
	}
}

void QueueSubmitter::SubmitBatches(const std::vector<Batch>& batches) {
	std::vector<vk::TimelineSemaphoreSubmitInfo> timelines(batches.size());
	std::vector<vk::SubmitInfo>                  infos(batches.size());
	for (size_t index = 0; index < batches.size(); index++) {
		const auto& batch                  = batches[index];
		auto&       timeline               = timelines[index];
		timeline.waitSemaphoreValueCount   = batch.submit.num_wait_semaphores;
		timeline.pWaitSemaphoreValues      = batch.submit.wait_ticks.data();
		timeline.signalSemaphoreValueCount = batch.submit.num_signal_semaphores;
		timeline.pSignalSemaphoreValues    = batch.submit.signal_ticks.data();

		auto& info                = infos[index];
		info.pNext                = &timeline;
		info.waitSemaphoreCount   = batch.submit.num_wait_semaphores;
		info.pWaitSemaphores      = batch.submit.wait_semaphores.data();
		info.pWaitDstStageMask    = batch.submit.wait_stages.data();
		info.commandBufferCount   = 1;
		info.pCommandBuffers      = &batch.buffer;
		info.signalSemaphoreCount = batch.submit.num_signal_semaphores;
		info.pSignalSemaphores    = batch.submit.signal_semaphores.data();
	}

	vk::Result result;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		result =
		    m_graphics.queue.submit(static_cast<uint32_t>(infos.size()), infos.data(), nullptr);
	}
	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, ticks=%" PRIu64 "..%" PRIu64 "\n",
		     vk::to_string(result).c_str(), batches.front().tick, batches.back().tick);
	}
}

} // namespace Libs::Graphics
