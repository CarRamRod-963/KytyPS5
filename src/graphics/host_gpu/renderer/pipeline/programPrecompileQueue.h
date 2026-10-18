#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMPRECOMPILEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMPRECOMPILEQUEUE_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Builds the pipelines of recorded programs (ProgramManifest) on background threads, one job
// per program key, in the order they were queued. A program the renderer needs before its job
// ran is expedited: the job runs on the renderer's thread, or the renderer waits for the running
// job, and the pipeline build that follows finds the driver cache filled. A pipeline is
// therefore never compiled twice at the same time, and nothing waits longer than its own build.
// Vulkan-free so that the scheduling can be tested without a device.
class ProgramPrecompileQueue {
public:
	using Job = Common::UniqueFunction<void>;

	explicit ProgramPrecompileQueue(uint32_t thread_count)
	    : m_queue(thread_count, [](uint64_t) {}) {}
	~ProgramPrecompileQueue() { Stop(); }
	KYTY_CLASS_NO_COPY(ProgramPrecompileQueue);

	// Queues the job of a program; a key that was queued before is ignored. Returns whether the
	// job was queued. Not allowed after Stop.
	bool Enqueue(uint64_t key, Job job) {
		std::lock_guard lock(m_mutex);
		if (m_stopped || !m_tickets.try_emplace(key, 0).second) {
			return false;
		}
		const auto ticket     = m_queue.Enqueue([job = std::move(job)]() mutable -> uint64_t {
			job();
			return 0;
		});
		m_tickets[key]        = ticket;
		m_ticket_keys[ticket] = key;
		m_queued++;
		return true;
	}

	// Returns once the program's job has finished: runs it on the calling thread if it has not
	// started, waits for it if it is running. Returns at once for a key without a job.
	void Expedite(uint64_t key) {
		uint64_t ticket = 0;
		{
			std::lock_guard lock(m_mutex);
			const auto      entry = m_tickets.find(key);
			if (m_stopped || entry == m_tickets.end() || entry->second == 0) {
				return;
			}
			ticket = entry->second;
		}
		m_queue.Expedite(ticket);
		Collect();
	}

	// Jobs queued or running (after Stop, also the dropped ones).
	[[nodiscard]] size_t Remaining() {
		Collect();
		std::lock_guard lock(m_mutex);
		return m_queued - m_finished;
	}
	[[nodiscard]] size_t Queued() {
		std::lock_guard lock(m_mutex);
		return m_queued;
	}

	void WaitIdle() {
		m_queue.WaitIdle();
		Collect();
	}

	// Drops the jobs that have not started and waits for the running ones. Idempotent.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
		}
		m_queue.Stop();
	}

private:
	// Marks finished jobs; their keys stay known so they are not queued again.
	void Collect() {
		std::lock_guard lock(m_mutex);
		m_completed.clear();
		m_queue.TakeCompleted(m_completed);
		for (const auto& completion: m_completed) {
			const auto key = m_ticket_keys.find(completion.ticket);
			if (key == m_ticket_keys.end()) {
				continue;
			}
			m_tickets[key->second] = 0;
			m_ticket_keys.erase(key);
			m_finished++;
		}
	}

	PipelineCompileQueue m_queue;
	std::mutex           m_mutex;
	// Ticket of each queued key; 0 once its job finished.
	std::unordered_map<uint64_t, uint64_t>        m_tickets;
	std::unordered_map<uint64_t, uint64_t>        m_ticket_keys;
	std::vector<PipelineCompileQueue::Completion> m_completed;
	size_t                                        m_queued   = 0;
	size_t                                        m_finished = 0;
	bool                                          m_stopped  = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMPRECOMPILEQUEUE_H_
