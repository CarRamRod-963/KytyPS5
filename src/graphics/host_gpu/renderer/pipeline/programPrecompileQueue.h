#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMPRECOMPILEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMPRECOMPILEQUEUE_H_

#include "common/assert.h"
#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

enum class PrecompileOutcome : uint8_t {
	// The driver cache already held the pipeline.
	Cached,
	// The driver compiled the pipeline into the driver cache.
	Compiled,
	// The program cannot be rebuilt from its record (a shader with calls, or a specialization
	// this recompiler no longer derives for the shader).
	Skipped,
};

// Counts the outcomes of a precompile run and tells exactly one caller that the run is complete:
// the job that finishes last, or EndQueueing when no job is left by then (also when nothing was
// queued). Queueing holds the run open, so a job finishing while others are still being queued
// never completes it, and keys the queue rejects are never waited for.
class PrecompileProgress {
public:
	struct Counts {
		uint64_t queued     = 0;
		uint64_t finished   = 0;
		uint64_t cached     = 0;
		uint64_t compiled   = 0;
		uint64_t skipped    = 0;
		uint64_t compile_ms = 0;
	};

	// Starts a run; not allowed while one is open.
	void BeginQueueing() {
		std::lock_guard lock(m_mutex);
		EXIT_IF(m_open != 0);
		m_open = 1;
	}
	// Accounts for a job the queue accepted; call before the job can run.
	void Queued() {
		std::lock_guard lock(m_mutex);
		EXIT_IF(m_open == 0);
		m_open++;
		m_counts.queued++;
	}
	// Takes back a Queued() whose job the queue did not accept.
	void Rejected() {
		std::lock_guard lock(m_mutex);
		EXIT_IF(m_open < 2 || m_counts.queued == 0);
		m_open--;
		m_counts.queued--;
	}
	// Ends queueing. Returns whether the run is complete.
	[[nodiscard]] bool EndQueueing() {
		std::lock_guard lock(m_mutex);
		EXIT_IF(m_open == 0);
		return --m_open == 0;
	}
	// Records a finished job. Returns whether it completed the run.
	[[nodiscard]] bool Finish(PrecompileOutcome outcome, uint64_t ms) {
		std::lock_guard lock(m_mutex);
		EXIT_IF(m_open == 0);
		m_counts.finished++;
		switch (outcome) {
			case PrecompileOutcome::Cached: m_counts.cached++; break;
			case PrecompileOutcome::Compiled:
				m_counts.compiled++;
				m_counts.compile_ms += ms;
				break;
			case PrecompileOutcome::Skipped: m_counts.skipped++; break;
		}
		return --m_open == 0;
	}
	[[nodiscard]] Counts Get() {
		std::lock_guard lock(m_mutex);
		return m_counts;
	}

private:
	std::mutex m_mutex;
	Counts     m_counts;
	// Jobs queued and not finished, plus one while queueing.
	uint64_t m_open = 0;
};

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
