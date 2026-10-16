#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_

#include "common/common.h"
#include "common/uniqueFunction.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Builds pipelines on background threads. A job returns an opaque handle (a Vulkan pipeline, or
// zero when the build failed); the owner collects finished handles on its own thread. Handles
// nobody collected when the queue stops go to the discard callback on the stopping thread.
// Vulkan-free so that the bookkeeping can be tested without a device.
class PipelineCompileQueue {
public:
	using Job     = Common::UniqueFunction<uint64_t>;
	using Discard = Common::UniqueFunction<void, uint64_t>;

	struct Completion {
		uint64_t ticket = 0;
		uint64_t handle = 0;
	};

	PipelineCompileQueue(uint32_t thread_count, Discard discard);
	~PipelineCompileQueue();
	KYTY_CLASS_NO_COPY(PipelineCompileQueue);

	// Returns the ticket that the job's completion carries. Not allowed after Stop.
	uint64_t Enqueue(Job job);
	// Appends the results finished since the last call, in completion order.
	void TakeCompleted(std::vector<Completion>& out);
	// Jobs queued, running, or finished and not collected yet.
	[[nodiscard]] size_t Outstanding();
	// Blocks until no job is queued or running.
	void WaitIdle();
	// Drops the queued jobs, waits for the running ones, joins the threads and discards every
	// uncollected handle. Idempotent.
	void Stop();

private:
	struct Entry {
		uint64_t ticket = 0;
		Job      job;
	};

	void Worker();

	Discard                  m_discard;
	std::mutex               m_mutex;
	std::condition_variable  m_job_available;
	std::condition_variable  m_job_finished;
	std::deque<Entry>        m_jobs;
	std::vector<Completion>  m_completed;
	std::vector<std::thread> m_threads;
	uint64_t                 m_next_ticket = 1;
	uint32_t                 m_running     = 0;
	bool                     m_stopped     = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_
