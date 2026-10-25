#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_

#include "common/common.h"
#include "common/uniqueFunction.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Builds pipelines on background threads. A job returns an opaque handle (a Vulkan pipeline, or
// zero when the build failed); the owner collects finished handles on its own thread. Handles
// nobody collected when the queue stops go to the discard callback on the stopping thread.
// Jobs run in the order of their `order` value, and in queueing order for equal values.
// Vulkan-free so that the bookkeeping can be tested without a device.
class PipelineCompileQueue {
public:
	using Job     = Common::UniqueFunction<uint64_t>;
	using Discard = Common::UniqueFunction<void, uint64_t>;

	struct Completion {
		uint64_t ticket = 0;
		uint64_t handle = 0;
	};

	struct Options {
		// Workers run below normal priority, so that the threads that drive the guest come first.
		bool low_priority = false;
	};

	PipelineCompileQueue(uint32_t thread_count, Discard discard);
	PipelineCompileQueue(uint32_t thread_count, Discard discard, Options options);
	~PipelineCompileQueue();
	KYTY_CLASS_NO_COPY(PipelineCompileQueue);

	// Returns the ticket that the job's completion carries. Jobs with a lower `order` start
	// first. Not allowed after Stop.
	uint64_t Enqueue(Job job, uint64_t order = 0);
	// Appends the results finished since the last call, in completion order.
	void TakeCompleted(std::vector<Completion>& out);
	// Jobs queued, running, or finished and not collected yet.
	[[nodiscard]] size_t Outstanding();
	// Jobs queued (not started) whose order is below `order`.
	[[nodiscard]] size_t QueuedBelow(uint64_t order);
	// Blocks until no job is queued or running.
	void WaitIdle();
	// Returns once the ticket's result is available to TakeCompleted (or was taken): a job that
	// has not started runs on the calling thread, a running one is waited for.
	void Expedite(uint64_t ticket);
	// Like Expedite, but waits at most `max_wait` for a running job. Returns whether the result
	// is available.
	bool Expedite(uint64_t ticket, std::chrono::milliseconds max_wait);
	// At most `count` (1 to the thread count) workers start jobs from now on; running jobs are
	// not interrupted.
	void                   SetActiveThreads(uint32_t count);
	[[nodiscard]] uint32_t ActiveThreads();
	[[nodiscard]] uint32_t ThreadCount() const { return m_thread_count; }
	// Gives up on a running job (a driver build that may never return): its ticket completes
	// now with a zero handle, a new worker takes the abandoned one's place, and the abandoned
	// worker ends by itself when the job returns (a handle it returns is discarded, or leaked
	// once the queue has stopped). The queue never waits for it. Returns false when the job is
	// not running.
	bool Abandon(uint64_t ticket);
	// Workers that run an abandoned job.
	[[nodiscard]] size_t AbandonedRunning();
	// Drops the queued jobs, waits for the running ones (except abandoned ones), joins the
	// threads and discards every uncollected handle. Idempotent.
	void Stop();

private:
	// Shared with the workers, which may outlive the queue while they run an abandoned job.
	struct State {
		Discard                 discard;
		std::mutex              mutex;
		std::condition_variable job_available;
		std::condition_variable job_finished;
		// Queued jobs by (order, ticket).
		std::map<std::pair<uint64_t, uint64_t>, Job> jobs;
		std::unordered_map<uint64_t, uint64_t>       job_orders;
		std::vector<Completion>                      completed;
		std::vector<uint64_t>                        running_tickets;
		std::unordered_set<uint64_t>                 abandoned;
		uint64_t                                     next_ticket = 1;
		uint32_t                                     active      = 0;
		// Workers running a job that is not abandoned.
		uint32_t busy         = 0;
		bool     stopped      = false;
		bool     low_priority = false;
		// Worker threads by id; an abandoned worker is detached and removed.
		std::unordered_map<uint64_t, std::thread> threads;
		uint64_t                                  next_worker = 0;
		// Which worker runs which ticket.
		std::unordered_map<uint64_t, uint64_t> ticket_workers;
	};

	static void Worker(const std::shared_ptr<State>& state, uint64_t worker);
	// Starts a worker; called with the mutex held.
	static void Spawn(const std::shared_ptr<State>& state);
	// Records a finished job; called with the mutex held.
	static void Finish(State& state, uint64_t ticket, uint64_t handle);

	std::shared_ptr<State> m_state;
	uint32_t               m_thread_count = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECOMPILEQUEUE_H_
