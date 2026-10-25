#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::PipelineCompileQueue;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "PipelineCompileQueueTests: failed: %s\n", text);
		std::abort();
	}
}

// Holds jobs until released, so that a test can observe a job while it runs.
class Gate {
public:
	void Release() {
		{
			std::lock_guard lock(m_mutex);
			m_open = true;
		}
		m_cv.notify_all();
	}
	void Wait() {
		std::unique_lock lock(m_mutex);
		m_cv.wait(lock, [this] { return m_open; });
	}

private:
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	bool                    m_open = false;
};

// Counts how many job inputs are alive, like the create-info copies a pipeline job owns.
struct Inputs {
	explicit Inputs(std::atomic<int>& alive): alive(alive) { alive++; }
	~Inputs() { alive--; }
	std::atomic<int>& alive;
};

void TestCompletion() {
	std::vector<uint64_t> discarded;
	PipelineCompileQueue  queue(2, [&](uint64_t handle) { discarded.push_back(handle); });
	std::atomic<int>      alive = 0;

	std::vector<uint64_t> tickets;
	for (uint64_t i = 0; i < 8; i++) {
		tickets.push_back(queue.Enqueue(
		    [inputs = std::make_unique<Inputs>(alive), i] { return 0x100 + i; }));
	}
	Check(std::adjacent_find(tickets.begin(), tickets.end(), std::greater_equal<>()) ==
	          tickets.end(),
	      "tickets increase");
	queue.WaitIdle();
	Check(alive == 0, "a finished job releases its inputs");
	Check(queue.Outstanding() == 8, "finished results stay outstanding until collected");

	std::vector<PipelineCompileQueue::Completion> completed;
	queue.TakeCompleted(completed);
	Check(completed.size() == 8 && queue.Outstanding() == 0, "every result is collected once");
	for (const auto& completion: completed) {
		const auto index = std::find(tickets.begin(), tickets.end(), completion.ticket);
		Check(index != tickets.end() &&
		          completion.handle == 0x100 + static_cast<uint64_t>(index - tickets.begin()),
		      "a result carries its job's ticket");
	}
	completed.clear();
	queue.TakeCompleted(completed);
	Check(completed.empty(), "nothing is collected twice");

	queue.Enqueue([] { return uint64_t {0}; });
	queue.WaitIdle();
	queue.TakeCompleted(completed);
	Check(completed.size() == 1 && completed[0].handle == 0, "a failed build reports zero");

	queue.Stop();
	Check(discarded.empty(), "collected results are not discarded");
}

void TestStop() {
	std::vector<uint64_t> discarded;
	auto                  queue = std::make_unique<PipelineCompileQueue>(
        1, [&](uint64_t handle) { discarded.push_back(handle); });
	std::atomic<int>      alive = 0;
	Gate                  started;
	Gate                  release;

	// The single worker blocks in the first job; the others stay queued.
	queue->Enqueue([&, inputs = std::make_unique<Inputs>(alive)] {
		started.Release();
		release.Wait();
		return uint64_t {0x11};
	});
	for (uint64_t i = 0; i < 4; i++) {
		queue->Enqueue([inputs = std::make_unique<Inputs>(alive)] { return uint64_t {0x22}; });
	}
	started.Wait();
	Check(alive == 5 && queue->Outstanding() == 5, "a running job and the queued ones are alive");

	std::thread releaser([&] { release.Release(); });
	queue->Stop();
	releaser.join();
	Check(alive == 0, "stopping drops the queued jobs and waits for the running one");
	Check(discarded.size() == 1 && discarded[0] == 0x11,
	      "the uncollected result of the running job is discarded, the dropped jobs never ran");
	Check(queue->Outstanding() == 0, "nothing is outstanding after a stop");
	queue->Stop();
	queue.reset();
	Check(discarded.size() == 1, "stopping twice discards nothing more");

	// A finished, uncollected result is discarded by the destructor.
	discarded.clear();
	{
		PipelineCompileQueue other(1, [&](uint64_t handle) { discarded.push_back(handle); });
		other.Enqueue([] { return uint64_t {0x33}; });
		other.Enqueue([] { return uint64_t {0}; });
		other.WaitIdle();
	}
	Check(discarded.size() == 1 && discarded[0] == 0x33,
	      "the destructor discards uncollected results and skips failed builds");
}

void TestExpedite() {
	std::vector<uint64_t> discarded;
	PipelineCompileQueue  queue(1, [&](uint64_t handle) { discarded.push_back(handle); });
	Gate                  started;
	Gate                  release;
	const auto            caller = std::this_thread::get_id();

	// The single worker blocks in the first job, so the second one stays queued.
	const auto running = queue.Enqueue([&] {
		started.Release();
		release.Wait();
		return uint64_t {0x44};
	});
	std::thread::id ran_on;
	const auto      queued = queue.Enqueue([&] {
        ran_on = std::this_thread::get_id();
        return uint64_t {0x55};
    });
	started.Wait();
	queue.Expedite(queued);
	Check(ran_on == caller, "a job that has not started runs on the expediting thread");
	std::vector<PipelineCompileQueue::Completion> completed;
	queue.TakeCompleted(completed);
	Check(completed.size() == 1 && completed[0].ticket == queued && completed[0].handle == 0x55,
	      "an expedited job's result is available when Expedite returns");
	Check(queue.Outstanding() == 1, "the running job is still outstanding");

	std::thread releaser([&] { release.Release(); });
	queue.Expedite(running);
	releaser.join();
	completed.clear();
	queue.TakeCompleted(completed);
	Check(completed.size() == 1 && completed[0].ticket == running && completed[0].handle == 0x44,
	      "expediting a running job waits for its result");

	queue.Expedite(running);
	queue.Expedite(0x1234);
	Check(queue.Outstanding() == 0, "expediting a collected or unknown ticket does nothing");
	queue.Stop();
	Check(discarded.empty(), "collected results are not discarded");
}

void TestOrder() {
	PipelineCompileQueue queue(1, [](uint64_t) {});
	Gate                 started;
	Gate                 release;
	std::mutex           mutex;
	std::vector<int>     ran;
	queue.Enqueue([&] {
		started.Release();
		release.Wait();
		return uint64_t {0};
	});
	started.Wait();
	// Queued while the only worker is busy: they start by order, equal orders by queueing.
	for (const auto& [order, id]:
	     std::vector<std::pair<uint64_t, int>> {{5, 1}, {1, 2}, {5, 3}, {0, 4}, {1, 5}}) {
		queue.Enqueue(
		    [&, id] {
			    std::lock_guard lock(mutex);
			    ran.push_back(id);
			    return uint64_t {0};
		    },
		    order);
	}
	Check(queue.QueuedBelow(1) == 1 && queue.QueuedBelow(5) == 3 && queue.QueuedBelow(6) == 5,
	      "queued jobs are counted by order");
	release.Release();
	queue.WaitIdle();
	Check(ran == std::vector<int> {4, 2, 5, 1, 3}, "jobs start by order, then in queueing order");
	Check(queue.QueuedBelow(~uint64_t {0}) == 0, "nothing is queued after the jobs ran");
}

void TestActiveThreads() {
	PipelineCompileQueue queue(4, [](uint64_t) {}, {.low_priority = true});
	Check(queue.ThreadCount() == 4 && queue.ActiveThreads() == 4, "all workers start active");
	queue.SetActiveThreads(1);
	std::atomic<int> running = 0;
	std::atomic<int> peak    = 0;
	for (int i = 0; i < 16; i++) {
		queue.Enqueue([&] {
			const auto now  = ++running;
			int        seen = peak.load();
			while (now > seen && !peak.compare_exchange_weak(seen, now)) {
			}
			std::this_thread::sleep_for(std::chrono::milliseconds {2});
			running--;
			return uint64_t {0};
		});
	}
	queue.WaitIdle();
	Check(peak == 1, "one active worker runs one job at a time");

	// Raising the limit lets the waiting workers take jobs again.
	queue.SetActiveThreads(4);
	Gate             release;
	std::atomic<int> started = 0;
	for (int i = 0; i < 4; i++) {
		queue.Enqueue([&] {
			started++;
			release.Wait();
			return uint64_t {0};
		});
	}
	while (started < 4) {
		std::this_thread::yield();
	}
	release.Release();
	queue.WaitIdle();
	Check(queue.ActiveThreads() == 4, "the active count is kept");
}

void TestAbandon() {
	// The abandoned job outlives the queue, so everything it touches is shared.
	struct Shared {
		Gate             started;
		Gate             release;
		Gate             returned;
		std::atomic<int> discarded = 0;
	};
	auto shared = std::make_shared<Shared>();
	auto queue  = std::make_unique<PipelineCompileQueue>(
	    1, [shared](uint64_t handle) { shared->discarded += static_cast<int>(handle); });
	const auto        stuck    = queue->Enqueue([shared] {
		shared->started.Release();
		shared->release.Wait();
		shared->returned.Release();
		return uint64_t {7};
	});
	std::atomic<bool> next_ran = false;
	const auto        next     = queue->Enqueue([&] {
		next_ran = true;
		return uint64_t {0};
	});
	shared->started.Wait();
	Check(!queue->Abandon(next), "a queued job cannot be abandoned");
	Check(queue->Abandon(stuck) && queue->AbandonedRunning() == 1, "a running job is abandoned");
	Check(!queue->Abandon(stuck), "a job is abandoned once");
	queue->Expedite(stuck);
	queue->WaitIdle();
	Check(next_ran, "a new worker takes the place of the abandoned one");
	std::vector<PipelineCompileQueue::Completion> completed;
	queue->TakeCompleted(completed);
	Check(std::ranges::count(completed, stuck, &PipelineCompileQueue::Completion::ticket) == 1 &&
	          std::ranges::find(completed, stuck, &PipelineCompileQueue::Completion::ticket)
	                  ->handle == 0,
	      "an abandoned job completes at once without a handle");

	// Its late result is discarded while the queue runs.
	shared->release.Release();
	shared->returned.Wait();
	while (queue->AbandonedRunning() != 0) {
		std::this_thread::yield();
	}
	Check(shared->discarded == 7, "the late handle of an abandoned job is discarded");

	// Stopping does not wait for an abandoned job, which may then return after the queue is
	// gone.
	auto       second = std::make_shared<Shared>();
	const auto id     = queue->Enqueue([second] {
		second->started.Release();
		second->release.Wait();
		second->returned.Release();
		return uint64_t {9};
	});
	second->started.Wait();
	Check(queue->Abandon(id), "a second running job is abandoned");
	queue->Stop();
	queue.reset();
	second->release.Release();
	second->returned.Wait();
	Check(shared->discarded == 7, "a handle returned after the stop is not discarded");
}

} // namespace

int main() {
	TestCompletion();
	TestStop();
	TestExpedite();
	TestOrder();
	TestActiveThreads();
	TestAbandon();
	std::printf("PipelineCompileQueueTests: all tests passed\n");
	return 0;
}
