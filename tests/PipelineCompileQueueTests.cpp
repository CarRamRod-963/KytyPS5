#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
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

} // namespace

int main() {
	TestCompletion();
	TestStop();
	std::printf("PipelineCompileQueueTests: all tests passed\n");
	return 0;
}
