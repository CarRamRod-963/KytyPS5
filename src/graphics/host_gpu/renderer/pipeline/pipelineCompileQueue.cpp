#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "common/threads.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace Libs::Graphics {

PipelineCompileQueue::PipelineCompileQueue(uint32_t thread_count, Discard discard)
    : PipelineCompileQueue(thread_count, std::move(discard), Options {}) {}

PipelineCompileQueue::PipelineCompileQueue(uint32_t thread_count, Discard discard, Options options)
    : m_state(std::make_shared<State>()), m_thread_count(thread_count) {
	EXIT_IF(thread_count == 0 || !discard);
	m_state->discard      = std::move(discard);
	m_state->active       = thread_count;
	m_state->low_priority = options.low_priority;
	std::lock_guard lock(m_state->mutex);
	for (uint32_t i = 0; i < thread_count; i++) {
		Spawn(m_state);
	}
}

PipelineCompileQueue::~PipelineCompileQueue() {
	Stop();
}

void PipelineCompileQueue::Spawn(const std::shared_ptr<State>& state) {
	const auto worker = state->next_worker++;
	state->threads.emplace(worker, std::thread([state, worker] { Worker(state, worker); }));
}

uint64_t PipelineCompileQueue::Enqueue(Job job, uint64_t order) {
	EXIT_IF(!job);
	auto&    state  = *m_state;
	uint64_t ticket = 0;
	{
		std::lock_guard lock(state.mutex);
		EXIT_IF(state.stopped);
		ticket = state.next_ticket++;
		state.jobs.emplace(std::pair {order, ticket}, std::move(job));
		state.job_orders.emplace(ticket, order);
	}
	state.job_available.notify_one();
	return ticket;
}

void PipelineCompileQueue::TakeCompleted(std::vector<Completion>& out) {
	auto&           state = *m_state;
	std::lock_guard lock(state.mutex);
	out.insert(out.end(), state.completed.begin(), state.completed.end());
	state.completed.clear();
}

size_t PipelineCompileQueue::Outstanding() {
	auto&           state = *m_state;
	std::lock_guard lock(state.mutex);
	return state.jobs.size() + state.running_tickets.size() + state.completed.size();
}

size_t PipelineCompileQueue::QueuedBelow(uint64_t order) {
	auto&           state = *m_state;
	std::lock_guard lock(state.mutex);
	return static_cast<size_t>(
	    std::distance(state.jobs.begin(), state.jobs.lower_bound(std::pair {order, uint64_t {0}})));
}

void PipelineCompileQueue::WaitIdle() {
	auto&            state = *m_state;
	std::unique_lock lock(state.mutex);
	state.job_finished.wait(lock,
	                        [&] { return state.jobs.empty() && state.running_tickets.empty(); });
}

void PipelineCompileQueue::Expedite(uint64_t ticket) {
	(void)Expedite(ticket, std::chrono::milliseconds::max());
}

bool PipelineCompileQueue::Expedite(uint64_t ticket, std::chrono::milliseconds max_wait) {
	auto&            state = *m_state;
	std::unique_lock lock(state.mutex);
	const auto       order = state.job_orders.find(ticket);
	if (order == state.job_orders.end()) {
		const auto finished = [&] {
			return std::ranges::find(state.running_tickets, ticket) == state.running_tickets.end();
		};
		if (max_wait == std::chrono::milliseconds::max()) {
			state.job_finished.wait(lock, finished);
			return true;
		}
		return state.job_finished.wait_for(lock, max_wait, finished);
	}
	const auto queued = state.jobs.find({order->second, ticket});
	EXIT_IF(queued == state.jobs.end());
	auto job = std::move(queued->second);
	state.jobs.erase(queued);
	state.job_orders.erase(order);
	state.running_tickets.push_back(ticket);
	lock.unlock();
	const auto handle = job();
	job               = Job {};
	lock.lock();
	Finish(state, ticket, handle);
	lock.unlock();
	state.job_finished.notify_all();
	return true;
}

void PipelineCompileQueue::SetActiveThreads(uint32_t count) {
	EXIT_IF(count == 0 || count > m_thread_count);
	auto& state = *m_state;
	{
		std::lock_guard lock(state.mutex);
		state.active = count;
	}
	state.job_available.notify_all();
}

uint32_t PipelineCompileQueue::ActiveThreads() {
	auto&           state = *m_state;
	std::lock_guard lock(state.mutex);
	return state.active;
}

bool PipelineCompileQueue::Abandon(uint64_t ticket) {
	auto& state = *m_state;
	{
		std::lock_guard lock(state.mutex);
		const auto      worker = state.ticket_workers.find(ticket);
		if (worker == state.ticket_workers.end() || state.abandoned.contains(ticket)) {
			// Not running, running on a thread that expedited it (which the caller owns), or
			// abandoned before.
			return false;
		}
		const auto thread = state.threads.find(worker->second);
		EXIT_IF(thread == state.threads.end());
		thread->second.detach();
		state.threads.erase(thread);
		state.abandoned.insert(ticket);
		std::erase(state.running_tickets, ticket);
		state.completed.push_back({ticket, 0});
		state.busy--;
		if (!state.stopped) {
			Spawn(m_state);
		}
	}
	state.job_finished.notify_all();
	state.job_available.notify_all();
	return true;
}

size_t PipelineCompileQueue::AbandonedRunning() {
	auto&           state = *m_state;
	std::lock_guard lock(state.mutex);
	return state.abandoned.size();
}

void PipelineCompileQueue::Finish(State& state, uint64_t ticket, uint64_t handle) {
	state.completed.push_back({ticket, handle});
	std::erase(state.running_tickets, ticket);
}

void PipelineCompileQueue::Stop() {
	auto&                                        state = *m_state;
	std::map<std::pair<uint64_t, uint64_t>, Job> dropped;
	std::unordered_map<uint64_t, std::thread>    threads;
	{
		std::unique_lock lock(state.mutex);
		state.stopped = true;
		dropped       = std::move(state.jobs);
		state.jobs.clear();
		state.job_orders.clear();
		state.job_available.notify_all();
		// Workers leave once their jobs return; an abandoned job no longer counts (Abandon may
		// still give one up while this waits).
		state.job_finished.wait(lock, [&] { return state.busy == 0; });
		threads = std::move(state.threads);
		state.threads.clear();
	}
	// A job owns its build inputs; destroy them outside the lock.
	dropped.clear();
	for (auto& [worker, thread]: threads) {
		(void)worker;
		if (thread.joinable()) {
			thread.join();
		}
	}

	std::vector<Completion> completed;
	TakeCompleted(completed);
	for (const auto& completion: completed) {
		if (completion.handle != 0) {
			state.discard(completion.handle);
		}
	}
}

void PipelineCompileQueue::Worker(const std::shared_ptr<State>& state_ptr, uint64_t worker) {
	KYTY_PROFILER_THREAD("Thread_PipelineCompile");
	auto& state = *state_ptr;
	if (state.low_priority) {
		Common::Thread::LowerCurrentThreadPriority();
	}
	for (;;) {
		Job      job;
		uint64_t ticket = 0;
		{
			std::unique_lock lock(state.mutex);
			state.job_available.wait(lock, [&] {
				return state.stopped || (!state.jobs.empty() && state.busy < state.active);
			});
			if (state.stopped) {
				return;
			}
			auto first = state.jobs.begin();
			ticket     = first->first.second;
			job        = std::move(first->second);
			state.jobs.erase(first);
			state.job_orders.erase(ticket);
			state.running_tickets.push_back(ticket);
			state.ticket_workers.emplace(ticket, worker);
			state.busy++;
		}
		const auto handle = job();
		job               = Job {};
		{
			std::lock_guard lock(state.mutex);
			state.ticket_workers.erase(ticket);
			if (state.abandoned.erase(ticket) != 0) {
				// Replaced by another worker when it was abandoned; nobody collects the result.
				// The owner keeps the device alive until Stop returns.
				if (handle != 0 && !state.stopped) {
					state.discard(handle);
				}
				return;
			}
			state.busy--;
			Finish(state, ticket, handle);
		}
		state.job_finished.notify_all();
		state.job_available.notify_one();
	}
}

} // namespace Libs::Graphics
