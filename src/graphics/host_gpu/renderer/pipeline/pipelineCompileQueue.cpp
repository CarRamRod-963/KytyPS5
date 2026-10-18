#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include "common/assert.h"
#include "common/profiler.h"

#include <algorithm>
#include <utility>

namespace Libs::Graphics {

PipelineCompileQueue::PipelineCompileQueue(uint32_t thread_count, Discard discard)
    : m_discard(std::move(discard)) {
	EXIT_IF(thread_count == 0 || !m_discard);
	m_threads.reserve(thread_count);
	for (uint32_t i = 0; i < thread_count; i++) {
		m_threads.emplace_back([this] { Worker(); });
	}
}

PipelineCompileQueue::~PipelineCompileQueue() {
	Stop();
}

uint64_t PipelineCompileQueue::Enqueue(Job job) {
	EXIT_IF(!job);
	uint64_t ticket = 0;
	{
		std::lock_guard lock(m_mutex);
		EXIT_IF(m_stopped);
		ticket = m_next_ticket++;
		m_jobs.push_back({ticket, std::move(job)});
	}
	m_job_available.notify_one();
	return ticket;
}

void PipelineCompileQueue::TakeCompleted(std::vector<Completion>& out) {
	std::lock_guard lock(m_mutex);
	out.insert(out.end(), m_completed.begin(), m_completed.end());
	m_completed.clear();
}

size_t PipelineCompileQueue::Outstanding() {
	std::lock_guard lock(m_mutex);
	return m_jobs.size() + m_running_tickets.size() + m_completed.size();
}

void PipelineCompileQueue::WaitIdle() {
	std::unique_lock lock(m_mutex);
	m_job_finished.wait(lock, [this] { return m_jobs.empty() && m_running_tickets.empty(); });
}

void PipelineCompileQueue::Expedite(uint64_t ticket) {
	std::unique_lock lock(m_mutex);
	const auto       queued = std::ranges::find(m_jobs, ticket, &Entry::ticket);
	if (queued == m_jobs.end()) {
		m_job_finished.wait(lock, [&] {
			return std::ranges::find(m_running_tickets, ticket) == m_running_tickets.end();
		});
		return;
	}
	auto job = std::move(queued->job);
	m_jobs.erase(queued);
	m_running_tickets.push_back(ticket);
	lock.unlock();
	const auto handle = job();
	job               = Job {};
	lock.lock();
	Finish(ticket, handle);
	lock.unlock();
	m_job_finished.notify_all();
}

void PipelineCompileQueue::Finish(uint64_t ticket, uint64_t handle) {
	m_completed.push_back({ticket, handle});
	std::erase(m_running_tickets, ticket);
}

void PipelineCompileQueue::Stop() {
	std::deque<Entry> dropped;
	{
		std::lock_guard lock(m_mutex);
		m_stopped = true;
		dropped   = std::move(m_jobs);
		m_jobs.clear();
	}
	// A job owns its build inputs; destroy them outside the lock.
	dropped.clear();
	m_job_available.notify_all();
	for (auto& thread: m_threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}
	m_threads.clear();

	std::vector<Completion> completed;
	TakeCompleted(completed);
	for (const auto& completion: completed) {
		if (completion.handle != 0) {
			m_discard(completion.handle);
		}
	}
}

void PipelineCompileQueue::Worker() {
	KYTY_PROFILER_THREAD("Thread_PipelineCompile");
	for (;;) {
		Entry entry;
		{
			std::unique_lock lock(m_mutex);
			m_job_available.wait(lock, [this] { return m_stopped || !m_jobs.empty(); });
			if (m_stopped) {
				return;
			}
			entry = std::move(m_jobs.front());
			m_jobs.pop_front();
			m_running_tickets.push_back(entry.ticket);
		}
		const auto handle = entry.job();
		entry.job         = Job {};
		{
			std::lock_guard lock(m_mutex);
			Finish(entry.ticket, handle);
		}
		m_job_finished.notify_all();
	}
}

} // namespace Libs::Graphics
