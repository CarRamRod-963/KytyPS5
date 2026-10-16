#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include "common/assert.h"
#include "common/profiler.h"

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
	return m_jobs.size() + m_running + m_completed.size();
}

void PipelineCompileQueue::WaitIdle() {
	std::unique_lock lock(m_mutex);
	m_job_finished.wait(lock, [this] { return m_jobs.empty() && m_running == 0; });
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
			m_running++;
		}
		const auto handle = entry.job();
		entry.job         = Job {};
		{
			std::lock_guard lock(m_mutex);
			m_completed.push_back({entry.ticket, handle});
			m_running--;
		}
		m_job_finished.notify_all();
	}
}

} // namespace Libs::Graphics
