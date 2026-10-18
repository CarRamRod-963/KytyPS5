#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SRTWALKERCOMPILEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SRTWALKERCOMPILEQUEUE_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace Libs::Graphics {

// Compiles the general SRT walker of resource plans on a background thread. Emitting it can take
// a hundred milliseconds, much of it for plans whose code outgrows the compiler's limit and
// stays interpreted (about a third of Ghost of Yotei's compute shaders), and it ran on the GPU
// thread at every first use of a shader.
//
// A job owns its own plan, extracted from the same program as the plan the GPU thread
// interprets meanwhile (extraction is deterministic, so the two are interchangeable). Only the
// job's thread touches the plan until Ready(); then the owner takes it and replaces its
// interpreted plan between two refreshes. Interpreted and compiled walks give the same results,
// so when the replacement happens does not change any refresh.
class SrtWalkerCompileQueue {
public:
	class Job {
	public:
		explicit Job(std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan)
		    : m_plan(std::move(plan)) {}
		[[nodiscard]] bool Ready() const noexcept {
			return m_ready.load(std::memory_order_acquire);
		}
		// The compiled plan, or null when the general walker does not cover it. Requires Ready().
		[[nodiscard]] std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> Take() {
			if (!Ready() || m_plan == nullptr || m_plan->srt_general_code == nullptr)
				return nullptr;
			return std::move(m_plan);
		}

	private:
		friend class SrtWalkerCompileQueue;
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> m_plan;
		std::atomic<bool>                                   m_ready {false};
	};

	SrtWalkerCompileQueue(): m_thread([this] { Run(); }) {}
	~SrtWalkerCompileQueue() {
		{
			std::scoped_lock lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		m_thread.join();
	}
	SrtWalkerCompileQueue(const SrtWalkerCompileQueue&)            = delete;
	SrtWalkerCompileQueue& operator=(const SrtWalkerCompileQueue&) = delete;

	std::shared_ptr<Job> Submit(std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan) {
		auto job = std::make_shared<Job>(std::move(plan));
		{
			std::scoped_lock lock(m_mutex);
			m_jobs.push_back(job);
		}
		m_wake.notify_one();
		return job;
	}

private:
	void Run() {
		for (;;) {
			std::shared_ptr<Job> job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return m_stop || !m_jobs.empty(); });
				if (m_stop) return;
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			// A job its owner dropped (the shader was re-extracted) is not compiled.
			if (job.use_count() > 1) ShaderRecompiler::IR::CompileGeneralSrtWalker(*job->m_plan);
			job->m_ready.store(true, std::memory_order_release);
		}
	}

	std::mutex                       m_mutex;
	std::condition_variable          m_wake;
	std::deque<std::shared_ptr<Job>> m_jobs;
	bool                             m_stop = false;
	std::thread                      m_thread;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SRTWALKERCOMPILEQUEUE_H_ */
