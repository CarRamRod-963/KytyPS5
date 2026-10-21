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

// Compiles the SRT walker of resource plans extracted with SrtWalkerCompilation::Deferred on a
// background thread, with reader loads: the plans are walked off the GPU thread
// (ResourceLookahead), where a load must neither fault into the caches nor read anything but the
// walk's reader.
//
// A job owns its own plan, extracted from the same program as the plan the GPU thread walks
// (extraction is deterministic, so the two are interchangeable). Only the job's thread touches
// the plan until Ready().
class SrtWalkerCompileQueue {
public:
	class Job {
	public:
		explicit Job(std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan)
		    : m_plan(std::move(plan)) {}
		[[nodiscard]] bool Ready() const noexcept {
			return m_ready.load(std::memory_order_acquire);
		}
		// The plan once Ready(), with its walker unless the plan is incomplete. Null before.
		[[nodiscard]] const ShaderRecompiler::IR::ResourcePlan* Plan() const {
			return Ready() ? m_plan.get() : nullptr;
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
			auto& plan = *job->m_plan;
			if (job.use_count() > 1 && plan.srt_plan_complete && plan.resource_tracking_complete) {
				ShaderRecompiler::IR::CompileSrtWalker(
				    plan, {}, ShaderRecompiler::IR::SrtWalkerLoads::Reader);
			}
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
