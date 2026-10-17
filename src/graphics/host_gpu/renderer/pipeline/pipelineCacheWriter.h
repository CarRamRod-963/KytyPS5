#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHEWRITER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHEWRITER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"

#include <atomic>
#include <thread>
#include <utility>

namespace Libs::Graphics {

// Runs driver pipeline-cache writes on a background thread, one at a time, so that the thread
// that creates pipelines never waits for the cache snapshot or the file. A single owner thread
// at a time calls TryStart and Wait. Vulkan-free so that it can be tested without a device.
class PipelineCacheWriter {
public:
	using Write = Common::UniqueFunction<void>;

	PipelineCacheWriter() = default;
	~PipelineCacheWriter() { Wait(); }
	KYTY_CLASS_NO_COPY(PipelineCacheWriter);

	// Starts the write unless the previous one is still running; returns whether it started.
	bool TryStart(Write write) {
		if (m_running.load(std::memory_order_acquire)) {
			return false;
		}
		if (m_thread.joinable()) {
			m_thread.join();
		}
		m_running.store(true, std::memory_order_relaxed);
		m_thread = std::thread([this, write = std::move(write)] {
			write();
			m_running.store(false, std::memory_order_release);
		});
		return true;
	}

	[[nodiscard]] bool Busy() const { return m_running.load(std::memory_order_acquire); }

	// Blocks until the running write, if any, has finished. Idempotent.
	void Wait() {
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}

private:
	std::thread       m_thread;
	std::atomic<bool> m_running {false};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHEWRITER_H_
