#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANRECORDING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANRECORDING_H_

#include <cstddef>
#include <cstdint>

// Vulkan command recording on a thread of its own (--vulkan-recording-thread).
//
// The GPU thread spends a large part of its time inside the driver: every vkCmd* validates and
// encodes there. With a ProducerScope, the calling thread (the producer) copies its Vulkan calls
// that only record, arguments and all, into a stream (recordingStream.h), and a recording thread
// makes them in the same order. The producer stays the only owner of the renderer's state; the
// recording thread only calls the driver.
//
// Install replaces entries of Vulkan-Hpp's default dispatcher, so no call site changes. On a
// producer thread:
// - every vkCmd* with copyable arguments, vkBegin/EndCommandBuffer, descriptor set updates, host
//   query pool resets and object destruction are queued (the destruction keeps its place behind
//   the calls that name the object);
// - command pool and descriptor pool operations, queue submissions, presentation, fences and
//   idle waits first wait until the recording thread made every queued call (Drain), then call
//   the driver;
// - every other call (creation, queries of properties and progress, memory mapping) goes to the
//   driver at once: it cannot depend on the queued calls.
// On every other thread the entries call the driver as before. Command buffers are handed to the
// submission thread behind their commands (CommandScheduler::Submit, QueueSubmitter::Reserve).
//
// The idea follows the Vulkan recording thread of yumlevi's KytyPS5 fork (commits 7830ebca0 and
// its follow-ups on saros-port), which builds on chenxiao07's fork; this implementation is
// written for this tree without a generated thunk table.
namespace Libs::Graphics::VulkanRecording {

// Replaces the default dispatcher's entries; after the dispatcher was initialized for the device
// and before the memory allocator is created. Without it, ProducerScope records nothing.
void               Install();
[[nodiscard]] bool Installed();

// The calling thread becomes a producer for the scope's lifetime: its recording thread starts
// here and ends, drained, with the scope. One scope per thread; no effect without Install or
// with `enabled` false.
class ProducerScope {
public:
	explicit ProducerScope(bool enabled = true);
	~ProducerScope();
	ProducerScope(const ProducerScope&)            = delete;
	ProducerScope& operator=(const ProducerScope&) = delete;

private:
	void* m_stream = nullptr;
};

// The calling thread stops being a producer for the scope's lifetime (its queued calls are made
// first): for a thread that hands the renderer to the GPU thread (RenderContext::InitializeGpu).
// Restored when the scope ends on the same thread.
class PauseScope {
public:
	PauseScope();
	~PauseScope();
	PauseScope(const PauseScope&)            = delete;
	PauseScope& operator=(const PauseScope&) = delete;

private:
	void* m_stream = nullptr;
	void* m_thread = nullptr;
};

// The calling thread is a producer: its recording calls are queued.
[[nodiscard]] bool Recording();

// Returns once the recording thread has made every call the calling thread queued. Needed
// before code that records with the driver's own entry points (an upscaler library) or hands
// the thread's command buffers to another thread. Does not wait for the GPU; no effect on a
// thread that is not a producer.
void Drain();

// Queues `function(data)` to run on the recording thread after everything queued so far, with
// a copy of the `size` bytes at `data` (trivially copyable), and hands it over at once. False
// when nothing was queued (the calling thread is not a producer, or the copy is too large): every
// call queued before has then been made, and the caller runs `function` itself.
[[nodiscard]] bool EnqueueOrdered(void (*function)(const void*), const void* data, size_t size);

} // namespace Libs::Graphics::VulkanRecording

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANRECORDING_H_ */
