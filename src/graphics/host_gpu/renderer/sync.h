#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SYNC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SYNC_H_

#include "kernel/eventQueue.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandBuffer;
class Buffer;
class RenderContext;

namespace Sync {

[[nodiscard]] bool     ScaleReferenceClock(uint64_t host_ticks, uint64_t host_frequency,
                                           uint64_t& value);
[[nodiscard]] uint64_t ReadReferenceClock();
// The host clock `elapsed_ns` before `host_ticks`, in host ticks; 0 if that is before tick 0.
[[nodiscard]] uint64_t HostTicksBefore(uint64_t host_ticks, uint64_t host_frequency,
                                       uint64_t elapsed_ns);
// What ReadReferenceClock() returned at the std::chrono::steady_clock time `steady_ns`; a time
// after now reads now.
[[nodiscard]] uint64_t ReferenceClockAt(int64_t steady_ns);
// `value`, or the largest value an earlier call returned: GPU timestamps never go backwards.
[[nodiscard]] uint64_t MonotonicReferenceClock(uint64_t value);

[[nodiscard]] uint64_t PrepareVideoOutFlip(CommandBuffer& buffer, int handle, int index,
                                           int flip_mode, int64_t flip_arg);
int  AddEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id,
                void* udata);
int  DeleteEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id);
void ReadGds(const Buffer& gds, uint32_t* dst, uint32_t dw_offset, uint32_t dw_size);

} // namespace Sync
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SYNC_H_
