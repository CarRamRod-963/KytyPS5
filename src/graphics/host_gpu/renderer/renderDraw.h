#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include <cstdint>
#include <utility>

namespace Libs::Graphics {

struct ShaderVertexInputInfo;

// Host vertex/instance offsets of a draw: the hardware index offset plus, for a fetch rewritten
// into Vulkan vertex inputs, the user SGPR offsets the guest fetch adds (or the unpatched
// indirect offsets when no such SGPR was recognized).
[[nodiscard]] std::pair<int32_t, uint32_t>
ResolveDrawOffsets(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info,
                   int32_t unpatched_vertex_offset = 0, uint32_t unpatched_instance_offset = 0);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
