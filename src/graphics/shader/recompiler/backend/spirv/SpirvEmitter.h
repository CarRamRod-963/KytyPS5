#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Host device capabilities that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	// NMin/NMax order -0 below +0 and return the other operand of a single NaN; a NaN pair gives a
	// canonical NaN instead of the guest's first source, and FP32 denormals may be flushed.
	bool ordered_float_min_max = false;
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
	// Storage buffer elements past the descriptor range, to the byte, read zero and drop writes.
	bool robust_buffer_access = false;
	// 8 and 8_8 sRGB textures sample through sRGB views; without one the shader decodes them.
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
};

inline constexpr Profile DefaultProfile {};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool           emit_debug_info       = false);

// Set once, before any emission, when the device enables VK_KHR_shader_maximal_reconvergence.
// Guest waves iterate scalar loops whose exits need every lane of the host subgroup to take
// part in each ballot and broadcast; only maximal reconvergence guarantees that.
void SetMaximalReconvergence(bool enabled);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
