#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects the shader interface, resource usage and instruction features after resource tracking.
void CollectShaderInfo(Program& program, ShaderStageInputInfo input_info);

// Rewrites the LDS addresses of a pixel shader to be relative to its own lane when every address
// is LaneId times one stride plus a lane-independent part (see ShaderInfoCollection.cpp).
void MakePixelLdsLaneRelative(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
