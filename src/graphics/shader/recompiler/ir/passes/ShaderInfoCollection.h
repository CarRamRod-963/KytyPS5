#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects the shader interface, resource usage and instruction features after resource tracking.
void CollectShaderInfo(Program& program, ShaderStageInputInfo input_info);

// Rewrites the LDS addresses of a pixel shader to be relative to its own lane when every address
// is LaneId times one stride plus a lane-independent part (see ShaderInfoCollection.cpp).
void MakePixelLdsLaneRelative(Program& program);

// Drops device-scope coherence from the GLC/DLC loads of a program that writes no memory other
// invocations read: no invocation of the same draw or dispatch can produce new data for them, and
// the renderer's barriers make earlier commands' writes visible. Loads in loops (polling) keep it.
void DropReadOnlyCoherence(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
