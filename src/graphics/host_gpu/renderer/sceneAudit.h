#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SCENEAUDIT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SCENEAUDIT_H_

#include "common/onceValue.h"

#include <cstdint>
#include <string_view>

namespace Libs::Graphics {

class RenderContext;
struct PreparedBindings;
struct RenderColorInfo;
struct RenderDepthInfo;
struct ShaderStageRuntime;

// KYTY_TRACE_AUDIT_FRAME=<n> (with KYTY_TRACE_FRAME_DIR; diagnostic, nothing of the frame changes): scene audit of
// presented frame n (the draws and dispatches recorded while DebugPresentCount() == n, like the AUDIT_USE lines).
// Every draw and dispatch gives one record in <KYTY_TRACE_FRAME_DIR>/scene_audit_f<n>.jsonl: sequence number, game
// debug-marker path (the pass), shader hashes, every bound image/buffer/sampler with address, format and extent, a
// flag for emulator placeholders (null/soft null images, GPU-selected or bindless descriptors), the render targets
// and storage outputs it writes, and whether the emulator skipped it or replaced its work (APPROXIMATION switches).
// Images and buffers are copied whole (the mip a read samples, every layer and slice; buffers up to
// KYTY_SCENE_AUDIT_BUFFER_MB) and analyzed on the host: zero / constant counts over the full extent and, for
// uncompressed formats, per-channel min/max/mean and NaN/Inf counts ("snap" records). A read is copied right
// before the command that reads it when its contents changed since the last copy (so an output is copied when the
// next pass reads it); outputs nobody read again are copied when the frame is over. scripts/scene_report.py turns
// the file into scene_report.md. The copies are fenced like the RT dumps; the batch is analyzed (submit and wait)
// before a command once KYTY_SCENE_AUDIT_BATCH_MB (default 1024) are pending.
// KYTY_SCENE_AUDIT_TOTAL_MB (default 16384): no more copies after that many bytes.
// KYTY_SCENE_AUDIT_MAX_MB (default 512): larger single images are listed but not copied.
// KYTY_SCENE_AUDIT_BUFFER_MB (default 64): larger buffers are copied partially (first bytes).
// KYTY_SCENE_AUDIT_REPEAT (default 16): an image mip or buffer range is copied for reads at most that many times in
// the frame (ping-pong blurs, iterated simulations); its frame-end copy is always taken.
// KYTY_SCENE_AUDIT_STATIC=0: images the GPU never wrote (streamed textures) are not copied.
// KYTY_SCENE_AUDIT_BUFFERS=0: buffers are listed but not copied.
namespace SceneAudit {

// The audited frame (0: off; also 0 without KYTY_TRACE_FRAME_DIR).
[[nodiscard]] uint32_t ConfiguredFrame();

[[nodiscard]] inline bool Enabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([&] { return ConfiguredFrame() != 0u; });
	return on;
}

// One shader stage of a draw or dispatch; bindings == nullptr lists the descriptors only.
struct Stage {
	const ShaderStageRuntime* runtime  = nullptr;
	const PreparedBindings*   bindings = nullptr;
};

// At the start of a draw or dispatch, before anything of it is recorded: writes the previous record, analyzes a
// full batch and, at the first command after the audited frame, copies the remaining outputs and closes the file
// (both submit the scheduler and wait).
void OnCommandStart(RenderContext& context);
// A draw or dispatch with its bindings prepared, before they are committed (copies need no rendering instance and
// leave images in the general layout; committing the bindings transitions them back).
void RecordDraw(RenderContext& context, const Stage* stages, uint32_t stage_count, const RenderColorInfo* colors,
                uint32_t color_count, const RenderDepthInfo* depth, uint32_t count, uint32_t instances, bool indexed,
                bool gpu_indirect, bool mesh);
void RecordDispatch(RenderContext& context, const Stage& stage, uint32_t groups_x, uint32_t groups_y,
                    uint32_t groups_z, bool indirect);
// A draw or dispatch the emulator does not run as the guest recorded it (dropped, skipped, replaced by a clear).
void RecordSkipped(const char* kind, const char* reason, uint64_t shader_hash = 0);
// A dispatch skipped by KYTY_DIAG_SKIP_CS_HASHES / KYTY_DIAG_SKIP_UNSUPPORTED_CS (shader code address).
void RecordSkippedDispatch(uint64_t code_address, bool indirect);
// The register state behind the skip of the last recorded command (a JSON object), written as its "skip_state".
void NoteSkipState(std::string_view json_object);
// The last recorded command was not run after all.
void NoteSkipped(const char* reason);
// A bindless heap key resolved to the dummy slot (descriptors.cpp BindlessReject).
void NoteBindlessReject(const char* stat, uint64_t heap, uint32_t entry);

} // namespace SceneAudit

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SCENEAUDIT_H_
