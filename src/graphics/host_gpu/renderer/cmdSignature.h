#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDSIGNATURE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDSIGNATURE_H_

#include "common/onceValue.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

// KYTY_TRACE_CMD_SIGNATURE=1|2 (diagnostic, default off; read once, "TRACE_CMD_SIGNATURE enabled mode=N" logged once):
// a fingerprint of the command stream the GPU thread records, from the stage F design (section 4
// item 2), so a step that moves work to another thread can show it records the same commands.
//
// At the recording points (CommitBindings, BeginRendering, the draw and dispatch calls) the GPU thread folds a canonical
// encoding of what it records into a 64-bit hash. The encoding uses identities, not handles: programs by shader hash and
// stage, images by guest address, formats, extent, view range and layout (and the image the texture cache found by its
// guest address, format and extent), samplers by their key (S# words and border class), buffers by guest range, shader
// data words, the flattened SRT, draw and dispatch arguments, and render-pass sizes, layouts, load ops and clear values.
// Each draw/dispatch call (as KYTY_TRACE_STAGEF's events) gets its own signature, which is folded into the signature of
// its debug-marker pass (the innermost two markers, DebugMarkerPath) and of the frame (events between two presents).
//
// Output (stderr): =1 one line per presented frame
//   CMD_SIGNATURE frame=<n> events=<n> draws=<n> dispatches=<n> outside=<folds outside events> passes=<n> sig=<hex>
// =2 also one line per pass of the frame, in the order each pass first appeared:
//   CMD_SIGNATURE_PASS frame=<n> pass=<marker path> events=<n> sig=<hex>
// KYTY_TRACE_CMD_SIGNATURE_FILE=<path> (with either mode): one line per event appended to the file
//   <frame> <event> <kind> <sig> <marker path>
// limited to frames KYTY_TRACE_CMD_SIGNATURE_FRAMES=<first>-<last> (default all; the file can grow large).
//
// Not covered yet (see the commit): barriers, copies, uploads and clears outside render passes, dynamic state, push
// constants as bytes, the pipeline's key; render-target views (BeginRendering folds their sizes, layouts and load ops,
// not which image).
// With the switch off every hook is one load of a constant-initialized flag and a not-taken branch.
namespace Libs::Graphics::CmdSignature {

[[nodiscard]] uint32_t ModeInit();
[[nodiscard]] inline uint32_t Mode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	return mode_once.Get([] { return ModeInit(); });
}
[[nodiscard]] inline bool Enabled() {
	return Mode() != 0;
}

// What a fold encodes (part of the hash).
enum class Kind : uint32_t {
	EventBegin = 1,
	Stage,          // a program's stage and shader hash (CommitBindings)
	Image,          // one image binding
	Sampler,        // one sampler binding
	Buffer,         // one buffer binding's guest range
	ShaderData,     // the user-data / push words of a stage
	Srt,            // the flattened SRT of a stage
	Draw,           // vkCmdDraw / DrawIndexed arguments
	DrawIndirect,   // GPU-driven draw: guest argument and count addresses, max count, stride
	DrawMesh,       // mesh task groups
	Dispatch,       // group counts
	DispatchIndirect,
	BeginRendering, // render-pass size, layers, attachments' layouts, load ops, clear values
};

// Kinds of events (draw/dispatch calls).
enum class EventKind : uint32_t { Draw = 1, DrawIndirect, Dispatch, DispatchIndirect };

// GuestGpu::ThreadRun: the calling thread is the GPU thread (the only one whose folds count).
void MarkGpuThread();
// GPU thread (other threads: ignored). Folds `size` bytes tagged with `kind` into the current event (outside events:
// into the frame and its pass directly).
void Fold(Kind kind, const void* data, size_t size);
inline void FoldWords(Kind kind, std::initializer_list<uint64_t> words) {
	Fold(kind, words.begin(), words.size() * sizeof(uint64_t));
}
[[nodiscard]] bool EventBegin(EventKind kind);
void               EventEnd();

// One per draw or dispatch call (next to StageF::EventScope); nested calls count as their outermost call.
class EventScope {
public:
	explicit EventScope(EventKind kind) {
		if (Enabled()) [[unlikely]] {
			m_on = EventBegin(kind);
		}
	}
	~EventScope() {
		if (m_on) EventEnd();
	}
	EventScope(const EventScope&)            = delete;
	EventScope& operator=(const EventScope&) = delete;

private:
	bool m_on = false;
};

} // namespace Libs::Graphics::CmdSignature

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDSIGNATURE_H_
