#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STAGEFTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STAGEFTRACE_H_

#include "common/diagFastPath.h"
#include "common/onceValue.h"

#include <cstdint>

// KYTY_TRACE_STAGEF=1 (diagnostic, default off; read once, "TRACE_STAGEF enabled" logged once): step F0 of the stage F
// design: how much of the GPU thread's work is a lookup that a helper thread could run
// ahead (L) and how much is an ordered side effect (C), whether the resolve thread already had the inputs of
// those lookups (coverage) and whether its view would still be current when the GPU thread uses it (staleness), and
// how long the GPU thread runs SendCommand lambdas. Nothing changes in what the emulator does.
//
// Time: the GPU thread's busy time is split into zones (RDTSC). Zones nest; each zone's own time excludes the zones
// inside it, so the sub-phases add up to the busy time. Only the GPU thread counts (other threads' zones are one
// thread-local load and return). The instrumentation's own cost is calibrated every window (empty zones) and
// subtracted per zone; the estimate of what is left is printed as self=.
//
// Every ~5 s the GPU thread prints one STAGEF line (see stageFTrace.cpp for the fields).
// With the switch off every hook is one load of a constant-initialized flag and a not-taken branch.
namespace Libs::Graphics::StageF {

// Sub-phases. L: lookups that are a pure function of their inputs and a registry state (a resolver thread could run
// them ahead, the GPU thread would validate them). C: ordered side effects. W: waits for the host GPU.
enum class Sub : uint8_t {
	// L
	Decode,     // T# decode and image description (ResolveTexture without the cache lookup and the tail)
	FindImage,  // TextureCache::FindImage (lookup; includes creation/merge decisions made inside it except InsertImage)
	Target,     // render-target description from the context registers (ResolveRender{Color,Depth}Target)
	FindBuffer, // BufferCache::FindBuffer (page-table lookup) and RenderExecutor::FindBuffers (V# decode)
	Sampler,    // S# decode and SamplerCache lookup (NativeSampler)
	ProgFind,   // ProgramCache::Get (key, find, permutation)
	SrtWalk,    // MaterializeResources run by the execution thread (the resolve thread's result was not taken)
	PipeFind,   // Get{Graphics,Compute}Pipeline key + find
	TrkQuery,   // page-tracker calls made for a draw/dispatch that only looked (queries; upload walks of clean ranges)
	// C
	Adopt,      // re-checking the resolve thread's read log (validation, its tracker queries included)
	Compile,    // ProgramCache::Get miss: shader recompilation
	PipeCreate, // pipeline creation on a miss, background-compile start, pipeline cache saving
	Create,     // TextureCache::InsertImage, BufferCache::CreateBuffer (with overlap joins)
	TexSync,    // FindTexture/FindRenderTarget/FindDepthTarget, ResolveTexture's tail: alias syncs, refreshes, views
	Obtain,     // BufferCache::ObtainBuffer (synchronization, copies, GPU-write marks)
	TrkMark,    // page-tracker calls that change state (marks, dirty-run walks, invalidations), and queries made
	            // outside draws/dispatches (faults, CP writes, SendCommand lambdas)
	Protect,    // page protection syscalls (PageManager::Protect)
	Staging,    // memcpy into staging/stream buffers and image uploads (detile recording)
	Record,     // CommitBindings and the recording after the "cannot touch guest memory" seam
	RecScan,    // CommitBindings: the scalar-read / shader-write overlap scan
	RecDesc,    // CommitBindings: building the descriptor writes and push data
	RecTransit, // CommitBindings: image layout transitions and the GDS / bindless barriers of the bound resources
	RecEmit,    // CommitBindings: the push-descriptor / push-constant / bind calls
	RecVtx,     // a draw: vertex and index buffer binds
	RecDyn,     // a draw: dynamic state (viewports, depth/stencil, blend constants, feedback loop)
	RecBegin,   // a draw: begin rendering and pipeline bind
	RecIssue,   // a draw: from the pipeline bind to the end of the draw call(s)
	RecPost,    // a draw: what follows the draw call(s)
	RecCPre,    // CommitBindings: counting the writes before the overlap scan
	RecCompute, // a dispatch: its recording outside CommitBindings
	RecFill,    // the record packets: writing the descriptor infos into the packet (inside rec_emit)
	RecRing,    // the record packets: reserving and committing the packet (inside rec_emit)
	// M26 / WATEK-GPU-PLAN M0: the parts of "ev_other" named (own time; the zones above stay inside them)
	EvBindless,  // PrepareBindlessHeaps
	EvImgLoop,   // RebindImages: the loop body outside the texture-cache calls
	EvBindImg,   // PrepareBindings: the image loop outside Decode / TexSync
	EvDispPre,   // a dispatch's preamble (EndRendering, meta clears, image-clear consumption)
	EvDrawPre,   // a draw's preamble (NoteGpuCommand ... topology)
	EvProgs,     // GetGraphicsPrograms outside ProgramCache::Get
	EvTargets,   // AcquireRenderTargets outside the texture-cache calls
	EvRebindBuf, // RebindBuffers outside ObtainBuffer (NativeUpload, Map)
	EvBufImgSync, // SyncBufferFromWrittenImage inside ObtainBuffer
	ObtPre,      // ObtainBuffer before the buffer lookup (direct ranges, merge / clean-range checks)
	ObtSync,     // ObtainBuffer: SynchronizeBuffer outside the tracker / staging zones
	ObtWrite,    // ObtainBuffer: NoteObtainedWrite (GPU-dirty set, write ticks, observers)
	ObtWDirty,   // NoteObtainedWrite: GpuDirtyAdd
	ObtWTick,    // NoteObtainedWrite: NoteWriteTick (per-range ticks)
	ObtWGran,    // NoteObtainedWrite: NoteGpuWriteTick (granule ticks)
	ObtWObs,     // NoteObtainedWrite: write observer, hot generation
	Submit,     // CommandScheduler::Submit / BeginCommand
	EventOther, // inside a draw/dispatch, in no zone above
	Outside,    // GPU-thread slice work in no draw/dispatch and no zone above (PM4, register state, CP writes)
	// The PM4 handlers' own time, by packet class (the part of "outside" inside a handler; WATEK-GPU-PLAN M0):
	OutEvPre,   // draw / dispatch packets outside their event (prologue, epilogue, render mutex)
	OutSetReg,  // SET_*_REG (incl. indirect / index forms), CONTEXT_CONTROL, CLEAR_STATE, index/instance setup
	OutWait,    // WAIT_REG_MEM, CE/DE counter waits, MEM_SEMAPHORE, COND_EXEC, SET_PREDICATION
	OutRelease, // EVENT_WRITE, EVENT_WRITE_EOP/EOS, RELEASE_MEM, ACQUIRE_MEM, SURFACE_SYNC, PFP_SYNC_ME
	OutWrite,   // WRITE_DATA, DMA_DATA, CP_DMA, COPY_DATA, constant RAM writes/dumps
	OutOther,   // every other packet (NOP markers, INDIRECT_BUFFER, ...)
	Command,    // SendCommand lambdas (faults, readbacks, ...) run by the GPU thread
	// W
	Wait,       // SyncTimeline::WaitScope (fence/semaphore waits, readback waits)
	Calib,      // calibration only
	Count
};

[[nodiscard]] bool EnabledInit();
[[nodiscard]] inline bool Enabled() {
	// No trace is on, so the answer is false; every Zone / EventScope then costs one load (diagFastPath.h).
	if (Common::DiagFastPath::Skip()) [[likely]] {
		return false;
	}
	static constinit Common::OnceValue<bool> enabled_once;
	return enabled_once.Get([] { return EnabledInit(); });
}

// GuestGpu::ThreadRun: the calling thread is the GPU thread (the only one whose zones count).
void MarkGpuThread();
// Zone bodies (only call them with Enabled()). Push returns false when the zone is not counted (another thread, or
// too deeply nested); only then must Pop not be called. PushRange: a page-tracker zone over [address, address + size).
[[nodiscard]] bool Push(Sub sub);
[[nodiscard]] bool PushRange(Sub sub, uint64_t address, uint64_t size);
void               Pop();
// The innermost zone, when it is a TrkQuery, becomes a TrkMark (it found something to do).
void Reclassify();
[[nodiscard]] bool EventBegin();
void               EventEnd();

// The resolve thread's view of the registries when it made a ring slot (taken on the resolve thread).
struct ResolveStamp {
	uint64_t image_epoch  = 0; // DrawRepeat::g_image_registry_epoch
	uint64_t buffer_epoch = 0; // DrawRepeat::g_buffer_registry_epoch
	uint64_t cpu_sequence = 0; // CpuDirtyMarkSequence()
	uint32_t region_clock = 0; // g_cpu_dirty_region_clock (4 MiB tracker regions marked after it are newer)
	bool     valid        = false;
};
[[nodiscard]] ResolveStamp TakeResolveStamp();
// Execution thread, ProgramCache::Get: a ring slot made at `stamp` replaced a MaterializeResources call.
void NoteAdopted(const ResolveStamp& stamp);
// GPU thread, ProgramCache::Get (cached program): whether a result was made for the stage, had a slot with the same key
// (taken) and its result was used (adopted).
void NoteStage(bool made, bool taken, bool adopted);
// GPU thread, between slices: the STAGEF line every ~5 s.
void MaybeReport();

class Zone {
public:
	explicit Zone(Sub sub) {
		if (Enabled()) [[unlikely]] {
			m_on = Push(sub);
		}
	}
	Zone(Sub sub, uint64_t address, uint64_t size) {
		if (Enabled()) [[unlikely]] {
			m_on = PushRange(sub, address, size);
		}
	}
	~Zone() {
		if (m_on) Pop();
	}
	Zone(const Zone&)            = delete;
	Zone& operator=(const Zone&) = delete;

private:
	bool m_on = false;
};

// One per draw or dispatch call; nested calls count as their outermost call.
class EventScope {
public:
	EventScope() {
		if (Enabled()) [[unlikely]] {
			m_on = EventBegin();
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

} // namespace Libs::Graphics::StageF

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STAGEFTRACE_H_
