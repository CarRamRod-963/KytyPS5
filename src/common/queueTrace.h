#ifndef KYTY_COMMON_QUEUETRACE_H_
#define KYTY_COMMON_QUEUETRACE_H_

#include "common/diagFastPath.h"
#include "common/onceValue.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// KYTY_TRACE_QUEUES=1 (diagnostic, default off): how the single GPU thread spends its time on each guest queue (the
// graphics ring and every async compute ring), how the queues wait for each other, and what one host thread per queue
// could gain. When the switch is off every hook is one load of a constant-initialized flag and a not-taken branch.
//
// Every ~5 s the GPU thread prints (times in ms within the window):
//   QUEUES window_ms frames presents busy cmd idle_blocked idle_empty wait_addrs
//       frames = SuspendPoints processed; presents = frames shown; busy = all GPU-thread work (queues + commands);
//       cmd = GPU-thread commands (readbacks, uploads, ... sent by other threads); idle_blocked = slept with every
//       queue blocked; idle_empty = slept with no work at all; wait_addrs = label addresses seen in WAIT_REG_MEM.
//   QUEUES q=<gfx|cqNN|cmd|other> one line per queue (cqNN = compute queue id NN) that did something:
//       subs (submissions finished; cmd: commands run) slices (Process calls) res (slices that resumed a started
//       submission) susp (slices that ended blocked) retry (blocked slices without a single packet) pk (type-3
//       packets) draws disp (host draws / dispatches) busy (GPU-thread wall time on the queue, nested commands
//       excluded) gpu (its waits for the host GPU, part of busy) susp_ms=label/cede/rewind/pred (time between a
//       blocked slice and the next slice of the queue, by the reason of the block) wrm=passed_at_once/
//       passed_after_block blocked (WAIT_REG_MEM: first failed test to pass) lag (from the label write that released
//       the wait to the pass: the queue was runnable but the thread was elsewhere) by=[releaser:ms/n ...] (who wrote
//       the label: another queue, the queue itself, or cpu = no command-processor write since the block)
//       xdep (waits that passed on a label another queue wrote) cexs (always 0, kept for the line's format) shared
//       tex/buf/page/pipe=n/ms (outermost calls into the texture cache (FindImage, InvalidateMemoryFromGPU), buffer cache (ObtainBuffer), page tracking (MemoryTracker::Iterate) and
//       pipeline cache (ProgramCache::Get, Get*Pipeline) made for the queue, with their time; kinds overlap: a buffer
//       lookup's page-tracking calls count in both).
//   QUEUES shared_other_threads tex/buf/page/pipe=n/ms (the same calls from other threads: faults, resolve thread, ...)
//   QUEUES ceiling serial perq perq_x two two_x free free_x stall_perq stall_two stall_free | per_frame serial perq
//       two free gfx_busy: a replay of the window with the same per-slice costs on several lanes (host threads):
//       perq = one lane per queue, two = graphics lane + one lane for all compute rings, free = perq without the
//       frame barrier. Commands run on the graphics lane. A wait that passed on a label written by another lane
//       cannot pass before that write's time on its lane (the replay delays the waiting lane: stall). perq and two
//       drain every lane at each SuspendPoint (frame barrier). The critical path is the advance of the latest lane;
//       x = serial / that. Waits released by the guest CPU add no delay (optimistic); everything a second thread
//       would add (locks, per-queue submissions, cross-queue semaphores) is not in the replay.
namespace Common::QueueTrace {

inline bool Enabled() {
	// No trace is on, so the answer is false (diagFastPath.h).
	if (Common::DiagFastPath::Skip()) [[likely]] {
		return false;
	}
	static constinit Common::OnceValue<bool> enabled_once;
	const bool enabled = enabled_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_QUEUES");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) std::fprintf(stderr, "TRACE_QUEUES enabled\n");
		return on;
	});
	return enabled;
}

// Slots: 0 = graphics ring, 1..56 = compute queue 0x20 + slot - 1 (GuestGpu's queue ids).
constexpr uint32_t kQueueSlots = 57;

enum class SliceKind : uint32_t { Graphics, Compute, FlipPreparation, SuspendPoint };
enum class SuspendReason : uint32_t { Label, CeDe, Rewind, Predication, Count };
enum class Shared : uint32_t { TextureCache, BufferCache, PageTracking, PipelineCache, Count };

// GuestGpu::ThreadRun: the calling thread is the GPU thread (the other hooks attribute only its work to queues).
void MarkGpuThread();

// GuestGpu::ThreadRun (GPU thread): one slice = one GuestGpu::Process call; complete = the submission finished.
void SliceBegin(uint32_t queue_slot, SliceKind kind, bool resumed);
void SliceEnd(bool complete);
// A GPU-thread command (GuestGpu::SendCommand), run between slices or inside one (ProcessCommands).
void CommandBegin();
void CommandEnd();
// The GPU thread slept: every queue blocked (blocked) or no work at all.
void NoteIdle(uint64_t ns, bool blocked);

// Command processor (GPU thread).
void NotePacket();
void NoteDraw();
void NoteDispatch();
void NoteWaitBlocked(uint64_t addr, uint32_t size);
void NoteWaitPassed(uint64_t addr, uint32_t size);
void NoteSuspend(SuspendReason reason);
// A command-processor write of guest memory (labels, WRITE_DATA, DMA/COPY_DATA, flips).
void NoteWrite(uint64_t addr, uint64_t size);

// The GPU thread waits for the host GPU (SyncTimeline::WaitScope sites); nested waits count once.
[[nodiscard]] bool GpuWaitBegin();
void               GpuWaitEnd(int64_t ns);

// Calls into structures a second recording thread would share. The outermost scope of each kind counts.
void SharedBegin(Shared kind);
void SharedEnd(Shared kind, int64_t ns);

class SharedScope {
public:
	explicit SharedScope(Shared kind): m_kind(kind) {
		if (!Enabled()) return;
		m_on = true;
		SharedBegin(kind);
		m_start = Now();
	}
	~SharedScope() {
		if (m_on) SharedEnd(m_kind, Now() - m_start);
	}
	SharedScope(const SharedScope&)            = delete;
	SharedScope& operator=(const SharedScope&) = delete;

private:
	static int64_t Now();

	Shared  m_kind;
	bool    m_on    = false;
	int64_t m_start = 0;
};

// The GPU thread's ~5 s QUEUES lines (presents: DebugPresentCount, for the window's present count).
void Report(uint32_t presents);

} // namespace Common::QueueTrace

#endif
