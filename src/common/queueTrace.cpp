#include "common/queueTrace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>

// KYTY_TRACE_QUEUES (queueTrace.h). Everything below runs on the GPU thread, except SharedBegin/SharedEnd and
// GpuWaitBegin/GpuWaitEnd, which other threads also call (they only touch thread-local depths and atomics there).
namespace Common::QueueTrace {

namespace {

constexpr uint32_t kCommandSlot = kQueueSlots;     // GPU-thread commands (SendCommand)
constexpr uint32_t kOtherSlot   = kQueueSlots + 1; // GPU-thread work outside slices and commands
constexpr uint32_t kSlots       = kQueueSlots + 2;
constexpr uint32_t kCpuReleaser = kSlots; // a wait released without a command-processor write since it blocked
constexpr uint32_t kReleasers   = kSlots + 1;
constexpr uint32_t kShared      = static_cast<uint32_t>(Shared::Count);
constexpr uint32_t kReasons     = static_cast<uint32_t>(SuspendReason::Count);

// Replay models (lanes = host threads): one lane per queue with a frame barrier, graphics + one compute lane with a
// frame barrier, one lane per queue without a barrier. Commands run on lane 0 (the graphics lane) in every model.
enum Model : uint32_t { kPerQueue, kTwo, kFree, kModels };

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

uint32_t Lane(uint32_t model, uint32_t slot) {
	if (slot >= kQueueSlots) return 0;
	if (model == kTwo) return slot == 0 ? 0u : 1u;
	return slot;
}

struct SlotStats {
	uint64_t subs       = 0;
	uint64_t slices     = 0;
	uint64_t resumed    = 0;
	uint64_t susp       = 0;
	uint64_t retry      = 0;
	uint64_t packets    = 0;
	uint64_t draws      = 0;
	uint64_t dispatches = 0;
	int64_t  busy_ns    = 0;
	int64_t  gpu_ns     = 0;
	int64_t  susp_ns[kReasons] {};
	uint64_t wrm_now    = 0;
	uint64_t wrm_late   = 0;
	int64_t  blocked_ns = 0;
	int64_t  lag_ns     = 0;
	int64_t  by_ns[kReleasers] {};
	uint32_t by_n[kReleasers] {};
	uint64_t xdep       = 0;
	uint64_t cexs       = 0;
	uint64_t shared_n[kShared] {};
	int64_t  shared_ns[kShared] {};
};

// Per queue, kept across windows.
struct SlotState {
	bool          waiting      = false; // a WAIT_REG_MEM of the queue failed and has not passed yet
	uint64_t      wait_addr    = 0;
	int64_t       wait_since   = 0;
	bool          suspended    = false; // the queue's last slice ended blocked
	SuspendReason reason       = SuspendReason::Label;
	int64_t       suspended_at = 0;
};

// An address some WAIT_REG_MEM tested, with its newest command-processor write.
struct WaitAddress {
	uint32_t size     = 4;
	uint32_t writer   = kCpuReleaser;
	int64_t  write_ns = 0;
	int64_t  write_sim[kModels] {};
};

struct State {
	std::array<SlotStats, kSlots>   stats {};
	std::array<SlotState, kSlots>   slots {};
	std::map<uint64_t, WaitAddress> waits;

	uint32_t      current      = kOtherSlot;
	bool          in_slice     = false;
	uint32_t      slice_slot   = 0;
	SliceKind     slice_kind   = SliceKind::Graphics;
	int64_t       slice_start  = 0;
	int64_t       slice_nested = 0; // commands run inside the slice
	uint64_t      slice_packets = 0;
	SuspendReason slice_reason = SuspendReason::Label;

	uint32_t cmd_depth = 0;
	int64_t  cmd_start = 0;
	uint32_t cmd_prev  = kOtherSlot;

	// Replay: each lane's clock, the current slice's start on its lane (plus its stalls), stalls in the window.
	int64_t clock[kModels][kQueueSlots] {};
	int64_t base[kModels] {};
	int64_t crit_start[kModels] {};
	int64_t stall_ns[kModels] {};

	int64_t  window_start    = 0;
	uint32_t presents_start  = 0;
	uint64_t frames          = 0;
	int64_t  idle_blocked_ns = 0;
	int64_t  idle_empty_ns   = 0;
};

State& S() {
	static State state;
	return state;
}

thread_local bool     t_gpu_thread            = false;
thread_local uint32_t t_gpu_wait_depth        = 0;
thread_local uint32_t t_shared_depth[kShared] = {};

std::atomic<uint64_t> g_other_shared_n[kShared] {};
std::atomic<int64_t>  g_other_shared_ns[kShared] {};

int64_t SliceElapsed(const State& s, int64_t now) {
	return now - s.slice_start - s.slice_nested;
}

int64_t MaxClock(const State& s, uint32_t model) {
	int64_t value = 0;
	for (uint32_t lane = 0; lane < kQueueSlots; lane++) value = std::max(value, s.clock[model][lane]);
	return value;
}

// The replay time of `now` on the current slot's lane.
int64_t SimNow(const State& s, uint32_t model, int64_t now) {
	if (s.in_slice && s.current == s.slice_slot) return s.base[model] + SliceElapsed(s, now);
	if (s.current == kCommandSlot && s.cmd_depth != 0) {
		const bool same_lane = s.in_slice && Lane(model, s.slice_slot) == 0;
		return (same_lane ? s.base[model] + SliceElapsed(s, s.cmd_start) : s.clock[model][0]) + (now - s.cmd_start);
	}
	return s.clock[model][Lane(model, s.current)];
}

void SlotName(uint32_t slot, char* out, size_t size) {
	if (slot == 0) {
		std::snprintf(out, size, "gfx");
	} else if (slot < kQueueSlots) {
		std::snprintf(out, size, "cq%02x", 0x20u + slot - 1u);
	} else if (slot == kCommandSlot) {
		std::snprintf(out, size, "cmd");
	} else if (slot == kOtherSlot) {
		std::snprintf(out, size, "other");
	} else {
		std::snprintf(out, size, "cpu");
	}
}

double Ms(int64_t ns) {
	return static_cast<double>(ns) / 1e6;
}

} // namespace

int64_t SharedScope::Now() {
	return NowNs();
}

void MarkGpuThread() {
	t_gpu_thread = true;
}

void SliceBegin(uint32_t queue_slot, SliceKind kind, bool resumed) {
	auto&      s   = S();
	const auto now = NowNs();
	if (queue_slot >= kQueueSlots) queue_slot = 0;
	auto& slot = s.slots[queue_slot];
	if (slot.suspended) {
		s.stats[queue_slot].susp_ns[static_cast<uint32_t>(slot.reason)] += now - slot.suspended_at;
		slot.suspended = false;
	}
	s.in_slice      = true;
	s.slice_slot    = queue_slot;
	s.slice_kind    = kind;
	s.slice_start   = now;
	s.slice_nested  = 0;
	s.slice_packets = 0;
	s.slice_reason  = SuspendReason::Label;
	s.current       = queue_slot;
	s.stats[queue_slot].slices++;
	if (resumed) s.stats[queue_slot].resumed++;
	if (kind == SliceKind::SuspendPoint) s.frames++;
	for (uint32_t m = 0; m < kModels; m++) s.base[m] = s.clock[m][Lane(m, queue_slot)];
}

void SliceEnd(bool complete) {
	auto& s = S();
	if (!s.in_slice) return;
	const auto now  = NowNs();
	const auto slot = s.slice_slot;
	const auto busy = SliceElapsed(s, now);
	auto&      st   = s.stats[slot];
	st.busy_ns += busy;
	if (complete) {
		st.subs++;
	} else {
		st.susp++;
		if (s.slice_packets == 0) st.retry++;
		auto& state        = s.slots[slot];
		state.suspended    = true;
		state.reason       = s.slice_reason;
		state.suspended_at = now;
	}
	for (uint32_t m = 0; m < kModels; m++) {
		s.clock[m][Lane(m, slot)] = s.base[m] + busy;
		if (s.slice_kind == SliceKind::SuspendPoint && m != kFree) {
			// Frame barrier: the suspend point drains every lane.
			const auto latest = MaxClock(s, m);
			for (uint32_t lane = 0; lane < kQueueSlots; lane++) s.clock[m][lane] = latest;
		}
	}
	s.in_slice = false;
	s.current  = kOtherSlot;
}

void CommandBegin() {
	auto& s = S();
	if (s.cmd_depth++ != 0) return;
	s.cmd_prev  = s.current;
	s.current   = kCommandSlot;
	s.cmd_start = NowNs();
}

void CommandEnd() {
	auto& s = S();
	if (s.cmd_depth == 0 || --s.cmd_depth != 0) return;
	const auto duration = NowNs() - s.cmd_start;
	auto&      st       = s.stats[kCommandSlot];
	st.busy_ns += duration;
	st.subs++;
	for (uint32_t m = 0; m < kModels; m++) {
		if (s.in_slice && Lane(m, s.slice_slot) == 0) {
			s.base[m] += duration; // the graphics lane runs the command inside its slice
		} else {
			s.clock[m][0] += duration;
		}
	}
	if (s.in_slice) s.slice_nested += duration;
	s.current = s.cmd_prev;
}

void NoteIdle(uint64_t ns, bool blocked) {
	auto& s = S();
	(blocked ? s.idle_blocked_ns : s.idle_empty_ns) += static_cast<int64_t>(ns);
}

void NotePacket() {
	auto& s = S();
	s.stats[s.current].packets++;
	s.slice_packets++;
}

void NoteDraw() {
	auto& s = S();
	s.stats[s.current].draws++;
}

void NoteDispatch() {
	auto& s = S();
	s.stats[s.current].dispatches++;
}

void NoteSuspend(SuspendReason reason) {
	S().slice_reason = reason;
}

void NoteWaitBlocked(uint64_t addr, uint32_t size) {
	auto& s = S();
	s.slice_reason = SuspendReason::Label;
	auto& slot     = s.slots[s.current];
	if (!slot.waiting || slot.wait_addr != addr) {
		slot.waiting    = true;
		slot.wait_addr  = addr;
		slot.wait_since = NowNs();
	}
	auto& entry = s.waits[addr];
	entry.size  = std::max(entry.size, size);
}

void NoteWaitPassed(uint64_t addr, uint32_t size) {
	auto&      s     = S();
	const auto now   = NowNs();
	auto&      entry = s.waits[addr];
	entry.size       = std::max(entry.size, size);
	auto& st         = s.stats[s.current];
	auto& slot       = s.slots[s.current];
	if (slot.waiting && slot.wait_addr == addr) {
		const auto blocked = now - slot.wait_since;
		const auto who     = entry.write_ns != 0 && entry.write_ns >= slot.wait_since ? entry.writer : kCpuReleaser;
		st.wrm_late++;
		st.blocked_ns += blocked;
		st.by_ns[who] += blocked;
		st.by_n[who]++;
		if (who != kCpuReleaser) st.lag_ns += now - entry.write_ns;
	} else {
		st.wrm_now++;
	}
	slot.waiting = false;
	if (entry.write_ns == 0 || entry.writer == s.current) return;
	st.xdep++;
	if (!s.in_slice || s.current != s.slice_slot) return;
	// Replay: the waiting lane cannot pass before the releasing write's time on the writer's lane.
	for (uint32_t m = 0; m < kModels; m++) {
		if (Lane(m, entry.writer) == Lane(m, s.current)) continue;
		const auto sim_now = s.base[m] + SliceElapsed(s, now);
		if (sim_now < entry.write_sim[m]) {
			const auto stall = entry.write_sim[m] - sim_now;
			s.base[m] += stall;
			s.stall_ns[m] += stall;
		}
	}
}

void NoteWrite(uint64_t addr, uint64_t size) {
	auto& s = S();
	if (s.waits.empty() || size == 0) return;
	const uint64_t end = addr + size;
	int64_t        now = 0;
	for (auto it = s.waits.lower_bound(addr > 7u ? addr - 7u : 0u); it != s.waits.end() && it->first < end; ++it) {
		if (it->first + it->second.size <= addr) continue;
		if (now == 0) now = NowNs();
		auto& entry    = it->second;
		entry.writer   = s.current;
		entry.write_ns = now;
		for (uint32_t m = 0; m < kModels; m++) entry.write_sim[m] = SimNow(s, m, now);
	}
}

bool GpuWaitBegin() {
	if (!t_gpu_thread) return false;
	t_gpu_wait_depth++;
	return true;
}

void GpuWaitEnd(int64_t ns) {
	if (t_gpu_wait_depth == 0 || --t_gpu_wait_depth != 0) return;
	auto& s = S();
	s.stats[s.current].gpu_ns += ns;
}

void SharedBegin(Shared kind) {
	t_shared_depth[static_cast<uint32_t>(kind)]++;
}

void SharedEnd(Shared kind, int64_t ns) {
	const auto index = static_cast<uint32_t>(kind);
	if (t_shared_depth[index] == 0 || --t_shared_depth[index] != 0) return;
	if (!t_gpu_thread) {
		g_other_shared_n[index].fetch_add(1, std::memory_order_relaxed);
		g_other_shared_ns[index].fetch_add(ns, std::memory_order_relaxed);
		return;
	}
	auto& s = S();
	s.stats[s.current].shared_n[index]++;
	s.stats[s.current].shared_ns[index] += ns;
}

void Report(uint32_t presents) {
	auto&      s   = S();
	const auto now = NowNs();
	if (s.window_start == 0) {
		s.window_start   = now;
		s.presents_start = presents;
		for (uint32_t m = 0; m < kModels; m++) s.crit_start[m] = MaxClock(s, m);
		return;
	}
	if (now - s.window_start < 5'000'000'000ll) return;
	const auto window = now - s.window_start;

	int64_t serial = 0;
	for (uint32_t slot = 0; slot < kSlots; slot++) serial += s.stats[slot].busy_ns;
	const auto frames   = s.frames;
	const auto presents_window = presents - s.presents_start;
	std::fprintf(stderr,
	             "QUEUES window_ms=%.0f frames=%llu presents=%u busy=%.1f cmd=%.1f idle_blocked=%.1f idle_empty=%.1f "
	             "wait_addrs=%zu\n",
	             Ms(window), static_cast<unsigned long long>(frames), presents_window, Ms(serial),
	             Ms(s.stats[kCommandSlot].busy_ns), Ms(s.idle_blocked_ns), Ms(s.idle_empty_ns), s.waits.size());

	static const char* shared_names[kShared] = {"tex", "buf", "page", "pipe"};
	for (uint32_t slot = 0; slot < kSlots; slot++) {
		const auto& st = s.stats[slot];
		if (st.slices == 0 && st.busy_ns == 0 && st.subs == 0 && st.gpu_ns == 0 && st.shared_n[0] == 0 &&
		    st.shared_n[1] == 0 && st.shared_n[2] == 0 && st.shared_n[3] == 0) {
			continue;
		}
		char name[16];
		SlotName(slot, name, sizeof(name));
		std::fprintf(stderr,
		             "QUEUES q=%s subs=%llu slices=%llu res=%llu susp=%llu retry=%llu pk=%llu draws=%llu disp=%llu "
		             "busy=%.1f gpu=%.1f susp_ms=%.1f/%.1f/%.1f/%.1f wrm=%llu/%llu blocked=%.1f lag=%.1f",
		             name, static_cast<unsigned long long>(st.subs), static_cast<unsigned long long>(st.slices),
		             static_cast<unsigned long long>(st.resumed), static_cast<unsigned long long>(st.susp), static_cast<unsigned long long>(st.retry),
		             static_cast<unsigned long long>(st.packets), static_cast<unsigned long long>(st.draws),
		             static_cast<unsigned long long>(st.dispatches), Ms(st.busy_ns), Ms(st.gpu_ns), Ms(st.susp_ns[0]),
		             Ms(st.susp_ns[1]), Ms(st.susp_ns[2]), Ms(st.susp_ns[3]), static_cast<unsigned long long>(st.wrm_now),
		             static_cast<unsigned long long>(st.wrm_late), Ms(st.blocked_ns), Ms(st.lag_ns));
		// The releasers of the blocked waits, by time (at most 4).
		std::array<uint32_t, kReleasers> order {};
		for (uint32_t i = 0; i < kReleasers; i++) order[i] = i;
		std::partial_sort(order.begin(), order.begin() + 4, order.end(),
		                  [&](uint32_t a, uint32_t b) { return st.by_ns[a] > st.by_ns[b]; });
		std::fprintf(stderr, " by=[");
		for (uint32_t i = 0; i < 4; i++) {
			const auto who = order[i];
			if (st.by_n[who] == 0) break;
			char releaser[16];
			SlotName(who, releaser, sizeof(releaser));
			std::fprintf(stderr, "%s%s:%.1f/%u", i == 0 ? "" : " ", releaser, Ms(st.by_ns[who]), st.by_n[who]);
		}
		std::fprintf(stderr, "] xdep=%llu cexs=%llu shared", static_cast<unsigned long long>(st.xdep),
		             static_cast<unsigned long long>(st.cexs));
		for (uint32_t k = 0; k < kShared; k++) {
			std::fprintf(stderr, " %s=%llu/%.1f", shared_names[k], static_cast<unsigned long long>(st.shared_n[k]),
			             Ms(st.shared_ns[k]));
		}
		std::fprintf(stderr, "\n");
	}
	std::fprintf(stderr, "QUEUES shared_other_threads");
	for (uint32_t k = 0; k < kShared; k++) {
		std::fprintf(stderr, " %s=%llu/%.1f", shared_names[k],
		             static_cast<unsigned long long>(g_other_shared_n[k].exchange(0, std::memory_order_relaxed)),
		             Ms(g_other_shared_ns[k].exchange(0, std::memory_order_relaxed)));
	}
	std::fprintf(stderr, "\n");

	int64_t crit[kModels] {};
	for (uint32_t m = 0; m < kModels; m++) {
		const auto latest = MaxClock(s, m);
		crit[m]           = latest - s.crit_start[m];
		s.crit_start[m]   = latest;
	}
	const auto ratio = [&](int64_t value) { return value > 0 ? static_cast<double>(serial) / static_cast<double>(value) : 0.0; };
	const auto per_frame = [&](int64_t value) { return frames != 0 ? Ms(value) / static_cast<double>(frames) : 0.0; };
	std::fprintf(stderr,
	             "QUEUES ceiling serial=%.1f perq=%.1f perq_x=%.2f two=%.1f two_x=%.2f free=%.1f free_x=%.2f "
	             "stall_perq=%.1f stall_two=%.1f stall_free=%.1f | per_frame serial=%.2f perq=%.2f two=%.2f free=%.2f "
	             "gfx_busy=%.2f\n",
	             Ms(serial), Ms(crit[kPerQueue]), ratio(crit[kPerQueue]), Ms(crit[kTwo]), ratio(crit[kTwo]),
	             Ms(crit[kFree]), ratio(crit[kFree]), Ms(s.stall_ns[kPerQueue]), Ms(s.stall_ns[kTwo]),
	             Ms(s.stall_ns[kFree]), per_frame(serial), per_frame(crit[kPerQueue]), per_frame(crit[kTwo]),
	             per_frame(crit[kFree]), per_frame(s.stats[0].busy_ns + s.stats[kCommandSlot].busy_ns));

	for (auto& st: s.stats) st = {};
	for (auto& stall: s.stall_ns) stall = 0;
	s.window_start    = now;
	s.presents_start  = presents;
	s.frames          = 0;
	s.idle_blocked_ns = 0;
	s.idle_empty_ns   = 0;
}

} // namespace Common::QueueTrace
