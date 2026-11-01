#include "graphics/host_gpu/renderer/stageFTrace.h"

#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

// KYTY_TRACE_STAGEF (see stageFTrace.h). One line per ~5 s window, printed by the GPU thread. Every time is in ms PER
// PRESENTED FRAME of the window (frames = presents in the window; with frames=0 the values are per window), after the
// calibrated cost of the instrumentation itself is taken out of every zone:
//  STAGEF window_ms frames fps
//   busy (GPU-thread time in zones = all its work) idle (the rest of the wall time: no work, or the trace's own report)
//   self (estimated instrumentation cost still inside busy's wall time; busy itself is corrected) zones (zones/frame)
//   cal=full/inner (ns per empty zone: as seen by the enclosing zone / as reported by the zone itself)
//   L (lookups that could run ahead: the L: group) L_ev (the part of L made for draws/dispatches, i.e. inside their
//   call and not inside a SendCommand lambda: what resolver threads would take over; the F2 decision value)
//   L_cov (the part of L_ev whose inputs the resolve thread had: every program stage of the event adopted from the ring; L_cov
//   never contains rt and pipe, see cov) C (ordered side effects: the C: group) W (waits for the host GPU)
//   cmd (SendCommand lambdas run by the GPU thread, inclusive of the waits and zones inside them) cmd_n (lambdas/frame)
//   dropped (zones nested deeper than 64: not counted; should be 0)
//  L:[name=ms/calls ...] C:[...] W:[...]: each sub-phase (stageFTrace.h, enum Sub) with its own time and its zones
//   per frame (calls are per frame too). Names: dec img rt buf samp prog srt pipe trkq | adopt compile pipe_new create
//   texsync obtain trkm protect staging record submit ev_other outside cmd | wait. trkq holds only page-tracker
//   queries made for a draw/dispatch that found nothing to do; tracker queries made by the adoption's re-check count
//   in adopt, every other tracker call (faults, CP writes, commands, marks, dirty-run walks) in trkm.
//  cov:[...]: lookups made inside a draw/dispatch, by kind (dec, samp, img = FindImage outside a render-target
//   resolution, buf = FindBuffer, trk = page-tracker queries, prog = program stages), the share whose inputs were in
//   the resolve thread's ring: an event counts as covered when every one of its program stages so far was adopted from the
//   ring (so its T#/S#/V# words are the resolve thread's). prog: stages adopted (slot: a slot with the same key existed).
//   rt (render-target FindImage) and pipe (pipeline lookups) are 0 by construction: the resolve thread shadows only the SH
//   registers, not the context registers they need. events: draws/dispatches with every stage adopted.
//   spec: program stages for which a result was made.
//  stale:[...]: of the covered lookups, the share whose registry moved between the resolve thread's slot and the GPU thread's
//   use: img (texture registry epoch), buf (buffer registry epoch), trk_seq (CpuDirtyMarkSequence: anything anywhere
//   became CPU-dirty; upper bound), trk_rgn (a 4 MiB tracker region of the queried range became CPU-dirty), prog
//   (slots with the same key whose reads had changed: rejected at adoption); reg = img and buf together (the F2
//   decision value). dec and samp are pure functions of the words and cannot be stale.
//  n/frame:[...]: lookups per frame made inside draws/dispatches (img, buf, trk, prog = program stages) and events.
//  verdict: f2 = GO when L_ev >= 35 ms/frame and reg <= 10% (DESIGN.md section 3, F0; UNKNOWN when no registry lookup
//   was covered); b_d3_before_f4 = YES when cmd >= 10 ms/frame (stage B D3 before F4).

namespace Libs::Graphics::StageF {

namespace {

constexpr size_t kSubCount = static_cast<size_t>(Sub::Count);

constexpr const char* kSubNames[kSubCount] = {"dec",     "img",     "rt",       "buf",     "samp",    "prog",
                                              "srt",     "pipe",    "trkq",     "adopt",   "compile", "pipe_new",
                                              "create",  "texsync", "obtain",   "trkm",    "protect", "staging",
                                              "record",  "rec_scan", "rec_desc", "rec_transit", "rec_emit", "rec_vtx", "rec_dyn", "rec_begin", "rec_issue", "rec_post", "rec_cpre", "rec_compute", "rec_fill", "rec_ring", "ev_bindless", "ev_imgloop", "ev_bindimg", "ev_disppre", "ev_drawpre", "ev_progs", "ev_targets", "ev_rebindbuf", "ev_bufimgsync", "obt_pre", "obt_sync", "obt_write", "obtw_dirty", "obtw_tick", "obtw_gran", "obtw_obs", "submit",  "ev_other", "outside", "out_evpre", "out_setreg", "out_wait", "out_release", "out_write", "out_other", "cmd",     "wait",
                                              "calib"};

constexpr size_t Index(Sub sub) {
	return static_cast<size_t>(sub);
}

constexpr bool IsLookup(size_t index) {
	return index <= Index(Sub::TrkQuery);
}

constexpr uint32_t kMaxDepth         = 64;
constexpr uint64_t kCalibrationZones = 16384;

uint64_t Ticks() {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

// Lookup kinds for coverage and staleness.
enum Kind : uint32_t { KDec, KSamp, KImg, KBuf, KTrk, KProg, KRt, KPipe, KCount };

constexpr const char* kKindNames[KCount] = {"dec", "samp", "img", "buf", "trk", "prog", "rt", "pipe"};

struct Cover {
	uint64_t n            = 0; // lookups inside a draw/dispatch
	uint64_t covered      = 0; // ... whose inputs the resolve thread had
	uint64_t stale        = 0; // ... covered, but the registry moved since the resolve thread's slot (trk: CpuDirtyMarkSequence)
	uint64_t stale_region = 0; // trk: a 4 MiB region of the range was marked CPU-dirty since the resolve thread's slot
	uint64_t taken        = 0; // prog: a ring slot with the same key existed
};

// What NoteLookup counted for a zone (undone when a tracker query turns out to be a mark), and FInEvent: the zone runs
// for a draw/dispatch (inside its event, not inside a SendCommand lambda).
enum LookupFlag : uint8_t { FCounted = 1, FCovered = 2, FStale = 4, FStaleRegion = 8, FInEvent = 16 };

struct Frame {
	uint64_t start = 0;
	uint64_t child = 0; // inclusive time of the zones directly inside
	Sub      sub   = Sub::Outside;
	uint8_t  flags = 0;
};

struct State {
	std::array<Frame, kMaxDepth> stack {};
	uint32_t                     depth         = 0;
	uint32_t                     target_depth  = 0;
	uint32_t                     command_depth = 0;
	uint32_t                     adopt_depth   = 0;
	// Window counters.
	std::array<uint64_t, kSubCount> ticks {};         // own time
	std::array<uint64_t, kSubCount> calls {};         // zones
	std::array<uint64_t, kSubCount> child_calls {};   // zones directly inside
	std::array<uint64_t, kSubCount> covered_ticks {}; // own time of zones whose lookup was covered
	std::array<uint64_t, kSubCount> event_ticks {};   // own time of zones made for a draw/dispatch (FInEvent)
	uint64_t                        command_ticks  = 0; // inclusive time of outermost SendCommand lambdas
	uint64_t                        command_calls  = 0;
	uint64_t                        command_nested = 0; // zones inside them
	uint64_t                        dropped        = 0; // zones deeper than kMaxDepth
	uint64_t                        events = 0, events_covered = 0;
	uint64_t                        stages = 0, stages_speculated = 0;
	std::array<Cover, KCount>       cover {};
	// The current draw/dispatch.
	uint32_t   event_depth = 0;
	uint32_t   ev_stages   = 0; // ProgramCache::Get calls in the event
	uint32_t   ev_adopted  = 0; // ... whose MaterializeResources was the resolve thread's
	ResolveStamp ev_stamp {};     // the oldest adopted slot's view (field by field)
	// Calibration (ticks per zone): full = what one empty zone costs the zone around it, inner = what it reports for
	// itself.
	double                                full  = 0.0;
	double                                inner = 0.0;
	std::chrono::steady_clock::time_point window_start {};
	uint64_t                              window_ticks = 0;
	uint32_t                              window_frame = 0;
};

thread_local State* t_state = nullptr;

bool EventCovered(const State& s) {
	return s.event_depth != 0 && s.ev_stages != 0 && s.ev_adopted == s.ev_stages && s.ev_stamp.valid;
}

bool RegionMarkedSince(uint64_t address, uint64_t size, uint32_t clock) {
	auto* epochs = g_cpu_dirty_region_epochs.load(std::memory_order_acquire);
	if (epochs == nullptr || size == 0 || size > UINT64_MAX - address) {
		return false;
	}
	const uint64_t first = address / TRACKER_REGION_SIZE;
	const uint64_t last  = (address + size - 1) / TRACKER_REGION_SIZE;
	for (uint64_t region = first; region <= last && region < CPU_DIRTY_REGION_COUNT && region - first < 64u; ++region) {
		if (static_cast<int32_t>(epochs[region].load(std::memory_order_relaxed) - clock) > 0) {
			return true;
		}
	}
	return false;
}

// Coverage and staleness of a lookup zone that starts now (GPU thread).
uint8_t NoteLookup(State& s, Sub sub, uint64_t address, uint64_t size) {
	if (s.event_depth == 0) {
		return 0;
	}
	Kind kind = KCount;
	switch (sub) {
		case Sub::Decode: kind = KDec; break;
		case Sub::Sampler: kind = KSamp; break;
		case Sub::FindImage: kind = s.target_depth != 0 ? KRt : KImg; break;
		case Sub::FindBuffer: kind = KBuf; break;
		case Sub::TrkQuery: kind = KTrk; break;
		case Sub::PipeFind: kind = KPipe; break;
		case Sub::ProgFind:
			// Counted here, judged by NoteStage.
			s.ev_stages++;
			s.cover[KProg].n++;
			return 0;
		default: return 0;
	}
	auto& c = s.cover[kind];
	c.n++;
	uint8_t flags = FCounted;
	// Render targets and pipelines need the context registers, which the resolve thread does not shadow.
	if (kind == KRt || kind == KPipe || !EventCovered(s)) {
		return flags;
	}
	c.covered++;
	flags |= FCovered;
	bool stale        = false;
	bool stale_region = false;
	switch (kind) {
		case KImg:
			stale = DrawRepeat::g_image_registry_epoch.load(std::memory_order_relaxed) != s.ev_stamp.image_epoch;
			break;
		case KBuf:
			stale = DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_relaxed) != s.ev_stamp.buffer_epoch;
			break;
		case KTrk:
			stale        = CpuDirtyMarkSequence() != s.ev_stamp.cpu_sequence;
			stale_region = RegionMarkedSince(address, size, s.ev_stamp.region_clock);
			break;
		default: break;
	}
	if (stale) {
		c.stale++;
		flags |= FStale;
	}
	if (stale_region) {
		c.stale_region++;
		flags |= FStaleRegion;
	}
	return flags;
}

bool PushImpl(Sub sub, uint64_t address, uint64_t size) {
	State* const s = t_state;
	if (s == nullptr) {
		return false;
	}
	if (s->depth >= kMaxDepth) {
		s->dropped++;
		return false;
	}
	const bool in_event = s->event_depth != 0 && s->command_depth == 0;
	if (sub == Sub::TrkQuery) {
		if (s->adopt_depth != 0) {
			sub = Sub::Adopt; // the adoption re-reading the resolve thread's log: validation
		} else if (!in_event) {
			sub = Sub::TrkMark; // faults, CP writes, commands: not a draw's lookup
		}
	}
	uint8_t flags = NoteLookup(*s, sub, address, size);
	if (in_event) flags |= FInEvent;
	if (sub == Sub::Target) s->target_depth++;
	if (sub == Sub::Command) s->command_depth++;
	if (sub == Sub::Adopt) s->adopt_depth++;
	Frame& frame = s->stack[s->depth++];
	frame.sub    = sub;
	frame.child  = 0;
	frame.flags  = flags;
	frame.start  = Ticks();
	return true;
}

// At depth 0 only (no enclosing zone).
void Calibrate(State& s) {
	using PushFn = bool (*)(Sub);
	using PopFn  = void (*)();
	PushFn volatile push = &Push; // calls, as at the hooks
	PopFn volatile  pop  = &Pop;
	const auto      run  = [&](uint64_t count) {
		for (uint64_t i = 0; i < count; i++) {
			if (push(Sub::Calib)) pop();
		}
	};
	run(256);
	const size_t calib = Index(Sub::Calib);
	s.ticks[calib]     = 0;
	const auto begin   = Ticks();
	run(kCalibrationZones);
	const auto end = Ticks();
	s.full         = static_cast<double>(end - begin) / static_cast<double>(kCalibrationZones);
	s.inner        = static_cast<double>(s.ticks[calib]) / static_cast<double>(kCalibrationZones);
	s.ticks[calib] = 0;
	s.calls[calib] = 0;
}

void StartWindow(State& s) {
	s.ticks             = {};
	s.calls             = {};
	s.child_calls       = {};
	s.covered_ticks     = {};
	s.event_ticks       = {};
	s.command_ticks     = 0;
	s.command_calls     = 0;
	s.command_nested    = 0;
	s.dropped           = 0;
	s.events            = 0;
	s.events_covered    = 0;
	s.stages            = 0;
	s.stages_speculated = 0;
	s.cover             = {};
	s.window_start      = std::chrono::steady_clock::now();
	s.window_ticks      = Ticks();
	s.window_frame      = DebugPresentCount().load(std::memory_order_relaxed);
}

double Percent(uint64_t part, uint64_t whole) {
	return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

void Report(State& s, std::chrono::steady_clock::time_point now) {
	const double window_ms = std::chrono::duration<double, std::milli>(now - s.window_start).count();
	const auto   elapsed   = Ticks() - s.window_ticks;
	const double per_ms    = window_ms > 0.0 && elapsed != 0 ? static_cast<double>(elapsed) / window_ms : 1.0;
	const auto   frame     = DebugPresentCount().load(std::memory_order_relaxed);
	const auto   frames    = frame - s.window_frame;
	const double divisor   = per_ms * static_cast<double>(std::max<uint32_t>(frames, 1u));
	const double count_div = static_cast<double>(std::max<uint32_t>(frames, 1u));
	const double outer     = std::max(s.full - s.inner, 0.0);

	std::array<double, kSubCount> ms {};
	double                        raw_busy = 0.0;
	double                        zones    = 0.0;
	double                        l_ms = 0.0, l_ev_ms = 0.0, l_cov_ms = 0.0, c_ms = 0.0, w_ms = 0.0;
	for (size_t i = 0; i < kSubCount; i++) {
		if (i == Index(Sub::Calib)) continue;
		const double raw = static_cast<double>(s.ticks[i]);
		const double own = std::max(raw - static_cast<double>(s.calls[i]) * s.inner -
		                                static_cast<double>(s.child_calls[i]) * outer,
		                            0.0);
		ms[i] = own / divisor;
		raw_busy += raw;
		zones += static_cast<double>(s.calls[i]);
		if (IsLookup(i)) {
			l_ms += ms[i];
			if (raw > 0.0) {
				l_ev_ms += ms[i] * static_cast<double>(s.event_ticks[i]) / raw;
				l_cov_ms += ms[i] * static_cast<double>(s.covered_ticks[i]) / raw;
			}
		} else if (i == Index(Sub::Wait)) {
			w_ms += ms[i];
		} else {
			c_ms += ms[i];
		}
	}
	const double cmd_ms = std::max(static_cast<double>(s.command_ticks) - static_cast<double>(s.command_calls) * s.inner -
	                                   static_cast<double>(s.command_nested) * s.full,
	                               0.0) /
	                      divisor;
	const double busy   = l_ms + c_ms + w_ms;
	const double idle   = std::max(window_ms - raw_busy / per_ms, 0.0) / count_div;
	const double self   = zones * s.full / divisor;

	std::string line;
	char        text[512];
	std::snprintf(text, sizeof(text),
	              "STAGEF window_ms=%.0f frames=%u fps=%.2f busy=%.1f idle=%.1f self=%.1f zones=%.0f cal=%.1f/%.1fns "
	              "L=%.1f L_ev=%.1f L_cov=%.1f C=%.1f W=%.1f cmd=%.1f cmd_n=%.1f dropped=%llu |",
	              window_ms, frames, window_ms > 0.0 ? 1000.0 * frames / window_ms : 0.0, busy, idle, self,
	              zones / count_div, s.full / per_ms * 1e6, s.inner / per_ms * 1e6, l_ms, l_ev_ms, l_cov_ms, c_ms, w_ms, cmd_ms,
	              static_cast<double>(s.command_calls) / count_div, static_cast<unsigned long long>(s.dropped));
	line += text;
	const auto group = [&](const char* name, size_t first, size_t last) {
		line += ' ';
		line += name;
		line += ":[";
		for (size_t i = first; i <= last; i++) {
			std::snprintf(text, sizeof(text), "%s%s=%.2f/%.0f", i == first ? "" : " ", kSubNames[i], ms[i],
			              static_cast<double>(s.calls[i]) / count_div);
			line += text;
		}
		line += ']';
	};
	group("L", Index(Sub::Decode), Index(Sub::TrkQuery));
	group("C", Index(Sub::Adopt), Index(Sub::Command));
	group("W", Index(Sub::Wait), Index(Sub::Wait));

	const auto& prog = s.cover[KProg];
	std::snprintf(text, sizeof(text), " | cov:[events=%.1f%% spec=%.1f%% prog=%.1f%%(slot=%.1f%%)",
	              Percent(s.events_covered, s.events), Percent(s.stages_speculated, s.stages), Percent(prog.covered, prog.n),
	              Percent(prog.taken, prog.n));
	line += text;
	for (const Kind kind: {KDec, KSamp, KImg, KBuf, KTrk, KRt, KPipe}) {
		std::snprintf(text, sizeof(text), " %s=%.1f%%", kKindNames[kind], Percent(s.cover[kind].covered, s.cover[kind].n));
		line += text;
	}
	const auto&    img       = s.cover[KImg];
	const auto&    buf       = s.cover[KBuf];
	const auto&    trk       = s.cover[KTrk];
	const uint64_t reg_den   = img.covered + buf.covered;
	const double   reg_stale = Percent(img.stale + buf.stale, reg_den);
	std::snprintf(text, sizeof(text),
	              "] stale:[img=%.1f%% buf=%.1f%% trk_seq=%.1f%% trk_rgn=%.1f%% prog=%.1f%% reg=%.1f%%] n/frame:[img=%.0f "
	              "buf=%.0f trk=%.0f prog=%.0f events=%.0f]",
	              Percent(img.stale, img.covered), Percent(buf.stale, buf.covered), Percent(trk.stale, trk.covered),
	              Percent(trk.stale_region, trk.covered), Percent(prog.stale, prog.taken), reg_stale,
	              static_cast<double>(img.n) / count_div, static_cast<double>(buf.n) / count_div,
	              static_cast<double>(trk.n) / count_div, static_cast<double>(prog.n) / count_div,
	              static_cast<double>(s.events) / count_div);
	line += text;

	// The decision uses the lookups made for draws and dispatches (L_ev): those are what resolvers would take over.
	const bool  l_ok = l_ev_ms >= 35.0;
	const char* f2   = !l_ok ? "NO" : (reg_den == 0 ? "UNKNOWN(no covered registry lookups)" : (reg_stale <= 10.0 ? "GO" : "NO"));
	std::snprintf(text, sizeof(text), " | verdict f2=%s (L_ev=%.1f%s35, reg_stale=%.1f%%%s10%%) b_d3_before_f4=%s (cmd=%.1f%s10)",
	              f2, l_ev_ms, l_ok ? ">=" : "<", reg_stale, reg_den == 0 ? "?" : (reg_stale <= 10.0 ? "<=" : ">"),
	              cmd_ms >= 10.0 ? "YES" : "no", cmd_ms, cmd_ms >= 10.0 ? ">=" : "<");
	line += text;
	std::fprintf(stderr, "%s\n", line.c_str());
}

} // namespace

bool EnabledInit() {
	const char* value = std::getenv("KYTY_TRACE_STAGEF");
	const bool  on    = value != nullptr && value[0] == '1';
	if (on) {
		// One epoch per 4 MiB tracker region (the BufferCache may have made it already): every CPU-dirty mark sets it to the
		// next g_cpu_dirty_region_clock value. 16 MiB.
		auto*                  epochs   = new std::atomic<uint32_t>[CPU_DIRTY_REGION_COUNT] {};
		std::atomic<uint32_t>* expected = nullptr;
		if (!g_cpu_dirty_region_epochs.compare_exchange_strong(expected, epochs, std::memory_order_acq_rel)) {
			delete[] epochs;
		}
		std::fprintf(stderr, "TRACE_STAGEF enabled\n");
	}
	return on;
}

void MarkGpuThread() {
	if (t_state != nullptr) {
		return;
	}
	auto* state = new State {}; // kept for the life of the thread
	t_state     = state;
	Calibrate(*state);
}

bool Push(Sub sub) {
	return PushImpl(sub, 0, 0);
}

bool PushRange(Sub sub, uint64_t address, uint64_t size) {
	return PushImpl(sub, address, size);
}

void Pop() {
	const uint64_t now = Ticks();
	State* const   s   = t_state;
	if (s == nullptr || s->depth == 0) {
		return;
	}
	Frame&         frame = s->stack[--s->depth];
	const uint64_t total = now - frame.start;
	const uint64_t own   = total > frame.child ? total - frame.child : 0;
	const size_t   index = Index(frame.sub);
	s->ticks[index] += own;
	s->calls[index]++;
	if ((frame.flags & FCovered) != 0) s->covered_ticks[index] += own;
	if ((frame.flags & FInEvent) != 0) s->event_ticks[index] += own;
	if (frame.sub == Sub::Target) s->target_depth--;
	if (frame.sub == Sub::Adopt) s->adopt_depth--;
	if (frame.sub == Sub::Command) s->command_depth--;
	if (s->command_depth != 0) {
		s->command_nested++;
	} else if (frame.sub == Sub::Command) {
		s->command_ticks += total;
		s->command_calls++;
	}
	if (s->depth != 0) {
		Frame& parent = s->stack[s->depth - 1];
		parent.child += total;
		s->child_calls[Index(parent.sub)]++;
	}
}

void Reclassify() {
	State* const s = t_state;
	if (s == nullptr || s->depth == 0) {
		return;
	}
	Frame& frame = s->stack[s->depth - 1];
	if (frame.sub != Sub::TrkQuery) {
		return;
	}
	frame.sub = Sub::TrkMark;
	// A mark is ordered work: it leaves the lookup statistics.
	auto& c = s->cover[KTrk];
	if ((frame.flags & FCounted) != 0) c.n--;
	if ((frame.flags & FCovered) != 0) c.covered--;
	if ((frame.flags & FStale) != 0) c.stale--;
	if ((frame.flags & FStaleRegion) != 0) c.stale_region--;
	frame.flags &= FInEvent;
}

bool EventBegin() {
	State* const s = t_state;
	if (s == nullptr) {
		return false;
	}
	if (s->event_depth == 0) {
		s->ev_stages  = 0;
		s->ev_adopted = 0;
		s->ev_stamp   = {};
	}
	if (!PushImpl(Sub::EventOther, 0, 0)) {
		return false;
	}
	if (s->event_depth++ == 0) {
		s->events++;
	}
	return true;
}

void EventEnd() {
	State* const s = t_state;
	if (s == nullptr) {
		return;
	}
	Pop();
	if (s->event_depth == 0) {
		return;
	}
	if (s->event_depth == 1 && s->ev_stages != 0 && s->ev_adopted == s->ev_stages && s->ev_stamp.valid) {
		s->events_covered++;
	}
	s->event_depth--;
}

ResolveStamp TakeResolveStamp() {
	ResolveStamp stamp;
	stamp.image_epoch  = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire);
	stamp.buffer_epoch = DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_acquire);
	stamp.cpu_sequence = CpuDirtyMarkSequence();
	stamp.region_clock = g_cpu_dirty_region_clock.load(std::memory_order_acquire);
	stamp.valid        = true;
	return stamp;
}

void NoteAdopted(const ResolveStamp& stamp) {
	State* const s = t_state;
	if (s == nullptr || s->event_depth == 0 || !stamp.valid) {
		return;
	}
	auto& oldest = s->ev_stamp;
	if (!oldest.valid) {
		oldest = stamp;
		return;
	}
	// Every field only grows: the oldest view of each is the smallest.
	oldest.image_epoch  = std::min(oldest.image_epoch, stamp.image_epoch);
	oldest.buffer_epoch = std::min(oldest.buffer_epoch, stamp.buffer_epoch);
	oldest.cpu_sequence = std::min(oldest.cpu_sequence, stamp.cpu_sequence);
	if (static_cast<int32_t>(stamp.region_clock - oldest.region_clock) < 0) {
		oldest.region_clock = stamp.region_clock;
	}
}

void NoteStage(bool made, bool taken, bool adopted) {
	State* const s = t_state;
	if (s == nullptr) {
		return;
	}
	s->stages++;
	if (made) s->stages_speculated++;
	if (s->event_depth != 0) {
		auto& c = s->cover[KProg];
		if (taken) c.taken++;
		if (adopted) {
			c.covered++;
			s->ev_adopted++;
		} else if (taken) {
			c.stale++; // the key matched, the reads had changed
		}
	}
	// The program lookup around this call (ProgramCache::Get) counts as covered when its result was the resolve thread's.
	if (adopted && s->depth != 0 && s->stack[s->depth - 1].sub == Sub::ProgFind) {
		s->stack[s->depth - 1].flags |= FCovered;
	}
}

void MaybeReport() {
	State* const s = t_state;
	if (s == nullptr || s->depth != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (s->window_ticks == 0) {
		StartWindow(*s);
		return;
	}
	if (now - s->window_start < std::chrono::seconds(5)) {
		return;
	}
	Report(*s, now);
	Calibrate(*s); // for the next window (the clock and the caches may have changed)
	StartWindow(*s);
}

} // namespace Libs::Graphics::StageF
