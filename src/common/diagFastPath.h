#ifndef KYTY_COMMON_DIAGFASTPATH_H_
#define KYTY_COMMON_DIAGFASTPATH_H_

#include <atomic>
#include <cstdint>

// The diagnostic hooks of the per-draw path (KYTY_TRACE_STAGEF zones, KYTY_TRACE_QUEUES shared scopes) would cost something
// on every draw even with their trace off: a function-local static's thread-safe init guard, a clock read. So every one of
// them first reads Skip() below (one relaxed load of a global and a test) and, when it is true, does nothing.
//
// Skip() is true only while neither trace is switched on in the environment (decided once at start by Init(), from the
// same environment variables, with the test widened rather than narrowed: a set, non-empty value that does not begin with
// '0' counts as on). So a skipped hook is always one whose own Enabled() would have answered false; the fast path only
// leaves out work that was a no-op, and with either trace on nothing is skipped at all.
//
// KYTY_PROFILER_BLOCK (profiler.h) uses the same word: tracy::ProfilerAvailable() is an out-of-line call per zone, so the
// zone asks the profiler bit (set by Profiler::Initialize, cleared by Shutdown) instead.
//
// Before Init() the word says "a trace may be on": every hook does its full work.
namespace Common::DiagFastPath {

inline constexpr uint8_t kAnyHot   = 2u; // one of the traces above is switched on in the environment (set until Init() proved none)
inline constexpr uint8_t kProfiler = 4u; // the Tracy profiler is running (Profiler::Initialize .. Profiler::Shutdown)

inline constinit std::atomic<uint8_t> g_state {kAnyHot};

// The hooks of the traces above: true = return at once (the trace is off).
[[nodiscard]] inline bool Skip() noexcept {
	return (g_state.load(std::memory_order_relaxed) & kAnyHot) == 0;
}

// Main thread, once at start (emulator.cpp Init), before any worker thread draws: reads the traces' environment variables.
void Init();
// Profiler::Initialize / Shutdown: whether the Tracy profiler is running.
void NoteProfiler(bool running);

} // namespace Common::DiagFastPath

#endif // KYTY_COMMON_DIAGFASTPATH_H_
