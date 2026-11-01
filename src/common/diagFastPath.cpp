#include "common/diagFastPath.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Common::DiagFastPath {
namespace {

// A trace the hooks may leave out is off when its variable is unset or empty or begins with '0'. Every reader of these
// variables accepts only '1', so this is the same test widened: a value this does not call "off" may still be "off" for
// the reader, which only keeps the hooks running.
bool EnvOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// The traces whose hooks Skip() leaves out: one entry per Enabled() in the headers that read Skip().
bool AnyTraceOn() {
	return EnvOn("KYTY_TRACE_STAGEF")  // renderer/stageFTrace.h
	    || EnvOn("KYTY_TRACE_QUEUES"); // common/queueTrace.h
}

} // namespace

void Init() {
	static std::once_flag once;
	std::call_once(once, [] {
		if (AnyTraceOn()) {
			std::fprintf(stderr, "DIAG_HOOKS: a trace is switched on, its hooks keep running\n");
		} else {
			g_state.fetch_and(static_cast<uint8_t>(~kAnyHot), std::memory_order_relaxed);
		}
	});
}

void NoteProfiler(bool running) {
	if (running) {
		g_state.fetch_or(kProfiler, std::memory_order_relaxed);
	} else {
		g_state.fetch_and(static_cast<uint8_t>(~kProfiler), std::memory_order_relaxed);
	}
}

} // namespace Common::DiagFastPath
