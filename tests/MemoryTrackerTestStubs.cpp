// Test-only definition for memory_tracker_tests: perfCounters.h's inline paths call a hook of the renderer's performance
// tracing, defined in perfSampler.cpp, which this test does not link. It is never reached; the body only satisfies the
// linker.
#include "graphics/host_gpu/renderer/perfCounters.h"

namespace Libs::Graphics {


void PerfDumpCurrentStack(const char* /*label*/) {}

} // namespace Libs::Graphics
