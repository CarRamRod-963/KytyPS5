#pragma once

#include "common/envRead.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {
class BufferCache;
} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

// KYTY_TRACE_SRT_READS=1: the resource walker's ordinary reads (pointer chasing at draw/dispatch
// preparation, on the GPU thread) are grouped by shader and 4 KiB page: all reads, reads of a page the
// GPU modified (the session reads their clean bytes from the backing store) and reads of bytes the
// GPU really wrote (these must wait for it either way). The top entries are printed every 8192 GPU-modified-page reads.
inline bool SrtReadTraceEnabled() {
	static const bool on = Common::EnvRead::SrtReads();
	return on;
}
void TraceSrtRead(Graphics::BufferCache& buffers, uint64_t vaddr, uint64_t size);

// KYTY_TRACE_SRT_CLASSES (diagnostic, default off): the GPU thread's SRT reads, their waits for GPU writes and the Adopt
// work by what the word is for (shader-only or needed by the CPU) and by address region; one SRT_CLASSES line per ~5 s
// with per-frame averages. Decides whether letting the shaders read the shader-only words pays.
inline uint64_t SrtClassesNowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
void SrtClassesNoteRead(uint64_t vaddr, uint64_t size);
void SrtClassesWait(const char* where, uint32_t cls, uint64_t vaddr, uint64_t ns);

} // namespace Libs::LibKernel::Memory
