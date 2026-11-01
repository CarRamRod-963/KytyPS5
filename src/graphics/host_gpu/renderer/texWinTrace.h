#ifndef KYTY_GRAPHICS_TEXWIN_TRACE_H_
#define KYTY_GRAPHICS_TEXWIN_TRACE_H_

#include "common/onceValue.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::TexWin {

// KYTY_TRACE_TEXWIN=1 (diagnostic, default off; one cached load per hook when off): stage S0 of
// notes\TEKSTURY-STREAMING-PROJEKT.md, the counters TW1-TW7 of notes\TEKSTURY-STREAMING-ANALIZA.md section 5: what the
// host holds and does for the textures the game streams into its private range 0x90_0000_0000-0xFC_0000_0000
// ("texman" below). Changes no behaviour: it only counts, times and reads. Every ~5 s the GPU thread prints (sizes in
// MiB with one decimal, times in microseconds, "a/b" = count/MiB):
//   TEXWIN images (TW1, RenderExecutor::FindResolvedTexture): every resolve of a T# whose base lies in texman.
//     resolves, base=<BASE_LEVEL>:<resolves>,... (histogram, non-zero buckets); over the distinct images of the window
//     (key: base address + full chain size): distinct, full = their full chains, prefix = the bytes of their levels
//     from the smallest BASE_LEVEL seen this window to the last (the span of those levels' mip_layout entries; an
//     image whose span does not start at the base address counts with its full chain), zeros = full - prefix,
//     streamed = images seen only with BASE_LEVEL > 0, not_prefix = images whose span does not start at the base,
//     excl = arrays, 3D, MSAA, metadata (DCC) and storage images (not in full/prefix), overflow = resolves not recorded
//     (map bound).
//   TEXWIN zero_fill (TW2): GuestBackingStore::TryReadSparseBacking (only caller: BufferCache::ObtainBufferForImage,
//     the image uploads): reads, bytes copied and zero-filled (unmapped pages), all and texman; then the sources of
//     ObtainBufferForImage (buffer = a cached buffer holds the range, gpu = GPU-modified range, staging = the sparse
//     read), all and texman.
//   TEXWIN batchmap (TW3): KernelBatchMap2 calls and entries by operation (n/MiB, texman in brackets); then the
//     GPU-side unmaps (RenderContext::UnmapMemory, a SendCommandSync wait for a game thread) by caller: map_free = a
//     fixed map over space without committed memory (ReplaceFixedRangeWithReserved: the texman page maps), map_replace
//     = a fixed map over committed memory, map_other = the other map paths (into a reserved range, or a free range the
//     kernel found), unmap = UnmapMemory (munmap and BatchMap2 unmap entries), reserve = a reserve of virtual space,
//     other = every other caller. Per caller n,tex,drain (Finish() ran),wall (the caller's time in
//     RenderContext::UnmapMemory),exec (the GPU thread's time in the command),max (longest wall); inline = calls that
//     ran on the calling thread (the GPU thread itself, or a stopping GPU).
//   TEXWIN unmap_drain (TW4): the GPU-side unmaps, those that did not drain, and the drains (Finish() +
//     WaitPriorityOperations) by the clause that fired, n/texman/us: image = an image registered in the range,
//     buffer = a buffer registered (no image), prio_only = only a priority operation pending anywhere, unknown = none
//     of them when evaluated (raced); then the images TextureCache::UnmapMemory freed (all, texman).
//   TEXWIN first_write (TW5): write faults (RenderContext::HandleFault) on texman pages; first = the first one on a
//     128 KiB page since RenderContext::MapMemory mapped it; mapped_pages = 128 KiB texman pages mapped;
//     unmapped_unwritten = pages unmapped again without such a fault; pending = pages mapped and not written yet; then
//     the uploads of texman images (TextureCache::InitializeImage): new = the image's first upload, again = a
//     re-upload of an image already uploaded (a CPU write or a buffer copy made it dirty: the whole chain again).
//     Best effort: a write to a page no registered image watches does not fault and is not seen.
//   TEXWIN game_pool (TW6): the game's own view, read once per window from guest memory with
//     the backing store (never faults, never waits for the GPU): pool free bytes, requests in flight (textures, meshes), number of textures,
//     texman update counter; "unreadable" when any read fails. Only with KYTY_TRACE_TEXWIN_GAME_POOL (the addresses
//     of those counters and the pool size, hex, see texWinTrace.cpp; default empty: no game_pool line).
//   TW7: the bindless table slots repointed to the placeholder already have their counters: PERF bindless_tables
//     dropped_written / dead_slot_reads (KYTY_TRACE_PERF, commit 9ca2f187, BindlessTable DropTableSlot).
// At device creation one "TEXWIN sparse" line: the device's sparse binding / residency features and properties.
inline bool Enabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		const char* value   = std::getenv("KYTY_TRACE_TEXWIN");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) std::fprintf(stderr, "TRACE_TEXWIN enabled\n");
		return enabled;
	});
	return on;
}

// The game's private range for streamed textures (the game maps pages there without reserving the range with the
// kernel first).
constexpr uint64_t kTexmanBegin = 0x90'0000'0000ull;
constexpr uint64_t kTexmanEnd   = 0xFC'0000'0000ull;

[[nodiscard]] constexpr bool InTexman(uint64_t address) noexcept {
	return address >= kTexmanBegin && address < kTexmanEnd;
}

[[nodiscard]] constexpr bool OverlapsTexman(uint64_t address, uint64_t size) noexcept {
	return size != 0 && address < kTexmanEnd && (address >= kTexmanBegin || kTexmanBegin - address < size);
}

// A texture cache image (ImageId index and generation: unique over the session).
[[nodiscard]] constexpr uint64_t ImageKey(uint32_t index, uint32_t generation) noexcept {
	return (static_cast<uint64_t>(index) << 32u) | generation;
}

[[nodiscard]] inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// ---- TW1 ----

// A T# resolve with a texman base: its BASE_LEVEL, the full chain (ImageInfo data.size), the bytes of the levels
// BASE_LEVEL..last (span of their mip_layout entries) and whether that span starts at the base address (prefix).
void NoteResolve(uint64_t address, uint64_t full_bytes, uint32_t base_level, uint64_t window_bytes, bool prefix,
                 bool excluded);

// ---- TW2 ----

// TryReadSparseBacking: `size` bytes read for an image upload, `zeroed` of them zero-filled (no guest page).
void NoteSparseRead(uint64_t vaddr, uint64_t size, uint64_t zeroed);

enum class UploadSource : uint32_t { Buffer, GpuModified, Staging, Count };
// BufferCache::ObtainBufferForImage: where the bytes of an image upload come from.
void NoteUploadSource(UploadSource source, uint64_t vaddr, uint64_t size);

// ---- TW3 ----

void NoteBatchCall();
// One KernelBatchMap2 entry about to run (operation 0..4: map direct, unmap, protect, map flexible, type protect).
void NoteBatchEntry(int32_t operation, uint64_t address, uint64_t size);

enum class UnmapCaller : uint32_t { MapFree, MapReplace, MapOther, Unmap, Reserve, Other, Count };

// The caller of the GPU-side unmaps of this scope (the calling thread; an inner scope wins). Nothing when off.
UnmapCaller SwapUnmapCaller(UnmapCaller caller) noexcept;
UnmapCaller CurrentUnmapCaller() noexcept;

class UnmapCallerScope {
public:
	explicit UnmapCallerScope(UnmapCaller caller) noexcept {
		if (Enabled()) {
			m_set      = true;
			m_previous = SwapUnmapCaller(caller);
		}
	}
	~UnmapCallerScope() {
		if (m_set) (void)SwapUnmapCaller(m_previous);
	}
	UnmapCallerScope(const UnmapCallerScope&)            = delete;
	UnmapCallerScope& operator=(const UnmapCallerScope&) = delete;

private:
	bool        m_set      = false;
	UnmapCaller m_previous = UnmapCaller::Other;
};

// ---- TW3, TW4: one RenderContext::UnmapMemory ----

struct UnmapRecord {
	UnmapCaller caller        = UnmapCaller::Other;
	bool        texman        = false;
	bool        buffer        = false; // a buffer registered in the range (GPU thread, before the drain decision)
	bool        image         = false; // an image registered in the range
	bool        priority      = false; // a priority operation pending anywhere
	bool        drained       = false; // Finish() + WaitPriorityOperations ran
	int64_t     call_ns       = 0;
	int64_t     exec_begin_ns = 0;
	int64_t     exec_end_ns   = 0;
	int64_t     drain_ns      = 0;
};

// The calling thread, before the command is sent (also TW5: texman pages unmapped).
void BeginUnmap(UnmapRecord& record, uint64_t vaddr, uint64_t size);
// The thread running the command: the clauses of the drain decision as they were when it ran.
void BeginUnmapExec(UnmapRecord& record, bool buffer, bool image, bool priority);
void EndUnmapExec(UnmapRecord& record);
// The calling thread, after the command (inline_call: it ran on the calling thread, nobody waited for the GPU thread).
void EndUnmap(const UnmapRecord& record, bool inline_call);

// The drain of RenderContext::UnmapMemory (null record: off).
class DrainScope {
public:
	explicit DrainScope(UnmapRecord* record) noexcept: m_record(record) {
		if (m_record != nullptr) m_start_ns = NowNs();
	}
	~DrainScope() {
		if (m_record != nullptr) {
			m_record->drained  = true;
			m_record->drain_ns = NowNs() - m_start_ns;
		}
	}
	DrainScope(const DrainScope&)            = delete;
	DrainScope& operator=(const DrainScope&) = delete;

private:
	UnmapRecord* m_record   = nullptr;
	int64_t      m_start_ns = 0;
};

// TextureCache::UnmapMemory: an image it frees.
void NoteUnmapFree(uint64_t address, uint64_t bytes);

// ---- TW5 ----

// RenderContext::MapMemory: a range became mapped (texman pages are remembered until their first write fault).
void NoteMapped(uint64_t vaddr, uint64_t size);
// RenderContext::HandleFault: a write fault.
void NoteWriteFault(uint64_t vaddr);
// TextureCache::InitializeImage: an image upload; TextureCache::UnregisterImage: the image is gone.
void NoteImageUpload(uint64_t image_key, uint64_t address, uint64_t bytes);
void NoteImageGone(uint64_t image_key, uint64_t address);

// ---- Report ----

// GPU thread (GuestGpu::ThreadRun, after the PERF and MEM_CENSUS lines): the TEXWIN lines every ~5 s.
void Report();

} // namespace Libs::Graphics::TexWin

#endif
