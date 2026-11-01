#include "graphics/host_gpu/renderer/texWinTrace.h"

#include "graphics/host_gpu/renderer/debugSample.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::TexWin {

namespace {

using Counter = std::atomic<uint64_t>;

void Add(Counter& counter, uint64_t value) {
	counter.fetch_add(value, std::memory_order_relaxed);
}

uint64_t Take(Counter& counter) {
	return counter.exchange(0, std::memory_order_relaxed);
}

void Max(Counter& counter, uint64_t value) {
	uint64_t current = counter.load(std::memory_order_relaxed);
	while (value > current && !counter.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
	}
}

double Mib(uint64_t bytes) {
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

unsigned long long Num(uint64_t value) {
	return static_cast<unsigned long long>(value);
}

unsigned long long Us(uint64_t ns) {
	return static_cast<unsigned long long>(ns / 1000u);
}

uint64_t Positive(int64_t value) {
	return value > 0 ? static_cast<uint64_t>(value) : 0u;
}

// ---- TW1 (GPU thread; locked anyway: resolves may run elsewhere) ----

struct ResolveKey {
	uint64_t address = 0;
	uint64_t full    = 0;
	bool     operator==(const ResolveKey&) const = default;
};

struct ResolveKeyHash {
	size_t operator()(const ResolveKey& key) const noexcept {
		return std::hash<uint64_t> {}((key.address * 0x9E3779B97F4A7C15ull) ^ key.full);
	}
};

struct ResolveEntry {
	uint64_t window   = 0;
	uint32_t min_base = UINT32_MAX;
	bool     prefix   = true;
	bool     excluded = false;
};

// Distinct images per window; a window with more is counted as overflow (cleared at every report).
constexpr size_t kMaxResolveImages = size_t {1} << 18u;

struct ResolveState {
	std::mutex                                                   mutex;
	std::unordered_map<ResolveKey, ResolveEntry, ResolveKeyHash> images;
	std::array<uint64_t, 16>                                     base {};
	uint64_t                                                     resolves = 0;
	uint64_t                                                     overflow = 0;
};

ResolveState& Resolves() {
	static ResolveState state;
	return state;
}

// ---- TW2 (any thread) ----

constexpr uint32_t kSources = static_cast<uint32_t>(UploadSource::Count);

struct SparseCounters {
	Counter                       reads {0};
	Counter                       copied {0};
	Counter                       zeroed {0};
	Counter                       tex_reads {0};
	Counter                       tex_copied {0};
	Counter                       tex_zeroed {0};
	std::array<Counter, kSources> source_n {};
	std::array<Counter, kSources> source_bytes {};
	std::array<Counter, kSources> tex_source_n {};
	std::array<Counter, kSources> tex_source_bytes {};
};

SparseCounters& Sparse() {
	static SparseCounters counters;
	return counters;
}

// ---- TW3, TW4 (game threads, GPU thread) ----

constexpr uint32_t kOps     = 5;
constexpr uint32_t kCallers = static_cast<uint32_t>(UnmapCaller::Count);

enum DrainCause : uint32_t { CauseImage, CauseBuffer, CausePriority, CauseUnknown, kCauses };

struct BatchCounters {
	Counter                       calls {0};
	Counter                       entries {0};
	std::array<Counter, kOps>     n {};
	std::array<Counter, kOps>     bytes {};
	std::array<Counter, kOps>     tex_n {};
	std::array<Counter, kOps>     tex_bytes {};
	std::array<Counter, kCallers> caller_n {};
	std::array<Counter, kCallers> caller_tex {};
	std::array<Counter, kCallers> caller_drains {};
	std::array<Counter, kCallers> caller_wall_ns {};
	std::array<Counter, kCallers> caller_exec_ns {};
	std::array<Counter, kCallers> caller_max_ns {};
	Counter                       inline_calls {0};
	// TW4
	Counter                       unmaps {0};
	Counter                       no_drain {0};
	std::array<Counter, kCauses>  drain_n {};
	std::array<Counter, kCauses>  drain_tex {};
	std::array<Counter, kCauses>  drain_ns {};
	Counter                       freed_n {0};
	Counter                       freed_bytes {0};
	Counter                       freed_tex_n {0};
	Counter                       freed_tex_bytes {0};
};

BatchCounters& Batch() {
	static BatchCounters counters;
	return counters;
}

thread_local UnmapCaller t_unmap_caller = UnmapCaller::Other;

// ---- TW5 ----

constexpr uint32_t kPageShift = 17u;               // the game's 128 KiB texture pages
constexpr size_t   kMaxPages  = size_t {1} << 20u; // bound of both sets (cleared, counted as overflow)

struct FirstWriteState {
	std::mutex                   mutex;
	std::unordered_set<uint64_t> fresh;    // texman pages mapped and not written since (address >> kPageShift)
	std::unordered_set<uint64_t> uploaded; // texman images uploaded at least once (ImageKey)
	uint64_t                     overflow = 0;
};

FirstWriteState& FirstWrite() {
	static FirstWriteState state;
	return state;
}

struct FirstWriteCounters {
	Counter faults {0};
	Counter first {0};
	Counter mapped_pages {0};
	Counter unmapped_unwritten {0};
	Counter new_n {0};
	Counter new_bytes {0};
	Counter again_n {0};
	Counter again_bytes {0};
};

FirstWriteCounters& FirstWriteCount() {
	static FirstWriteCounters counters;
	return counters;
}

// The texman pages of [vaddr, vaddr + size) (clipped to the range).
template <typename F>
void ForEachTexmanPage(uint64_t vaddr, uint64_t size, F&& f) {
	if (!OverlapsTexman(vaddr, size)) return;
	const uint64_t begin = std::max(vaddr, kTexmanBegin);
	const uint64_t end   = std::min(vaddr + size, kTexmanEnd);
	for (uint64_t page = begin >> kPageShift; page <= (end - 1u) >> kPageShift; page++) {
		f(page);
	}
}

// ---- TW6 ----

// KYTY_TRACE_TEXWIN_GAME_POOL=<free>,<in_flight>,<textures>,<update>,<pool_bytes> (hex, default empty): where the game
// keeps its texture pool counters, given by the user for the game build being traced; the source holds no game
// addresses. free = u64 free bytes of the texture page pool, in_flight = i32 texture requests and i32 mesh requests in
// flight, textures = u32 textures known to texman, update = i32 texman update counter, pool_bytes = the pool's size
// (a number, not an address). Empty or not five values: TW6 prints nothing.
struct GamePool {
	uint64_t free_bytes = 0, in_flight = 0, textures = 0, update = 0, pool_bytes = 0;
	bool     set        = false;
};

const GamePool& GamePoolAddresses() {
	static const GamePool pool = [] {
		GamePool    p;
		const char* value = std::getenv("KYTY_TRACE_TEXWIN_GAME_POOL");
		if (value == nullptr || value[0] == '\0') return p;
		uint64_t*   fields[] = {&p.free_bytes, &p.in_flight, &p.textures, &p.update, &p.pool_bytes};
		const char* s        = value;
		for (size_t i = 0; i < std::size(fields); i++) {
			char* end  = nullptr;
			*fields[i] = std::strtoull(s, &end, 16);
			const char separator = i + 1 == std::size(fields) ? '\0' : ',';
			if (end == s || *end != separator) return GamePool {};
			s = end + 1;
		}
		p.set = true;
		std::fprintf(stderr, "TRACE_TEXWIN game_pool addresses set\n");
		return p;
	}();
	return pool;
}

void ReportImages(uint32_t frame) {
	auto&                                                        state = Resolves();
	std::unordered_map<ResolveKey, ResolveEntry, ResolveKeyHash> images;
	std::array<uint64_t, 16>                                     base {};
	uint64_t                                                     resolves = 0, overflow = 0;
	{
		std::scoped_lock lock(state.mutex);
		images.swap(state.images);
		base     = std::exchange(state.base, {});
		resolves = std::exchange(state.resolves, 0);
		overflow = std::exchange(state.overflow, 0);
	}
	uint64_t full = 0, prefix = 0, streamed_n = 0, streamed_bytes = 0, not_prefix_n = 0, not_prefix_bytes = 0;
	uint64_t excluded_n = 0, excluded_bytes = 0;
	for (const auto& [key, entry]: images) {
		if (entry.excluded) {
			excluded_n++;
			excluded_bytes += key.full;
			continue;
		}
		full += key.full;
		if (entry.prefix) {
			prefix += std::min(entry.window, key.full);
		} else {
			prefix += key.full;
			not_prefix_n++;
			not_prefix_bytes += key.full;
		}
		if (entry.min_base > 0) {
			streamed_n++;
			streamed_bytes += key.full;
		}
	}
	std::string histogram;
	for (size_t level = 0; level < base.size(); level++) {
		if (base[level] == 0) continue;
		histogram += (histogram.empty() ? "" : ",") + std::to_string(level) + ":" + std::to_string(base[level]);
	}
	std::fprintf(stderr,
	             "TEXWIN images frame=%u resolves=%llu base=%s distinct=%zu full=%.1f prefix=%.1f zeros=%.1f "
	             "streamed=%llu/%.1f not_prefix=%llu/%.1f excl=%llu/%.1f overflow=%llu\n",
	             frame, Num(resolves), histogram.empty() ? "-" : histogram.c_str(), images.size(), Mib(full),
	             Mib(prefix), Mib(full - prefix), Num(streamed_n), Mib(streamed_bytes), Num(not_prefix_n),
	             Mib(not_prefix_bytes), Num(excluded_n), Mib(excluded_bytes), Num(overflow));
}

void ReportZeroFill(uint32_t frame) {
	auto& c = Sparse();
	std::fprintf(stderr,
	             "TEXWIN zero_fill frame=%u reads=%llu copied=%.1f zeroed=%.1f texman_reads=%llu texman_copied=%.1f "
	             "texman_zeroed=%.1f",
	             frame, Num(Take(c.reads)), Mib(Take(c.copied)), Mib(Take(c.zeroed)), Num(Take(c.tex_reads)),
	             Mib(Take(c.tex_copied)), Mib(Take(c.tex_zeroed)));
	static constexpr const char* names[kSources] = {"buffer", "gpu", "staging"};
	std::fprintf(stderr, " | src");
	for (uint32_t i = 0; i < kSources; i++) {
		std::fprintf(stderr, " %s=%llu/%.1f", names[i], Num(Take(c.source_n[i])), Mib(Take(c.source_bytes[i])));
	}
	std::fprintf(stderr, " texman");
	for (uint32_t i = 0; i < kSources; i++) {
		std::fprintf(stderr, " %s=%llu/%.1f", names[i], Num(Take(c.tex_source_n[i])), Mib(Take(c.tex_source_bytes[i])));
	}
	std::fprintf(stderr, "\n");
}

void ReportBatchMap(uint32_t frame) {
	auto& c = Batch();
	static constexpr const char* ops[kOps] = {"map_direct", "unmap", "protect", "map_flexible", "type_protect"};
	std::fprintf(stderr, "TEXWIN batchmap frame=%u calls=%llu entries=%llu", frame, Num(Take(c.calls)),
	             Num(Take(c.entries)));
	for (uint32_t i = 0; i < kOps; i++) {
		std::fprintf(stderr, " %s=%llu/%.1f(tex %llu/%.1f)", ops[i], Num(Take(c.n[i])), Mib(Take(c.bytes[i])),
		             Num(Take(c.tex_n[i])), Mib(Take(c.tex_bytes[i])));
	}
	static constexpr const char* callers[kCallers] = {"map_free", "map_replace", "map_other",
	                                                  "unmap",    "reserve",     "other"};
	std::fprintf(stderr, " | gpu_unmap");
	for (uint32_t i = 0; i < kCallers; i++) {
		std::fprintf(stderr, " %s=%llu,tex=%llu,drain=%llu,wall=%lluus,exec=%lluus,max=%lluus", callers[i],
		             Num(Take(c.caller_n[i])), Num(Take(c.caller_tex[i])), Num(Take(c.caller_drains[i])),
		             Us(Take(c.caller_wall_ns[i])), Us(Take(c.caller_exec_ns[i])), Us(Take(c.caller_max_ns[i])));
	}
	std::fprintf(stderr, " inline=%llu\n", Num(Take(c.inline_calls)));
}

void ReportUnmapDrain(uint32_t frame) {
	auto& c = Batch();
	static constexpr const char* causes[kCauses] = {"image", "buffer", "prio_only", "unknown"};
	std::fprintf(stderr, "TEXWIN unmap_drain frame=%u unmaps=%llu no_drain=%llu", frame, Num(Take(c.unmaps)),
	             Num(Take(c.no_drain)));
	for (uint32_t i = 0; i < kCauses; i++) {
		std::fprintf(stderr, " %s=%llu/%llu/%lluus", causes[i], Num(Take(c.drain_n[i])), Num(Take(c.drain_tex[i])),
		             Us(Take(c.drain_ns[i])));
	}
	std::fprintf(stderr, " | freed=%llu/%.1f texman=%llu/%.1f\n", Num(Take(c.freed_n)), Mib(Take(c.freed_bytes)),
	             Num(Take(c.freed_tex_n)), Mib(Take(c.freed_tex_bytes)));
}

void ReportFirstWrite(uint32_t frame) {
	auto&    c       = FirstWriteCount();
	size_t   pending = 0;
	uint64_t overflow = 0;
	{
		auto&            state = FirstWrite();
		std::scoped_lock lock(state.mutex);
		pending  = state.fresh.size();
		overflow = std::exchange(state.overflow, 0);
	}
	std::fprintf(stderr,
	             "TEXWIN first_write frame=%u texman_faults=%llu first=%llu mapped_pages=%llu unmapped_unwritten=%llu "
	             "pending=%zu overflow=%llu | uploads new=%llu/%.1f again=%llu/%.1f\n",
	             frame, Num(Take(c.faults)), Num(Take(c.first)), Num(Take(c.mapped_pages)),
	             Num(Take(c.unmapped_unwritten)), pending, Num(overflow), Num(Take(c.new_n)), Mib(Take(c.new_bytes)),
	             Num(Take(c.again_n)), Mib(Take(c.again_bytes)));
}

void ReportGamePool(uint32_t frame) {
	const GamePool& pool = GamePoolAddresses();
	if (!pool.set) return;
	uint64_t free_bytes = 0;
	int32_t  in_flight[2] {};
	uint32_t textures = 0;
	int32_t  update   = 0;
	using LibKernel::Memory::TryReadBacking;
	const bool ok = TryReadBacking(pool.free_bytes, &free_bytes, sizeof(free_bytes)) &&
	                TryReadBacking(pool.in_flight, in_flight, sizeof(in_flight)) &&
	                TryReadBacking(pool.textures, &textures, sizeof(textures)) &&
	                TryReadBacking(pool.update, &update, sizeof(update));
	if (!ok) {
		std::fprintf(stderr, "TEXWIN game_pool frame=%u unreadable\n", frame);
		return;
	}
	const uint64_t used = free_bytes <= pool.pool_bytes ? pool.pool_bytes - free_bytes : 0;
	std::fprintf(stderr,
	             "TEXWIN game_pool frame=%u free=%.1f used=%.1f pool=%.1f inflight_tex=%d inflight_mesh=%d textures=%u "
	             "update=%d\n",
	             frame, Mib(free_bytes), Mib(used), Mib(pool.pool_bytes), in_flight[0], in_flight[1], textures, update);
}

} // namespace

// ---- TW1 ----

void NoteResolve(uint64_t address, uint64_t full_bytes, uint32_t base_level, uint64_t window_bytes, bool prefix,
                 bool excluded) {
	auto&            state = Resolves();
	std::scoped_lock lock(state.mutex);
	state.resolves++;
	state.base[std::min<uint32_t>(base_level, 15u)]++;
	auto found = state.images.find({address, full_bytes});
	if (found == state.images.end()) {
		if (state.images.size() >= kMaxResolveImages) {
			state.overflow++;
			return;
		}
		found = state.images.emplace(ResolveKey {address, full_bytes}, ResolveEntry {}).first;
	}
	auto& entry    = found->second;
	entry.excluded = entry.excluded || excluded;
	if (base_level < entry.min_base) {
		entry.min_base = base_level;
		entry.window   = window_bytes;
		entry.prefix   = prefix;
	}
}

// ---- TW2 ----

void NoteSparseRead(uint64_t vaddr, uint64_t size, uint64_t zeroed) {
	auto&          c      = Sparse();
	const uint64_t copied = size >= zeroed ? size - zeroed : 0;
	Add(c.reads, 1);
	Add(c.copied, copied);
	Add(c.zeroed, zeroed);
	if (OverlapsTexman(vaddr, size)) {
		Add(c.tex_reads, 1);
		Add(c.tex_copied, copied);
		Add(c.tex_zeroed, zeroed);
	}
}

void NoteUploadSource(UploadSource source, uint64_t vaddr, uint64_t size) {
	auto&          c     = Sparse();
	const uint32_t index = std::min(static_cast<uint32_t>(source), kSources - 1u);
	Add(c.source_n[index], 1);
	Add(c.source_bytes[index], size);
	if (OverlapsTexman(vaddr, size)) {
		Add(c.tex_source_n[index], 1);
		Add(c.tex_source_bytes[index], size);
	}
}

// ---- TW3 ----

void NoteBatchCall() {
	Add(Batch().calls, 1);
}

void NoteBatchEntry(int32_t operation, uint64_t address, uint64_t size) {
	if (operation < 0 || static_cast<uint32_t>(operation) >= kOps) return;
	auto&          c     = Batch();
	const uint32_t index = static_cast<uint32_t>(operation);
	Add(c.entries, 1);
	Add(c.n[index], 1);
	Add(c.bytes[index], size);
	if (OverlapsTexman(address, size)) {
		Add(c.tex_n[index], 1);
		Add(c.tex_bytes[index], size);
	}
}

UnmapCaller SwapUnmapCaller(UnmapCaller caller) noexcept {
	return std::exchange(t_unmap_caller, caller);
}

UnmapCaller CurrentUnmapCaller() noexcept {
	return t_unmap_caller;
}

// ---- TW3, TW4 ----

void BeginUnmap(UnmapRecord& record, uint64_t vaddr, uint64_t size) {
	record.caller  = CurrentUnmapCaller();
	record.texman  = OverlapsTexman(vaddr, size);
	record.call_ns = NowNs();
	if (!record.texman) return;
	// TW5: pages unmapped before any write fault.
	uint64_t unwritten = 0;
	{
		auto&            state = FirstWrite();
		std::scoped_lock lock(state.mutex);
		if (!state.fresh.empty()) {
			ForEachTexmanPage(vaddr, size, [&](uint64_t page) { unwritten += state.fresh.erase(page); });
		}
	}
	Add(FirstWriteCount().unmapped_unwritten, unwritten);
}

void BeginUnmapExec(UnmapRecord& record, bool buffer, bool image, bool priority) {
	record.buffer        = buffer;
	record.image         = image;
	record.priority      = priority;
	record.exec_begin_ns = NowNs();
}

void EndUnmapExec(UnmapRecord& record) {
	record.exec_end_ns = NowNs();
}

void EndUnmap(const UnmapRecord& record, bool inline_call) {
	auto&          c       = Batch();
	const uint32_t caller  = std::min(static_cast<uint32_t>(record.caller), kCallers - 1u);
	const uint64_t wall_ns = Positive(NowNs() - record.call_ns);
	Add(c.caller_n[caller], 1);
	Add(c.caller_tex[caller], record.texman ? 1u : 0u);
	Add(c.caller_drains[caller], record.drained ? 1u : 0u);
	Add(c.caller_wall_ns[caller], wall_ns);
	Add(c.caller_exec_ns[caller], Positive(record.exec_end_ns - record.exec_begin_ns));
	Max(c.caller_max_ns[caller], wall_ns);
	if (inline_call) Add(c.inline_calls, 1);
	Add(c.unmaps, 1);
	if (!record.drained) {
		Add(c.no_drain, 1);
		return;
	}
	const uint32_t cause = record.image ? CauseImage : record.buffer ? CauseBuffer : record.priority ? CausePriority
	                                                                                                  : CauseUnknown;
	Add(c.drain_n[cause], 1);
	Add(c.drain_tex[cause], record.texman ? 1u : 0u);
	Add(c.drain_ns[cause], Positive(record.drain_ns));
}

void NoteUnmapFree(uint64_t address, uint64_t bytes) {
	auto& c = Batch();
	Add(c.freed_n, 1);
	Add(c.freed_bytes, bytes);
	if (InTexman(address)) {
		Add(c.freed_tex_n, 1);
		Add(c.freed_tex_bytes, bytes);
	}
}

// ---- TW5 ----

void NoteMapped(uint64_t vaddr, uint64_t size) {
	if (!OverlapsTexman(vaddr, size)) return;
	uint64_t pages = 0;
	{
		auto&            state = FirstWrite();
		std::scoped_lock lock(state.mutex);
		ForEachTexmanPage(vaddr, size, [&](uint64_t page) {
			state.fresh.insert(page);
			pages++;
		});
		if (state.fresh.size() > kMaxPages) {
			state.overflow += state.fresh.size();
			state.fresh.clear();
		}
	}
	Add(FirstWriteCount().mapped_pages, pages);
}

void NoteWriteFault(uint64_t vaddr) {
	if (!InTexman(vaddr)) return;
	auto& c = FirstWriteCount();
	Add(c.faults, 1);
	bool first = false;
	{
		auto&            state = FirstWrite();
		std::scoped_lock lock(state.mutex);
		first = state.fresh.erase(vaddr >> kPageShift) != 0;
	}
	if (first) Add(c.first, 1);
}

void NoteImageUpload(uint64_t image_key, uint64_t address, uint64_t bytes) {
	if (!InTexman(address)) return;
	bool first = false;
	{
		auto&            state = FirstWrite();
		std::scoped_lock lock(state.mutex);
		if (state.uploaded.size() >= kMaxPages) {
			state.overflow += state.uploaded.size();
			state.uploaded.clear();
		}
		first = state.uploaded.insert(image_key).second;
	}
	auto& c = FirstWriteCount();
	Add(first ? c.new_n : c.again_n, 1);
	Add(first ? c.new_bytes : c.again_bytes, bytes);
}

void NoteImageGone(uint64_t image_key, uint64_t address) {
	if (!InTexman(address)) return;
	auto&            state = FirstWrite();
	std::scoped_lock lock(state.mutex);
	state.uploaded.erase(image_key);
}

// ---- Report ----

void Report() {
	static auto window_start = std::chrono::steady_clock::now();
	const auto  now          = std::chrono::steady_clock::now();
	if (now - window_start < std::chrono::seconds(5)) return;
	window_start         = now;
	const uint32_t frame = DebugPresentCount().load();
	ReportImages(frame);
	ReportZeroFill(frame);
	ReportBatchMap(frame);
	ReportUnmapDrain(frame);
	ReportFirstWrite(frame);
	ReportGamePool(frame);
}

} // namespace Libs::Graphics::TexWin
