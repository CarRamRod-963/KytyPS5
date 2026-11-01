#pragma once

#include "common/envRead.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class RenderContext;
struct TextureBinding;

// KYTY_TRACE_TABLE_CENSUS=1 (diagnostic, default off; F:\Claude\analysis_1016\TABLE-MODE-DESIGN.md Step 0): per stage,
// whether its image set (T# words, compare functions, S# words, soft-resolve flag) was seen for the same program in this
// or the previous frame (a "would-hit"); for would-hits the set's per-use proof is run dry (page-epoch sums, the
// TEX_REPLAY no-op predicate, id/generation: TextureCache::TableProofDry) and timed (maintenance), today's image and
// sampler work is timed (ceiling; RebindImages' part too), and the stored ids are compared with today's result
// (held_differs, must be 0). Nothing the emulator does changes. One PERF table_census line per ~5 s, gfx and cs.
inline bool TableCensusEnabled() {
	static const bool on = [] {
		const char* value   = std::getenv("KYTY_TRACE_TABLE_CENSUS");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) std::fprintf(stderr, "TRACE_TABLE_CENSUS enabled\n");
		return enabled;
	}();
	return on;
}

struct TableCensusSet {
	std::vector<uint32_t> words;
	std::vector<ImageId>  ids;
	std::vector<uint64_t> page_sums;
	std::vector<std::pair<uint64_t, uint64_t>> ranges; // desc data address, size
	std::vector<uint8_t>  memo;
	uint32_t              frame = 0;
};
struct TableCensusCounts {
	uint64_t stages = 0, would_hit = 0, hit_same_frame = 0, all_memo = 0, proof_ok = 0, page_changed = 0, held_differs = 0;
	uint64_t slots = 0, slots_storage = 0, slots_other = 0;
	uint64_t proof_fail[8] {};
	uint64_t loop_ns_all = 0, loop_ns_hit = 0, proof_ns = 0, rebind_ns_hit = 0;
};
struct TableCensusState {
	std::unordered_map<const void*, std::vector<TableCensusSet>> sets;
	TableCensusCounts                                            counts[2];
	std::chrono::steady_clock::time_point                        window_start = std::chrono::steady_clock::now();
	uint32_t                                                     window_frame = 0;
};
TableCensusState& TableCensus();
void              TableCensusReport();

namespace DescriptorTrace {

// KYTY_TRACE_TEX_ADDRS, KYTY_TRACE_TEX_HASH, KYTY_TRACE_TEX_WRITES_EXTENT: a binding of PrepareBindings' image loop.
[[nodiscard]] inline bool ImageBindingsTraced() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] {
		return std::getenv("KYTY_TRACE_TEX_ADDRS") != nullptr || std::getenv("KYTY_TRACE_TEX_HASH") != nullptr ||
		       std::getenv("KYTY_TRACE_TEX_WRITES_EXTENT") != nullptr;
	});
}
void ImageBinding(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t i,
                  const TextureBinding& binding);

// KYTY_TRACE_SRT_DUMP.
[[nodiscard]] inline bool SrtDumpOn() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] { return std::getenv("KYTY_TRACE_SRT_DUMP") != nullptr; });
}
void SrtDump(uint64_t hash, const std::vector<uint32_t>& words);

// KYTY_TRACE_AUDIT_FRAME (0: not set).
[[nodiscard]] inline uint32_t AuditFrame() {
	static constinit Common::OnceValue<uint32_t> frame_once;
	return frame_once.Get([] { return Common::EnvRead::AuditFrame(); });
}
void AuditImageBinding(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t index,
                       const TextureBinding& binding, Image& image, bool written);

} // namespace DescriptorTrace

} // namespace Libs::Graphics
