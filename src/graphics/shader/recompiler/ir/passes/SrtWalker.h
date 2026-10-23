#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

// Image descriptors at base + first + k * step, k < count, within size bytes that a sampled
// root selects by GPU-computed byte offset. The host keeps them resident instead of
// materializing them.
struct ImageHeapRequest {
	uint64_t             base  = 0;
	uint64_t             size  = 0;
	uint64_t             step  = 0;
	uint64_t             first = 0;
	uint64_t             count = 0;
	const ImageResource* root  = nullptr;
};

struct ImageHeapView {
	uint64_t                         slots      = 0; // Device address of one slot per heap dword.
	uint32_t                         slot_count = 0; // Keys past it select a trailing null slot.
	std::span<const DescriptorValue> kinds; // One representative descriptor per kind.
};

class ImageHeapSource {
public:
	// False when the host does not keep the table resident.
	virtual bool Acquire(const ImageHeapRequest& request, ImageHeapView& view) = 0;

protected:
	~ImageHeapSource() = default;
};

// A direct-mapped cache of guest memory in 128-byte chunks that read_memory keeps: a DWORD read
// whose chunk address equals the tag of entry (address / ChunkBytes) & index_mask may load the
// DWORD from that entry, at data + index * stride + address % ChunkBytes, instead of calling
// read_memory, with the same result. Only read_memory calls change the tags and the entries.
struct SrtReadCacheView {
	static constexpr uint64_t ChunkBytes = 128;
	const uint64_t*           tags       = nullptr;
	const uint8_t*            data       = nullptr;
	uint32_t                  stride     = 0;
	uint32_t                  index_mask = 0;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	// Guest memory for the loads of a walker compiled with SrtWalkerLoads::Reader: the walk can
	// run on a thread that must not fault into the caches, or over a snapshot. A refused read
	// leaves its value unknown.
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	ImageHeapSource*          image_heap            = nullptr;
	// Optional: the cache read_memory reads through. A reader load whose chunk it holds loads the
	// DWORD from it instead of calling read_memory.
	const SrtReadCacheView* read_cache = nullptr;
	// A walk ahead of the commands that may still write its guest data (ResourceLookahead):
	// MaterializeResources reports none of its failures, the caller counts them.
	bool speculative = false;
};

// How walker loads read guest memory: in place (a fault completes the load as a zero read), or
// through SrtRuntime::read_memory.
enum class SrtWalkerLoads : uint8_t { Direct, Reader };

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadMemory(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uniform across a compute wave, including values derived from its workgroup ID.
// This does not imply that the host can evaluate the value.
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
// Compiles the plan's x86 walker. Without values it walks the plan's resources; otherwise it
// only evaluates the values, with every block reachable.
void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values = {},
                      SrtWalkerLoads loads = SrtWalkerLoads::Direct);
// Runs the walker. flat receives the scalar reads of reachable blocks and reads, when non-null,
// one DWORD range per guest load. False when a reachable read is unknown. The results go to
// scratch when it is not empty (at least plan.srt_scratch.size() words), else to the plan's
// scratch: threads with their own scratch can walk one plan at the same time.
bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads, std::span<uint64_t> scratch = {});
// Results of the last walk with that scratch; false when unknown.
bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result,
                 std::span<const uint64_t> scratch = {});
bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result,
                      std::span<const uint64_t> scratch = {});
bool WalkedActive(const ResourcePlan& plan, uint32_t source,
                  std::span<const uint64_t> scratch = {});
// Length of the guest load at rip if it belongs to the walker running on this thread, else 0.
// The exception handler completes such a load as a zero read.
uint32_t SrtWalkerLoadLength(uint64_t rip);

// Nonnegative affine coefficients: element zero is constant, remaining elements are
// caller-selected leaves. Runtime range proofs must additionally exclude U32 wrap.
template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
