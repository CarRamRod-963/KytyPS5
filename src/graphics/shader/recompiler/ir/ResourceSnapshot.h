#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

// A bindless image root: the records (a V#) whose 64-bit pointer at pointer_offset leads to a
// T# at pointer + table_offset + layer * layer_stride. The renderer gives the heap a region of
// the translation table and patches [region, entry count] into the flattened SRT at
// mapping_offset; the shader looks key = record * layer_count + layer up there.
struct BindlessUse {
	// pointer_offset of a table whose records hold the T# itself (at base + key * record_stride +
	// table_offset) instead of a pointer to it.
	static constexpr uint32_t NoPointer = UINT32_MAX;

	uint32_t resource       = 0;
	uint64_t base           = 0;
	uint64_t size           = 0;
	uint32_t record_stride  = 0;
	uint32_t pointer_offset = 0;
	uint32_t table_offset   = 0;
	uint32_t layer_stride   = 0;
	uint32_t layer_count    = 1;
	uint32_t mapping_offset = 0;

	bool operator==(const BindlessUse&) const = default;
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<BindlessUse>     bindless_uses;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	// Set when a refresh fails only because of this draw's guest data: a GPU-selected candidate
	// set reads memory that is not mapped or needs more probes than the host enumerates. The
	// shader is supported; the draw that supplied the data cannot be specialized.
	bool unavailable = false;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
