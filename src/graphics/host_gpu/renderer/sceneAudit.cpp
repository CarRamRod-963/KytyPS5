#include "graphics/host_gpu/renderer/sceneAudit.h"
#include "common/envDefaults.h"
#include "common/envRead.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics::SceneAudit {

namespace {

// Image slot index, slot generation, mip, aspect (AspectColor / AspectDepth / AspectStencil).
using ImageKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
// Guest address, size.
using RangeKey = std::pair<uint64_t, uint64_t>;

constexpr uint32_t AspectColor   = 0;
constexpr uint32_t AspectDepth   = 1;
constexpr uint32_t AspectStencil = 2;

// Positions in the frame: a copy taken before command n is at 2n, the writes of command n at 2n + 1; the copies
// taken when the frame is over come after every write.
constexpr int64_t  FrameEndTime = std::numeric_limits<int64_t>::max() / 2;
constexpr uint64_t ChunkBytes   = 256ull << 20u;

// How the elements of a copy are read: texels (uncompressed formats, one entry per component), blocks
// (compressed formats: zero/constant counts only) or raw dwords (buffers).
struct Layout {
	uint32_t                elem_bytes = 0;
	bool                    block      = false;
	bool                    packed     = false; // components share one little-endian word
	uint32_t                channels   = 0;
	std::array<uint32_t, 4> bits {};
	std::array<uint32_t, 4> offset {}; // packed: bit position, otherwise byte offset
	// 'u' unorm/srgb, 's' snorm, 'U' uint, 'S' sint, 'f' float, 'F' unsigned small float, 'r' raw
	std::array<char, 4>     numeric {};
	std::string             names;
};

struct Chunk {
	vk::Buffer    buffer;
	VmaAllocation allocation = nullptr;
	uint8_t*      mapped     = nullptr;
	uint64_t      size       = 0;
	uint64_t      used       = 0;
};

struct PendingSnapshot {
	int32_t     id       = 0;
	uint32_t    chunk    = 0;
	uint64_t    offset   = 0;
	uint64_t    bytes    = 0;
	uint64_t    elements = 0;
	Layout      layout;
	std::string head; // JSON members naming the source
};

struct ImageVersion {
	Image*   image      = nullptr; // checked against the cache before the frame-end copy
	uint64_t address    = 0;
	uint64_t serial     = 0;       // Image::content_serial when copied
	int32_t  snap       = -1;
	int64_t  snap_time  = -1;
	int64_t  last_write = -1;
	uint32_t taken      = 0;       // copies before reads in this frame (KYTY_SCENE_AUDIT_REPEAT)
};

struct BufferVersion {
	int32_t  snap      = -1;
	int64_t  snap_time = -1;
	uint32_t taken     = 0;
};

struct WriteMark {
	bool     buffer = false;
	ImageKey key {};
	Image*   image = nullptr;
	RangeKey range {};
};

struct Stats {
	uint64_t                zero     = 0;
	uint64_t                eq_first = 0;
	uint64_t                nan      = 0;
	uint64_t                inf      = 0;
	std::array<double, 4>   min {};
	std::array<double, 4>   max {};
	std::array<double, 4>   sum {};
	std::array<uint64_t, 4> finite {};
	std::string             first;
};

struct AuditState {
	uint32_t    frame   = ConfiguredFrame();
	bool        started = false;
	bool        done    = false;
	std::string path;
	FILE*       file = nullptr;
	std::chrono::steady_clock::time_point start_time;

	uint64_t batch_limit   = 0;
	uint64_t total_limit   = 0;
	uint64_t single_limit  = 0;
	uint64_t buffer_limit  = 0;
	uint32_t repeat_limit  = 0;
	bool     static_images = true;
	bool     buffers       = true;

	uint32_t                        next_seq    = 0;
	bool                            has_pending = false;
	uint32_t                        pending_seq = 0;
	std::string                     pending_body;
	std::string                     pending_skip;
	std::map<std::string, uint32_t> pending_rejects;
	std::map<std::string, uint32_t> held_rejects;  // before the record of the command they belong to
	std::map<std::string, uint32_t> frame_rejects;

	std::vector<Chunk>           chunks;
	std::vector<PendingSnapshot> snapshots;
	int32_t                      next_snapshot = 0;
	uint64_t                     batch_bytes   = 0;
	uint64_t                     total_bytes   = 0;
	uint32_t                     flushes       = 0;
	uint32_t                     analyzed      = 0;
	uint32_t                     not_taken     = 0;
	uint32_t                     frame_end_gone = 0;
	bool                         copies         = false; // copies recorded since the last closing barrier
	bool                         buffer_barrier = false; // the opening barrier of buffer copies is recorded

	std::map<ImageKey, ImageVersion>  images;
	std::map<RangeKey, BufferVersion> buffer_versions;
	std::map<RangeKey, int64_t>       buffer_writes; // last write time per written range
};

AuditState& Audit() {
	static AuditState state;
	return state;
}

uint64_t EnvMegabytes(const char* name, uint64_t fallback) {
	const char*    value = std::getenv(name);
	const uint64_t mb    = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : fallback;
	return mb << 20u;
}

bool EnvOn(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return fallback;
	return value[0] != '0';
}

void AppendString(std::string& out, std::string_view text) {
	out += '"';
	for (const char ch: text) {
		const auto c = static_cast<unsigned char>(ch);
		if (c == '"' || c == '\\') {
			out += '\\';
			out += ch;
		} else if (c < 0x20u) {
			out += fmt::format("\\u{:04x}", static_cast<uint32_t>(c));
		} else {
			out += ch;
		}
	}
	out += '"';
}

std::string JsonString(std::string_view text) {
	std::string out;
	AppendString(out, text);
	return out;
}

const char* JsonBool(bool value) {
	return value ? "true" : "false";
}

std::string JsonNumber(double value) {
	return std::isfinite(value) ? fmt::format("{:.6g}", value) : std::string("null");
}

std::string Join(const std::vector<std::string>& items) {
	std::string out;
	for (size_t i = 0; i < items.size(); i++) {
		if (i != 0) out += ',';
		out += items[i];
	}
	return out;
}

std::string KeyText(const ImageKey& key) {
	return fmt::format("\"{}:{}:{}:{}\"", std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
}

const char* AspectName(uint32_t aspect) {
	return aspect == AspectDepth ? "depth" : aspect == AspectStencil ? "stencil" : "color";
}

const char* StageName(ShaderType type) {
	switch (type) {
		case ShaderType::Vertex: return "vs";
		case ShaderType::Pixel: return "ps";
		case ShaderType::Fetch: return "fs";
		case ShaderType::Compute: return "cs";
		case ShaderType::Mesh: return "ms";
		case ShaderType::Local: return "ls";
		case ShaderType::TessellationControl: return "hs";
		case ShaderType::TessellationEvaluation: return "ds";
		default: return "??";
	}
}

// The whole debug-marker stack of the current command stream ("Frame>Deferred>SSR").
std::string MarkerPath() {
	std::scoped_lock lock(DebugMarkerMutex());
	const auto&      stack = DebugMarkerStack();
	if (stack.empty()) return "-";
	std::string path;
	for (const auto& name: stack) {
		if (!path.empty()) path += '>';
		path += name;
	}
	return path;
}

// Descriptor dwords without DecodeNativeDescriptor's size check (R128 images carry 4 dwords).
template <size_t N>
bool CopyDwords(const ShaderRecompiler::IR::DescriptorValue& value, uint32_t (&out)[N]) {
	bool       zero  = true;
	const auto count = std::min<size_t>({static_cast<size_t>(value.dword_count), value.dwords.size(), N});
	for (size_t k = 0; k < count; k++) {
		out[k] = value.dwords[k];
		zero &= value.dwords[k] == 0u;
	}
	return zero;
}

Layout MakeLayout(vk::Format format) {
	Layout layout;
	switch (format) {
		case vk::Format::eUndefined: return layout;
		case vk::Format::eD16Unorm:
			layout.elem_bytes = 2;
			layout.channels   = 1;
			layout.bits[0]    = 16;
			layout.numeric[0] = 'u';
			layout.names      = "D";
			return layout;
		case vk::Format::eX8D24UnormPack32:
			layout.elem_bytes = 4;
			layout.packed     = true;
			layout.channels   = 1;
			layout.bits[0]    = 24;
			layout.offset[0]  = 0;
			layout.numeric[0] = 'u';
			layout.names      = "D";
			return layout;
		case vk::Format::eD32Sfloat:
			layout.elem_bytes = 4;
			layout.channels   = 1;
			layout.bits[0]    = 32;
			layout.numeric[0] = 'f';
			layout.names      = "D";
			return layout;
		case vk::Format::eS8Uint:
			layout.elem_bytes = 1;
			layout.channels   = 1;
			layout.bits[0]    = 8;
			layout.numeric[0] = 'U';
			layout.names      = "S";
			return layout;
		case vk::Format::eE5B9G9R9UfloatPack32:
			// Shared exponent: zero/constant counts only.
			layout.elem_bytes = 4;
			return layout;
		default: break;
	}
	layout.elem_bytes = vk::blockSize(format);
	if (layout.elem_bytes == 0u) return layout;
	if (vk::isCompressed(format)) {
		layout.block = true;
		return layout;
	}
	if (vk::planeCount(format) > 1u) {
		layout.elem_bytes = 0;
		return layout;
	}
	const uint32_t count  = std::min<uint32_t>(vk::componentCount(format), 4u);
	const uint32_t packed = vk::packed(format);
	uint32_t       total  = 0;
	for (uint32_t c = 0; c < count; c++) {
		const uint32_t bits = vk::componentBits(format, static_cast<uint8_t>(c));
		if (bits == 0u || bits > 64u || (packed == 0u && bits % 8u != 0u)) return layout;
		total += bits;
	}
	if ((packed != 0u && total > packed) || (packed == 0u && total / 8u > layout.elem_bytes)) return layout;
	layout.packed     = packed != 0u;
	uint32_t position = layout.packed ? packed : 0u;
	for (uint32_t c = 0; c < count; c++) {
		const uint32_t bits = vk::componentBits(format, static_cast<uint8_t>(c));
		layout.bits[c]      = bits;
		if (layout.packed) {
			// Packed formats name their components from the most significant bits down.
			position -= bits;
			layout.offset[c] = position;
		} else {
			layout.offset[c] = position / 8u;
			position += bits;
		}
		const std::string_view numeric = vk::componentNumericFormat(format, static_cast<uint8_t>(c));
		layout.numeric[c]              = numeric == "UNORM" || numeric == "SRGB"     ? 'u'
		                                 : numeric == "SNORM"                        ? 's'
		                                 : numeric == "UINT" || numeric == "USCALED" ? 'U'
		                                 : numeric == "SINT" || numeric == "SSCALED" ? 'S'
		                                 : numeric == "SFLOAT"                       ? 'f'
		                                 : numeric == "UFLOAT"                       ? 'F'
		                                                                             : 'r';
		const char* name = vk::componentName(format, static_cast<uint8_t>(c));
		layout.names += name != nullptr && name[0] != '\0' ? name[0] : '?';
	}
	layout.channels = count;
	return layout;
}

float HalfToFloat(uint32_t half) {
	const uint32_t sign     = (half >> 15u) & 1u;
	const uint32_t exponent = (half >> 10u) & 0x1fu;
	const uint32_t mantissa = half & 0x3ffu;
	float          value    = 0.0f;
	if (exponent == 0u) {
		value = std::ldexp(static_cast<float>(mantissa), -24);
	} else if (exponent == 31u) {
		value = mantissa != 0u ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
	} else {
		value = std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
	}
	return sign != 0u ? -value : value;
}

const std::vector<float>& HalfTable() {
	static const std::vector<float> table = [] {
		std::vector<float> result(65536u);
		for (uint32_t i = 0; i < 65536u; i++) result[i] = HalfToFloat(i);
		return result;
	}();
	return table;
}

// 11- and 10-bit unsigned floats (5-bit exponent).
float SmallUfloat(uint64_t value, uint32_t mantissa_bits) {
	const auto exponent = static_cast<uint32_t>(value >> mantissa_bits) & 0x1fu;
	const auto mantissa = static_cast<uint32_t>(value) & ((1u << mantissa_bits) - 1u);
	if (exponent == 0u) return std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissa_bits));
	if (exponent == 31u) {
		return mantissa != 0u ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
	}
	return std::ldexp(static_cast<float>(mantissa | (1u << mantissa_bits)),
	                  static_cast<int>(exponent) - 15 - static_cast<int>(mantissa_bits));
}

int64_t SignExtend(uint64_t value, uint32_t bits) {
	if (bits == 0u || bits >= 64u) return static_cast<int64_t>(value);
	const uint64_t sign = uint64_t {1} << (bits - 1u);
	return static_cast<int64_t>((value ^ sign) - sign);
}

double Component(const uint8_t* element, const Layout& layout, uint32_t c) {
	const uint32_t bits = layout.bits[c];
	const uint64_t mask = bits >= 64u ? ~uint64_t {0} : (uint64_t {1} << bits) - 1u;
	uint64_t       raw  = 0;
	if (layout.packed) {
		uint64_t word = 0;
		std::memcpy(&word, element, std::min<uint32_t>(layout.elem_bytes, 8u));
		raw = (word >> layout.offset[c]) & mask;
	} else {
		std::memcpy(&raw, element + layout.offset[c], std::min<uint32_t>(bits / 8u, 8u));
	}
	switch (layout.numeric[c]) {
		case 'u': return static_cast<double>(raw) / static_cast<double>(mask);
		case 's':
			return std::max(static_cast<double>(SignExtend(raw, bits)) / static_cast<double>(mask >> 1u), -1.0);
		case 'S': return static_cast<double>(SignExtend(raw, bits));
		case 'f':
			if (bits == 16u) return HalfTable()[raw & 0xffffu];
			if (bits == 32u) {
				const auto word  = static_cast<uint32_t>(raw);
				float      value = 0.0f;
				std::memcpy(&value, &word, sizeof(value));
				return value;
			}
			if (bits == 64u) {
				double value = 0.0;
				std::memcpy(&value, &raw, sizeof(value));
				return value;
			}
			return static_cast<double>(raw);
		case 'F':
			if (bits == 11u) return SmallUfloat(raw, 6);
			if (bits == 10u) return SmallUfloat(raw, 5);
			return static_cast<double>(raw);
		default: return static_cast<double>(raw);
	}
}

// Full-content statistics of elements whose layout is `layout` (every texel, block or dword is looked at).
Stats Analyze(const uint8_t* data, uint64_t elements, const Layout& layout) {
	Stats          stats;
	const uint32_t size = layout.elem_bytes;
	if (data == nullptr || elements == 0u || size == 0u) return stats;
	for (uint32_t k = 0; k < std::min(size, 16u); k++) stats.first += fmt::format("{:02x}", static_cast<uint32_t>(data[k]));
	static const std::array<uint8_t, 64> zeros {};
	for (uint64_t i = 0; i < elements; i++) {
		const uint8_t* element = data + i * size;
		bool           zero    = true;
		if (size <= zeros.size()) {
			zero = std::memcmp(element, zeros.data(), size) == 0;
		} else {
			for (uint32_t k = 0; k < size && zero; k++) zero = element[k] == 0u;
		}
		stats.zero += zero ? 1u : 0u;
		stats.eq_first += std::memcmp(element, data, size) == 0 ? 1u : 0u;
	}
	if (layout.block || layout.channels == 0u) return stats;
	stats.min.fill(std::numeric_limits<double>::infinity());
	stats.max.fill(-std::numeric_limits<double>::infinity());
	// A constant image decodes one element and counts it for all of them.
	const bool     constant = stats.eq_first == elements;
	const uint64_t decoded  = constant ? 1u : elements;
	for (uint64_t i = 0; i < decoded; i++) {
		const uint8_t* element = data + i * size;
		for (uint32_t c = 0; c < layout.channels; c++) {
			const double value = Component(element, layout, c);
			if (std::isnan(value)) {
				stats.nan++;
			} else if (std::isinf(value)) {
				stats.inf++;
			} else {
				stats.min[c] = std::min(stats.min[c], value);
				stats.max[c] = std::max(stats.max[c], value);
				stats.sum[c] += value;
				stats.finite[c]++;
			}
		}
	}
	if (constant && elements > 1u) {
		stats.nan *= elements;
		stats.inf *= elements;
		for (uint32_t c = 0; c < layout.channels; c++) {
			stats.sum[c] *= static_cast<double>(elements);
			stats.finite[c] *= elements;
		}
	}
	return stats;
}

void WriteLine(const std::string& line) {
	auto& audit = Audit();
	if (audit.file != nullptr) std::fwrite(line.data(), 1, line.size(), audit.file);
}

void WriteSnapshot(const PendingSnapshot& snapshot, const Stats& stats) {
	std::string line = fmt::format("{{\"t\":\"snap\",\"id\":{},{},\"bytes\":{},\"elems\":{},\"zero\":{},\"eq_first\":{}",
	                               snapshot.id, snapshot.head, snapshot.bytes, snapshot.elements, stats.zero,
	                               stats.eq_first);
	if (!stats.first.empty()) line += ",\"first\":\"" + stats.first + "\"";
	const auto& layout = snapshot.layout;
	if (!layout.block && layout.channels != 0u) {
		std::vector<std::string> lows, highs, means;
		for (uint32_t c = 0; c < layout.channels; c++) {
			const bool any = stats.finite[c] != 0u;
			lows.push_back(any ? JsonNumber(stats.min[c]) : std::string("null"));
			highs.push_back(any ? JsonNumber(stats.max[c]) : std::string("null"));
			means.push_back(any ? JsonNumber(stats.sum[c] / static_cast<double>(stats.finite[c])) : std::string("null"));
		}
		line += ",\"ch\":" + JsonString(layout.names) + ",\"min\":[" + Join(lows) + "],\"max\":[" + Join(highs) +
		        "],\"mean\":[" + Join(means) + "]";
		line += fmt::format(",\"nan\":{},\"inf\":{}", stats.nan, stats.inf);
	}
	line += "}\n";
	WriteLine(line);
}

bool Allocate(GraphicContext& graphics, uint64_t bytes, uint64_t alignment, uint32_t& chunk, uint64_t& offset) {
	auto& audit = Audit();
	if (!audit.chunks.empty()) {
		auto&          last  = audit.chunks.back();
		const uint64_t start = (last.used + alignment - 1u) / alignment * alignment;
		if (start + bytes <= last.size) {
			last.used = start + bytes;
			chunk     = static_cast<uint32_t>(audit.chunks.size() - 1u);
			offset    = start;
			return true;
		}
	}
	vk::BufferCreateInfo info {};
	info.size  = std::max(bytes, ChunkBytes);
	info.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationInfo result {};
	VkBuffer          native = VK_NULL_HANDLE;
	Chunk             created;
	if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&info), &allocation_info,
	                    &native, &created.allocation, &result) != VK_SUCCESS) {
		return false;
	}
	created.buffer = native;
	created.mapped = static_cast<uint8_t*>(result.pMappedData);
	created.size   = info.size;
	created.used   = bytes;
	if (created.mapped == nullptr) {
		vmaDestroyBuffer(graphics.allocator, created.buffer, created.allocation);
		return false;
	}
	audit.chunks.push_back(created);
	chunk  = static_cast<uint32_t>(audit.chunks.size() - 1u);
	offset = 0;
	return true;
}

void ReleaseChunks(GraphicContext& graphics) {
	auto& audit = Audit();
	for (auto& chunk: audit.chunks) vmaDestroyBuffer(graphics.allocator, chunk.buffer, chunk.allocation);
	audit.chunks.clear();
}

// After the copies of one command: the next commands may write what was copied (write-after-read), and the host
// reads the copies after the scheduler finished.
void CloseCopies(RenderContext& context) {
	auto& audit = Audit();
	if (!audit.copies) return;
	audit.copies = false;
	auto& scheduler = context.GetCommandScheduler();
	scheduler.EndRendering();
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask =
	    vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                             vk::PipelineStageFlagBits::eAllCommands |
	                                                 vk::PipelineStageFlagBits::eHost,
	                                             {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

void QueueSnapshot(const Layout& layout, uint32_t chunk, uint64_t offset, uint64_t bytes, uint64_t elements,
                   std::string&& head) {
	auto&           audit = Audit();
	PendingSnapshot snapshot;
	snapshot.id       = audit.next_snapshot++;
	snapshot.chunk    = chunk;
	snapshot.offset   = offset;
	snapshot.bytes    = bytes;
	snapshot.elements = elements;
	snapshot.layout   = layout;
	snapshot.head     = std::move(head);
	audit.batch_bytes += bytes;
	audit.total_bytes += bytes;
	audit.snapshots.push_back(std::move(snapshot));
}

// The whole mip `key` names (every layer, every slice of a volume), as the GPU holds it now.
int32_t SnapshotImage(RenderContext& context, Image& image, const ImageKey& key, int64_t seq, std::string& reason) {
	auto&       audit   = Audit();
	const auto& backing = image.backing;
	if (backing.image == nullptr) {
		reason = "no_backing";
		return -1;
	}
	if (backing.samples != 1u) {
		reason = "msaa";
		return -1;
	}
	if (audit.total_bytes >= audit.total_limit) {
		reason = "budget";
		audit.not_taken++;
		return -1;
	}
	const uint32_t aspect      = std::get<3>(key);
	const uint32_t mip         = std::min(std::get<2>(key), std::max(backing.mip_levels, 1u) - 1u);
	const bool     has_depth   = vk::hasDepthComponent(backing.format);
	const bool     has_stencil = vk::hasStencilComponent(backing.format);
	vk::Format     copy_format = backing.format;
	bool           color       = true;
	auto           copy_aspect = vk::ImageAspectFlags {vk::ImageAspectFlagBits::eColor};
	if (has_depth || has_stencil) {
		color = false;
		if ((aspect == AspectStencil && has_stencil) || !has_depth) {
			copy_format = vk::Format::eS8Uint;
			copy_aspect = vk::ImageAspectFlagBits::eStencil;
		} else {
			copy_aspect = vk::ImageAspectFlagBits::eDepth;
			switch (backing.format) {
				case vk::Format::eD16Unorm:
				case vk::Format::eD16UnormS8Uint: copy_format = vk::Format::eD16Unorm; break;
				case vk::Format::eD24UnormS8Uint:
				case vk::Format::eX8D24UnormPack32: copy_format = vk::Format::eX8D24UnormPack32; break;
				default: copy_format = vk::Format::eD32Sfloat; break;
			}
		}
	}
	const Layout layout = MakeLayout(copy_format);
	if (layout.elem_bytes == 0u) {
		reason = "format";
		return -1;
	}
	std::array<uint8_t, 3> block {1, 1, 1};
	if (color) block = vk::blockExtent(copy_format);
	const bool     volume   = backing.image_type == vk::ImageType::e3D;
	const uint32_t width    = std::max(backing.extent.width >> mip, 1u);
	const uint32_t height   = std::max(backing.extent.height >> mip, 1u);
	const uint32_t depth    = volume ? std::max(backing.extent.depth >> mip, 1u) : 1u;
	const uint32_t layers   = volume ? 1u : std::max(backing.layers, 1u);
	const uint32_t columns  = (width + block[0] - 1u) / std::max<uint32_t>(block[0], 1u);
	const uint32_t rows     = (height + block[1] - 1u) / std::max<uint32_t>(block[1], 1u);
	const uint64_t elements = uint64_t {columns} * rows * depth * layers;
	const uint64_t bytes    = elements * layout.elem_bytes;
	if (bytes == 0u) {
		reason = "empty";
		return -1;
	}
	if (bytes > audit.single_limit) {
		reason = "too_big";
		audit.not_taken++;
		return -1;
	}
	uint32_t chunk  = 0;
	uint64_t offset = 0;
	if (!Allocate(context.GetGraphics(), bytes, std::lcm<uint64_t>(layout.elem_bytes, 4u), chunk, offset)) {
		reason = "alloc";
		audit.not_taken++;
		return -1;
	}
	auto& scheduler = context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	vk::BufferImageCopy region {};
	region.bufferOffset     = offset;
	region.imageSubresource = vk::ImageSubresourceLayers {copy_aspect, mip, 0, layers};
	region.imageExtent      = vk::Extent3D {width, height, depth};
	command.copyImageToBuffer(backing.image, vk::ImageLayout::eTransferSrcOptimal, audit.chunks[chunk].buffer, 1,
	                          &region);
	image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {},
	              command);
	audit.copies = true;
	QueueSnapshot(layout, chunk, offset, bytes, elements,
	              fmt::format("\"seq\":{},\"kind\":\"img\",\"key\":{},\"addr\":\"0x{:x}\",\"fmt\":\"{}\",\"ifmt\":\"{}\","
	                          "\"ext\":[{},{},{}],\"layers\":{},\"mip\":{},\"aspect\":\"{}\"",
	                          seq, KeyText(key), image.info.data.address, vk::to_string(copy_format),
	                          vk::to_string(backing.format), width, height, depth, layers, mip, AspectName(aspect)));
	return audit.next_snapshot - 1;
}

int32_t SnapshotBuffer(RenderContext& context, vk::Buffer source, uint64_t source_offset, uint64_t address,
                       uint64_t size, int64_t seq, std::string& reason) {
	auto& audit = Audit();
	if (!audit.buffers) {
		reason = "off";
		return -1;
	}
	if (!source) {
		reason = "no_buffer";
		return -1;
	}
	if (audit.total_bytes >= audit.total_limit) {
		reason = "budget";
		audit.not_taken++;
		return -1;
	}
	const bool     partial = size > audit.buffer_limit;
	const uint64_t bytes   = std::min(size, audit.buffer_limit) & ~uint64_t {3};
	if (bytes == 0u) {
		reason = "small";
		return -1;
	}
	uint32_t chunk  = 0;
	uint64_t offset = 0;
	if (!Allocate(context.GetGraphics(), bytes, 4u, chunk, offset)) {
		reason = "alloc";
		audit.not_taken++;
		return -1;
	}
	auto& scheduler = context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	if (!audit.buffer_barrier) {
		// Earlier shader and transfer writes land before the copies (read-after-write).
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
		                        &barrier, 0, nullptr, 0, nullptr);
		audit.buffer_barrier = true;
	}
	const vk::BufferCopy copy {source_offset, offset, bytes};
	command.copyBuffer(source, audit.chunks[chunk].buffer, 1, &copy);
	audit.copies = true;
	Layout layout;
	layout.elem_bytes = 4;
	QueueSnapshot(layout, chunk, offset, bytes, bytes / 4u,
	              fmt::format("\"seq\":{},\"kind\":\"buf\",\"key\":\"{:x}:{:x}\",\"addr\":\"0x{:x}\",\"size\":{},"
	                          "\"partial\":{}",
	                          seq, address, size, address, size, JsonBool(partial)));
	return audit.next_snapshot - 1;
}

// The copy a read sees: the last one, unless the image was written since (content serial or a recorded write).
int32_t ReadImage(RenderContext& context, Image& image, const ImageKey& key, std::string& reason) {
	auto& audit   = Audit();
	auto& version = audit.images[key];
	version.image   = &image;
	version.address = image.info.data.address;
	const bool fresh = version.snap < 0 || version.serial != image.content_serial || version.last_write > version.snap_time;
	if (!fresh) return version.snap;
	if (!audit.static_images && image.content_serial == 0u && version.last_write < 0) {
		reason = "static_off";
		return -1;
	}
	// An image rewritten and read again and again (ping-pong blurs, iterated simulations): the first copies only.
	if (version.taken >= audit.repeat_limit) {
		reason = "repeat";
		return -1;
	}
	const int32_t snap = SnapshotImage(context, image, key, audit.pending_seq, reason);
	if (snap >= 0) {
		version.snap      = snap;
		version.snap_time = int64_t {audit.pending_seq} * 2;
		version.serial    = image.content_serial;
		version.taken++;
	}
	return snap;
}

bool WrittenSince(uint64_t address, uint64_t size, int64_t time) {
	for (const auto& [range, write_time]: Audit().buffer_writes) {
		if (write_time > time && range.first < address + size && address < range.first + range.second) return true;
	}
	return false;
}

int32_t ReadBuffer(RenderContext& context, vk::Buffer source, uint64_t source_offset, uint64_t address, uint64_t size,
                   std::string& reason) {
	auto& audit   = Audit();
	auto& version = audit.buffer_versions[{address, size}];
	if (version.snap >= 0 && !WrittenSince(address, size, version.snap_time)) return version.snap;
	if (version.taken >= audit.repeat_limit) {
		reason = "repeat";
		return -1;
	}
	const int32_t snap = SnapshotBuffer(context, source, source_offset, address, size, audit.pending_seq, reason);
	if (snap >= 0) {
		version.snap      = snap;
		version.snap_time = int64_t {audit.pending_seq} * 2;
		version.taken++;
	}
	return snap;
}

void ApplyWrites(const std::vector<WriteMark>& writes) {
	auto&         audit = Audit();
	const int64_t time  = int64_t {audit.pending_seq} * 2 + 1;
	for (const auto& write: writes) {
		if (write.buffer) {
			audit.buffer_writes[write.range] = time;
			continue;
		}
		auto& version      = audit.images[write.key];
		version.image      = write.image;
		version.address    = write.image->info.data.address;
		version.last_write = time;
	}
}

void Start() {
	auto& audit = Audit();
	if (audit.started || audit.done) return;
	audit.started       = true;
	audit.start_time    = std::chrono::steady_clock::now();
	audit.batch_limit   = EnvMegabytes("KYTY_SCENE_AUDIT_BATCH_MB", 1024);
	audit.total_limit   = EnvMegabytes("KYTY_SCENE_AUDIT_TOTAL_MB", 16384);
	audit.single_limit  = EnvMegabytes("KYTY_SCENE_AUDIT_MAX_MB", 512);
	audit.buffer_limit  = std::max<uint64_t>(EnvMegabytes("KYTY_SCENE_AUDIT_BUFFER_MB", 64), 4u);
	audit.repeat_limit  = [] {
		const char* value = std::getenv("KYTY_SCENE_AUDIT_REPEAT");
		return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 16u;
	}();
	audit.static_images = EnvOn("KYTY_SCENE_AUDIT_STATIC", true);
	audit.buffers       = EnvOn("KYTY_SCENE_AUDIT_BUFFERS", true);
	const char* dir     = Common::EnvRead::FrameDir();
	if (dir == nullptr) {
		audit.done = true;
		return;
	}
	std::error_code error;
	std::filesystem::create_directories(dir, error);
	audit.path = fmt::format("{}/scene_audit_f{}.jsonl", dir, audit.frame);
	audit.file = std::fopen(audit.path.c_str(), "wb");
	if (audit.file == nullptr) {
		std::fprintf(stderr, "SCENE_AUDIT cannot create %s\n", audit.path.c_str());
		audit.done = true;
		return;
	}
	// The switches that change what a frame shows (the report lists them).
	static const char* const switches[] = {
	    "KYTY_DIAG_NULL_TEXTURE_GRAY",          "KYTY_DIAG_NULL_TEXTURE_COLOR",
	    "KYTY_DIAG_SKIP_CS_HASHES",             "KYTY_DIAG_SKIP_UNSUPPORTED_CS",
	    "KYTY_DIAG_SKIP_PS_HASHES",             "KYTY_DIAG_SKIP_VS_HASHES",
	    "KYTY_DIAG_SKIP_DRAW_MARKERS",          "KYTY_DIAG_FILL_CS_HASHES",
	    "KYTY_DIAG_TEXTURE_CLAMP_LAST_LEVEL",   "KYTY_DIAG_TEXTURE_ALLOW_MISALIGNED",
	    "KYTY_DIAG_STENCIL_STORAGE_SHADOW",     "KYTY_DIAG_STENCIL_ALIAS_COPY",
	    "KYTY_DIAG_DCC_FAST_CLEAR",
	    "KYTY_DIAG_HTILE_RMW_CLEAR",            "KYTY_SHADER_LOOP_LIMIT_HASHES",
	    "KYTY_BINDLESS",                        "KYTY_PERF_BINDLESS_CLASSES",
	};
	std::vector<std::string> env;
	for (const char* name: switches) {
		// Effective values: the code defaults (Common::EnvDefaults) are listed too.
		if (const char* value = Common::EnvOrDefault(name); value != nullptr) {
			env.push_back(JsonString(name) + ":" + JsonString(value));
		}
	}
	WriteLine(fmt::format("{{\"t\":\"audit\",\"version\":1,\"frame\":{},\"file\":{},\"env\":{{{}}}}}\n", audit.frame,
	                      JsonString(audit.path), Join(env)));
	std::fprintf(stderr, "SCENE_AUDIT start frame=%u file=%s\n", audit.frame, audit.path.c_str());
	std::fflush(stderr);
}

// True while the audited frame is being recorded (the first event of the frame opens the file).
bool Recording() {
	if (!Enabled()) return false;
	auto& audit = Audit();
	if (audit.done || DebugPresentCount().load() != audit.frame) return false;
	if (!audit.started) Start();
	return !audit.done;
}

std::string Counts(const std::map<std::string, uint32_t>& counts) {
	std::vector<std::string> items;
	for (const auto& [name, count]: counts) items.push_back(JsonString(name) + ":" + std::to_string(count));
	return "{" + Join(items) + "}";
}

void WritePending() {
	auto& audit = Audit();
	if (!audit.has_pending) return;
	audit.has_pending = false;
	std::string line = "{\"t\":\"cmd\"," + audit.pending_body;
	line += ",\"skip\":" + JsonString(audit.pending_skip);
	if (!audit.pending_rejects.empty()) line += ",\"bindless_rejects\":" + Counts(audit.pending_rejects);
	line += "}\n";
	WriteLine(line);
	audit.pending_body.clear();
	audit.pending_skip.clear();
	audit.pending_rejects.clear();
}

void BeginRecord() {
	auto& audit = Audit();
	WritePending();
	audit.has_pending     = true;
	audit.pending_seq     = audit.next_seq++;
	audit.pending_rejects = std::move(audit.held_rejects);
	audit.held_rejects.clear();
	audit.copies         = false;
	audit.buffer_barrier = false;
}

// Submits what was recorded, waits, analyzes the pending copies and frees their memory.
void Flush(RenderContext& context) {
	auto& audit    = Audit();
	auto& graphics = context.GetGraphics();
	if (!audit.snapshots.empty()) {
		auto& scheduler = context.GetCommandScheduler();
		scheduler.EndRendering();
		scheduler.Finish();
		for (auto& chunk: audit.chunks) (void)vmaInvalidateAllocation(graphics.allocator, chunk.allocation, 0, VK_WHOLE_SIZE);
		for (const auto& snapshot: audit.snapshots) {
			const auto& chunk = audit.chunks[snapshot.chunk];
			WriteSnapshot(snapshot, Analyze(chunk.mapped + snapshot.offset, snapshot.elements, snapshot.layout));
			audit.analyzed++;
		}
		audit.snapshots.clear();
		audit.flushes++;
	}
	ReleaseChunks(graphics);
	audit.batch_bytes = 0;
	if (audit.file != nullptr) std::fflush(audit.file);
}

// Outputs written after their last copy (nobody read them again in the frame), and written buffer ranges.
void SnapshotFrameEnd(RenderContext& context) {
	auto& audit          = Audit();
	audit.copies         = false;
	audit.buffer_barrier = false;
	auto& cache          = context.GetTextureCache();
	for (auto& [key, version]: audit.images) {
		if (version.last_write < 0 || version.last_write <= version.snap_time || version.image == nullptr ||
		    version.address == 0u) {
			continue;
		}
		// The image may have been replaced since its write: only a cached image of the same address is copied.
		const auto alive = cache.DebugImagesAt(version.address);
		if (std::find(alive.begin(), alive.end(), version.image) == alive.end()) {
			audit.frame_end_gone++;
			continue;
		}
		std::string   reason;
		const int32_t snap = SnapshotImage(context, *version.image, key, -1, reason);
		if (snap >= 0) {
			version.snap      = snap;
			version.snap_time = FrameEndTime;
		}
	}
	for (const auto& [range, write_time]: audit.buffer_writes) {
		auto& version = audit.buffer_versions[range];
		if (version.snap >= 0 && version.snap_time > write_time) continue;
		const auto [buffer, offset] = context.GetBufferCache().FindCachedBuffer(range.first, range.second);
		if (buffer == nullptr) {
			audit.frame_end_gone++;
			continue;
		}
		std::string   reason;
		const int32_t snap = SnapshotBuffer(context, buffer->Handle(), offset, range.first, range.second, -1, reason);
		if (snap >= 0) {
			version.snap      = snap;
			version.snap_time = FrameEndTime;
		}
	}
	CloseCopies(context);
}

void Finalize(RenderContext& context) {
	auto& audit = Audit();
	WritePending();
	SnapshotFrameEnd(context);
	Flush(context);
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
	                                                                      audit.start_time)
	                    .count();
	WriteLine(fmt::format("{{\"t\":\"end\",\"frame\":{},\"cmds\":{},\"snapshots\":{},\"analyzed\":{},\"not_taken\":{},"
	                      "\"frame_end_gone\":{},\"mib\":{},\"flushes\":{},\"ms\":{},\"bindless_rejects\":{}}}\n",
	                      audit.frame, audit.next_seq, audit.next_snapshot, audit.analyzed, audit.not_taken,
	                      audit.frame_end_gone, audit.total_bytes >> 20u, audit.flushes, static_cast<long long>(ms),
	                      Counts(audit.frame_rejects)));
	if (audit.file != nullptr) {
		std::fclose(audit.file);
		audit.file = nullptr;
	}
	audit.done = true;
	std::fprintf(stderr,
	             "SCENE_AUDIT done frame=%u commands=%u snapshots=%d MiB=%llu flushes=%u not_taken=%u ms=%lld file=%s\n",
	             audit.frame, audit.next_seq, audit.next_snapshot,
	             static_cast<unsigned long long>(audit.total_bytes >> 20u), audit.flushes, audit.not_taken,
	             static_cast<long long>(ms), audit.path.c_str());
	std::fflush(stderr);
	audit.images.clear();
	audit.buffer_versions.clear();
	audit.buffer_writes.clear();
}

// One shader stage of a draw or dispatch: its hash, images, buffers and samplers.
struct StageLists {
	std::vector<std::string> stages;
	std::vector<std::string> images;
	std::vector<std::string> buffers;
	uint32_t                 samplers     = 0;
	uint32_t                 gpu_samplers = 0;
	uint32_t                 bindless     = 0;
};

void AddStage(RenderContext& context, const Stage& stage, const std::vector<ImageId>& targets, StageLists& lists,
              std::vector<WriteMark>& writes) {
	if (stage.runtime == nullptr || stage.runtime->program == nullptr || stage.runtime->resources == nullptr) return;
	const auto& program   = *stage.runtime->program;
	const auto& resources = *stage.runtime->resources;
	const char* name      = StageName(program.stage);
	lists.stages.push_back(fmt::format("{{\"st\":\"{}\",\"h\":\"{:016x}\"}}", name, program.shader_hash));
	auto& cache = context.GetTextureCache();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto&           info = program.info.images[i];
		ShaderTextureResource descriptor {};
		const bool zero_descriptor = i < resources.images.size() ? CopyDwords(resources.images[i], descriptor.fields) : true;
		const bool written = info.written;
		const bool read    = info.read || !info.written;
		const char* cls    = info.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled   ? "sampled"
		                     : info.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage ? "storage"
		                                                                                                : "none";
		std::string entry = fmt::format(
		    "{{\"st\":\"{}\",\"slot\":{},\"acc\":\"{}\",\"cls\":\"{}\",\"addr\":\"0x{:x}\",\"gfmt\":{},\"gext\":[{},{},{}],"
		    "\"gtype\":{},\"pc\":\"0x{:x}\"",
		    name, i, written ? (info.read ? "rw" : "w") : "r", cls, descriptor.Base40(),
		    static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(descriptor.Width5()) + 1u,
		    static_cast<uint32_t>(descriptor.Height5()) + 1u, static_cast<uint32_t>(descriptor.Depth()) + 1u,
		    static_cast<uint32_t>(descriptor.Type()), info.first_use_pc);
		if (info.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage) {
			entry += fmt::format(",\"gpu_sel\":{}", std::max<size_t>(info.indirect_resources.size(), 1u));
		}
		if (info.bindless) entry += ",\"bindless\":true";
		if (info.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) entry += ",\"dyn_mip\":true";
		if (stage.bindings == nullptr || i >= stage.bindings->images.size()) {
			lists.images.push_back(entry + ",\"bound\":false}");
			continue;
		}
		const auto& binding     = stage.bindings->images[i];
		const bool  placeholder = binding.desc.info.data.Empty();
		if (placeholder) {
			// The emulator's 1x1 null image: the game bound nothing (guest), a descriptor that resolved to all zeros
			// (unresolved GPU-selected descriptor) or a descriptor the emulator could not use (placeholder).
			const char* kind = !descriptor.IsNull() ? "placeholder" : zero_descriptor ? "zero_desc" : "guest";
			entry += fmt::format(",\"null\":\"{}\",\"nullfmt\":\"{}\"", kind, vk::to_string(binding.desc.info.pixel_format));
		}
		if (!binding.image_id) {
			lists.images.push_back(entry + ",\"noimage\":true}");
			continue;
		}
		auto&       image   = cache.GetImage(binding.image_id);
		const auto& view    = binding.desc.view_info;
		const auto& backing = image.backing;
		uint32_t    aspect  = AspectColor;
		if (vk::hasDepthComponent(backing.format) || vk::hasStencilComponent(backing.format)) {
			aspect = vk::hasStencilComponent(backing.format) &&
			                 (!vk::hasDepthComponent(backing.format) || vk::blockSize(view.format) == 1u)
			             ? AspectStencil
			             : AspectDepth;
		}
		// The largest mip the shader can sample (streamed textures publish a minimum LOD before their mips are
		// resident); a storage image is accessed at its base level.
		const uint32_t last_mip = std::max(backing.mip_levels, 1u) - 1u;
		const uint32_t base     = std::min(view.base_level, last_mip);
		const uint32_t lod_mip  = std::min({view.base_level + (view.min_lod >> 8u),
		                                    view.base_level + std::max(view.level_count, 1u) - 1u, last_mip});
		const uint32_t mip      = written || info.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage
		                              ? base
		                              : lod_mip;
		const ImageKey key {binding.image_id.index, binding.image_id.generation, mip, aspect};
		entry += fmt::format(",\"fmt\":\"{}\",\"vfmt\":\"{}\",\"ext\":[{},{},{}],\"layers\":{},\"mips\":{},\"samples\":{},"
		                     "\"vmip\":{},\"vmips\":{},\"minlod\":{},\"mip\":{},\"aspect\":\"{}\",\"key\":{},\"iaddr\":\"0x{:x}\"",
		                     vk::to_string(backing.format), vk::to_string(view.format), backing.extent.width,
		                     backing.extent.height, backing.extent.depth, backing.layers, backing.mip_levels,
		                     backing.samples, view.base_level, view.level_count, view.min_lod, mip, AspectName(aspect),
		                     KeyText(key), image.info.data.address);
		const bool feedback = std::find(targets.begin(), targets.end(), binding.image_id) != targets.end();
		if (feedback) entry += ",\"fb\":true";
		if (read && !placeholder && !feedback) {
			std::string   reason;
			const int32_t snap = ReadImage(context, image, key, reason);
			entry += fmt::format(",\"snap\":{}", snap);
			if (!reason.empty()) entry += ",\"nosnap\":" + JsonString(reason);
		}
		if (written && !placeholder) writes.push_back({false, key, &image, {}});
		lists.images.push_back(entry + "}");
	}
	for (uint32_t r = 0; r < program.info.buffers.size(); r++) {
		const auto&          info = program.info.buffers[r];
		ShaderBufferResource descriptor {};
		if (r < resources.buffers.size()) (void)CopyDwords(resources.buffers[r], descriptor.fields);
		const bool  written = info.written || info.atomic;
		const bool  read    = info.read || info.atomic || !info.written;
		std::string entry   = fmt::format(
            "{{\"st\":\"{}\",\"slot\":{},\"acc\":\"{}\",\"addr\":\"0x{:x}\",\"stride\":{},\"records\":{},\"dfmt\":{},"
              "\"pc\":\"0x{:x}\"",
            name, r, written ? (read ? "rw" : "w") : "r", descriptor.Base48(), static_cast<uint32_t>(descriptor.Stride()),
            descriptor.NumRecords(), static_cast<uint32_t>(descriptor.RawFormat()), info.first_use_pc);
		if (info.indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer) {
			entry += fmt::format(",\"gpu_sel\":{}", std::max<size_t>(info.indirect_resources.size(), 1u));
		}
		if (info.atomic) entry += ",\"atomic\":true";
		const uint32_t index = info.descriptor_index;
		if (stage.bindings == nullptr || index == UINT32_MAX || index >= stage.bindings->buffer_sources.size() ||
		    index >= stage.bindings->buffers.size()) {
			// Not live in the translated shader (no descriptor), or not prepared.
			lists.buffers.push_back(entry + (index == UINT32_MAX ? ",\"dead\":true}" : ",\"bound\":false}"));
			continue;
		}
		const auto& source = stage.bindings->buffer_sources[index];
		const auto& bound  = stage.bindings->buffers[index];
		if (source.address == 0u || source.size == 0u) {
			lists.buffers.push_back(entry + ",\"null\":true}");
			continue;
		}
		entry += fmt::format(",\"gaddr\":\"0x{:x}\",\"size\":{}", source.address, source.size);
		if (read) {
			// The binding starts up to 255 bytes before the guest address (storage buffer offset alignment).
			uint64_t data_offset = bound.offset;
			if (bound.range != VK_WHOLE_SIZE && bound.range >= source.size) data_offset += bound.range - source.size;
			std::string   reason;
			const int32_t snap = ReadBuffer(context, bound.buffer, data_offset, source.address, source.size, reason);
			entry += fmt::format(",\"snap\":{}", snap);
			if (!reason.empty()) entry += ",\"nosnap\":" + JsonString(reason);
		}
		if (written) writes.push_back({true, {}, nullptr, {source.address, source.size}});
		lists.buffers.push_back(entry + "}");
	}
	for (const auto& sampler: program.info.samplers) {
		lists.samplers++;
		if (sampler.indirect_root != ShaderRecompiler::IR::SamplerResource::NoIndirectSampler) lists.gpu_samplers++;
	}
	if (stage.bindings != nullptr) lists.bindless += static_cast<uint32_t>(stage.bindings->bindless_images.size());
}

std::string StageListsJson(const StageLists& lists) {
	return fmt::format(",\"stages\":[{}],\"images\":[{}],\"buffers\":[{}],\"samplers\":{},\"gpu_samplers\":{},"
	                   "\"bindless\":{}",
	                   Join(lists.stages), Join(lists.images), Join(lists.buffers), lists.samplers,
	                   lists.gpu_samplers, lists.bindless);
}

} // namespace

uint32_t ConfiguredFrame() {
	static const uint32_t frame = [] {
		const uint32_t n = Common::EnvRead::AuditFrame();
		if (n != 0u && Common::EnvRead::FrameDir() == nullptr) {
			std::fprintf(stderr, "SCENE_AUDIT off: KYTY_TRACE_AUDIT_FRAME needs KYTY_TRACE_FRAME_DIR\n");
			return 0u;
		}
		return n;
	}();
	return frame;
}

void OnCommandStart(RenderContext& context) {
	if (!Enabled()) return;
	auto& audit = Audit();
	if (audit.done) return;
	const auto frame = DebugPresentCount().load();
	if (frame < audit.frame) return;
	if (frame > audit.frame) {
		if (audit.started) {
			Finalize(context);
		} else {
			audit.done = true;
			std::fprintf(stderr, "SCENE_AUDIT frame=%u had no draws or dispatches\n", audit.frame);
		}
		return;
	}
	if (!audit.started) Start();
	if (audit.done) return;
	WritePending();
	if (audit.batch_bytes >= audit.batch_limit) Flush(context);
}

void RecordDraw(RenderContext& context, const Stage* stages, uint32_t stage_count, const RenderColorInfo* colors,
                uint32_t color_count, const RenderDepthInfo* depth, uint32_t count, uint32_t instances, bool indexed,
                bool gpu_indirect, bool mesh) {
	if (!Recording()) return;
	auto& audit = Audit();
	BeginRecord();
	auto&                    cache = context.GetTextureCache();
	std::vector<ImageId>     target_ids;
	std::vector<WriteMark>   writes;
	std::vector<std::string> targets;
	for (uint32_t i = 0; colors != nullptr && i < color_count; i++) {
		const auto& color = colors[i];
		if (!color.image_id) continue;
		auto&          image = cache.GetImage(color.image_id);
		const uint32_t mip   = std::min(color.desc.view_info.base_level, std::max(image.backing.mip_levels, 1u) - 1u);
		const ImageKey key {color.image_id.index, color.image_id.generation, mip, AspectColor};
		targets.push_back(fmt::format(
		    "{{\"slot\":{},\"addr\":\"0x{:x}\",\"fmt\":\"{}\",\"gext\":[{},{}],\"ext\":[{},{}],\"mip\":{},\"layer\":{},"
		    "\"key\":{}}}",
		    color.target_slot, color.desc.info.data.address, vk::to_string(image.backing.format),
		    color.desc.info.extent.width, color.desc.info.extent.height, image.backing.extent.width,
		    image.backing.extent.height, mip, color.desc.view_info.base_layer, KeyText(key)));
		target_ids.push_back(color.image_id);
		writes.push_back({false, key, &image, {}});
	}
	std::string depth_json = "null";
	if (depth != nullptr && depth->image_id) {
		auto&          image         = cache.GetImage(depth->image_id);
		const uint32_t mip           = std::min(depth->desc.view_info.base_level, std::max(image.backing.mip_levels, 1u) - 1u);
		const bool     stencil       = vk::hasStencilComponent(image.backing.format);
		const bool     write_depth   = depth->depth_write_enable || depth->depth_clear_enable;
		const bool     write_stencil = stencil && (depth->stencil_clear_enable ||
		                                       (depth->stencil_test_enable &&
		                                        (depth->stencil_front.writeMask != 0u || depth->stencil_back.writeMask != 0u)));
		const ImageKey depth_key {depth->image_id.index, depth->image_id.generation, mip, AspectDepth};
		const ImageKey stencil_key {depth->image_id.index, depth->image_id.generation, mip, AspectStencil};
		depth_json = fmt::format(
		    "{{\"addr\":\"0x{:x}\",\"fmt\":\"{}\",\"gext\":[{},{}],\"ext\":[{},{}],\"mip\":{},\"test\":{},\"write\":{},"
		    "\"stest\":{},\"swrite\":{},\"clear\":{},\"key\":{},\"skey\":{}}}",
		    depth->desc.info.data.address, vk::to_string(image.backing.format), depth->desc.info.extent.width,
		    depth->desc.info.extent.height, image.backing.extent.width, image.backing.extent.height, mip,
		    JsonBool(depth->depth_test_enable), JsonBool(write_depth), JsonBool(depth->stencil_test_enable),
		    JsonBool(write_stencil), JsonBool(depth->depth_clear_enable || depth->stencil_clear_enable), KeyText(depth_key),
		    KeyText(stencil_key));
		target_ids.push_back(depth->image_id);
		if (write_depth) writes.push_back({false, depth_key, &image, {}});
		if (write_stencil) writes.push_back({false, stencil_key, &image, {}});
	}
	StageLists lists;
	for (uint32_t i = 0; stages != nullptr && i < stage_count; i++) AddStage(context, stages[i], target_ids, lists, writes);
	ApplyWrites(writes);
	audit.pending_body = fmt::format("\"seq\":{},\"kind\":\"{}\",\"pass\":{},\"count\":{},\"inst\":{},\"indexed\":{}",
	                                 audit.pending_seq, mesh ? "mesh" : gpu_indirect ? "draw_indirect_gpu" : "draw",
	                                 JsonString(MarkerPath()), count, instances, JsonBool(indexed));
	audit.pending_body += StageListsJson(lists);
	audit.pending_body += ",\"targets\":[" + Join(targets) + "],\"depth\":" + depth_json;
	CloseCopies(context);
}

void RecordDispatch(RenderContext& context, const Stage& stage, uint32_t groups_x, uint32_t groups_y,
                    uint32_t groups_z, bool indirect) {
	if (!Recording()) return;
	auto& audit = Audit();
	BeginRecord();
	std::vector<WriteMark> writes;
	StageLists             lists;
	AddStage(context, stage, {}, lists, writes);
	ApplyWrites(writes);
	audit.pending_body = fmt::format("\"seq\":{},\"kind\":\"{}\",\"pass\":{},\"groups\":[{},{},{}]", audit.pending_seq,
	                                 indirect ? "dispatch_indirect" : "dispatch", JsonString(MarkerPath()), groups_x,
	                                 groups_y, groups_z);
	audit.pending_body += StageListsJson(lists);
	CloseCopies(context);
}

void RecordSkipped(const char* kind, const char* reason, uint64_t shader_hash) {
	if (!Recording()) return;
	auto& audit = Audit();
	BeginRecord();
	const std::string_view kind_text = kind != nullptr ? kind : "command";
	audit.pending_body = fmt::format("\"seq\":{},\"kind\":{},\"pass\":{}", audit.pending_seq, JsonString(kind_text),
	                                 JsonString(MarkerPath()));
	if (shader_hash != 0u) {
		audit.pending_body += fmt::format(",\"stages\":[{{\"st\":\"{}\",\"h\":\"{:016x}\"}}]",
		                                  kind_text.starts_with("dispatch") ? "cs" : "ps", shader_hash);
	}
	audit.pending_skip = reason != nullptr && reason[0] != '\0' ? reason : "skipped";
}

void RecordSkippedDispatch(uint64_t code_address, bool indirect) {
	if (!Recording()) return;
	uint64_t hash = 0;
	(void)ShaderLookupMapped(code_address, nullptr, &hash);
	static const std::string skip_list = Common::EnvRead::SkipCsHashes();
	const bool listed = hash != 0u && skip_list.find(fmt::format("{:016x}", hash)) != std::string::npos;
	RecordSkipped(indirect ? "dispatch_indirect" : "dispatch",
	              listed ? "KYTY_DIAG_SKIP_CS_HASHES: shader not run"
	                     : "KYTY_DIAG_SKIP_UNSUPPORTED_CS: shader not run (unsupported instructions)",
	              hash);
}

void NoteSkipState(std::string_view json_object) {
	if (!Recording() || json_object.empty()) return;
	auto& audit = Audit();
	if (!audit.has_pending) return;
	audit.pending_body += ",\"skip_state\":";
	audit.pending_body += json_object;
}

void NoteSkipped(const char* reason) {
	if (!Recording()) return;
	auto& audit = Audit();
	if (!audit.has_pending) return;
	audit.pending_skip = reason != nullptr && reason[0] != '\0' ? reason : "skipped";
}

void NoteBindlessReject(const char* stat, uint64_t heap, uint32_t entry) {
	(void)heap;
	(void)entry;
	if (!Recording()) return;
	auto&             audit = Audit();
	const std::string name  = stat != nullptr ? stat : "other";
	(audit.has_pending ? audit.pending_rejects : audit.held_rejects)[name]++;
	audit.frame_rejects[name]++;
}

} // namespace Libs::Graphics::SceneAudit
