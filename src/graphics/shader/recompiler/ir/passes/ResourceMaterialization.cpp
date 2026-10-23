#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/DescriptorInterner.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <fmt/format.h>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Vectors MaterializeResources fills and empties again within one refresh. They keep their
// capacity in the plan, so a refresh of a known shape allocates nothing; every user clears or
// assigns its vector before reading it.
struct MaterializationScratch {
	std::vector<uint32_t>                                     bindless_mapping;
	std::vector<uint32_t>                                     buffer_headers;
	std::vector<uint32_t>                                     image_headers;
	std::vector<uint32_t>                                     sampler_headers;
	std::vector<uint32_t>                                     padding;
	std::vector<std::pair<uint32_t, uint32_t>>                class_spares;
	std::vector<uint32_t>                                     class_ranks;
	std::vector<uint32_t>                                     class_list;
	std::vector<uint32_t>                                     class_starts;
	std::vector<uint32_t>                                     class_order;
	std::vector<DescriptorValue>                              grouped;
	std::vector<ResourceSpecialization::Image>                grouped_images;
	std::vector<uint32_t>                                     renumber;
	std::vector<uint32_t>                                     row_lengths;
	std::vector<uint32_t>                                     row_batch;
	// RangeOverlapIndex storage for the specialization reads.
	std::vector<std::pair<uint64_t, uint64_t>>                read_ends;
	DescriptorInterner                                        candidates;
	// The usable candidates of an image heap (a T# at any DWORD of a table) and whether all of
	// them are float 2D, for the heap's last contents. Candidates are a function of the table's
	// DWORDs and the resource, and heaps are rewritten far less often than they are read: a
	// heap with unchanged DWORDs takes them from here instead of normalizing every DWORD start.
	struct HeapCandidates {
		uint32_t                                          resource = UINT32_MAX;
		uint64_t                                          first    = 0;
		uint64_t                                          step     = 0;
		uint64_t                                          count    = 0;
		uint32_t                                          dwords   = 0;
		bool                                              affine   = false;
		bool                                              float_2d = false;
		std::vector<uint32_t>                             table;
		std::vector<std::pair<uint64_t, DescriptorValue>> candidates;
	};
	std::vector<HeapCandidates> heaps;
};

namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
// An image table with more keys than this is not enumerated: its program would bake the
// candidate array and recompile whenever the table's contents change (Ghost of Yotei streams
// hundreds of entries), so the renderer resolves the keys the GPU asks for instead.
constexpr size_t BindlessTableKeys = 32u;

// The reasons a MaterializeResources call on this thread gave up, joined into the one line it
// reports when it fails (a reason recorded on a path that still succeeds is dropped).
struct FailureReasons {
	bool        speculative = false;
	std::string text;
};
thread_local FailureReasons t_failure_reasons;

std::atomic<uint64_t> g_failure_reports {0};

bool SpecializationFail(std::string_view message) {
	auto& reasons = t_failure_reasons;
	if (!reasons.speculative) {
		if (!reasons.text.empty()) reasons.text += "; ";
		reasons.text += message;
	}
	return false;
}

// A shader that cannot be specialized usually fails the same way for every draw; its first
// failure names the reasons and later ones are not printed.
void ReportSpecializationFailure(uint64_t shader_hash, const std::string& reasons) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> reported;
	{
		std::scoped_lock lock(mutex);
		if (!reported.insert(shader_hash).second) return;
	}
	g_failure_reports.fetch_add(1u, std::memory_order_relaxed);
	std::fprintf(stderr, "shader resource specialization failed: hash=%016llx %s\n",
	             static_cast<unsigned long long>(shader_hash), reasons.c_str());
}

// The draw's guest data cannot be enumerated (see ResourceSnapshot::unavailable). The data can
// stay that way for many draws, so only the first reports are printed.
bool SpecializationUnavailable(uint64_t shader_hash, ResourceSnapshot& snapshot,
                               std::string_view message) {
	snapshot.unavailable = true;
	static std::atomic<uint32_t> reports {0};
	if (!t_failure_reasons.speculative && reports.fetch_add(1u, std::memory_order_relaxed) < 64u) {
		std::fprintf(stderr, "shader resource specialization unavailable: hash=%016llx %.*s\n",
		             static_cast<unsigned long long>(shader_hash), static_cast<int>(message.size()),
		             message.data());
	}
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool NullBufferDescriptor(Value value) {
	const auto* handle = value.Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
	    handle->NumArgs() != 4u) return false;
	for (size_t index = 0; index < 4u; ++index)
		if (handle->Arg(index).Resolve() != Value(0u)) return false;
	return true;
}

// A texture can only be placed at a base aligned to its tile block; a table record that is not
// (stale or unrelated data) is not a texture the shader samples.
bool PlaceableImageDescriptor(const DescriptorValue& descriptor) {
	const auto base =
	    ((descriptor.dwords[0] | (uint64_t {descriptor.dwords[1]} << 32u)) & 0xFFFFFFFFFFull) << 8u;
	uint64_t alignment = 0;
	switch (static_cast<Prospero::TileMode>((descriptor.dwords[3] >> 20u) & 0x1fu)) {
		case Prospero::TileMode::kLinear:
		case Prospero::TileMode::kStandard256B: alignment = 256u; break;
		case Prospero::TileMode::kStandard4KB: alignment = 4096u; break;
		case Prospero::TileMode::kStandard64KB:
		case Prospero::TileMode::kPrt:
		case Prospero::TileMode::kDepth:
		case Prospero::TileMode::kRenderTarget: alignment = 65536u; break;
		default: return false;
	}
	return (base & (alignment - 1u)) == 0u;
}

// Whether an image operation can use a T# with this format; it mirrors the format checks of
// BuildResourceSpecialization. FMASK surfaces are only supported through a direct image load.
bool ImageDescriptorFormatUsable(const ImageResource& use, const DescriptorValue& descriptor) {
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (Prospero::IsFmaskTextureFormat(format)) {
		return false;
	}
	if (use.atomic) {
		return use.atomic64 ? format == Prospero::BufferFormat::k32_32UInt
		                    : format == Prospero::BufferFormat::k32UInt ||
		                          format == Prospero::BufferFormat::k32SInt ||
		                          format == Prospero::BufferFormat::k32Float;
	}
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (numeric_class == Prospero::TextureNumericClass::Unsupported) {
		return false;
	}
	if (use.resource_class == ImageResourceClass::Storage) {
		return numeric_class != Prospero::TextureNumericClass::Sint ||
		       RawSignedStorageFormat(use, format) != Prospero::BufferFormat::kInvalid;
	}
	return !use.depth_compare || numeric_class == Prospero::TextureNumericClass::Float;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	if (!IsValidImageSwizzle(words[3] & 0xfffu)) return false;
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	const auto base_level = (words[3] >> 12u) & 0xfu;
	const auto last_level = (words[3] >> 16u) & 0xfu;
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && last_level >= 1 && last_level <= 3 &&
		       (r128 || max_mip == last_level);
	}
	return base_level <= last_level;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

bool ValidatePackedUnormSampling(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                                 uint32_t image_index) {
	ShaderTextureResource texture;
	std::copy_n(snapshot.images[image_index].dwords.begin(), std::size(texture.fields),
	            texture.fields);
	if (texture.Type() != Prospero::ImageType::kColor2D ||
	    texture.BaseLevel() != texture.LastLevel() || texture.MinLod() != 0u ||
	    texture.CornerSample()) {
		return SpecializationFail("packed UNorm sampling requires a centered, single-mip 2D image");
	}
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.image != image_index) continue;
		if (pair.sampler >= snapshot.samplers.size() ||
		    program.descriptor_sources[program.info.samplers[pair.sampler].source].indirect_descriptor) {
			return SpecializationFail("packed UNorm sampling requires a direct sampler");
		}
		ShaderSamplerResource sampler;
		std::copy_n(snapshot.samplers[pair.sampler].dwords.begin(), std::size(sampler.fields),
		            sampler.fields);
		const auto linear = static_cast<uint32_t>(Prospero::SamplerFilter::kBilinear);
		const auto clamp = static_cast<uint32_t>(Prospero::SamplerClampMode::kClampLastTexel);
		if (sampler.XyMagFilter() != linear || sampler.XyMinFilter() != linear ||
		    sampler.ClampX() != clamp || sampler.ClampY() != clamp ||
		    sampler.ForceUnormCoords() || sampler.MaxAnisoRatio() != 0u ||
		    sampler.FilterMode() != 0u || sampler.MinLod() != 0u || sampler.ForceSrgb() ||
		    sampler.TruncCoord() || (sampler.fields[0] & (1u << 19u)) != 0u) { // MC_COORD_TRUNC
			return SpecializationFail(fmt::format(
			    "packed UNorm image {} requires normalized bilinear clamp sampling (sampler {})",
			    image_index, pair.sampler));
		}
	}
	return true;
}

enum class SamplerClass : uint8_t { Float, Integer };

SamplerClass ClassifySampler(const ImageResource& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Uint ||
	               image.numeric_class == Prospero::TextureNumericClass::Sint
	           ? SamplerClass::Integer : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return last - base + 1u;
}

// What decides whether two typed candidates of one image root can share a dynamically indexed
// access: the numeric class (it selects the binding array), the dimension and the cube flag, as
// BuildResourceSpecialization derives them.
std::array<uint32_t, 3> ImageCandidateClass(const ImageResource&   image,
                                            const DescriptorValue& descriptor) {
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	auto numeric_class = Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (image.resource_class == ImageResourceClass::Storage &&
	    (image.atomic || RawSignedStorageFormat(image, format) != Prospero::BufferFormat::kInvalid))
		numeric_class = Prospero::TextureNumericClass::Uint;
	return {static_cast<uint32_t>(numeric_class),
	        static_cast<uint32_t>(DescriptorDimension(descriptor, image.dimension)),
	        DescriptorIsCube(descriptor) ? 1u : 0u};
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanReads(const std::unordered_set<const Inst*>& planned_reads,
                    const DescriptorSource* source, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value> pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (planned_reads.contains(inst)) {
			auto flags = inst->Flags<SrtReadFlags>();
			flags.clean = 1u;
			inst->SetFlags(flags);
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadTable(uint64_t base, uint64_t size, uint64_t offset,
               const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	if (base > AddressMask || offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	return ReadTable(base & AddressMask & ~uint64_t {3}, size, offset & ~uint64_t {3}, runtime, words);
}

bool NormalizeIndirectBuffer(DescriptorValue& value, bool indexed, const BufferResource& resource) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	const bool formatted_read = resource.formatted && resource.read && !resource.written;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    (!formatted_read && (descriptor.Base48() == 0u || descriptor.GetSize() == 0u))) {
		value.dwords.fill(0u);
		return true;
	}
	if (descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	    descriptor.OutOfBounds() != (indexed ? 0u : 3u)) {
		return SpecializationFail(fmt::format(
		    "indirect buffers require exact linear index or byte bounds: "
		    "descriptor={:08x},{:08x},{:08x},{:08x} swizzle={} add_tid={} oob={} "
		    "indexed={} formatted={} read={} written={} max_extent={}",
		    value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
		    descriptor.SwizzleEnabled(), descriptor.AddTid(), descriptor.OutOfBounds(), indexed,
		    resource.formatted, resource.read, resource.written, resource.max_byte_extent));
	}
	const auto rejected = [&](std::string_view reason) {
		return SpecializationFail(fmt::format(
		    "{}: descriptor={:08x},{:08x},{:08x},{:08x} format={} stride={} dst_sel={:#x} "
		    "max_extent={}",
		    reason, value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
		    static_cast<uint32_t>(descriptor.Format()), descriptor.Stride(),
		    descriptor.DstSelXYZW(), resource.max_byte_extent));
	};
	if (resource.formatted) {
		const auto format = Format::GetFormatInfo(descriptor.Format());
		if (!indexed || format.type == Format::ComponentType::Unknown)
			return rejected("indirect formatted buffers require a supported indexed format");
		if (!formatted_read) {
			const bool scalar = resource.max_byte_extent == 4u &&
			    (descriptor.Format() == Prospero::BufferFormat::k32UInt ||
			     descriptor.Format() == Prospero::BufferFormat::k8SInt);
			const bool rgba16 = resource.max_byte_extent == 16u &&
			    descriptor.Format() == Prospero::BufferFormat::k16_16_16_16Float;
			if ((!scalar && !rgba16) || descriptor.Stride() != format.byte_size ||
			    descriptor.DstSelXYZW() != (rgba16 ? DstSel(4, 5, 6, 7) : DstSel(4, 0, 0, 0)))
				return rejected("indirect formatted stores require tight identity SInt8, UInt32 or "
				                "RGBA16Float records");
		}
		if (formatted_read &&
		    (descriptor.Base48() == 0u || descriptor.Stride() == 0u || descriptor.NumRecords() == 0u)) {
			// Every index is out of bounds. Preserve the format and swizzle constants.
			value.dwords[1] &= ~(0x3fffu << 16u);
			value.dwords[2] = 0u;
			return true;
		}
		const auto alignment = std::min(4u, format.byte_size);
		if (descriptor.Stride() < format.byte_size ||
		    (descriptor.Stride() & (alignment - 1u)) != 0u ||
		    (descriptor.Base48() & (alignment - 1u)) != 0u)
			return rejected("indirect formatted buffers require aligned full-format records");
	} else if ((!indexed && descriptor.Stride() != 1u) ||
	           (indexed && resource.max_byte_extent != 1u && descriptor.Stride() != 4u)) {
		return rejected("indirect buffer stores require exact linear DWORD or byte bounds");
	}
	return true;
}

bool IsBoundedDescriptorTable(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect) {
	const auto* table = Source(program, indirect.table_source);
	return indirect.sources.empty() && (!indirect.selector || indirect.table_scalar) &&
	       (indirect.key_count.IsEmpty() || indirect.table_maximum) && indirect.workgroup_axis == UINT32_MAX &&
	       table != nullptr && table->dword_count == 4u;
}

bool CanReadIndexedSelector(const ShaderBufferResource& material, uint32_t offset) {
	if (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
	    material.OutOfBounds() != 0u) return false;
	if (material.Format() == Prospero::BufferFormat::kInvalid || material.NumRecords() == 0u ||
	    offset >= material.Stride()) return true;
	return material.Base48() != 0u &&
	       ((material.Base48() | material.Stride() | offset) & 3u) == 0u &&
	       material.NumRecords() <= MaxIndirectDescriptorProbes &&
	       uint64_t {material.NumRecords() - 1u} * material.Stride() + offset + 4u <=
	           uint64_t {UINT32_MAX} + 1u;
}

bool EvaluateIndirectCount(const DescriptorSource::IndirectDescriptor& indirect,
                           const SrtRuntime& runtime, SrtWalker& clean, uint32_t& count) {
	if (indirect.workgroup_axis != UINT32_MAX) {
		if (runtime.workgroup_counts.size() != 3u || indirect.workgroup_axis >= 3u) return false;
		count = std::ranges::find(runtime.workgroup_counts, 0u) != runtime.workgroup_counts.end()
		    ? 0u : runtime.workgroup_counts[indirect.workgroup_axis];
		return true;
	}
	if (!clean.Evaluate(indirect.key_count, count)) return false;
	if (indirect.signed_count && std::bit_cast<int32_t>(count) < 0) count = 0;
	return true;
}

// `unavailable` is set when the keys depend on guest data that cannot be enumerated: unmapped
// memory, a runtime count beyond the probe limit or a key range past 32 bits.
bool ResolveSelectorKeys(const DescriptorSource::IndirectDescriptor& indirect,
                         const DescriptorValue& material_value, const SrtRuntime& runtime,
                         SrtWalker& clean, std::vector<uint32_t>& keys, bool& unavailable,
                         MaterializationScratch& program_scratch) {
	const auto* selector = &*indirect.selector;
	keys.clear();
	unavailable           = false;
	const auto data_limit = [&] {
		unavailable = true;
		return false;
	};
	const auto read_scalar = [&](uint64_t base, uint64_t size, uint64_t offset,
	                             std::span<uint32_t> words) {
		return ReadScalarTable(base, size, offset, runtime, words) || data_limit();
	};
	const auto read_selector = [&](uint64_t base, uint64_t size, uint64_t offset,
	                               std::span<uint32_t> words) {
		return selector->global ? ReadTable(base, size, offset, runtime, words) || data_limit()
		                        : read_scalar(base, size, offset, words);
	};
	const auto read_keys = [&](uint64_t base, uint64_t size, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (count == 0u) return true;
		if (step == 4u) {
			return read_selector(base, size, first, keys);
		}
		if (step < 4u) return false;
		for (uint64_t copied = 0; copied < count;) {
			const auto offset = first + copied * step;
			// Scalar reads align each strided offset independently.
			const auto bias = selector->global ? 0u : offset & 3u;
			const auto length = std::min(count - copied,
			    1u + (uint64_t {MaxIndirectDescriptorProbes - 1u} * 4u - bias) / step);
			keys.resize(count + (bias + (length - 1u) * step) / 4u + 1u);
			const auto batch = std::span(keys).subspan(count);
			if (!read_selector(base, size, offset, batch)) {
				keys.resize(count);
				return false;
			}
			for (uint64_t index = 0; index < length; ++index)
				keys[copied + index] = batch[(bias + index * step) / 4u];
			copied += length;
		}
		keys.resize(count);
		return true;
	};
	if (selector->rows) {
		const auto& rows = *selector->rows;
		DescriptorValue count_value;
		ShaderBufferResource counts, material;
		if (!clean.EvaluateDescriptor(rows.source, count_value)) return data_limit();
		if (!DecodeBufferDescriptor(count_value, counts) ||
		    !DecodeBufferDescriptor(material_value, material) || selector->stride != 4u ||
		    selector->offset != 0u || rows.stride == 0u || (rows.stride & 3u) != 0u)
			return false;
		if (counts.GetSize() / 4u > MaxIndirectDescriptorProbes) return data_limit();
		auto& lengths = program_scratch.row_lengths;
		lengths.assign(counts.GetSize() / 4u, 0u);
		if (!read_scalar(counts.Base48(), counts.GetSize(), 0u, lengths)) return false;
		uint32_t total = 0;
		for (size_t row = 0; row < lengths.size(); ++row) {
			const auto length = lengths[row];
			if (length > MaxIndirectDescriptorProbes - total ||
			    (length != 0u && uint64_t {row} * rows.stride + uint64_t {length} * 4u >
			                         UINT32_MAX + 1ull))
				return data_limit();
			total += length;
		}
		keys.reserve(total);
		auto& batch = program_scratch.row_batch;
		for (size_t row = 0; row < lengths.size();) {
			if (lengths[row] == 0u) { ++row; continue; }
			const auto first = uint64_t {row} * rows.stride / 4u;
			auto end = first + lengths[row];
			size_t next = row + 1u;
			for (; next < lengths.size(); ++next) {
				if (lengths[next] == 0u) continue;
				const auto candidate_end =
				    std::max(end, uint64_t {next} * rows.stride / 4u + lengths[next]);
				if (candidate_end - first > MaxIndirectDescriptorProbes) break;
				end = candidate_end;
			}
			// Coalesce small gaps, but interpret only selected prefixes. The strict
			// reader captures the entire fetched span for writable-alias rejection.
			batch.resize(end - first);
			if (!read_scalar(material.Base48(), material.GetSize(), first * 4u, batch))
				return false;
			for (; row < next; ++row) {
				if (lengths[row] == 0u) continue;
				const auto begin = batch.begin() + (uint64_t {row} * rows.stride / 4u - first);
				keys.insert(keys.end(), begin, begin + lengths[row]);
			}
		}
	} else if (selector->bitset) {
		ShaderBufferResource material;
		std::array<uint32_t, 4> masks {};
		uint32_t first = 0, count = 0;
		if (!DecodeBufferDescriptor(material_value, material)) return false;
		if (!clean.Evaluate(indirect.selector_first, first) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count))
			return data_limit();
		if (count != masks.size() * 32u ||
		    selector->stride != sizeof(masks) || selector->offset != 0u ||
		    material.Type() != 0u || material.Stride() != sizeof(masks) ||
		    material.SwizzleEnabled() || material.AddTid() || material.OutOfBounds() != 0u ||
		    (material.Base48() & 3u) != 0u) return false;
		if (material.Format() != Prospero::BufferFormat::kInvalid) {
			const auto words = material.GetSize() / sizeof(uint32_t);
			// Bitmap storage can grow with resolution while its candidate domain stays
			// at 128 keys. Bound scratch memory, preserving each coherent read's capture.
			for (uint64_t first_word = 0; first_word < words; first_word += keys.size()) {
				keys.resize(std::min<uint64_t>(words - first_word, MaxIndirectDescriptorProbes));
				if (!read_scalar(material.Base48(), material.GetSize(), first_word * 4u, keys))
					return false;
				for (size_t word = 0; word < keys.size(); ++word) masks[word % masks.size()] |= keys[word];
				if (std::ranges::all_of(masks, [](uint32_t mask) { return mask == UINT32_MAX; })) break;
			}
			keys.clear();
			for (uint32_t word = 0; word < masks.size(); ++word) {
				auto mask = masks[word];
				while (mask != 0u) {
					const auto rank = word * 32u + std::countr_zero(mask);
					keys.push_back(first + count - 1u - rank);
					mask &= mask - 1u;
				}
			}
		}
	} else if (!selector->byte_ranges.empty()) {
		if (material_value.dword_count != 2u) return false;
		const auto base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		std::array<bool, 256> selected {};
		for (const auto [offset, size]: selector->byte_ranges) {
			if (size > MaxIndirectDescriptorProbes || ((offset | size) & 3u) != 0u) return false;
			keys.resize(size / 4u);
			if (!read_scalar(base, UINT64_MAX, offset, keys)) return false;
			for (const auto word: keys)
				for (uint32_t shift = 0; shift < 32u; shift += 8u)
					selected[(word >> shift) & 255u] = true;
		}
		keys.clear();
		for (uint32_t key = 0; key < selected.size(); ++key)
			if (selected[key]) keys.push_back(key);
	} else if (!indirect.selector_first.IsEmpty()) {
		ShaderBufferResource material;
		uint64_t             base = 0, size = UINT64_MAX;
		uint32_t             first = 0, count = 0;
		if (material_value.dword_count == 2u) {
			if (selector->stride == 0u) return false;
			base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		} else {
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			base = material.Base48();
			size = material.GetSize();
		}
		if (!clean.Evaluate(indirect.selector_first, first) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count > MaxIndirectDescriptorProbes)
			return data_limit();
		// SMEM uses its instruction's byte stride; V# stride only supplies its bounds.
		const auto step = selector->stride != 0u ? selector->stride : material.Stride();
		if (selector->stride == 0u &&
		    (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
		     material.OutOfBounds() != 0u || uint64_t {selector->offset} + 4u > step ||
		     uint64_t {first} + count > material.NumRecords())) return false;
		const auto last = count != 0u ? uint64_t {first} + count - 1u : 0u;
		if (last > UINT32_MAX) return data_limit();
		const auto end = count != 0u ? last * step + selector->offset + 4u : 0u;
		if (end > uint64_t {UINT32_MAX} + 1u ||
		    (indirect.workgroup_axis == UINT32_MAX && end > size))
			return data_limit();
		if (!read_keys(base, size, uint64_t {first} * step + selector->offset, step, count))
			return false;
	} else if (!indirect.selector_mask.IsEmpty()) {
		uint32_t mask = 0;
		uint32_t count = 0;
		if (material_value.dword_count != 2u) return false;
		if (!clean.Evaluate(indirect.selector_mask, mask) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count))
			return data_limit();
		if (count == 0u || count > 32u) return false;
		if (count < 32u) mask &= (1u << count) - 1u;
		const auto material_base =
		    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
		keys.reserve(std::popcount(mask));
		while (mask != 0u) {
			const auto index = std::countr_zero(mask);
			const auto offset = static_cast<uint64_t>(selector->offset) +
			                    static_cast<uint64_t>(index) * selector->stride;
			if (offset > UINT32_MAX) return false;
			uint32_t key = 0;
			if (!read_selector(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
			                   {&key, 1})) return false;
			keys.push_back(key);
			mask &= mask - 1u;
		}
	} else if (selector->stride == 0u) {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !CanReadIndexedSelector(material, selector->offset)) return false;
		if (material.Format() != Prospero::BufferFormat::kInvalid && material.NumRecords() != 0u &&
		    selector->offset < material.Stride() &&
		    !read_keys(material.Base48(), material.GetSize(), selector->offset,
		               material.Stride(), material.NumRecords())) return false;
		keys.push_back(0u); // Every out-of-bounds indexed load returns zero.
	} else {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) || selector->stride < 4u ||
		    !std::has_single_bit(selector->stride) || (selector->offset & 3u) != 0u)
			return false;
		// Empty selector bounds denote the complete U32 congruence class. SMEM
		// adds its aligned immediate after the dynamic offset, without U32 wrap.
		const auto domain_count = (uint64_t {1} << 32u) / selector->stride;
		const auto available = material.GetSize() > selector->offset
		    ? material.GetSize() - selector->offset : 0u;
		const auto count = std::min(domain_count,
		    available < 4u ? 0u : (available - 4u) / selector->stride + 1u);
		if (count > MaxIndirectDescriptorProbes) return data_limit();
		if (!read_keys(material.Base48(), material.GetSize(), selector->offset, selector->stride,
		               count))
			return false;
		if (count < domain_count) keys.push_back(0u); // An out-of-bounds scalar read returns zero.
	}
	if (selector->byte_ranges.empty()) {
		if (selector->bit_offset != 0u || selector->bit_mask != UINT32_MAX || selector->maximum != UINT32_MAX)
			for (auto& key: keys)
				key = std::min((key >> selector->bit_offset) & selector->bit_mask, selector->maximum);
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	}
	return true;
}

template <typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(
    const ResourcePlan& program, uint32_t source_index, uint32_t resource_index,
    uint32_t dword_count, uint32_t read_dwords, const SrtRuntime& runtime, SrtWalker& clean,
    ResourceSnapshot& snapshot, std::vector<DescriptorValue>& descriptors,
    std::vector<Specialization>& specializations, uint32_t bindless_mapping,
    uint32_t mapping_header, std::vector<uint32_t>& padding,
    std::vector<std::pair<uint32_t, uint32_t>>& class_spares, Normalize&& normalize) {
	const auto& indirect = *program.descriptor_sources[source_index].indirect_descriptor;
	const auto  descriptor_bytes = read_dwords * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	const auto* selector = indirect.selector ? &*indirect.selector : nullptr;
	const bool byte_keys = IsBoundedDescriptorTable(program, indirect);
	auto* selector_keys = selector != nullptr
	    ? &program.selector_keys[program.selector_groups[source_index]] : nullptr;
	auto& keys = selector_keys != nullptr ? selector_keys->values : program.material_keys;
	auto& byte_offsets = program.material_keys;
	const auto children_begin = descriptors.size();
	const auto root_resource = specializations[resource_index];
	// A table yields thousands of keys but a few hundred distinct candidates; the interner finds
	// the first equal child as a linear search would, so ordinals follow first occurrence.
	auto& interner = program.materialization_scratch->candidates;
	interner.Reset();
	const auto children = [&] {
		return std::span<const DescriptorValue>(descriptors).subspan(children_begin);
	};
	const auto intern_candidate = [&](const DescriptorValue& candidate) {
		if (candidate == descriptors[resource_index]) return 0u;
		const auto found = interner.Find(children(), candidate);
		if (found != DescriptorInterner::NotFound) return found + 1u;
		const auto ordinal = static_cast<uint32_t>(descriptors.size() - children_begin + 1u);
		descriptors.push_back(candidate);
		interner.Added(children());
		auto child          = root_resource;
		child.indirect_root = resource_index;
		specializations.push_back(child);
		return ordinal;
	};
	// Spare candidate slots hold null descriptors: the program bakes the candidate count, so it
	// is padded to a power of two and a table whose contents change with the scene does not
	// compile a permutation per change. No key maps to a spare slot.
	const auto pad_candidates = [&] {
		const auto unique = descriptors.size() - children_begin + 1u;
		for (auto padded = std::bit_ceil(unique); descriptors.size() - children_begin + 1u < padded;) {
			padding.push_back(static_cast<uint32_t>(descriptors.size()));
			descriptors.push_back({.dword_count = dword_count});
			auto child          = root_resource;
			child.indirect_root = resource_index;
			specializations.push_back(child);
		}
	};
	// Image candidates of one class share a dynamically indexed access only while their ordinals
	// are contiguous, and the program bakes where each class starts. A table enumerated in row
	// order (Ghost of Yotei's 440-byte object records, probed every 8 bytes) interleaves classes
	// differently with every object streamed in, so each scene compiled a new program with another
	// switch arm. The children are therefore grouped by class - the root's class first, since the
	// root is ordinal 0 - and each group is padded to a power of two with null spares that take the
	// group's class (class_spares: spare, donor). Growing a group within its power of two leaves
	// the specialization unchanged; a new class still changes it. The keys in the map
	// (entry_stride words apart from map_first) are renumbered.
	const auto group_image_candidates = [&](size_t map_first, size_t map_entries,
	                                        size_t entry_stride) {
		if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
			const auto& image = program.info.images[resource_index];
			const auto  count = descriptors.size() - children_begin;
			if (count == 0u) return true;
			// A class packed into one word orders like the triple: each part is below 256.
			const auto class_of = [&](const DescriptorValue& value) {
				const auto type = ImageCandidateClass(image, value);
				return (type[0] << 16u) | (type[1] << 8u) | type[2];
			};
			auto& scratch = *program.materialization_scratch;
			const auto&             root    = descriptors[resource_index];
			std::optional<uint32_t> lead;
			if (!NullImageDescriptor(root)) lead = class_of(root);
			// A null root takes the class of the first typed candidate, i.e. of the first group.
			for (uint32_t child = 0; child < count && NullImageDescriptor(root); ++child) {
				const auto& value = descriptors[children_begin + child];
				if (NullImageDescriptor(value)) continue;
				const auto type = class_of(value);
				if (!lead || type < *lead) lead = type;
			}
			if (!lead) return true;
			// The lead class first, then the others in ascending order; children keep their table
			// order within a class. Tables hold a handful of classes, so the children are counted
			// into their class's run instead of sorted.
			auto& ranks   = scratch.class_ranks;
			auto& classes = scratch.class_list;
			auto& starts  = scratch.class_starts;
			auto& order   = scratch.class_order;
			ranks.resize(count);
			classes.clear();
			for (uint32_t child = 0; child < count; ++child) {
				const auto& value = descriptors[children_begin + child];
				// Null children (only behind a typed root) take the root's class as well.
				const auto type = NullImageDescriptor(value) ? *lead : class_of(value);
				ranks[child]    = type;
				if (std::ranges::find(classes, type) == classes.end()) classes.push_back(type);
			}
			std::ranges::sort(classes, [&](uint32_t a, uint32_t b) {
				return std::pair {a != *lead, a} < std::pair {b != *lead, b};
			});
			starts.assign(classes.size() + 1u, 0u);
			for (auto& rank: ranks) {
				rank = static_cast<uint32_t>(std::ranges::find(classes, rank) - classes.begin());
				++starts[rank + 1u];
			}
			for (size_t rank = 1; rank < starts.size(); ++rank)
				starts[rank] += starts[rank - 1u];
			// Placing a child advances its class's start, so starts[rank] ends as the end of the
			// class's run.
			order.resize(count);
			for (uint32_t child = 0; child < count; ++child) {
				order[starts[ranks[child]]++] = child;
			}
			// Dynamic-mip candidates never share an access; spares would only add switch arms.
			const bool pad = image.mip_mode != ImageMipMode::Dynamic;
			auto&      grouped                 = scratch.grouped;
			auto&      grouped_specializations = scratch.grouped_images;
			auto&      renumber                = scratch.renumber;
			grouped.clear();
			grouped_specializations.clear();
			renumber.assign(count + 1u, 0u);
			auto child_template          = root_resource;
			child_template.indirect_root = resource_index;
			for (size_t rank = 0, first = 0; rank < classes.size(); ++rank) {
				const size_t last  = starts[rank];
				const auto   donor = static_cast<uint32_t>(children_begin + grouped.size());
				for (auto member = first; member < last; ++member) {
					renumber[order[member] + 1u] = static_cast<uint32_t>(grouped.size() + 1u);
					grouped.push_back(descriptors[children_begin + order[member]]);
					grouped_specializations.push_back(
					    specializations[children_begin + order[member]]);
				}
				// The root (ordinal 0) belongs to the lead group.
				const bool lead_group = first == 0u;
				const auto members = last - first + (lead_group ? 1u : 0u);
				for (auto slots = pad ? std::bit_ceil(members) : members; slots > members; --slots) {
					if (!lead_group)
						class_spares.emplace_back(static_cast<uint32_t>(children_begin + grouped.size()), donor);
					grouped.push_back({.dword_count = dword_count});
					grouped_specializations.push_back(child_template);
				}
				first = last;
			}
			descriptors.resize(children_begin);
			specializations.resize(children_begin);
			descriptors.insert(descriptors.end(), grouped.begin(), grouped.end());
			specializations.insert(specializations.end(), grouped_specializations.begin(),
			                       grouped_specializations.end());
			for (size_t entry = 0; entry < map_entries; ++entry) {
				auto& ordinal = snapshot.flattened_srt[map_first + entry * entry_stride];
				if (ordinal != 0u) ordinal = renumber[ordinal];
			}
		}
		return true;
	};
	// A heap's two words live at a place fixed by the plan, so the mapping offsets the program
	// bakes do not move with the length of other tables' key maps.
	// A key map's offset depends on the lengths of the maps before it, which change with the
	// scene; a module that baked it would be compiled again for every change. The plan reserves
	// a header word per indirect root that holds the offset (MaterializeResources), and the
	// module reads it from there.
	const auto publish_mapping = [&](Specialization& root, size_t offset) {
		if (mapping_header == UINT32_MAX) {
			root.indirect_mapping_offset = static_cast<uint32_t>(offset);
			return;
		}
		snapshot.flattened_srt[mapping_header] = static_cast<uint32_t>(offset);
		root.indirect_mapping_offset           = mapping_header;
		root.indirect_mapping_header           = true;
	};
	const auto bindless_words = [&] {
		if (bindless_mapping != UINT32_MAX) return bindless_mapping;
		const auto offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
		snapshot.flattened_srt.resize(offset + 2u, 0u);
		return offset;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	DescriptorValue table_value;
	ShaderBufferResource table;
	const auto describe = [&](std::string_view reason, std::string_view details) {
		const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
		return fmt::format(
		    "indirect {} resource={} source={} table_source={} words={:08x},{:08x},{:08x},{:08x} "
		    "base={:#x} size={} type={} runtime_stride={} records={} swizzle={} addtid={} oob={} "
		    "maximum={} offset={} immediate={} record_bytes={} child_count={} child_stride={} "
		    "child_offset={} indexed_pointer={} candidates={} {}",
		    reason, resource_index, source_index, indirect.table_source,
		    table_value.dwords[0], table_value.dwords[1], table_value.dwords[2], table_value.dwords[3],
		    table_base, table_size, table.Type(), table.Stride(), table.NumRecords(),
		    table.SwizzleEnabled(), table.AddTid(), table.OutOfBounds(),
		    indirect.table_maximum ? fmt::format("{}", *indirect.table_maximum) : "none",
		    indirect.table_offset, indirect.table_immediate, indirect.table_record_bytes,
		    pointer != nullptr ? pointer->count : 1u, pointer != nullptr ? pointer->stride : 0u,
		    pointer != nullptr ? pointer->offset : 0u, pointer != nullptr && pointer->indexed,
		    descriptors.size(), details);
	};
	const auto failed = [&](std::string_view reason, std::string_view details = {}) {
		return SpecializationFail(describe(reason, details));
	};
	// Failures caused by this draw's guest data rather than by the shader.
	const auto unavailable = [&](std::string_view reason, std::string_view details = {}) {
		return SpecializationUnavailable(program.shader_hash, snapshot, describe(reason, details));
	};
	const auto read_failed = [&](std::string_view reason, uint32_t row, uint32_t child,
	                             uint64_t base, uint64_t offset, size_t bytes) {
		return unavailable(reason, fmt::format("row={} child={} read_base={:#x} read_offset={:#x} "
		                                 "read_address={:#x} requested_bytes={}", row, child, base, offset,
		    (base & AddressMask & ~uint64_t {3}) + (offset & ~uint64_t {3}), bytes));
	};
	const auto normalization_failed = [&](uint64_t offset, uint32_t key, uint64_t step,
	                                      std::string_view key_kind) {
		return failed("candidate", fmt::format("candidate_offset={:#x} {}={} stride={} step={}",
		    offset & ~uint64_t {3}, key_kind, key, indirect.table_stride, step));
	};
	if (sources.empty()) {
		if (selector == nullptr) keys.clear();
		DescriptorValue material_value;
		if ((selector != nullptr && (byte_keys || !selector_keys->resolved) &&
		     !clean.EvaluateDescriptor(selector->source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return unavailable("table evaluation");
		}
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return false;
		}
		if (indirect.bindless) {
			// Nothing to enumerate: the renderer resolves the keys the GPU asks for between
			// frames. The snapshot describes the heap, the flattened SRT reserves the words the
			// renderer patches per draw, and the root samples the null image until then.
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				if (table_value.dword_count != 4u || table.Type() != 0u || table.Stride() == 0u ||
				    table.SwizzleEnabled() || table.AddTid() || table_size > UINT32_MAX)
					return false;
				const auto mapping_offset = bindless_words();
				snapshot.bindless_uses.push_back({.resource       = resource_index,
				                                  .base           = table_base,
				                                  .size           = table_size,
				                                  .record_stride  = table.Stride(),
				                                  .pointer_offset = indirect.pointer_offset,
				                                  .table_offset   = indirect.table_offset,
				                                  .layer_stride   = indirect.table_stride,
				                                  .layer_count    = indirect.layer_count,
				                                  .mapping_offset = static_cast<uint32_t>(mapping_offset)});
				descriptors[resource_index]     = {.dword_count = dword_count};
				auto& root                      = specializations[resource_index];
				root.indirect_root              = resource_index;
				root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
				root.indirect_search_iterations = 0u;
				root.bindless                   = true;
				return true;
			} else {
				return false;
			}
		}
		if (byte_keys && table_size > UINT32_MAX) return failed("table geometry");
		bool use_selector = selector != nullptr;
		if (byte_keys && selector != nullptr) {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			use_selector = CanReadIndexedSelector(material, selector->offset);
			if (use_selector && material.Format() != Prospero::BufferFormat::kInvalid &&
			    material.NumRecords() != 0u && selector->offset < material.Stride()) {
				const auto begin = material.Base48() + selector->offset;
				const auto end = begin + uint64_t {material.NumRecords() - 1u} * material.Stride() + 4u;
				for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
					const auto& buffer = program.info.buffers[i];
					if (!buffer.written) continue;
					ShaderBufferResource written;
					if (program.descriptor_sources[buffer.source].indirect_descriptor ||
					    !DecodeBufferDescriptor(snapshot.buffers[i], written) ||
					    (written.Base48() != 0u && written.GetSize() != 0u &&
					     begin < written.Base48() + written.GetSize() &&
					     written.Base48() < end)) {
						use_selector = false;
						break;
					}
				}
			}
		}
		if (byte_keys && !use_selector) {
			if (!indirect.table_scalar && table.Format() == Prospero::BufferFormat::kInvalid) {
				descriptors[resource_index] = {.dword_count = dword_count};
				return true;
			}
			if (indirect.table_record_bytes != 0u &&
			     (table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			      table.OutOfBounds() != 0u || (table_base & 3u) != 0u ||
			      (table.Stride() & 3u) != 0u || table.Stride() < indirect.table_record_bytes))
				return failed("table geometry");
			const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
			const bool indexed_pointer = pointer != nullptr && pointer->indexed;
			const auto child_count = pointer != nullptr ? pointer->count : 1u;
			uint64_t domain = indirect.table_maximum ? uint64_t {*indirect.table_maximum} + 1u
			                                       : uint64_t {1} << 32u;
			if (!indirect.key_count.IsEmpty()) {
				uint32_t key_count = 0u;
				if (!EvaluateIndirectCount(indirect, runtime, clean, key_count))
					return unavailable("key count evaluation");
				domain = std::min(domain, uint64_t {key_count});
			}
			if (indexed_pointer &&
			    (!indirect.table_maximum || table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			     table.OutOfBounds() != 0u || (table_base & 3u) != 0u || (table.Stride() & 3u) != 0u ||
			     uint64_t {indirect.table_immediate} + 8u > table.Stride() ||
			     domain > table.NumRecords()))
				return failed("indexed pointer geometry", fmt::format("rows={}", domain));
			const bool affine = indirect.table_stride != 0u || indexed_pointer;
			const bool image_heap = dword_count == 8u && pointer == nullptr;
			const uint64_t step = indexed_pointer ? table.Stride()
			    : indirect.table_maximum ? indirect.table_stride
			    : affine ? std::max(uint64_t {1} << std::countr_zero(indirect.table_stride), uint64_t {4})
			             : 4u;
			const uint64_t first = ((indirect.table_maximum ? indirect.table_offset
			    : indirect.table_offset % step) & ~uint64_t {3}) +
			                       (indirect.table_immediate & ~3u);
			if (!indirect.table_maximum) domain /= step;
			const auto count = table_size >= first + 4u
			    ? std::min((table_size - first - 4u) / step + 1u, domain)
			    : 0u;
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				// Children behind parent pointers the GPU reads are whatever those pointers lead
				// to in this draw (Ghost of Yotei's material records: 4 instance rows x 255
				// layers), so a large domain is not enumerated: its program would bake the
				// candidates and recompile for every material. The renderer resolves the keys the
				// GPU asks for; key = row * child count + child is the heap's record * layer_count
				// + layer.
				const auto& image = program.info.images[resource_index];
				if (pointer != nullptr && dword_count == 8u && read_dwords == 8u &&
				    child_count != 0u && count * child_count > BindlessTableKeys &&
				    image.resource_class == ImageResourceClass::Sampled && !image.written &&
				    !image.atomic && !image.depth_compare && !image.cube &&
				    image.dimension == Decoder::ImageDimension::Dim2D && step != 0u &&
				    step <= UINT32_MAX && count * step <= UINT32_MAX && first <= UINT32_MAX - 8u &&
				    (first & 3u) == 0u && (pointer->offset & 3u) == 0u &&
				    (pointer->stride & 3u) == 0u) {
					const auto mapping_offset = bindless_words();
					snapshot.bindless_uses.push_back(
					    {.resource       = resource_index,
					     .base           = table_base,
					     .size           = count * step,
					     .record_stride  = static_cast<uint32_t>(step),
					     .pointer_offset = static_cast<uint32_t>(first),
					     .table_offset   = pointer->offset,
					     .layer_stride   = pointer->stride,
					     .layer_count    = child_count,
					     .mapping_offset = static_cast<uint32_t>(mapping_offset)});
					descriptors[resource_index]     = {.dword_count = dword_count};
					auto& root                      = specializations[resource_index];
					root.indirect_root              = resource_index;
					root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
					root.indirect_search_iterations = 0u;
					root.bindless                   = true;
					return true;
				}
			}
			if (child_count == 0u) return failed("candidate count");
			if (!image_heap && count * child_count > MaxIndirectDescriptorProbes)
				return unavailable("candidate count", fmt::format("rows={} limit={}", count,
				                                                  MaxIndirectDescriptorProbes));
			// Affine SMEM offsets wrap in U32 before adding the aligned immediate.
			// Image heaps retain one bulk snapshot and their absolute byte-offset mapping.
			auto& heap_words = program.material_keys;
			if (image_heap) {
				heap_words.resize(table_size / 4u);
				if (!ReadScalarTable(table_base, table_size, 0u, runtime, heap_words))
					return read_failed("image heap read", 0u, 0u, table_base, 0u, heap_words.size() * sizeof(uint32_t));
			}
			// The T# an image heap holds at a table offset (zero past its end).
			const auto heap_candidate = [&](uint64_t table_offset) {
				DescriptorValue candidate {.dword_count = dword_count};
				const auto      table_word = table_offset / 4u;
				std::copy_n(heap_words.begin() + table_word,
				            std::min<size_t>(read_dwords, heap_words.size() - table_word),
				            candidate.dwords.begin());
				return candidate;
			};
			// An image heap's usable candidates, normalized once per change of its DWORDs (see
			// MaterializationScratch::HeapCandidates); null while any DWORD start fails to
			// normalize, which the enumeration below reports.
			const MaterializationScratch::HeapCandidates* heap = nullptr;
			if (image_heap) {
				auto& heaps = program.materialization_scratch->heaps;
				auto  found = std::ranges::find(heaps, resource_index,
				                                &MaterializationScratch::HeapCandidates::resource);
				if (found == heaps.end()) {
					found = heaps.insert(heaps.end(), {.resource = resource_index});
				}
				auto& cached = *found;
				if (cached.first == first && cached.step == step && cached.count == count &&
				    cached.dwords == read_dwords && cached.affine == affine &&
				    cached.table == heap_words) {
					heap = &cached;
				} else {
					cached.table.clear();
					cached.candidates.clear();
					const auto&                   image = program.info.images[resource_index];
					const std::array<uint32_t, 3> float_2d {
					    static_cast<uint32_t>(Prospero::TextureNumericClass::Float),
					    static_cast<uint32_t>(Decoder::ImageDimension::Dim2D), 0u};
					const DescriptorValue null_candidate {.dword_count = dword_count};
					bool                  normalized = true;
					bool                  all_float  = true;
					for (uint32_t word = 0; word < count && normalized; ++word) {
						const auto table_offset =
						    affine ? first + word * step : uint64_t {word} * 4u;
						auto candidate = heap_candidate(table_offset);
						normalized     = normalize(candidate);
						if (!normalized || candidate == null_candidate) continue;
						all_float =
						    all_float && (NullImageDescriptor(candidate) ||
						                  ImageCandidateClass(image, candidate) == float_2d);
						cached.candidates.emplace_back(table_offset, candidate);
					}
					if (normalized) {
						cached.first    = first;
						cached.step     = step;
						cached.count    = count;
						cached.dwords   = read_dwords;
						cached.affine   = affine;
						cached.float_2d = all_float;
						cached.table.assign(heap_words.begin(), heap_words.end());
						heap = &cached;
					} else {
						cached.candidates.clear();
					}
				}
			}
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				// An image heap yields a candidate per DWORD start it may hold a T# at, and every
				// distinct one is bound at every draw (Ghost of Yotei's PS 2a1e4953d280871c: 33,573
				// per draw). When every usable T# in a large heap is a float 2D texture, the type
				// the bindless image array holds, the heap is a bindless heap instead: the key is
				// the T#'s byte offset in the table, and its records are the table's DWORDs. A heap
				// that also holds T#s of another class (object records with fields for other
				// operations) keeps its enumerated, class-grouped candidates.
				const auto&                   image = program.info.images[resource_index];
				const std::array<uint32_t, 3> float_2d {
				    static_cast<uint32_t>(Prospero::TextureNumericClass::Float),
				    static_cast<uint32_t>(Decoder::ImageDimension::Dim2D), 0u};
				bool bindless = image_heap && read_dwords == 8u && count > BindlessTableKeys &&
				                image.resource_class == ImageResourceClass::Sampled &&
				                !image.written && !image.atomic && !image.depth_compare &&
				                !image.cube && image.dimension == Decoder::ImageDimension::Dim2D;
				if (heap != nullptr) {
					bindless = bindless && heap->float_2d;
				} else {
					for (uint32_t word = 0; word < count && bindless; ++word) {
						const auto table_offset =
						    affine ? first + word * step : uint64_t {word} * 4u;
						auto candidate = heap_candidate(table_offset);
						if (!normalize(candidate))
							return normalization_failed(table_offset, word, step, "probe");
						bindless = NullImageDescriptor(candidate) ||
						           ImageCandidateClass(image, candidate) == float_2d;
					}
				}
				if (bindless) {
					const auto mapping_offset = bindless_words();
					snapshot.bindless_uses.push_back(
					    {.resource       = resource_index,
					     .base           = table_base & AddressMask & ~uint64_t {3},
					     .size           = table_size,
					     .record_stride  = sizeof(uint32_t),
					     .pointer_offset = BindlessUse::NoPointer,
					     .table_offset   = 0u,
					     .layer_stride   = 0u,
					     .layer_count    = 1u,
					     .mapping_offset = static_cast<uint32_t>(mapping_offset)});
					descriptors[resource_index]     = {.dword_count = dword_count};
					auto& root                      = specializations[resource_index];
					root.numeric_class              = Prospero::TextureNumericClass::Float;
					root.dimension                  = Decoder::ImageDimension::Dim2D;
					root.cube                       = false;
					root.indirect_root              = resource_index;
					root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
					root.indirect_search_iterations = 0u;
					root.bindless                   = true;
					root.bindless_byte_keys         = true;
					return true;
				}
			}
			const auto mapping_offset = snapshot.flattened_srt.size();
			descriptors[resource_index] = {.dword_count = dword_count};
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			// A DWORD start whose candidate normalizes to null interns as the null root (ordinal
			// 0) and maps nothing, so only the heap's usable candidates are visited.
			for (const auto& [table_offset, candidate]:
			     heap != nullptr ? std::span(heap->candidates)
			                     : std::span<const std::pair<uint64_t, DescriptorValue>> {}) {
				const auto ordinal = intern_candidate(candidate);
				if (ordinal == 0u) continue;
				const auto mapping_word = table_offset / 4u;
				snapshot.flattened_srt.resize(mapping_offset + 2u + mapping_word, 0u);
				snapshot.flattened_srt[mapping_offset + 1u + mapping_word] = ordinal;
			}
			for (uint32_t word = 0; word < (heap != nullptr ? 0u : count); ++word) {
				const auto table_offset = affine ? first + word * step : uint64_t {word} * 4u;
				uint64_t pointer_base = 0u;
				if (pointer != nullptr) {
					std::array<uint32_t, 2> address {};
					if (!ReadScalarTable(table_base, table_size, table_offset, runtime, address))
						return read_failed("parent pointer read", word, 0u, table_base, table_offset, sizeof(address));
					pointer_base = (uint64_t {address[1]} << 32u) | address[0];
				}
				for (uint32_t child = 0; child < child_count; ++child) {
					DescriptorValue candidate {.dword_count = dword_count};
					if (pointer != nullptr) {
						if (!ReadScalarTable(pointer_base, UINT64_MAX,
						        uint64_t {child} * pointer->stride + pointer->offset, runtime,
						        std::span(candidate.dwords).first(read_dwords)))
							return read_failed("child descriptor read", word, child, pointer_base,
							    uint64_t {child} * pointer->stride + pointer->offset, descriptor_bytes);
					} else if (!image_heap) {
						if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
						                     std::span(candidate.dwords).first(read_dwords)))
							return read_failed("table row read", word, child, table_base,
							                   table_offset, descriptor_bytes);
					} else {
						candidate = heap_candidate(table_offset);
					}
					if (!normalize(candidate))
						return normalization_failed(table_offset, word, step,
						                            indirect.table_maximum ? "key" : "probe");
					const auto ordinal = intern_candidate(candidate);
					if (ordinal == 0u) continue;
					const auto mapping_word = image_heap ? table_offset / 4u : word * child_count + child;
					snapshot.flattened_srt.resize(mapping_offset + 2u + mapping_word, 0u);
					snapshot.flattened_srt[mapping_offset + 1u + mapping_word] = ordinal;
				}
			}

			if (descriptors.size() == children_begin) {
				snapshot.flattened_srt.resize(mapping_offset);
			} else {
				const auto entries = snapshot.flattened_srt.size() - mapping_offset - 1u;
				snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(entries);
				// The key map ends at the last row with a candidate; its reserved length is a
				// power of two so that the maps after it keep their offsets.
				snapshot.flattened_srt.resize(mapping_offset + 1u + std::bit_ceil(entries), 0u);
				// Store candidates are bound buffers; tables of them are as large as the scene's
				// object list (Ghost of Yotei CS 0x8457901d80b91921: about 60 per scene).
				if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Buffer>)
					pad_candidates();
				if (!group_image_candidates(mapping_offset + 1u, entries, 1u)) return false;
				auto& root = specializations[resource_index];
				root.indirect_root = resource_index;
				publish_mapping(root, mapping_offset);
				root.indirect_search_iterations = 0u;
			}
			return true;
		}
		if (selector == nullptr) {
			uint32_t key_count = 0;
			uint32_t first_key = 0;
			if (!EvaluateIndirectCount(indirect, runtime, clean, key_count) ||
			    (!indirect.selector_first.IsEmpty() &&
			     !clean.Evaluate(indirect.selector_first, first_key)))
				return unavailable("key range evaluation");
			if (indirect.table_stride == 0u) return false;
			if (key_count > MaxIndirectDescriptorProbes ||
			    (indirect.selector_first.IsEmpty() && key_count != 0u &&
			         uint64_t {indirect.table_offset} + indirect.table_immediate +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return unavailable("key range",
				                   fmt::format("first={} count={} limit={}", first_key, key_count,
				                               MaxIndirectDescriptorProbes));
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), first_key);
			if (!keys.empty() && keys.back() < keys.front())
				std::rotate(keys.begin(), keys.begin() + (UINT32_MAX - first_key + 1ull), keys.end());
		} else {
			if (!indirect.selector_mask.IsEmpty() && table_value.dword_count != 2u) return false;
			if (!selector_keys->resolved) {
				bool data_limit = false;
				if (!ResolveSelectorKeys(indirect, material_value, runtime, clean, keys, data_limit,
				                         *program.materialization_scratch)) {
					const auto details = fmt::format(
					    "selector_source={} stride={} offset={} workgroup_axis={} limit={}",
					    selector->source, selector->stride, selector->offset,
					    indirect.workgroup_axis, MaxIndirectDescriptorProbes);
					return data_limit ? unavailable("selector keys", details)
					                  : failed("selector keys", details);
				}
				selector_keys->resolved = true;
			}
		}
		if (byte_keys) {
			// Keep cached selector values independent of each destination table's
			// U32 arithmetic, and use the same byte keys as the full-heap GPU path.
			byte_offsets.clear();
			byte_offsets.reserve(keys.size());
			for (const auto key: keys) {
				const auto offset = uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
				                    (indirect.table_immediate & ~3u);
				byte_offsets.push_back(offset <= UINT32_MAX ? static_cast<uint32_t>(offset) : UINT32_MAX);
			}
			std::ranges::sort(byte_offsets);
			byte_offsets.erase(std::unique(byte_offsets.begin(), byte_offsets.end()), byte_offsets.end());
		}
	}

	if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
		// A large table of T#s at base + key * stride: a bindless heap of direct records. A
		// selector only narrows which keys the host enumerates; the shader still forms the key
		// itself, so the heap covers every record up to the largest key.
		const uint64_t records =
		    keys.empty() ? 0u : uint64_t {*std::ranges::max_element(keys)} + 1u;
		if (sources.empty() && indirect.workgroup_axis == UINT32_MAX &&
		    indirect.pointer_offset == UINT32_MAX && keys.size() > BindlessTableKeys &&
		    dword_count == 8u && indirect.table_stride != 0u &&
		    (indirect.table_stride & 3u) == 0u && (indirect.table_offset & 3u) == 0u &&
		    (indirect.table_immediate & 3u) == 0u && records * indirect.table_stride <= UINT32_MAX) {
			const auto mapping_offset = bindless_words();
			snapshot.bindless_uses.push_back(
			    {.resource       = resource_index,
			     .base           = table_base,
			     .size           = records * indirect.table_stride,
			     .record_stride  = indirect.table_stride,
			     .pointer_offset = BindlessUse::NoPointer,
			     .table_offset   = indirect.table_offset + indirect.table_immediate,
			     .layer_stride   = 0u,
			     .layer_count    = 1u,
			     .mapping_offset = static_cast<uint32_t>(mapping_offset)});
			descriptors[resource_index]     = {.dword_count = dword_count};
			auto& root                      = specializations[resource_index];
			root.indirect_root              = resource_index;
			root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
			root.indirect_search_iterations = 0u;
			root.bindless                   = true;
			return true;
		}
	}
	// The program bakes the search depth and the candidate array size, so a table whose entry
	// count changes with the scene (Ghost of Yotei's streaming tables: 69 then 80 candidates)
	// would compile a permutation per change. Both are padded to a power of two; the shader
	// searches the count the SRT holds, and spare candidate slots hold null descriptors.
	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto selected_keys = byte_keys ? std::span<const uint32_t>(byte_offsets)
	                                    : std::span<const uint32_t>(keys);
	const auto key_count = sources.empty() ? selected_keys.size() : sources.size();
	const auto padded_keys = std::bit_ceil(std::max<size_t>(key_count, 1u));
	snapshot.flattened_srt.resize(mapping_offset + 1u + padded_keys * 2u, 0u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? selected_keys[entry] : entry;
		uint64_t table_offset = 0;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			table_offset = byte_keys ? key
			    : uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
			          (indirect.table_immediate & ~3u);
			if (!(byte_keys && key == UINT32_MAX) &&
			    !ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(read_dwords))) {
				return read_failed("table descriptor read", key, 0u, table_base, table_offset, descriptor_bytes);
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return unavailable("candidate evaluation", fmt::format("entry={}", entry));
		}
		if (!normalize(candidate))
			return normalization_failed(table_offset, key, indirect.table_stride, "key");
		uint32_t ordinal = 0;
		if (entry == 0 && !byte_keys) {
			descriptors[resource_index] = candidate;
		} else {
			ordinal = intern_candidate(candidate);
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		const auto unique = descriptors.size() - children_begin + 1u;
		if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
			if (!group_image_candidates(mapping_offset + 2u, key_count, 2u)) return false;
		} else {
			pad_candidates();
		}
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		publish_mapping(root, mapping_offset);
		root.indirect_search_iterations = std::bit_width(padded_keys);
	}
	return true;
}

} // namespace

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::vector<std::array<uint32_t, 2>> mapping;
	std::vector<Binding>                bindings;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

// Where an indirect root's key map starts in the flattened SRT (SIZE_MAX when out of range).
template <typename Specialization>
size_t KeyMapOffset(std::span<const uint32_t> flat, const Specialization& root) {
	if (!root.indirect_mapping_header) return root.indirect_mapping_offset;
	return root.indirect_mapping_offset < flat.size() ? flat[root.indirect_mapping_offset]
	                                                  : SIZE_MAX;
}

size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Buffer& root) {
	return KeyMapOffset(std::span<const uint32_t>(snapshot.flattened_srt), root);
}
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Image& root) {
	return KeyMapOffset(std::span<const uint32_t>(snapshot.flattened_srt), root);
}
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Sampler& root) {
	return KeyMapOffset(std::span<const uint32_t>(snapshot.flattened_srt), root);
}

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization) {
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const auto raw_sint_format = RawSignedStorageFormat(base, format);
		const bool raw_sint_storage = raw_sint_format != Prospero::BufferFormat::kInvalid;
		if (raw_sint_format == Prospero::BufferFormat::k16UInt) image.conversion_format = format;
		image.numeric_class = Prospero::SampledTextureNumericClass(
		    Prospero::RemapTextureFormat(format));
		if (!storage && format == Prospero::BufferFormat::k11_11_10UNorm) {
			if (image.indirect_root != ImageResource::NoIndirectImage) {
				return SpecializationFail("packed UNorm sampling requires a direct image");
			}
			if (std::ranges::any_of(program.info.sampled_pairs,
			                       [&](const auto& pair) { return pair.image == i; }) &&
			    !ValidatePackedUnormSampling(program, snapshot, i)) return false;
		}
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			auto message = fmt::format(
			    "sampled image descriptor {} uses unsupported format {}: root={} source={} "
			    "first_use_pc={:#x} r128={} words={:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, static_cast<uint32_t>(format), base_index, base.source, base.first_use_pc,
			    base.r128, descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2],
			    descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5],
			    descriptor.dwords[6], descriptor.dwords[7]);
			const auto* source = Source(program, base.source);
			if (i >= program.info.images.size() && source != nullptr &&
			    source->indirect_descriptor &&
			    IsBoundedDescriptorTable(program, *source->indirect_descriptor) &&
			    specialization.images[base_index].indirect_search_iterations == 0u &&
			    !source->indirect_descriptor->pointer_table) {
				const auto& indirect = *source->indirect_descriptor;
				const auto&  flat     = snapshot.flattened_srt;
				const size_t mapping  = KeyMapOffset(flat, specialization.images[base_index]);
				if (mapping < flat.size() && flat[mapping] <= flat.size() - mapping - 1u) {
					uint32_t ordinal = 0;
					for (size_t child = program.info.images.size(); child <= i; ++child)
						ordinal += specialization.images[child].indirect_root == base_index;
					uint32_t matches = 0;
					uint64_t first_offset = 0;
					for (uint32_t probe = 0; probe < flat[mapping]; ++probe) {
						if (flat[mapping + 1u + probe] != ordinal) continue;
						if (matches++ == 0u) first_offset = uint64_t {probe} * 4u;
					}
					message += fmt::format(" table_source={} ordinal={} dense_matches={} "
					                       "first_candidate_offset={:#x}",
					                       indirect.table_source, ordinal, matches, first_offset);
				}
			}
			return SpecializationFail(message);
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue; // bindless: resolved at run time, the class comes from the instruction
		}
		const auto map_offset = KeyMapOffset(snapshot.flattened_srt, root);
		const auto key_count =
		    map_offset < snapshot.flattened_srt.size() ? snapshot.flattened_srt[map_offset] : 0u;
		if (key_count == 0u || map_offset + 1u +
		                               static_cast<size_t>(key_count) *
		                                   (root.indirect_search_iterations == 0u ? 1u : 2u) >
		                           snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto& base = program.info.images[root_index];
		const bool storage_swizzle = base.unfiltered_access_only &&
		    base.resource_class == ImageResourceClass::Storage;
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		const auto is_1d_or_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim1D ||
			       dimension == Decoder::ImageDimension::Dim2D;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			const bool unfiltered_coordinates = base.unfiltered_access_only &&
			    base.dimension == Decoder::ImageDimension::Dim2D &&
			    !image.cube && !image_class.cube && is_1d_or_2d(image.dimension) &&
			    is_1d_or_2d(image_class.dimension);
			if ((!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension)) &&
			     !unfiltered_coordinates) ||
			    image.conversion_format != image_class.conversion_format ||
			    (!storage_swizzle && image.shader_swizzle != image_class.shader_swizzle)) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	plan.mapping.resize(base.samplers.size());
	plan.bindings.resize(base.samplers.size());
	std::vector<uint8_t> usage(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
		for (const auto candidate: base.images[pair.image].indirect_resources) {
			if (candidate >= base.images.size()) return false;
			usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[candidate]));
		}
		for (const auto candidate: base.samplers[pair.sampler].indirect_resources) {
			if (candidate >= base.samplers.size()) return false;
			usage[candidate] |= usage[pair.sampler];
		}
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : static_cast<uint32_t>(plan.bindings.size());
			if (!first) plan.bindings.emplace_back();
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	// The renderer checks captured scalar reads against final buffer and image write ranges.
	if (program.has_address_writes) {
		return {};
	}
	std::unordered_map<const Block*, uint32_t> indices;
	for (uint32_t i = 0; i < program.blocks.size(); i++) {
		if (!indices.emplace(program.blocks[i], i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&             info       = *program.blocks[i];
		const auto&           terminator = info.terminator;
		const std::array        targets {terminator.true_block, terminator.false_block};
		std::span<Block* const> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors = std::span(targets).first(1); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors      = targets;
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch: successors = info.ImmSuccessors(); break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	for (auto& block: blocks) block.condition = predicate(block.condition);
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard) {
	return AffineIndex<3>(value, [&](Value at) { return ResolveActiveU32(at, guard); },
	    [&](Value at) -> std::optional<size_t> {
		    const auto* inst = at.TryInstruction();
		    if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetBuiltin ||
		        inst->Arg(1) != Value(axis)) return {};
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) return 1u;
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) return 2u;
		    return {};
	    });
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const auto*                      block = program.blocks.front();
	const Inst*                      store = nullptr;
	for (;;) {
		if (block == nullptr || !visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		block = term.true_block;
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true)) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

// Whether every formatted untyped read of a base buffer returns zero out of bounds: only a
// constant-one selector makes an out-of-bounds component nonzero. A plan without component
// masks counts every component.
static bool FormattedOutOfBoundsZero(const ResourcePlan& program, uint32_t resource,
                                     uint32_t swizzle) {
	const uint32_t components = resource < program.formatted_components.size()
	                                ? program.formatted_components[resource]
	                                : 0xfu;
	for (uint32_t component = 0; component < 4u; ++component) {
		if (((components >> component) & 1u) != 0u && GetDstSel(swizzle, component) == 1u)
			return false;
	}
	return true;
}

// Whether a module compiled with `compiled` behaves as one compiled with `current`. The flag
// itself never changes the module, so the same layout always does. An unbound buffer is bound as
// the null buffer behind an offset past its end, so every access of any layout misses: loads
// read the out-of-bounds value, stores and atomics are dropped. Another layout of a direct
// buffer therefore behaves identically whenever both layouts read zero out of bounds.
static bool ServesBuffer(const ResourcePlan& program, uint32_t resource,
                         const ResourceSpecialization::Buffer& compiled,
                         const ResourceSpecialization::Buffer& current) {
	auto layout    = compiled;
	layout.unbound = current.unbound;
	if (layout == current) return true;
	return current.unbound && resource < program.info.buffers.size() &&
	       compiled.indirect_root == BufferResource::NoIndirectBuffer &&
	       current.indirect_root == BufferResource::NoIndirectBuffer &&
	       compiled.indirect_mapping_offset == current.indirect_mapping_offset &&
	       compiled.indirect_search_iterations == current.indirect_search_iterations &&
	       FormattedOutOfBoundsZero(program, resource, compiled.descriptor_swizzle) &&
	       FormattedOutOfBoundsZero(program, resource, current.descriptor_swizzle);
}

bool SpecializationServes(const ResourcePlan& program, const ResourceSpecialization& compiled,
                          const ResourceSpecialization& current) {
	if (compiled.buffers.size() != current.buffers.size() || compiled.images != current.images ||
	    compiled.samplers != current.samplers)
		return false;
	for (uint32_t i = 0; i < current.buffers.size(); ++i) {
		if (compiled.buffers[i] != current.buffers[i] &&
		    !ServesBuffer(program, i, compiled.buffers[i], current.buffers[i]))
			return false;
	}
	return true;
}

void AdoptServingBuffers(const ResourcePlan& program, ResourceSpecialization& target,
                         const ResourceSpecialization& donor) {
	if (donor.buffers.size() != target.buffers.size()) return;
	for (uint32_t i = 0; i < target.buffers.size(); ++i) {
		if (target.buffers[i].unbound && !donor.buffers[i].unbound &&
		    ServesBuffer(program, i, donor.buffers[i], target.buffers[i])) {
			target.buffers[i]         = donor.buffers[i];
			target.buffers[i].unbound = true;
		}
	}
}

// Candidate layouts across draws. A GPU-selected image table's candidates differ from draw to
// draw (Ghost of Yotei streams the objects whose records the tables hold), and so do the class
// groups and their power-of-two padding, which the module bakes. A module compiled for a layout
// with at least the slots a draw needs serves it once the draw's candidates are moved into those
// slots; and a draw that no module serves gets a module for the union of its layout and the last
// one, so a shader's layouts only grow and settle after a few modules.
namespace {

using SpecImage = ResourceSpecialization::Image;

// A deeper key search finds every key a shallower one finds: the search is bounded by the
// map's run-time key count.
bool SearchServes(uint32_t compiled, uint32_t current) {
	return (compiled == 0u) == (current == 0u) && compiled >= current;
}

template <typename T>
bool SameExceptSearch(const T& compiled, const T& current) {
	auto layout                       = compiled;
	layout.indirect_search_iterations = current.indirect_search_iterations;
	return layout == current &&
	       SearchServes(compiled.indirect_search_iterations, current.indirect_search_iterations);
}

template <typename T>
T DeeperSearch(const T& a, const T& b) {
	auto result = b;
	result.indirect_search_iterations =
	    std::max(a.indirect_search_iterations, b.indirect_search_iterations);
	return result;
}

bool MappedImageRoot(const std::vector<SpecImage>& images, uint32_t root) {
	return images[root].indirect_root == root && !images[root].bindless;
}

bool SameImageClass(const SpecImage& a, const SpecImage& b) {
	return a.numeric_class == b.numeric_class && a.dimension == b.dimension &&
	       a.mip_count == b.mip_count && a.conversion_format == b.conversion_format &&
	       a.shader_swizzle == b.shader_swizzle && a.cube == b.cube && a.fmask == b.fmask &&
	       a.bindless == b.bindless && a.bindless_byte_keys == b.bindless_byte_keys;
}

// A root the draw does not map (an inactive source, or a table without candidates) still has
// its header naming the empty map. A layout that maps the root then selects the root's slot
// for every key, as the draw does, provided that slot reads the root's descriptor alike.
bool EmptyRootServed(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                     const SpecImage& compiled, const SpecImage& current, uint32_t root,
                     const DescriptorValue& root_descriptor) {
	const auto* source = Source(program, program.info.images[root].source);
	if (source == nullptr || !source->indirect_descriptor || current.bindless ||
	    current.indirect_root == root || !compiled.indirect_mapping_header)
		return false;
	const auto& flat   = snapshot.flattened_srt;
	const auto  header = compiled.indirect_mapping_offset;
	if (header >= flat.size() || flat[header] >= flat.size() || flat[flat[header]] != 0u)
		return false;
	return NullImageDescriptor(root_descriptor) || SameImageClass(compiled, current);
}

// Visits the key map words of a mapped root that hold candidate ordinals: a bounded map holds
// one ordinal per entry, a searched map a (key, ordinal) pair per entry.
template <typename Root, typename Visit>
bool VisitOrdinals(std::vector<uint32_t>& flat, const Root& root, Visit&& visit) {
	const auto offset = KeyMapOffset(flat, root);
	if (offset >= flat.size()) return false;
	const size_t count  = flat[offset];
	const size_t stride = root.indirect_search_iterations == 0u ? 1u : 2u;
	const size_t first  = offset + stride;
	if (count != 0u && first + (count - 1u) * stride >= flat.size()) return false;
	for (size_t entry = 0; entry < count; ++entry)
		visit(flat[first + entry * stride]);
	return true;
}

// Moves the draw's image candidates into the slots of `compiled` (see AdaptToSpecialization);
// with `apply` false, only answers whether they fit.
bool AdaptImages(const ResourcePlan& program, const std::vector<SpecImage>& compiled,
                 ResourceSnapshot& snapshot, const std::vector<SpecImage>& current, bool apply) {
	const auto direct = static_cast<uint32_t>(program.info.images.size());
	if (compiled.size() < direct || current.size() < direct) return false;
	// Snapshot images skip FMASK entries (BuildResourceSpecialization), which are direct only.
	std::vector<uint32_t> snapshot_index(current.size(), UINT32_MAX);
	for (uint32_t index = 0, position = 0; index < current.size(); ++index) {
		if (!current[index].fmask) snapshot_index[index] = position++;
	}
	const auto descriptor = [&](uint32_t index) -> const DescriptorValue* {
		const auto position = snapshot_index[index];
		return position < snapshot.images.size() ? &snapshot.images[position] : nullptr;
	};
	for (uint32_t root = 0; root < direct; ++root) {
		if (compiled[root].fmask != current[root].fmask) return false;
		if (SameExceptSearch(compiled[root], current[root])) continue;
		const auto* root_descriptor = descriptor(root);
		if (root_descriptor == nullptr || !MappedImageRoot(compiled, root) ||
		    !EmptyRootServed(program, snapshot, compiled[root], current[root], root,
		                     *root_descriptor))
			return false;
	}
	std::vector<std::vector<uint32_t>> slots(direct);
	std::vector<std::vector<uint32_t>> children(direct);
	for (uint32_t index = direct; index < compiled.size(); ++index) {
		const auto root = compiled[index].indirect_root;
		if (root >= direct || !MappedImageRoot(compiled, root)) return false;
		slots[root].push_back(index);
	}
	for (uint32_t index = direct; index < current.size(); ++index) {
		const auto root = current[index].indirect_root;
		if (root >= direct || !MappedImageRoot(current, root) || !MappedImageRoot(compiled, root))
			return false;
		children[root].push_back(index);
	}
	// Per root: the draw's candidate ordinal -> the compiled layout's (0 stays the root).
	std::vector<std::vector<uint32_t>> renumber(direct);
	std::vector<uint8_t>               used;
	std::vector<uint8_t>               referenced;
	for (uint32_t root = 0; root < direct; ++root) {
		if (children[root].empty()) continue;
		referenced.assign(children[root].size() + 1u, 0u);
		bool valid_ordinals = true;
		if (!VisitOrdinals(snapshot.flattened_srt, current[root],
		                   [&](uint32_t ordinal) {
			                   if (ordinal < referenced.size()) {
				                   referenced[ordinal] = 1u;
			                   } else {
				                   valid_ordinals = false;
			                   }
		                   }) ||
		    !valid_ordinals)
			return false;
		auto& map = renumber[root];
		map.assign(children[root].size() + 1u, 0u);
		used.assign(slots[root].size(), 0u);
		// A typed candidate needs a slot of its own layout; a null one reads the null descriptor
		// through any slot, so it takes what the typed ones left.
		for (const bool typed: {true, false}) {
			for (uint32_t ordinal = 1; ordinal < referenced.size(); ++ordinal) {
				if (!referenced[ordinal]) continue;
				const auto  child = children[root][ordinal - 1u];
				const auto* value = descriptor(child);
				if (value == nullptr) return false;
				if (NullImageDescriptor(*value) == typed) continue;
				uint32_t found = UINT32_MAX;
				for (uint32_t slot = 0; slot < slots[root].size() && found == UINT32_MAX; ++slot) {
					if (!used[slot] && compiled[slots[root][slot]] == current[child]) found = slot;
				}
				for (uint32_t slot = 0; !typed && slot < slots[root].size() && found == UINT32_MAX;
				     ++slot) {
					if (!used[slot]) found = slot;
				}
				if (found == UINT32_MAX) return false;
				used[found]  = 1u;
				map[ordinal] = found + 1u;
			}
		}
	}
	if (!apply) return true;
	std::vector<uint32_t> slot_source(compiled.size(), UINT32_MAX);
	for (uint32_t root = 0; root < direct; ++root) {
		for (uint32_t ordinal = 1; ordinal < renumber[root].size(); ++ordinal) {
			if (renumber[root][ordinal] != 0u)
				slot_source[slots[root][renumber[root][ordinal] - 1u]] =
				    children[root][ordinal - 1u];
		}
	}
	std::vector<DescriptorValue> images;
	images.reserve(compiled.size());
	for (uint32_t index = 0; index < compiled.size(); ++index) {
		if (compiled[index].fmask) continue;
		const auto source = index < direct ? index : slot_source[index];
		images.push_back(source != UINT32_MAX ? *descriptor(source)
		                                      : DescriptorValue {.dword_count = 8u});
	}
	for (uint32_t root = 0; root < direct; ++root) {
		if (children[root].empty()) continue;
		VisitOrdinals(snapshot.flattened_srt, current[root],
		              [&](uint32_t& ordinal) { ordinal = renumber[root][ordinal]; });
	}
	snapshot.images = std::move(images);
	return true;
}

using SpecSampler = ResourceSpecialization::Sampler;

bool MappedSamplerRoot(const std::vector<SpecSampler>& samplers, uint32_t root) {
	return samplers[root].indirect_root == root;
}

// A sampler root the draw does not map still names the empty map (see EmptyRootServed).
bool EmptySamplerRootServed(const ResourceSnapshot& snapshot, const SpecSampler& compiled,
                            const SpecSampler& current, uint32_t root) {
	const auto& flat   = snapshot.flattened_srt;
	const auto  header = compiled.indirect_mapping_offset;
	return current.indirect_root != root && compiled.indirect_mapping_header &&
	       header < flat.size() && flat[header] < flat.size() && flat[flat[header]] == 0u;
}

// Moves the draw's GPU-selected sampler candidates into the slots of `compiled`, as AdaptImages
// does for images (the sampler candidates of a root share one layout).
bool AdaptSamplers(const ResourcePlan& program, const std::vector<SpecSampler>& compiled,
                   ResourceSnapshot& snapshot, const std::vector<SpecSampler>& current,
                   bool apply) {
	const auto direct = static_cast<uint32_t>(program.info.samplers.size());
	if (compiled.size() < direct || current.size() < direct ||
	    snapshot.samplers.size() != current.size())
		return false;
	for (uint32_t root = 0; root < direct; ++root) {
		if (SameExceptSearch(compiled[root], current[root])) continue;
		if (!MappedSamplerRoot(compiled, root) ||
		    !EmptySamplerRootServed(snapshot, compiled[root], current[root], root))
			return false;
	}
	std::vector<std::vector<uint32_t>> slots(direct);
	std::vector<std::vector<uint32_t>> children(direct);
	for (uint32_t index = direct; index < compiled.size(); ++index) {
		const auto root = compiled[index].indirect_root;
		if (root >= direct || !MappedSamplerRoot(compiled, root)) return false;
		slots[root].push_back(index);
	}
	for (uint32_t index = direct; index < current.size(); ++index) {
		const auto root = current[index].indirect_root;
		if (root >= direct || !MappedSamplerRoot(current, root) ||
		    !MappedSamplerRoot(compiled, root))
			return false;
		children[root].push_back(index);
	}
	std::vector<std::vector<uint32_t>> renumber(direct);
	std::vector<uint8_t>               referenced;
	for (uint32_t root = 0; root < direct; ++root) {
		if (children[root].empty()) continue;
		referenced.assign(children[root].size() + 1u, 0u);
		bool valid_ordinals = true;
		if (!VisitOrdinals(snapshot.flattened_srt, current[root],
		                   [&](uint32_t ordinal) {
			                   if (ordinal < referenced.size()) {
				                   referenced[ordinal] = 1u;
			                   } else {
				                   valid_ordinals = false;
			                   }
		                   }) ||
		    !valid_ordinals)
			return false;
		auto& map = renumber[root];
		map.assign(children[root].size() + 1u, 0u);
		uint32_t next = 0;
		for (uint32_t ordinal = 1; ordinal < referenced.size(); ++ordinal) {
			if (!referenced[ordinal]) continue;
			while (next < slots[root].size() &&
			       compiled[slots[root][next]] != current[children[root][ordinal - 1u]])
				++next;
			if (next == slots[root].size()) return false;
			map[ordinal] = ++next;
		}
	}
	if (!apply) return true;
	std::vector<DescriptorValue> samplers(compiled.size(), DescriptorValue {.dword_count = 4u});
	std::copy_n(snapshot.samplers.begin(), direct, samplers.begin());
	for (uint32_t root = 0; root < direct; ++root) {
		for (uint32_t ordinal = 1; ordinal < renumber[root].size(); ++ordinal) {
			if (renumber[root][ordinal] != 0u)
				samplers[slots[root][renumber[root][ordinal] - 1u]] =
				    snapshot.samplers[children[root][ordinal - 1u]];
		}
		if (children[root].empty()) continue;
		VisitOrdinals(snapshot.flattened_srt, current[root],
		              [&](uint32_t& ordinal) { ordinal = renumber[root][ordinal]; });
	}
	snapshot.samplers = std::move(samplers);
	return true;
}

// The slot layouts of a root's children in `layout` plus those `mine` has beyond them: a layout
// the draw has more of grows to the next power of two (so that a growing table takes few
// modules), next to the slots of the same layout so that runs stay contiguous.
template <typename Spec>
std::vector<Spec> UnionLayout(std::vector<Spec> layout, const std::vector<Spec>& mine) {
	for (const auto& child: mine) {
		const auto need = static_cast<size_t>(std::ranges::count(mine, child));
		const auto have = static_cast<size_t>(std::ranges::count(layout, child));
		if (need <= have) continue;
		const auto last = std::find(layout.rbegin(), layout.rend(), child);
		const auto at   = last == layout.rend() ? layout.end() : last.base();
		layout.insert(at, std::bit_ceil(need) - have, child);
	}
	return layout;
}

} // namespace

bool AdaptToSpecialization(const ResourcePlan& program, const ResourceSpecialization& compiled,
                           ResourceSnapshot& snapshot, ResourceSpecialization& current) {
	if (compiled.buffers.size() != current.buffers.size()) return false;
	for (uint32_t i = 0; i < current.buffers.size(); ++i) {
		if (!SameExceptSearch(compiled.buffers[i], current.buffers[i]) &&
		    !ServesBuffer(program, i, compiled.buffers[i], current.buffers[i]))
			return false;
	}
	const bool samplers = compiled.samplers != current.samplers;
	const bool images   = compiled.images != current.images;
	if ((samplers &&
	     !AdaptSamplers(program, compiled.samplers, snapshot, current.samplers, false)) ||
	    (images && !AdaptImages(program, compiled.images, snapshot, current.images, false)))
		return false;
	if (samplers) AdaptSamplers(program, compiled.samplers, snapshot, current.samplers, true);
	if (images) AdaptImages(program, compiled.images, snapshot, current.images, true);
	for (uint32_t i = 0; i < current.buffers.size(); ++i) {
		current.buffers[i].indirect_search_iterations =
		    compiled.buffers[i].indirect_search_iterations;
	}
	current.images   = compiled.images;
	current.samplers = compiled.samplers;
	return true;
}

void WidenSpecialization(const ResourcePlan& program, const ResourceSpecialization& prior,
                         ResourceSnapshot& snapshot, ResourceSpecialization& current) {
	const auto direct = static_cast<uint32_t>(program.info.images.size());
	if (prior.images.size() < direct || current.images.size() < direct) return;
	const auto either_way = [](const auto& a, const auto& b) {
		return SameExceptSearch(a, b) || SameExceptSearch(b, a);
	};
	auto widened = current;
	if (prior.buffers.size() == current.buffers.size()) {
		for (uint32_t i = 0; i < current.buffers.size(); ++i) {
			if (either_way(prior.buffers[i], current.buffers[i]))
				widened.buffers[i] = DeeperSearch(prior.buffers[i], current.buffers[i]);
		}
	}
	const auto children_of = [](const auto& entries, size_t first, uint32_t root) {
		std::remove_cvref_t<decltype(entries)> result;
		for (size_t index = first; index < entries.size(); ++index) {
			if (entries[index].indirect_root == root) result.push_back(entries[index]);
		}
		return result;
	};
	const auto sampler_roots = static_cast<uint32_t>(program.info.samplers.size());
	if (prior.samplers.size() >= sampler_roots && current.samplers.size() >= sampler_roots) {
		std::vector<std::vector<SpecSampler>> layouts(sampler_roots);
		for (uint32_t root = 0; root < sampler_roots; ++root) {
			const auto& old  = prior.samplers[root];
			const auto& draw = current.samplers[root];
			if (either_way(old, draw)) {
				widened.samplers[root] = DeeperSearch(old, draw);
				if (MappedSamplerRoot(current.samplers, root))
					layouts[root] = UnionLayout(children_of(prior.samplers, sampler_roots, root),
					                            children_of(current.samplers, sampler_roots, root));
			} else if (MappedSamplerRoot(prior.samplers, root) &&
			           EmptySamplerRootServed(snapshot, old, draw, root)) {
				widened.samplers[root] = old;
				layouts[root]          = children_of(prior.samplers, sampler_roots, root);
			} else if (MappedSamplerRoot(current.samplers, root)) {
				layouts[root] = children_of(current.samplers, sampler_roots, root);
			}
		}
		widened.samplers.resize(sampler_roots);
		for (const auto& layout: layouts)
			widened.samplers.insert(widened.samplers.end(), layout.begin(), layout.end());
	}
	std::vector<uint32_t> snapshot_index(current.images.size(), UINT32_MAX);
	for (uint32_t index = 0, position = 0; index < current.images.size(); ++index) {
		if (!current.images[index].fmask) snapshot_index[index] = position++;
	}
	// Per root, the children layouts in order (UnionLayout).
	std::vector<std::vector<SpecImage>> layouts(direct);
	for (uint32_t root = 0; root < direct; ++root) {
		const auto& old      = prior.images[root];
		const auto& draw     = current.images[root];
		const auto  position = snapshot_index[root];
		if (either_way(old, draw)) {
			widened.images[root] = DeeperSearch(old, draw);
			if (!MappedImageRoot(current.images, root)) continue;
			layouts[root] = UnionLayout(children_of(prior.images, direct, root),
			                            children_of(current.images, direct, root));
		} else if (MappedImageRoot(prior.images, root) && position < snapshot.images.size() &&
		           EmptyRootServed(program, snapshot, old, draw, root, snapshot.images[position])) {
			widened.images[root] = old;
			layouts[root]        = children_of(prior.images, direct, root);
		} else if (MappedImageRoot(current.images, root)) {
			layouts[root] = children_of(current.images, direct, root);
		}
	}
	widened.images.resize(direct);
	for (uint32_t root = 0; root < direct; ++root) {
		widened.images.insert(widened.images.end(), layouts[root].begin(), layouts[root].end());
	}
	// A layout that keeps growing is not worth its descriptors; the draw keeps its own.
	if (widened.images.size() > std::max<size_t>(2u * current.images.size(), 1024u)) return;
	auto adapted = snapshot;
	auto result  = current;
	if (AdaptToSpecialization(program, widened, adapted, result)) {
		snapshot = std::move(adapted);
		current  = std::move(result);
	}
}

ResourcePlan ExtractResourcePlan(const Program& program, SrtWalkerCompilation walkers) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.formatted_components.assign(program.info.buffers.size(), 0u);
	for (const auto& memory: program.memory_info) {
		if (memory.kind != ResourceKind::Buffer || !memory.formatted || memory.typed ||
		    memory.resource >= plan.formatted_components.size())
			continue;
		const auto end = std::min(
		    memory.component_index + std::max(memory.data_dwords, memory.component_count), 4u);
		for (uint32_t component = memory.component_index; component < end; ++component)
			plan.formatted_components[memory.resource] |= static_cast<uint8_t>(1u << component);
	}
	plan.source_reads               = program.source_reads;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = source->Arg(1).Resolve();
			EXIT_IF(!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			        slot.U32() >= program.srt_reads.size());
			return Clone(program.srt_reads[slot.U32()].value);
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		if (source->GetOpcode() == ValueOpcode::LoadAddressU32 ||
		    source->GetOpcode() == ValueOpcode::ReadConstBuffer ||
		    source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			target.SetFlags(SrtReadFlags {.index = source->Flags<MemoryFlags>().index});
		}
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	bool capture_indirect_reads = false;
	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		if (target.indirect_descriptor.has_value()) {
			plan.requires_specialization_memory = true;
			capture_indirect_reads = true;
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	// Destination tables share keys when their complete selector domains agree.
	// Resolve those domains lazily at dispatch time, without repeated guest reads.
	plan.selector_groups.resize(plan.descriptor_sources.size(), UINT32_MAX);
	for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source].indirect_descriptor;
		if (!descriptor || !descriptor->selector) continue;
		const auto& a = *descriptor;
		uint32_t group = 0;
		for (; group < plan.selector_keys.size(); ++group) {
			const auto& b = *program.descriptor_sources[plan.selector_keys[group].source].indirect_descriptor;
			if (EquivalentSelectorDomain(program, a, b)) break;
		}
		if (group == plan.selector_keys.size()) plan.selector_keys.push_back({.source = source});
		plan.selector_groups[source] = group;
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// Retain integer control expressions with opaque varying leaves. Their identities
	// matter when an EXEC factor also guards a selected value; never clone float work.
	bool capture_control_reads = false;
	bool uniform_control = false;
	const auto control_value_count = plan.value_storage.size();
	struct ControlValue {
		Value value;
		bool complete = false;
	};
	std::unordered_map<const Inst*, ControlValue> predicates;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			uniform_control |= value.GetType() == Type::U1;
			return ValidateRuntimeValue(program, value, RuntimeValueType::Integer) ? value : Value {};
		}
		const auto [found, inserted] = predicates.try_emplace(inst);
		auto& entry = found->second;
		auto& result = entry.value;
		const auto opaque = [&]() -> Value {
			if (value.GetType() != Type::U1 && value.GetType() != Type::U32) return {};
			return result = Value(&plan.value_storage.emplace_back(
			    value.GetType() == Type::U1 ? ValueOpcode::UndefU1 : ValueOpcode::UndefU32));
		};
		// A recursive projection remains opaque, including unstructured loop Phis.
		if (!inserted) return entry.complete || !result.IsEmpty() ? result : opaque();
		const auto finish = [&](Value projected) {
			entry.complete = true;
			return result = projected;
		};
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			uniform_control = true;
			capture_control_reads |=
			    !ValidateRuntimeValue(program, value, RuntimeValueType::ImmutableInteger);
			return finish(Clone(value));
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef) return finish(ClonePredicate(inst->Arg(0)));
		if (op == ValueOpcode::Phi) {
			if (inst->Parent() == nullptr || inst->Parent()->terminator.loop_header ||
			    inst->NumPhiBlocks() != inst->NumArgs() || inst->NumArgs() == 0u ||
			    (value.GetType() != Type::U1 && value.GetType() != Type::U32)) return finish(opaque());
			std::vector<uint32_t> predecessors;
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				const auto predecessor = std::ranges::find(program.blocks, inst->PhiBlock(index));
				if (predecessor == program.blocks.end()) return finish(opaque());
				predecessors.push_back(static_cast<uint32_t>(predecessor - program.blocks.begin()));
			}
			auto& node = plan.value_storage.emplace_back(op, inst->Flags<uint64_t>());
			plan.control_phi_predecessors.emplace(&node, std::move(predecessors));
			for (size_t index = 0; index < inst->NumArgs(); ++index)
				node.AddPhiOperand(nullptr, ClonePredicate(inst->Arg(index)));
			return finish(Value(&node));
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr &&
		    op != ValueOpcode::LogicalNot && op != ValueOpcode::IEqual32 &&
		    op != ValueOpcode::INotEqual32 && op != ValueOpcode::ULessThan32 &&
		    op != ValueOpcode::ULessThanEqual32 &&
		    op != ValueOpcode::SelectU1 && op != ValueOpcode::SelectU32) return finish(opaque());
		auto& node = plan.value_storage.emplace_back(op);
		for (size_t index = 0; index < inst->NumArgs(); ++index)
			node.SetArg(index, ClonePredicate(inst->Arg(index)));
		return finish(Value(&node));
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	if (!uniform_control) {
		plan.control_flow.clear();
		plan.control_phi_predecessors.clear();
		auto first = std::next(plan.value_storage.begin(), control_value_count);
		for (auto it = first; it != plan.value_storage.end(); ++it) it->Invalidate();
		plan.value_storage.erase(first, plan.value_storage.end());
	}
	plan.uniform_fill = program.uniform_fill.fill.kind != UniformFillKind::None
	    ? program.uniform_fill : AnalyzeUniformFill(program);
	for (auto& value: plan.uniform_fill.values) value = Clone(value);
	plan.uniform_fill.pattern_size = Clone(plan.uniform_fill.pattern_size);
	plan.uniform_fill.element_count = Clone(plan.uniform_fill.element_count);
	plan.pattern_copy               = program.pattern_copy;
	plan.pattern_copy.pitch         = Clone(plan.pattern_copy.pitch);
	plan.capture_specialization_reads |=
	    capture_indirect_reads || capture_control_reads || !plan.source_reads.empty();
	if (plan.capture_specialization_reads) {
		// Every former flat alias now targets its retained raw read. Mark those reads
		// on the normalized graph, so discarded Phi/EXEC dependencies stay discarded.
		std::unordered_set<const Inst*> planned_reads;
		for (const auto& read: plan.srt_reads) {
			auto* inst = read.value.ResolveInstruction();
			EXIT_IF(inst == nullptr || (inst->GetOpcode() != ValueOpcode::LoadAddressU32 &&
			                           inst->GetOpcode() != ValueOpcode::ReadConstBuffer));
			planned_reads.insert(inst);
		}
		for (const auto& source: plan.descriptor_sources) {
			if (!source.indirect_descriptor.has_value()) continue;
			const auto& indirect = *source.indirect_descriptor;
			if (indirect.selector) {
				MarkCleanReads(planned_reads, Source(plan, indirect.selector->source));
				if (indirect.selector->rows)
					MarkCleanReads(planned_reads, Source(plan, indirect.selector->rows->source));
			}
			MarkCleanReads(planned_reads, nullptr, indirect.selector_mask);
			MarkCleanReads(planned_reads, nullptr, indirect.selector_first);
			MarkCleanReads(planned_reads, nullptr, indirect.key_count);
			if (indirect.sources.empty()) {
				MarkCleanReads(planned_reads, Source(plan, indirect.table_source));
			} else {
				for (const auto candidate: indirect.sources) {
					MarkCleanReads(planned_reads, Source(plan, candidate));
				}
			}
		}
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.written) MarkCleanReads(planned_reads, Source(plan, buffer.source));
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanReads(planned_reads, Source(plan, image.source));
		}
	}
	if (capture_indirect_reads || !plan.source_reads.empty())
		plan.resource_tracking_complete &= !program.has_address_writes;
	plan.user_data_reads = PlanUserDataReads(plan);
	if (walkers == SrtWalkerCompilation::All) {
		CompileSrtWalkers(plan);
	} else {
		CompileSrtWalker(plan);
	}
	return plan;
}

uint64_t PlanUserDataReads(const ResourcePlan& plan) {
	uint64_t slots = 0;
	for (const auto& inst: plan.value_storage) {
		if (inst.GetOpcode() != ValueOpcode::GetUserData) continue;
		const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
		if (reg < plan.user_data_base || reg - plan.user_data_base >= 64u) return ~uint64_t {0};
		slots |= uint64_t {1} << (reg - plan.user_data_base);
	}
	return slots;
}

bool RangeOverlapIndex::Overlaps(std::span<const std::pair<uint64_t, uint64_t>> ranges,
                                 std::vector<std::pair<uint64_t, uint64_t>>& index, uint64_t base,
                                 uint64_t size) {
	if (size == 0u) return false;
	if (!m_built) {
		index.clear();
		for (const auto [address, bytes]: ranges) {
			if (bytes != 0u) index.emplace_back(address, address + bytes);
		}
		std::ranges::sort(index);
		for (size_t position = 1; position < index.size(); ++position) {
			index[position].second = std::max(index[position].second, index[position - 1].second);
		}
		m_built = true;
	}
	// The ranges that start before the query ends; one of them overlaps it when the furthest
	// end among them lies past its start.
	const auto after = std::ranges::upper_bound(index, base + size - 1u, {},
	                                            [](const auto& range) { return range.first; });
	return after != index.begin() && std::prev(after)->second > base;
}

uint64_t SpecializationFailureReports() {
	return g_failure_reports.load(std::memory_order_relaxed);
}

static bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                                     ResourceSnapshot&       snapshot,
                                     ResourceSpecialization& specialization) {
	if (program.materialization_scratch == nullptr) {
		program.materialization_scratch = std::make_shared<MaterializationScratch>();
	}
	auto& scratch = *program.materialization_scratch;
	for (auto& keys: program.selector_keys) keys.resolved = false;
	snapshot.unavailable = false;
	if (!program.resource_tracking_complete ||
	    ((program.requires_specialization_memory || !program.source_reads.empty()) &&
	     runtime.read_specialization_memory == nullptr)) {
		return false;
	}
	const auto uses_workgroup_count = [](const DescriptorSource& source) {
		return source.indirect_descriptor && source.indirect_descriptor->workgroup_axis != UINT32_MAX;
	};
	const bool capture_reads = program.capture_specialization_reads || !program.source_reads.empty() ||
	    (runtime.indirect_args_address != 0u &&
	     std::ranges::any_of(program.descriptor_sources, uses_workgroup_count));
	auto& reads = snapshot.specialization_reads;
	reads = program.source_reads;
	SrtReadCapture capture {runtime, reads};
	auto observed = capture_reads ? capture.ObservedRuntime() : runtime;
	SrtWalker clean(program, CleanRuntime(observed));
	SrtWalker walker(program, observed,
	                 capture_reads || program.requires_specialization_memory ? &clean : nullptr);
	if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
		return SpecializationFail(fmt::format("SRT refresh complete={} reads={} user_data={}",
		    program.srt_plan_complete, program.srt_reads.size(), runtime.user_data.size()));
	}
	const auto active = std::span<const uint8_t>(program.active_sources);
	std::array<uint32_t, 3> workgroup_counts {};
	if (runtime.indirect_args_address != 0u) {
		for (uint32_t source = 0; source < program.descriptor_sources.size(); ++source) {
			if ((!active.empty() && !active[source]) ||
			    !uses_workgroup_count(program.descriptor_sources[source])) continue;
			if ((runtime.indirect_args_address & 3u) != 0u) return false;
			if (!ReadTable(runtime.indirect_args_address, sizeof(workgroup_counts), 0u, observed,
			               workgroup_counts))
				return SpecializationUnavailable(
				    program.shader_hash, snapshot,
				    fmt::format("dispatch arguments at {:#x}", runtime.indirect_args_address));
			observed.workgroup_counts = workgroup_counts;
			break;
		}
	}
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	if (uniform_fill && !fill.pattern_size.IsEmpty())
		uniform_fill = clean.Evaluate(fill.pattern_size, words) && words >= 1 && words <= 4;
	uint32_t elements = 0;
	if (uniform_fill && !fill.element_count.IsEmpty())
		uniform_fill = clean.Evaluate(fill.element_count, elements) && elements != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = clean.Evaluate(fill.values[i], stored[i]) && stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
		snapshot.uniform_fill.bounded_elements = elements;
	}
	snapshot.pattern_copy = {};
	if (uint32_t pitch = 0; program.pattern_copy.copy.kind != PatternCopyKind::None &&
	                        clean.Evaluate(program.pattern_copy.pitch, pitch)) {
		snapshot.pattern_copy       = program.pattern_copy.copy;
		snapshot.pattern_copy.pitch = pitch;
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return (capture_reads && written ? clean : walker).EvaluateDescriptor(source, value) ||
			    SpecializationFail(fmt::format("descriptor evaluation source={}", source));
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	const auto resource_failed = [&](std::string_view kind, uint32_t resource, uint32_t source,
	                                 size_t candidates) {
		// An unavailable candidate set has already named its reason.
		if (snapshot.unavailable) return false;
		return SpecializationFail(fmt::format("indirect {} resource={} source={} candidates={}",
		    kind, resource, source, candidates));
	};
	// Every image table may become a bindless heap; its two words are reserved ahead of the
	// variable-length key maps.
	auto& bindless_mapping = scratch.bindless_mapping;
	bindless_mapping.assign(program.info.images.size(), UINT32_MAX);
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto* source = Source(program, program.info.images[i].source);
		if (source == nullptr || !source->indirect_descriptor) continue;
		bindless_mapping[i] = static_cast<uint32_t>(snapshot.flattened_srt.size());
		snapshot.flattened_srt.resize(bindless_mapping[i] + 2u, 0u);
	}
	// Every indirect root's key map offset lives in a header word at a place fixed by the plan
	// (see MaterializeIndirectDescriptor). Until a map is published, a header names the shared
	// empty map, which maps no key: the root's own descriptor (ordinal 0) is selected.
	// The empty map is reserved with the first header.
	uint32_t   empty_map       = UINT32_MAX;
	const auto reserve_headers = [&](std::vector<uint32_t>& headers, size_t count,
	                                 auto&& indirect) {
		headers.assign(count, UINT32_MAX);
		for (uint32_t i = 0; i < count; ++i) {
			if (!indirect(i)) continue;
			if (empty_map == UINT32_MAX) {
				empty_map = static_cast<uint32_t>(snapshot.flattened_srt.size());
				snapshot.flattened_srt.push_back(0u);
			}
			headers[i] = static_cast<uint32_t>(snapshot.flattened_srt.size());
			snapshot.flattened_srt.push_back(empty_map);
		}
	};
	const auto indirect_source = [&](uint32_t source_index) {
		const auto* source = Source(program, source_index);
		return source != nullptr
		           ? (source->indirect_descriptor ? &*source->indirect_descriptor : nullptr)
		           : nullptr;
	};
	reserve_headers(scratch.buffer_headers, program.info.buffers.size(), [&](uint32_t i) {
		return indirect_source(program.info.buffers[i].source) != nullptr;
	});
	reserve_headers(scratch.image_headers, program.info.images.size(), [&](uint32_t i) {
		return indirect_source(program.info.images[i].source) != nullptr;
	});
	reserve_headers(scratch.sampler_headers, program.info.samplers.size(), [&](uint32_t i) {
		const auto* indirect = indirect_source(program.info.samplers[i].source);
		return indirect != nullptr && indirect->sources.empty();
	});
	auto& padding      = scratch.padding;
	auto& class_spares = scratch.class_spares;
	padding.clear();
	class_spares.clear();
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return false;
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return false;
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (active.empty() || active[base.source]) {
					if (!MaterializeIndirectDescriptor(
					        program, base.source, i, 4u, 4u, observed, clean, snapshot,
					        snapshot.buffers, specialization.buffers, UINT32_MAX,
					        scratch.buffer_headers[i], padding, class_spares,
					        [&](DescriptorValue& value) {
						        return NormalizeIndirectBuffer(
						            value, source->indirect_descriptor->indexed, base);
					        }))
						return resource_failed("buffer", i, base.source, snapshot.buffers.size());
				}
			} else if (!evaluate(base.source, snapshot.buffers[i], base.written)) {
				return false;
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		// Only whether the stride is zero changes the module; the value is a shader-data word.
		if (stride != 0u) packed_stride = (packed_stride & ~0x3fffu) | 1u;
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		buffer.scalar_empty       = base.scalar && descriptor.GetSize() == 0u;
		// The renderer binds a direct V# without base or records as the null buffer, which every
		// access misses, so other layouts of the buffer can serve it (see SpecializationServes).
		buffer.unbound = i < program.info.buffers.size() &&
		                 buffer.indirect_root == BufferResource::NoIndirectBuffer &&
		                 !Source(program, base.source)->indirect_descriptor.has_value() &&
		                 (descriptor.Base48() == 0u || descriptor.GetSize() == 0u);
	}
	// A spare candidate is never selected; it takes the layout of the first real candidate so
	// that candidates sharing one layout keep a single dynamically indexed access.
	for (const auto spare: padding) {
		const auto root = specialization.buffers[spare].indirect_root;
		for (uint32_t candidate = static_cast<uint32_t>(program.info.buffers.size());
		     candidate < specialization.buffers.size(); ++candidate) {
			if (specialization.buffers[candidate].indirect_root != root ||
			    std::ranges::all_of(snapshot.buffers[candidate].dwords,
			                        [](uint32_t word) { return word == 0u; }))
				continue;
			specialization.buffers[spare] = specialization.buffers[candidate];
			break;
		}
	}
	// A snapshot is evaluated repeatedly; the heaps found this time replace the previous ones,
	// whose mapping offsets belong to the flattened SRT just rebuilt.
	snapshot.bindless_uses.clear();
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube = image.cube,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return false;
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			const bool bounded_table = IsBoundedDescriptorTable(program, indirect);
			// Candidates read from guest tables (rather than enumerated by the program) may
			// belong to other image operations.
			const bool table_candidates = bounded_table || indirect.sources.empty();
			if (!MaterializeIndirectDescriptor(
			        program, image.source, i, 8u, image.r128 ? 4u : 8u, observed, clean, snapshot,
			        snapshot.images, specialization.images, bindless_mapping[i],
			        scratch.image_headers[i], padding, class_spares, [&](DescriptorValue& value) {
				        // A broad heap also contains resources for other typed image operations;
				        // a selector names exactly the records the shader uses, so a malformed one
				        // still fails the specialization. Another dimension is only nulled in a
				        // bounded table: an address table's 1D candidates stay typed for unfiltered
				        // reads and writes (BuildResourceSpecialization checks the other uses).
				        if (NullImageDescriptor(value) ||
				            !ValidImageDescriptor(value, image.r128) ||
				            !PlaceableImageDescriptor(value) ||
				            (bounded_table &&
				             DescriptorDimension(value, image.dimension) != image.dimension) ||
				            (table_candidates && !indirect.selector &&
				             !ImageDescriptorFormatUsable(image, value)))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				return resource_failed("image", i, image.source, snapshot.images.size());
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return false;
			}
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < snapshot.samplers.size(); ++i) {
		const auto base_index =
		    i < program.info.samplers.size() ? i : specialization.samplers[i].indirect_root;
		if (base_index >= program.info.samplers.size()) return false;
		if (i < program.info.samplers.size()) {
			auto source = program.info.samplers[i].source;
			const auto* descriptor = Source(program, source);
			if (descriptor == nullptr) return false;
			const auto* indirect = descriptor->indirect_descriptor ? &*descriptor->indirect_descriptor : nullptr;
			if (indirect != nullptr && indirect->sources.empty()) {
				snapshot.samplers[i] = {.dword_count = 4u};
				if ((active.empty() || active[source]) &&
				    !MaterializeIndirectDescriptor(
				        program, source, i, 4u, 4u, observed, clean, snapshot, snapshot.samplers,
				        specialization.samplers, UINT32_MAX, scratch.sampler_headers[i], padding,
				        class_spares, [](DescriptorValue&) { return true; }))
					return resource_failed("sampler", i, source, snapshot.samplers.size());
			} else {
				if (indirect != nullptr) source = indirect->sources[0];
				if (!evaluate(source, snapshot.samplers[i])) return false;
			}
		}
		if (program.info.samplers[base_index].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			const auto perf_mip = (snapshot.samplers[i].dwords[1] >> 24u) & 15u;
			// MipNone always selects the base level. Point/linear selection requires zero
			// primary/secondary bias; linear mip snapping is not implemented.
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u) ||
			    (filter == 2u && perf_mip != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires None, unbiased Point, or unbiased Linear without mip snapping");
			}
		}
	}
	// A GPU-selected store candidate whose range covers descriptor memory this dispatch reads is
	// not the buffer the shader writes (it would overwrite its own descriptors): a table record the
	// key never selects can hold a heap-wide V#. Such candidates are null, so the renderer's
	// write-overlap check applies to the candidates that remain. A table of store candidates is
	// read record by record, so both lists can hold thousands of entries: the reads are sorted
	// by address once, with the furthest end among each prefix, and each candidate is a binary
	// search.
	RangeOverlapIndex read_index;
	for (uint32_t i = 0; i < snapshot.buffers.size() && i < specialization.buffers.size(); ++i) {
		const auto root = specialization.buffers[i].indirect_root;
		if (root >= program.info.buffers.size() || !program.info.buffers[root].written) continue;
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(snapshot.buffers[i], descriptor)) continue;
		const auto base = descriptor.Base48();
		const auto size = descriptor.GetSize();
		if (base == 0u || size == 0u) continue;
		if (read_index.Overlaps(snapshot.specialization_reads, scratch.read_ends, base, size))
			snapshot.buffers[i].dwords.fill(0u);
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	if (!BuildResourceSpecialization(program, snapshot, specialization)) return false;
	// A spare image slot is never selected; it takes the class of the group it pads, which the
	// null descriptor alone would not give it.
	for (const auto& [spare, donor]: class_spares)
		specialization.images[spare] = specialization.images[donor];
	return true;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	auto& reasons       = t_failure_reasons;
	reasons.speculative = runtime.speculative;
	reasons.text.clear();
	const bool done = MaterializeResourcesImpl(program, runtime, snapshot, specialization);
	// An unavailable draw has already named its reason.
	if (!done && !snapshot.unavailable && !reasons.text.empty())
		ReportSpecializationFailure(program.shader_hash, reasons.text);
	reasons.text.clear();
	return done;
}

bool SpecializationApplies(const Program& program, const ResourceSpecialization& specialization) {
	if (!program.resource_tracking_complete || program.shader_info_complete) {
		return false;
	}
	// Mirrors the checks of ApplyResourceSpecialization: entries past the program's resources
	// copy a root among them, and every root names an entry.
	const auto roots_fit = [](const auto& entries, size_t original, uint32_t none) {
		if (original > entries.size()) {
			return false;
		}
		for (size_t index = 0; index < entries.size(); index++) {
			const auto root = entries[index].indirect_root;
			if ((index >= original && root >= original) ||
			    (root != none && root >= entries.size())) {
				return false;
			}
		}
		return true;
	};
	const auto& info = program.info;
	return roots_fit(specialization.buffers, info.buffers.size(),
	                 BufferResource::NoIndirectBuffer) &&
	       roots_fit(specialization.images, info.images.size(), ImageResource::NoIndirectImage) &&
	       roots_fit(specialization.samplers, info.samplers.size(),
	                 SamplerResource::NoIndirectSampler);
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size() ||
	        program.info.samplers.size() > specialization.samplers.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.runtime_stride             = (source.packed_stride & 0x3fffu) != 0u;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_mapping_header    = source.indirect_mapping_header;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.indirect_mapping_header    = source.indirect_mapping_header;
		image.cube                       = source.cube;
		image.bindless                   = source.bindless;
		image.bindless_byte_keys         = source.bindless_byte_keys;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto base_sampler_count = samplers.size();
	samplers.reserve(specialization.samplers.size());
	for (uint32_t index = 0; index < specialization.samplers.size(); ++index) {
		const auto& source = specialization.samplers[index];
		if (index >= samplers.size()) {
			EXIT_IF(source.indirect_root >= base_sampler_count);
			samplers.push_back(samplers[source.indirect_root]);
			samplers.back().indirect_resources.clear();
		}
		samplers[index].indirect_root = source.indirect_root;
		samplers[index].indirect_mapping_offset = source.indirect_mapping_offset;
		samplers[index].indirect_search_iterations = source.indirect_search_iterations;
		samplers[index].indirect_mapping_header    = source.indirect_mapping_header;
	}
	for (uint32_t index = 0; index < samplers.size(); ++index) {
		const auto root = samplers[index].indirect_root;
		if (root != SamplerResource::NoIndirectSampler) {
			EXIT_IF(root >= samplers.size());
			samplers[root].indirect_resources.push_back(index);
		}
	}
	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	const auto original_sampler_count = samplers.size();
	samplers.reserve(sampler_plan.bindings.size());
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::Integer;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		for (auto& candidate: samplers[index].indirect_resources) {
			candidate = sampler_plan.mapping[candidate][static_cast<uint32_t>(binding.type)];
			EXIT_IF(candidate == UINT32_MAX);
		}
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
		for (const auto candidate: samplers[pair.sampler].indirect_resources)
			samplers[candidate].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	// Resolve empty scalar-table words before folding the indirect descriptors that use them.
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
			const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
			if (memory.kind == ResourceKind::ScalarBuffer &&
			    specialization.buffers[memory.resource].scalar_empty &&
			    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer)
				inst.ReplaceUsesWith(Value(0u));
		}
	}
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				const bool native_oob = memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob &&
				    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer;
				const bool null_indirect = memory.kind == ResourceKind::IndirectBuffer &&
				    NullBufferDescriptor(inst.Arg(0));
				if (native_oob || null_indirect) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds. An all-zero V# is unbound.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (native_oob && memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						const bool d16 = memory.data_bits == 16u;
						const auto components = d16 ? memory.component_count : count;
						std::array<uint32_t, 4> constants {};
						for (uint32_t component = 0; component < components; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							if (d16 && one == 0x3f800000u) one = 0x3c00u;
							constants[d16 ? component / 2u : component] |=
							    one << (d16 ? (component % 2u) * 16u : 0u);
						}
						for (uint32_t component = 0; component < count; component++) {
							if (constants[component] == 0u) continue;
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(constants[component]), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			EXIT_IF(image.indirect_root == memory.resource &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageGatherRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageQueryDimensions &&
			        inst.GetOpcode() != ValueOpcode::ImageRead &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite);
		}
	}
	for (auto& memory: memory_info) {
		if (memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap[buffer.image_alias];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
}

void FillBufferStrides(const ShaderInfo& info, const Bindings& bindings,
                       const ResourceSnapshot& snapshot, std::span<uint32_t> shader_data) {
	if (!bindings.memory_strides) return;
	EXIT_IF(shader_data.size() < bindings.ShaderDataDwords() ||
	        snapshot.buffers.size() < info.buffers.size());
	for (uint32_t resource = 0; resource < info.buffers.size(); ++resource) {
		const auto& buffer = info.buffers[resource];
		if (!buffer.runtime_stride || buffer.descriptor_index == UINT32_MAX) continue;
		// Spare candidates (never selected) may carry no descriptor; their stride is zero.
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(snapshot.buffers[resource], descriptor) ||
		    descriptor.Type() != 0)
			continue;
		bindings.PackMemoryStride(shader_data, buffer.descriptor_index,
		                          descriptor.PackedStride() & 0x3fffu);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
