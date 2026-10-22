#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		// The shader adds the byte offset to every address and bounds it by the binding, so an
		// offset of the whole range makes every access out of bounds: loads read the
		// out-of-bounds value and stores are dropped, as for a V# without records.
		constexpr uint32_t NullRange = 16;
		buffer_offset                = NullRange;
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, NullRange};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
	    descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

// The view a T# selects; nullopt for a T# the view cannot express when `optional`, which exits
// otherwise.
static std::optional<ImageViewInfo>
TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                const ShaderTextureResource& descriptor, vk::Format format,
                const SurfaceFormatInfo& surface_format, bool storage, uint32_t view_levels,
                uint32_t image_layers, bool optional) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		if (optional) {
			return std::nullopt;
		}
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	const bool mapped = !storage && resource.conversion_format == Prospero::BufferFormat::kInvalid;
	if (mapped && optional && !IsValidImageSwizzle(descriptor.DstSelXYZW())) {
		return std::nullopt;
	}
	view.mapping =
	    mapped ? TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage)
	           : vk::ComponentMapping {};
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type        = vk::ImageViewType::e1D;
			view.base_layer  = image_layers > 1u ? descriptor.BaseArray5() : 0u;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type        = vk::ImageViewType::e1DArray;
			view.base_layer  = descriptor.BaseArray5();
			view.layer_count = image_layers - std::min(view.base_layer, image_layers);
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type        = vk::ImageViewType::e2DArray;
			view.base_layer  = descriptor.BaseArray5();
			view.layer_count = image_layers - std::min(view.base_layer, image_layers);
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type = vk::ImageViewType::e2D;
			// A T# without array slices has nothing for BASE_ARRAY to select.
			view.base_layer  = image_layers > 1u ? descriptor.BaseArray5() : 0u;
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	if (resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	    view.base_layer >= image_layers) {
		if (optional) {
			return std::nullopt;
		}
		EXIT("texture base layer is out of bounds: base=%u layers=%u dimension=%u "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     view.base_layer, image_layers, static_cast<uint32_t>(resource.dimension),
		     descriptor.fields[0], descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
		     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6],
		     descriptor.fields[7]);
	}
	return view;
}

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical        = info;
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              TextureSource                                source) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	// Every check below that a T# can fail returns an imageless binding for an optional source.
	const bool optional = source == TextureSource::Optional;
	EXIT_IF(optional && storage);
	if (optional && static_cast<uint32_t>(descriptor.Type()) <
	                    static_cast<uint32_t>(Prospero::ImageType::kColor1D)) {
		return {};
	}

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		if (optional) {
			return {};
		}
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : descriptor.Format();
	const auto found_format   = TextureFindSurfaceFormatInfo(format);
	if (!found_format) {
		if (optional) {
			return {};
		}
		EXIT("unknown texture format: fmt=%u dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     static_cast<uint32_t>(format), descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto& surface_format = *found_format;
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		if (optional) {
			return {};
		}
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : raw_sint_format != Prospero::BufferFormat::kInvalid
	                                     ? VulkanFormat(raw_sint_format)
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (optional) {
		// The layouts the mip view below computes, and the shape the image must have.
		auto probe = desc.info;
		if ((desc.info.type == Prospero::ImageType::kColor1D && height != 1u) ||
		    probe.TryUpdateSize() != nullptr) {
			return {};
		}
		probe                  = desc.info;
		probe.resources.levels = physical_levels;
		if (levels > physical_levels && probe.TryUpdateSize() != nullptr) {
			return {};
		}
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           !resource.r128 && descriptor.MetaCompress(), view_levels,
		                           view_base)) {
			// No expansion of the allocation stores the mips past MAX_MIP (a streamed texture
			// whose T# describes the full chain): the view ends at the allocation's last level,
			// and a BASE_LEVEL past it samples that level.
			view_base                  = std::min<uint32_t>(base_level, max_mip);
			view_levels                = physical_levels - view_base;
			levels                     = physical_levels;
			desc.info.resources.levels = levels;
			desc.info.UpdateSize();
		}
	} else {
		desc.info.UpdateSize();
	}
	if (!desc.info.DataPlaced()) {
		if (optional) {
			return {};
		}
		EXIT("texture data cannot be placed: addr=0x%016" PRIx64 " size=0x%" PRIx64
		     " alignment=0x%x format=%u tile=%u extent=%ux%ux%u levels=%u "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     address, desc.info.data.size, desc.info.DataAlignment(), static_cast<uint32_t>(format),
		     static_cast<uint32_t>(tile), width, height, desc.info.extent.depth, levels,
		     descriptor.fields[0], descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
		     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6],
		     descriptor.fields[7]);
	}
	// Image heaps keep the descriptors of unmapped textures, which PS5 never samples.
	if (resource.heap && !m_context.IsMapped(address, desc.info.data.size)) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	if (optional && !m_context.IsMapped(address, desc.info.data.size)) {
		return {};
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
		const auto& metadata             = desc.info.metadata.range;
		if (optional &&
		    !m_context.IsMapped(metadata.address, std::max<uint64_t>(metadata.size, 1))) {
			return {};
		}
	}
	const auto view_info =
	    TextureViewInfo(resource, descriptor, view_format, surface_format, storage, view_levels,
	                    desc.info.resources.layers, optional);
	if (!view_info) {
		return {};
	}
	desc.view_info            = *view_info;
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;

	auto       id                  = texture_cache.FindImage(desc, shader_conversion);
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (optional) {
			// Optional sources sample color textures.
			return {};
		}
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else if (optional && !IsSupportedSampledColorView(image->info.pixel_format, pixel_format,
	                                                    descriptor.DstSelXYZW())) {
		return {};
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	return {id, nullptr, std::move(desc)};
}

bool RenderExecutor::TextureMemoEntryValid(const TextureMemoEntry& entry) {
	auto& texture_cache = m_context.GetTextureCache();
	if (texture_cache.LayoutChangedSince(entry.stamp, entry.desc.info.data.address,
	                                     entry.desc.info.data.size)) {
		return false;
	}
	const auto* image = texture_cache.m_slot_images.try_get(entry.image_id);
	return image != nullptr && image->registered && !image->binding.needs_rebind &&
	       !image->depth_id;
}

void RenderExecutor::ResolveTextureMemo(const ShaderRecompiler::IR::ImageResource&   resource,
                                        const ShaderRecompiler::IR::DescriptorValue& value,
                                        TextureBinding&                              binding) {
	auto&          texture_cache = m_context.GetTextureCache();
	TextureMemoKey key {};
	std::copy_n(value.dwords.begin(), key.words.size(), key.words.begin());
	key.traits = uint64_t {resource.written} | uint64_t {resource.read} << 1u |
	             uint64_t {resource.atomic} << 2u | uint64_t {resource.atomic64} << 3u |
	             uint64_t {resource.r128} << 4u | uint64_t {resource.depth_compare} << 5u |
	             uint64_t {static_cast<uint8_t>(resource.resource_class)} << 8u |
	             uint64_t {static_cast<uint8_t>(resource.mip_mode)} << 16u |
	             uint64_t {static_cast<uint8_t>(resource.dimension)} << 24u |
	             uint64_t {static_cast<uint8_t>(resource.numeric_class)} << 32u |
	             uint64_t {value.dword_count} << 40u;
	// The binding is written in place: its description is about a kilobyte, and the callers keep
	// the mip-view vector's capacity.
	binding.image_view = nullptr;
	binding.layout     = vk::ImageLayout::eUndefined;
	binding.mip_views.clear();
	if (const auto found = m_texture_memo.find(key); found != m_texture_memo.end()) {
		auto& entry = found->second;
		if (TextureMemoEntryValid(entry)) {
			auto& image               = texture_cache.GetImage(entry.image_id);
			entry.stamp               = texture_cache.LayoutNow();
			image.tick_accessed_last  = m_context.GetCommandScheduler().CurrentTick();
			texture_cache.TouchImage(image);
			texture_cache.MaterializeColorClear(entry.image_id, entry.desc,
			                                    entry.desc.metadata_base_layer);
			binding.image_id = entry.image_id;
			// A binding refilled with the same entry (the same stage's slot draw after draw)
			// already holds its description.
			if (binding.memo_serial != entry.serial) {
				binding.desc        = entry.desc;
				binding.memo_serial = entry.serial;
			}
			return;
		}
		m_texture_memo.erase(found);
	}
	auto resolved      = ResolveTexture(resource, value);
	binding.image_id   = resolved.image_id;
	binding.image_view = resolved.image_view;
	binding.desc       = resolved.desc;
	binding.layout     = resolved.layout;
	// Entries whose image went away or whose memory changed layout never hit again; dropping
	// only those keeps the memo as large as the scene's live textures instead of emptying it
	// (a town binds more distinct T#s than a fixed cap held, so every lookup missed).
	if (m_texture_memo.size() >= m_texture_memo_sweep_at) {
		std::erase_if(m_texture_memo,
		              [&](const auto& item) { return !TextureMemoEntryValid(item.second); });
		m_texture_memo_sweep_at = std::max(TextureMemoFirstSweep, m_texture_memo.size() * 2u);
	}
	binding.memo_serial = ++m_texture_memo_serial;
	m_texture_memo.insert_or_assign(key, TextureMemoEntry {binding.image_id, binding.desc,
	                                                       texture_cache.LayoutNow(),
	                                                       binding.memo_serial});
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (program.info.images[i].heap) {
			prepared.images[i] = {};
			continue;
		}
		auto& binding = prepared.images[i];
		ResolveTextureMemo(program.info.images[i], snapshot.images[i], binding);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
	}
	prepared.bindless_heaps.clear();
	prepared.bindless_srt.clear();
	if (!snapshot.bindless_uses.empty()) {
		PrepareBindlessHeaps(prepared);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto& prepared = *stage;
		EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				// OOB_SELECT=2 only tests NumRecords != 0; a zero-stride fetch
				// still needs its full payload when NumRecords is smaller than it.
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address == 0 || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				//  PPSA15246
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
			}
			// Finite GPU descriptors can span sparse PRT holes; clamp only oversized/sentinel ranges.
			if (size >= (uint64_t {1} << 30u) || descriptor.NumRecords() == UINT32_MAX) {
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			}
			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		prepared.buffers[index] = NativeStorageBuffer(m_context, prepared.buffer_sources[index],
		                                              resource, buffer_offset);
		pack_memory_offset(index, buffer_offset);
	}
	ShaderRecompiler::IR::FillBufferStrides(program.info, layout, snapshot, prepared.shader_data);
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt = NativeUpload(m_context, prepared.bindless_srt.empty()
		                                                     ? snapshot.flattened_srt
		                                                     : prepared.bindless_srt);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || !old_image->registered ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			ResolveTextureMemo(program.info.images[i], snapshot.images[i], images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
	}
}

// A heap is the records of one V# a stage reaches T#s through (ResourceSnapshot::BindlessUse);
// its region and entry count go into the stage's copy of the flattened SRT.
void RenderExecutor::PrepareBindlessHeaps(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       table    = m_context.GetBindlessTable();
	if (!table.Enabled()) {
		EXIT("bindless images need descriptor indexing with update-after-bind sampled images\n");
	}
	auto& translation = table.Translation();
	EnsureBindlessPlaceholder();
	prepared.bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
	for (const auto& use: snapshot.bindless_uses) {
		EXIT_IF(use.layer_count == 0 || use.mapping_offset + 1u >= prepared.bindless_srt.size());
		auto* words = prepared.bindless_srt.data() + use.mapping_offset;
		// Keys past the heap's entries sample the placeholder (EmitBindlessSlot).
		constexpr uint64_t max_entries = BindlessTable::Entries / 4u;
		const auto records = static_cast<uint32_t>(std::min<uint64_t>(
		    use.record_stride == 0 ? 1u : use.size / use.record_stride,
		    max_entries / use.layer_count));
		const BindlessTranslation::HeapKey key {use.base,         use.record_stride,
		                                        use.pointer_offset, use.table_offset,
		                                        use.layer_stride, use.layer_count};
		const auto heap = translation.FindOrCreateHeap(key, records);
		if (heap == BindlessTranslation::NoHeap) {
			static bool reported = false;
			if (!std::exchange(reported, true)) {
				LOGF("bindless translation table is full; keys of heap 0x%016" PRIx64
				     " sample the placeholder\n",
				     use.base);
			}
			words[0] = 0;
			words[1] = 0;
			continue;
		}
		if (heap >= m_bindless_resources.size()) {
			m_bindless_resources.resize(heap + 1u);
			m_bindless_resources[heap] = program.info.images[use.resource];
			FillBindlessHeap(heap);
		}
		const auto& found = translation.GetHeap(heap);
		words[0]          = found.region;
		words[1]          = found.Entries();
		if (std::ranges::find(prepared.bindless_heaps, heap) == prepared.bindless_heaps.end()) {
			prepared.bindless_heaps.push_back(heap);
		}
	}
}

void RenderExecutor::EnsureBindlessPlaceholder() {
	// Unresolved keys sample a native null descriptor, as unbound textures do.
	if (m_bindless_placeholder_ready) return;
	m_context.GetBindlessTable().WriteSlot(BindlessTranslation::PlaceholderSlot, nullptr,
	                                       vk::ImageLayout::eGeneral);
	m_bindless_placeholder_ready = true;
}

// Every image a key of the draw's heaps resolved to is sampled in read-only layout; an image
// the draw also renders to keeps its attachment layout, as an enumerated binding would.
void RenderExecutor::CommitBindlessImages(const PreparedBindings& prepared,
                                          vk::CommandBuffer       vk_buffer) {
	KYTY_PROFILER_FUNCTION();
	auto& cache       = m_context.GetTextureCache();
	auto& translation = m_context.GetBindlessTable().Translation();
	if (cache.DeletionEpoch() != m_bindless_deletion_epoch) {
		ValidateBindlessSlots();
	}
	for (const auto heap: prepared.bindless_heaps) {
		for (const auto slot: translation.GetHeap(heap).slots) {
			auto& image = cache.GetImage(translation.GetSlot(slot).image);
			if (image.binding.is_target) continue;
			image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
			              {}, vk_buffer);
		}
	}
}

// Slots of images the texture cache deleted go back to the placeholder; their keys resolve
// again when a shader next flags them.
void RenderExecutor::ValidateBindlessSlots() {
	auto& cache       = m_context.GetTextureCache();
	auto& table       = m_context.GetBindlessTable();
	auto& translation = table.Translation();
	m_bindless_deletion_epoch = cache.DeletionEpoch();
	for (uint32_t slot = BindlessTranslation::PlaceholderSlot + 1u; slot < translation.SlotCount();
	     ++slot) {
		const auto& entry = translation.GetSlot(slot);
		if (!entry.InUse()) continue;
		const auto* image = cache.m_slot_images.try_get(entry.image);
		if (image != nullptr && (image->registered || image->info.data.Empty())) continue;
		table.WriteSlot(slot, nullptr, vk::ImageLayout::eGeneral);
		translation.FreeSlot(slot);
	}
	table.FlushTranslation();
}

// The slot of the T# a key names: the record's pointer, then the layer's T# behind it, read
// from guest memory as the shader would. Anything unreadable or unsupported is the placeholder.
uint32_t RenderExecutor::ResolveBindlessKey(uint32_t                            heap,
                                            const BindlessTranslation::Request& request) {
	auto&       table       = m_context.GetBindlessTable();
	auto&       translation = table.Translation();
	const auto& key         = translation.GetHeap(heap).key;
	uint64_t    pointer     = 0;
	const auto  record = key.base + uint64_t {request.record} * key.record_stride;
	if (key.pointer_offset == ShaderRecompiler::IR::BindlessUse::NoPointer) {
		// The record holds the T# block itself.
		pointer = record;
	} else {
		const bool readable = LibKernel::Memory::TryReadBacking(record + key.pointer_offset, &pointer,
		                                                        sizeof(pointer));
		// The shader sign-extends bit 47; the bits above it carry what the game stores there.
		pointer = static_cast<uint64_t>(static_cast<int64_t>(pointer << 16u) >> 16u);
		if (!readable || pointer == 0) {
			return BindlessTranslation::PlaceholderSlot;
		}
	}
	ShaderRecompiler::IR::DescriptorValue value;
	value.dword_count = static_cast<uint32_t>(value.dwords.size());
	const auto descriptor = pointer + uint64_t {request.layer} * key.layer_stride + key.table_offset;
	if (!LibKernel::Memory::TryReadBacking(descriptor, value.dwords.data(), sizeof(value.dwords)) ||
	    DecodeNativeDescriptor<ShaderTextureResource>(value).IsNull()) {
		return BindlessTranslation::PlaceholderSlot;
	}
	EXIT_IF(heap >= m_bindless_resources.size());
	// A T# a plain binding would refuse keeps the placeholder: the key may name a record the
	// game never meant to sample.
	const auto tsharp  = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const auto type    = tsharp.Type();
	const bool layered = type == Prospero::ImageType::kColor1DArray ||
	                     type == Prospero::ImageType::kColor2DArray ||
	                     type == Prospero::ImageType::kColor2DMsaaArray ||
	                     type == Prospero::ImageType::kCube;
	const auto layers  = layered ? static_cast<uint32_t>(tsharp.Depth()) + 1u : 1u;
	if (tsharp.BaseArray5() >= layers || tsharp.MinLod() > tsharp.LastLevel() * 256u ||
	    tsharp.BaseLevel() > tsharp.LastLevel()) {
		static uint32_t reported = 0;
		if (reported++ < 8u) {
			LOGF("bindless key names an unsupported T#: %08x %08x %08x %08x %08x %08x %08x %08x\n",
			     value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
			     value.dwords[4], value.dwords[5], value.dwords[6], value.dwords[7]);
		}
		return BindlessTranslation::PlaceholderSlot;
	}
	auto& cache   = m_context.GetTextureCache();
	auto  binding = ResolveTexture(m_bindless_resources[heap], value, TextureSource::Optional);
	if (!binding.image_id) {
		static uint32_t reported = 0;
		if (reported++ < 8u) {
			LOGF("bindless key names a T# that cannot be bound: %08x %08x %08x %08x %08x %08x "
			     "%08x %08x\n",
			     value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
			     value.dwords[4], value.dwords[5], value.dwords[6], value.dwords[7]);
		}
		return BindlessTranslation::PlaceholderSlot;
	}
	auto& image = cache.GetImage(binding.image_id);
	if (image.info.IsDepth() || image.info.data.Empty()) {
		return BindlessTranslation::PlaceholderSlot;
	}
	const auto view = cache.FindTexture(binding.image_id, binding.desc);
	const auto handle = reinterpret_cast<uint64_t>(static_cast<VkImageView>(view));
	auto       slot   = translation.FindSlot(binding.image_id, handle);
	if (slot == BindlessTranslation::NoSlot) {
		slot = translation.AllocateSlot(binding.image_id, handle);
		if (slot == BindlessTranslation::NoSlot) {
			static bool reported = false;
			if (!std::exchange(reported, true)) {
				LOGF("bindless image slots are exhausted; later keys sample the placeholder\n");
			}
			return BindlessTranslation::PlaceholderSlot;
		}
		table.WriteSlot(slot, view, vk::ImageLayout::eShaderReadOnlyOptimal);
	}
	auto& resolved = m_bindless_resolved[pointer + key.table_offset];
	if (std::ranges::find(resolved, request.layer, &BindlessResolved::layer) == resolved.end()) {
		resolved.push_back({request.layer, slot, binding.image_id});
	}
	return slot;
}

// A new heap's records point at material blocks whose keys may have resolved through an
// earlier heap: those translations are written right away.
void RenderExecutor::FillBindlessHeap(uint32_t heap) {
	auto&       table       = m_context.GetBindlessTable();
	auto&       translation = table.Translation();
	const auto& found       = translation.GetHeap(heap);
	const auto  key         = found.key;
	bool        filled      = false;
	for (uint32_t record = 0; record < found.records; ++record) {
		uint64_t   pointer = key.base + uint64_t {record} * key.record_stride;
		if (key.pointer_offset != ShaderRecompiler::IR::BindlessUse::NoPointer) {
			if (!LibKernel::Memory::TryReadBacking(pointer + key.pointer_offset, &pointer,
			                                       sizeof(pointer))) continue;
			pointer = static_cast<uint64_t>(static_cast<int64_t>(pointer << 16u) >> 16u);
		}
		const auto resolved = m_bindless_resolved.find(pointer + key.table_offset);
		if (pointer == 0 || resolved == m_bindless_resolved.end()) continue;
		for (const auto& entry: resolved->second) {
			const auto& slot = translation.GetSlot(entry.slot);
			if (entry.layer >= key.layer_count || !slot.InUse() || slot.image != entry.image) continue;
			translation.Translate(found.region + record * key.layer_count + entry.layer, entry.slot);
			filled = true;
		}
	}
	if (filled) {
		table.FlushTranslation();
	}
}

void RenderExecutor::ResolveBindlessImages(CommandBuffer& buffer) {
	auto& table = m_context.GetBindlessTable();
	if (!table.Created()) {
		return;
	}
	KYTY_PROFILER_FUNCTION();
	constexpr uint32_t budget = 512;
	buffer.EndRendering();
	table.MakeFeedbackVisible(buffer.Handle());
	m_bindless_feedback.clear();
	table.TakeFeedback(m_bindless_feedback, budget);
	auto& translation = table.Translation();
	auto& cache       = m_context.GetTextureCache();
	if (cache.DeletionEpoch() != m_bindless_deletion_epoch) {
		ValidateBindlessSlots();
	}
	// A resolved image is sampled through its slot without a lookup, so guest writes to it
	// (streamed-in texture data) are picked up here, once per frame.
	for (uint32_t slot = BindlessTranslation::PlaceholderSlot + 1u; slot < translation.SlotCount();
	     ++slot) {
		if (translation.GetSlot(slot).InUse()) {
			cache.UpdateImage(translation.GetSlot(slot).image);
		}
	}
	for (const auto entry: m_bindless_feedback) {
		BindlessTranslation::Request request;
		if (!translation.Decode(entry, request)) continue;
		translation.Translate(entry, ResolveBindlessKey(request.heap, request));
	}
	table.FlushTranslation();
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma        = false;
	bool uses_image_heap = false;
	FindBuffers(stages);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->program->info.uses_dma;
		uses_image_heap |= stage->runtime->program->info.uses_image_heap;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	if (uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || !old_image->registered ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	// Binding commands access no memory. Image transitions order pending dispatches first.
	auto                           vk_buffer              = buffer.StateHandle();
	size_t image_descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += std::popcount(program.bindings.descriptor_mask);
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// Specialization reads are proven disjoint from every guest range the commit writes. The
	// written ranges depend only on the commit, so they are collected once instead of per read.
	const bool reads_specialization = std::ranges::any_of(prepared_bindings, [](const auto* stage) {
		return !stage->runtime->resources->specialization_reads.empty();
	});
	if (reads_specialization) {
		for (const auto* writer: prepared_bindings) {
			if (writer->runtime->program->has_address_writes) {
				EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
			}
		}
		auto& written = m_commit_writes;
		written.Clear();
		bool valid = true;
		for (const auto id: m_bound_images) {
			const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
			if (image == nullptr || (!image->binding.shader_write && !image->binding.is_target)) {
				continue;
			}
			for (const auto range:
			     {image->info.data, image->info.stencil, image->info.metadata.range}) {
				valid &= written.Add(range.address, range.size, GuestWriteSet::Kind::Image);
			}
		}
		for (const auto* writer: prepared_bindings) {
			for (const auto& resource: writer->runtime->program->info.buffers) {
				if (!resource.written || resource.descriptor_index == UINT32_MAX) continue;
				const auto& range = writer->buffer_sources[resource.descriptor_index];
				// ADD_TID and swizzle move fixed-offset writes past write_extent.
				const bool linear = (resource.packed_stride & ((1u << 14u) | (1u << 20u))) == 0u;
				const auto written_size =
				    linear ? std::min<uint64_t>(range.size, resource.write_extent) : range.size;
				valid &= written.Add(range.address, written_size, GuestWriteSet::Kind::Buffer);
			}
		}
		if (!valid) {
			EXIT("invalid image overlap range\n");
		}
		for (const auto* reader: prepared_bindings) {
			if (written.Empty()) break;
			for (const auto [address, size]: reader->runtime->resources->specialization_reads) {
				if (!GuestWriteSet::ValidRange(address, size)) {
					EXIT("invalid image overlap range\n");
				}
				const auto* overlap = written.FindOverlap(address, size);
				if (overlap == nullptr) continue;
				if (overlap->kind == GuestWriteSet::Kind::Image) {
					EXIT("scalar resource reads overlap an image or attachment write\n");
				}
				EXIT("scalar resource reads overlap a shader buffer write\n");
			}
		}
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
	}
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	m_descriptor_writes.reserve(write_count);
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;
	bool                                    bindless = false;

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			buffer.BarrierHandle().pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& binding = descriptors.images[i];
			if (!binding.image_id) continue;
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}
		if (!descriptors.bindless_heaps.empty()) {
			CommitBindlessImages(descriptors, vk_buffer);
			bindless = true;
		}

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != write.descriptorCount);
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
						}
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							EXIT_IF(sampler == nullptr);
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			if (resource.heap) continue;
			EXIT_IF(texture.image_id &&
			        resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()));
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(), 0,
			    nullptr);
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
	if (bindless) {
		const auto set = m_context.GetBindlessTable().Set();
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout,
		                             ShaderRecompiler::IR::BindlessDescriptorSet, 1, &set, 0,
		                             nullptr);
	}
}

} // namespace Libs::Graphics
