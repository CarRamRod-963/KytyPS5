#include "graphics/host_gpu/renderer/pipeline/descriptorsTrace.h"

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envDefaults.h"
#include "common/envRead.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/cmdSignature.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/recordPackets.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"
#include "graphics/host_gpu/renderer/stageFTrace.h"
#include "graphics/host_gpu/renderer/texWinTrace.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

// The binding paths' logs and censuses (descriptors.cpp calls them; each is off unless its name is set).
namespace Libs::Graphics {

TableCensusState& TableCensus() {
	static TableCensusState state;
	return state;
}

void TableCensusReport() {
	auto&      s   = TableCensus();
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window_start < std::chrono::seconds(5)) return;
	const uint32_t frame  = DebugPresentCount().load();
	const double   frames = std::max(1.0, static_cast<double>(frame - s.window_frame));
	for (int k = 0; k < 2; k++) {
		const auto& c = s.counts[k];
		std::fprintf(stderr,
		             "PERF table_census %s frames=%.0f stages=%.0f would_hit=%.0f same_frame=%.0f all_memo=%.0f proof_ok=%.0f "
		             "page_changed=%.0f fail[gone=%llu unreg=%llu rebind=%llu depth=%llu dirty=%llu alias=%llu amem=%llu] "
		             "held_differs=%llu slots=%.0f storage=%.0f nonmemo=%.0f loop_ms_all=%.2f loop_ms_hit=%.2f "
		             "rebind_ms_hit=%.2f proof_ms=%.2f (per frame)\n",
		             k == 0 ? "gfx" : "cs", frames, c.stages / frames, c.would_hit / frames, c.hit_same_frame / frames,
		             c.all_memo / frames, c.proof_ok / frames, c.page_changed / frames,
		             static_cast<unsigned long long>(c.proof_fail[1]), static_cast<unsigned long long>(c.proof_fail[2]),
		             static_cast<unsigned long long>(c.proof_fail[3]), static_cast<unsigned long long>(c.proof_fail[4]),
		             static_cast<unsigned long long>(c.proof_fail[5]), static_cast<unsigned long long>(c.proof_fail[6]),
		             static_cast<unsigned long long>(c.proof_fail[7]), static_cast<unsigned long long>(c.held_differs),
		             c.slots / frames, c.slots_storage / frames, c.slots_other / frames, c.loop_ns_all / frames / 1e6,
		             c.loop_ns_hit / frames / 1e6, c.rebind_ns_hit / frames / 1e6, c.proof_ns / frames / 1e6);
		s.counts[k] = {};
	}
	s.window_start = now;
	s.window_frame = frame;
}

void RenderExecutor::FoldCommitSignature(vk::PipelineBindPoint bind_point, std::span<PreparedBindings* const> stages) {
	auto&      sampler_cache = m_context.GetSamplerCache();
	const auto pack          = [](uint64_t low, uint64_t high) { return (low & 0xffffffffull) | (high << 32u); };
	for (const auto* prepared: stages) {
		if (prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime) {
			continue;
		}
		const auto& program = *prepared->runtime->program;
		CmdSignature::FoldWords(CmdSignature::Kind::Stage, {static_cast<uint64_t>(bind_point),
		                                                    static_cast<uint64_t>(program.stage), program.shader_hash});
		// Images: what the descriptor asked for, the view, the layout, and the image the texture cache found.
		for (size_t i = 0; i < prepared->images.size(); i++) {
			const auto& binding       = prepared->images[i];
			const auto& info          = binding.desc.info;
			const auto& view          = binding.desc.view_info;
			uint64_t    image_address = 0, image_format = 0, image_extent = 0;
			if (const auto* image = m_context.GetTextureCache().m_slot_images.try_get(binding.image_id); image != nullptr) {
				image_address = image->info.data.address;
				image_format  = pack(static_cast<uint64_t>(image->info.guest_format),
				                     static_cast<uint64_t>(image->backing.format));
				image_extent  = pack(image->backing.extent.width, image->backing.extent.height);
			}
			const uint64_t words[] = {
			    i,
			    info.data.address,
			    info.data.size,
			    pack(static_cast<uint64_t>(info.guest_format), static_cast<uint64_t>(info.pixel_format)),
			    pack(static_cast<uint64_t>(info.type), static_cast<uint64_t>(info.tile_mode)),
			    pack(info.extent.width, info.extent.height),
			    pack(info.extent.depth, info.resources.levels),
			    pack(info.resources.layers, info.samples),
			    pack(static_cast<uint64_t>(view.format), static_cast<uint64_t>(view.type)),
			    pack(view.base_level, view.level_count),
			    pack(view.base_layer, view.layer_count),
			    pack(view.min_lod, static_cast<VkImageUsageFlags>(view.usage)),
			    pack(static_cast<uint64_t>(view.mapping.r) | (static_cast<uint64_t>(view.mapping.g) << 8u) |
			             (static_cast<uint64_t>(view.mapping.b) << 16u) | (static_cast<uint64_t>(view.mapping.a) << 24u),
			         static_cast<VkImageAspectFlags>(view.aspect)),
			    pack(static_cast<uint64_t>(binding.desc.type), static_cast<uint64_t>(binding.layout)),
			    binding.mip_views.size(),
			    image_address,
			    image_format,
			    image_extent,
			};
			CmdSignature::Fold(CmdSignature::Kind::Image, words, sizeof(words));
		}
		// Samplers by their key (the handle names it only within this process).
		for (size_t i = 0; i < prepared->samplers.size(); i++) {
			std::array<uint32_t, 5> key {};
			const bool              known = sampler_cache.KeyOf(prepared->samplers[i], key);
			CmdSignature::FoldWords(CmdSignature::Kind::Sampler,
			                        {i, known ? 1u : 0u, pack(key[0], key[1]), pack(key[2], key[3]), key[4]});
		}
		for (const auto& source: prepared->buffer_sources) {
			CmdSignature::FoldWords(CmdSignature::Kind::Buffer, {source.address, source.size});
		}
		if (!prepared->shader_data.empty()) {
			CmdSignature::Fold(CmdSignature::Kind::ShaderData, prepared->shader_data.data(),
			                   prepared->shader_data.size() * sizeof(uint32_t));
		}
		const auto& srt = prepared->runtime->resources->flattened_srt;
		if (!srt.empty()) {
			CmdSignature::Fold(CmdSignature::Kind::Srt, srt.data(), srt.size() * sizeof(uint32_t));
		}
	}
}

namespace DescriptorTrace {

namespace {
const std::string& SrtDumpList() {
	static constinit Common::OnceObject<std::string> list_once;
	return list_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_SRT_DUMP");
		return value != nullptr ? std::string(value) : std::string();
	});
}
} // namespace

void ImageBinding(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t i,
                  const TextureBinding& binding) {
	// KYTY_TRACE_TEX_ADDRS=<hex,hex,...>: every distinct (shader, binding) view of those guest
	// addresses, with the extent/format the descriptor asked for.
	static constinit Common::OnceObject<std::string> tex_addrs_once;
	const std::string& tex_addrs = tex_addrs_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_TEX_ADDRS");
		return value != nullptr ? std::string(value) : std::string();
	});
	// KYTY_TRACE_TEX_HASH=<hex hash>: the same for every binding of that shader.
	static constinit Common::OnceValue<uint64_t> tex_hash_once;
	const uint64_t tex_hash = tex_hash_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_TEX_HASH");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	});
	if ((!tex_addrs.empty() || tex_hash != 0) && binding.desc.info.data.address != 0) {
		char text[24];
		std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(binding.desc.info.data.address));
		if ((!tex_addrs.empty() && tex_addrs.find(text) != std::string::npos) ||
		    program.shader_hash == tex_hash) {
			static std::mutex                                            mutex;
			static std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> seen;
			std::scoped_lock                                             lock(mutex);
			uint32_t image_width = 0, image_height = 0, image_tile = 0, image_depth = 0;
			std::string image_format;
			if (binding.image_id) {
				const auto& bound = context.GetTextureCache().GetImage(binding.image_id);
				image_width       = bound.backing.extent.width;
				image_height      = bound.backing.extent.height;
				image_tile        = static_cast<uint32_t>(bound.info.tile_mode);
				image_depth       = bound.info.IsDepth() ? 1u : 0u;
				image_format      = vk::to_string(bound.backing.format);
			}
			const auto key = std::make_tuple(program.shader_hash, i, binding.desc.info.extent.width * 65536u + image_width);
			if (std::ranges::find(seen, key) == seen.end() && seen.size() < 400u) {
				seen.push_back(key);
				std::fprintf(stderr, "TEX_VIEW addr=0x%s stage=%u hash=0x%016llx binding=%u %s extent=%ux%u image=%ux%u fmt=%s tile=0x%x guest_fmt=%u | bound_fmt=%s bound_tile=0x%x bound_depth=%u marker=[%s]\n",
				             text, static_cast<uint32_t>(program.stage),
				             static_cast<unsigned long long>(program.shader_hash), i,
				             program.info.images[i].written ? "write" : "read", binding.desc.info.extent.width,
				             binding.desc.info.extent.height, image_width, image_height,
				             vk::to_string(binding.desc.info.pixel_format).c_str(),
				             static_cast<uint32_t>(binding.desc.info.tile_mode),
				             static_cast<uint32_t>(binding.desc.info.guest_format), image_format.c_str(),
				             image_tile, image_depth, DebugMarkerPath().c_str());
			}
		}
	}
	// KYTY_TRACE_TEX_WRITES_EXTENT=<W>x<H>: every distinct (shader, binding) storage write to an
	// image of that extent, with its address (finds the pass that writes a small intermediate image).
	static constinit Common::OnceObject<std::pair<uint32_t, uint32_t>> write_extent_once;
	const std::pair<uint32_t, uint32_t>& write_extent = write_extent_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_TEX_WRITES_EXTENT");
		uint32_t    w = 0, h = 0;
		if (value != nullptr && std::sscanf(value, "%ux%u", &w, &h) != 2) w = h = 0;
		return std::pair<uint32_t, uint32_t> {w, h};
	});
	if (write_extent.first != 0 && program.info.images[i].written &&
	    binding.desc.info.extent.width == write_extent.first && binding.desc.info.extent.height == write_extent.second) {
		static std::mutex                                  mutex;
		static std::vector<std::pair<uint64_t, uint64_t>>  seen;
		std::scoped_lock                                   lock(mutex);
		const auto key = std::make_pair(program.shader_hash, binding.desc.info.data.address);
		if (std::ranges::find(seen, key) == seen.end() && seen.size() < 200u) {
			seen.push_back(key);
			uint32_t bw = 0, bh = 0;
			if (binding.image_id) {
				const auto& bound = context.GetTextureCache().GetImage(binding.image_id);
				bw = bound.backing.extent.width;
				bh = bound.backing.extent.height;
			}
			std::fprintf(stderr, "TEX_WRITE frame=%u addr=0x%llx stage=%u hash=0x%016llx binding=%u extent=%ux%u image=%ux%u fmt=%s marker=[%s]\n",
			             DebugPresentCount().load(), static_cast<unsigned long long>(binding.desc.info.data.address),
			             static_cast<uint32_t>(program.stage), static_cast<unsigned long long>(program.shader_hash), i,
			             binding.desc.info.extent.width, binding.desc.info.extent.height, bw, bh,
			             vk::to_string(binding.desc.info.pixel_format).c_str(), DebugMarkerPath().c_str());
		}
	}
}

void SrtDump(uint64_t hash, const std::vector<uint32_t>& words) {
	// KYTY_TRACE_SRT_DUMP=<hex hash,...>: the listed shaders' flattened SRT words (as floats), every
	// 150th use, first 12 dumps; for reading the constants a pass computes with.
	if (SrtDumpList().find(fmt::format("{:016x}", hash)) != std::string::npos) {
		static std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> counters;
		auto& [uses, dumps] = counters[hash];
		if (uses++ % 150u == 0u && dumps++ < 12u) {
			std::string line = fmt::format("SRT_DUMP hash=0x{:016x} words={}", hash, words.size());
			for (size_t w = 0; w < words.size(); w++) {
				const auto word = words[w];
				line += fmt::format(" {}:{:08x}/{:g}", w, word, std::bit_cast<float>(word));
			}
			std::fprintf(stderr, "%s\n", line.c_str());
		}
	}
}

// KYTY_TRACE_AUDIT_FRAME=<n> (with KYTY_TRACE_FRAME_DIR and KYTY_TRACE_FLIPS): in presented frame n each
// image a shader binds is listed (AUDIT_USE: shader, stage, binding, read/write, image) and every
// distinct one is sampled once (8x8 grid of texels or blocks of mip 0, middle slice) into
// <dir>/audit_<id>.bin; bindings that resolve to no guest memory give AUDIT_NULL. scripts/frameaudit.py
// turns these lines into the list of suspects (constant inputs, null bindings).
void AuditImageBinding(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                       uint32_t index, const TextureBinding& binding, Image& image, bool written) {
	// Read once: getenv per image binding took ~12% of the GPU thread (it locks and scans the
	// environment on every call).
	static constinit Common::OnceObject<const char*> dir_once;
	const char* const dir = dir_once.Get([] { return Common::EnvRead::FrameDir(); });
	if (dir == nullptr || DebugPresentCount().load() != AuditFrame()) return;
	static uint32_t lines = 0;
	if (lines++ > 20000u) return;
	const auto id = binding.image_id.index;
	if (binding.desc.info.data.Empty()) {
		std::fprintf(stderr, "AUDIT_NULL shader=%016llx stage=%u binding=%u %s fmt=%s\n",
		             static_cast<unsigned long long>(program.shader_hash), static_cast<uint32_t>(program.stage), index,
		             written ? "write" : "read", vk::to_string(binding.desc.view_info.format).c_str());
		return;
	}
	const auto mip = std::min<uint32_t>(binding.desc.view_info.base_level, image.backing.mip_levels - 1u);
	std::fprintf(stderr,
	             "AUDIT_USE shader=%016llx stage=%u binding=%u %s id=%u addr=0x%llx %ux%ux%u fmt=%s mips=%u view_mip=%u+%u min_lod=%g pc=0x%x\n",
	             static_cast<unsigned long long>(program.shader_hash), static_cast<uint32_t>(program.stage), index,
	             written ? "write" : "read", id, static_cast<unsigned long long>(binding.desc.info.data.address),
	             image.backing.extent.width, image.backing.extent.height, image.backing.extent.depth,
	             vk::to_string(image.backing.format).c_str(), image.backing.mip_levels, binding.desc.view_info.base_level,
	             binding.desc.view_info.level_count, static_cast<double>(binding.desc.view_info.min_lod),
	             program.info.images[index].first_use_pc);
	static std::set<uint64_t> sampled;
	if (written || image.backing.samples != 1 || !sampled.insert((uint64_t {id} << 8u) | mip).second ||
	    sampled.size() > 3000u)
		return;
	const auto format  = image.backing.format;
	const bool depth   = image.info.IsDepth();
	const auto block   = depth ? std::array<uint8_t, 3> {1, 1, 1} : vk::blockExtent(format);
	uint32_t   texel   = depth ? ((format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint) ? 2u : 4u)
	                           : vk::blockSize(format);
	const auto stride  = (std::max(texel, 4u) + 3u) & ~3u;
	// The first mip the view exposes (streamed textures may not hold the larger ones yet).
	const auto w       = std::max(image.backing.extent.width >> mip, 1u);
	const auto h       = std::max(image.backing.extent.height >> mip, 1u);
	if (texel == 0 || w == 0 || h == 0) return;
	std::vector<vk::BufferImageCopy> regions;
	for (uint32_t gy = 0; gy < 8u; gy++) {
		for (uint32_t gx = 0; gx < 8u; gx++) {
			uint32_t x = static_cast<uint32_t>((uint64_t {w - 1u} * (2u * gx + 1u)) / 16u) / block[0] * block[0];
			uint32_t y = static_cast<uint32_t>((uint64_t {h - 1u} * (2u * gy + 1u)) / 16u) / block[1] * block[1];
			vk::BufferImageCopy region {};
			region.bufferOffset      = regions.size() * stride;
			region.imageSubresource  = {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor, mip, 0, 1};
			region.imageOffset       = vk::Offset3D {static_cast<int32_t>(x), static_cast<int32_t>(y),
			                                         static_cast<int32_t>(std::max(image.backing.extent.depth >> mip, 1u) / 2u)};
			region.imageExtent       = vk::Extent3D {std::min<uint32_t>(block[0], w - x), std::min<uint32_t>(block[1], h - y), 1};
			regions.push_back(region);
		}
	}
	auto&                graphics = context.GetGraphics();
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = regions.size() * stride;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationInfo result {};
	VkBuffer          native = VK_NULL_HANDLE;
	DebugSample       sample;
	if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&buffer_info), &allocation_info,
	                    &native, &sample.allocation, &result) != VK_SUCCESS) {
		return;
	}
	auto& scheduler = context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, native,
	                          static_cast<uint32_t>(regions.size()), regions.data());
	image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {},
	              command);
	sample.buffer    = native;
	sample.mapped    = result.pMappedData;
	sample.dump_size = buffer_info.size;
	sample.dump_path = fmt::format("{}/audit_{}_m{}_{}x{}_{}_s{}_t{}.bin", dir, id, mip, w, h, vk::to_string(format), stride, texel);
	DebugPushSample(std::move(sample));
}

} // namespace DescriptorTrace

} // namespace Libs::Graphics
