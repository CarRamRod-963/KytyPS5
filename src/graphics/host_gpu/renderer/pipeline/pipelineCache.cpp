#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/resourceLookahead.h"
#include "graphics/host_gpu/renderer/pipeline/scalarReadCache.h"
#include "graphics/host_gpu/renderer/pipeline/srtWalkerCompileQueue.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

// The guest memory the walk reads on the GPU thread, synchronized with recorded GPU writes like
// the specialization reader's TryReadBufferBacking.
struct GuestScalarMemory {
	static bool Read(uint64_t address, void* data, uint64_t size) {
		return Libs::LibKernel::Memory::TryReadBacking(address, data, size);
	}
	static bool HasGpuDirtyBytes(uint64_t address, uint64_t size) {
		return Libs::LibKernel::Memory::HasGpuDirtyBytes(address, size);
	}
	static void ReadGpuDirtyBytes(std::span<const ScalarReadRange> ranges) {
		Libs::LibKernel::Memory::ReadGpuDirtyBytes(ranges);
	}
	static uint64_t GpuWriteSequence() { return Libs::LibKernel::Memory::GpuWriteSequence(); }
};
using GuestScalarReadCache = ScalarReadCache<GuestScalarMemory>;

// Out of line: the cache is several KiB, and a frame that size costs every cached read a stack
// probe.
[[gnu::noinline]] void ReadShaderScalarMemoryUncached(uint64_t            address,
                                                      std::span<uint32_t> values) {
	GuestScalarMemory    memory;
	GuestScalarReadCache uncached(memory);
	uncached.Read(address, values);
}

bool ReadShaderScalarMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (auto* cache = static_cast<GuestScalarReadCache*>(userdata); cache != nullptr) {
		cache->Read(address, values);
		return true;
	}
	ReadShaderScalarMemoryUncached(address, values);
	return true;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

// The SPIR-V features of the host device every module of this run is emitted for.
ShaderRecompiler::Spirv::Profile HostShaderProfile(const GraphicContext& graphics) {
	// AMD GPUs (vendor 0x1002) run NMin/NMax as v_min/v_max, which give the guest's NaN and
	// signed-zero results; GLSL.std.450 alone does not order -0 below +0.
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u,
	        .device_clock_period_ps = graphics.device_clock_period_ps};
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
		uint64_t                                     spirv_hash = 0;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		// Replaces the interpreted plan with the one compiled in the background once it is ready.
		void AdoptCompiledWalker() {
			if (walker_job == nullptr || !walker_job->Ready()) return;
			if (auto compiled = walker_job->Take()) {
				compiled->source_reads = std::move(resource_plan.source_reads);
				resource_plan          = std::move(*compiled);
			}
			walker_job.reset();
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		// Refreshes resources and specialization for a draw. False when the draw cannot be
		// specialized: its guest data is unavailable, or the plan cannot serve one of its
		// descriptors (reported once per shader); the caller skips the draw. A dispatch is run
		// instead with the GPU-selected roots that cannot be served bound as null: its writes
		// through its other resources may be what the host or later work reads, and a skipped
		// dispatch would leave all of them stale. Once a dispatch needed that, its later walks
		// bind failed roots as null directly (a walk where every root is served is unchanged).
		bool Materialize(const ShaderRecompiler::IR::SrtRuntime& runtime, ShaderType stage) {
			auto       walk        = runtime;
			const auto materialize = [&] {
				const bool done = ShaderRecompiler::IR::MaterializeResources(
				    resource_plan, walk, resources, specialization);
				// The first report of each shader also goes to the log.
				if (const auto report = ShaderRecompiler::IR::TakeSpecializationFailureReport();
				    !report.empty()) {
					PipelineCacheLog("{}", report);
				}
				return done;
			};
			walk.null_failed_roots = null_failed_roots;
			if (!materialize()) {
				if (resources.unavailable || stage != ShaderType::Compute || null_failed_roots)
					return false;
				null_failed_roots      = true;
				walk.null_failed_roots = true;
				if (!materialize()) return false;
			}
			NoteUserDataReads();
			return true;
		}

		// After a refresh (Materialize, or a lookahead snapshot taken in its place).
		void NoteUserDataReads() {
			resources.user_data_reads =
			    resource_plan.user_data_reads |
			    (call_source != nullptr
			         ? ShaderRecompiler::IR::PlanUserDataReads(call_source->call_targets)
			         : 0u);
		}

		// Requires a successful Materialize for the current draw.
		template <bool HasCalls>
		Permutation* FindPermutation(uint32_t push_data_cursor) {
			for (auto& candidate: permutations) {
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword !=
				        ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                                 layout.ShaderDataDwords()) ||
				    !ShaderRecompiler::IR::SpecializationServes(
				        resource_plan, candidate.specialization, specialization)) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				return &candidate;
			}
			// A module whose candidate slots hold this draw's candidates serves it once they are
			// moved there; the latest module has the widest layout.
			for (auto it = permutations.rbegin(); it != permutations.rend(); ++it) {
				const auto& layout = it->program.bindings;
				if (layout.push_data_start_dword !=
				    ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                             layout.ShaderDataDwords())) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(it->function_code, FunctionCode())) continue;
				}
				if (ShaderRecompiler::IR::AdaptToSpecialization(resource_plan, it->specialization,
				                                                resources, specialization)) {
					return &*it;
				}
			}
			return nullptr;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		// The same plan with its general walker being compiled (see SrtWalkerCompileQueue).
		std::shared_ptr<SrtWalkerCompileQueue::Job>  walker_job;
		// The ranges the last walk read back (ScalarReadCache::Prefetch).
		std::vector<ScalarReadRange>                 walk_read_backs;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		// The plan the lookahead helpers evaluate (null when they are off or for stages the
		// command streams do not plan).
		std::shared_ptr<LookaheadPlan>             lookahead;
		bool                                       skip_dispatch = false;
		// A dispatch of this shader had a GPU-selected root the plan cannot serve (Materialize).
		bool null_failed_roots = false;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	// A draw whose GPU-selected resources cannot be enumerated (unmapped guest memory, more
	// probes than the host reads) or served (a descriptor the plan does not support) is skipped
	// rather than run with descriptors that may name the wrong memory; the shader's next draw is
	// specialized afresh.
	template <typename InputInfo>
	static ShaderProgram SkipUnavailableDraw(ShaderType stage, uint64_t hash,
	                                         InputInfo& input_info) {
		input_info.stage = {};
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1u, std::memory_order_relaxed) < 64u) {
			PipelineCacheLog("Shader {} {:016x}: resources cannot be specialized for this draw, "
			                 "skipped",
			                 static_cast<uint32_t>(stage), hash);
		}
		return {};
	}

	// Names the first resource whose specialization changed between a shader's last permutation
	// and the one about to be compiled, so a shader that keeps compiling can be understood.
	static void LogPermutationDifference(const char* stage_name, uint64_t hash,
	                                     const ShaderRecompiler::IR::ResourceSpecialization& old,
	                                     const ShaderRecompiler::IR::ResourceSpecialization& now) {
		if (old.buffers.size() != now.buffers.size() || old.images.size() != now.images.size() ||
		    old.samplers.size() != now.samplers.size()) {
			LOGF("Permutation: %s %016" PRIx64 " resource counts buffers %zu->%zu images %zu->%zu "
			     "samplers %zu->%zu\n",
			     stage_name, hash, old.buffers.size(), now.buffers.size(), old.images.size(),
			     now.images.size(), old.samplers.size(), now.samplers.size());
			return;
		}
		for (size_t i = 0; i < now.buffers.size(); ++i) {
			const auto& a = old.buffers[i];
			const auto& b = now.buffers[i];
			if (a == b) continue;
			LOGF("Permutation: %s %016" PRIx64 " buffer %zu stride %u->%u format %u->%u swizzle "
			     "%u->%u zero_oob %d->%d scalar_empty %d->%d unbound %d->%d root %u->%u mapping "
			     "%u->%u iterations %u->%u\n",
			     stage_name, hash, i, a.packed_stride, b.packed_stride,
			     static_cast<uint32_t>(a.descriptor_format),
			     static_cast<uint32_t>(b.descriptor_format), a.descriptor_swizzle,
			     b.descriptor_swizzle, a.zero_stride_oob ? 1 : 0, b.zero_stride_oob ? 1 : 0,
			     a.scalar_empty ? 1 : 0, b.scalar_empty ? 1 : 0, a.unbound ? 1 : 0,
			     b.unbound ? 1 : 0, a.indirect_root, b.indirect_root, a.indirect_mapping_offset,
			     b.indirect_mapping_offset, a.indirect_search_iterations,
			     b.indirect_search_iterations);
			return;
		}
		for (size_t i = 0; i < now.images.size(); ++i) {
			const auto& a = old.images[i];
			const auto& b = now.images[i];
			if (a == b) continue;
			LOGF("Permutation: %s %016" PRIx64 " image %zu numeric %u->%u dimension %u->%u mips "
			     "%u->%u conversion %u->%u swizzle %u->%u root %u->%u mapping %u->%u iterations "
			     "%u->%u cube %d->%d fmask %d->%d bindless %d->%d\n",
			     stage_name, hash, i, static_cast<uint32_t>(a.numeric_class),
			     static_cast<uint32_t>(b.numeric_class), static_cast<uint32_t>(a.dimension),
			     static_cast<uint32_t>(b.dimension), a.mip_count, b.mip_count,
			     static_cast<uint32_t>(a.conversion_format), static_cast<uint32_t>(b.conversion_format),
			     a.shader_swizzle, b.shader_swizzle, a.indirect_root, b.indirect_root,
			     a.indirect_mapping_offset, b.indirect_mapping_offset, a.indirect_search_iterations,
			     b.indirect_search_iterations, a.cube ? 1 : 0, b.cube ? 1 : 0, a.fmask ? 1 : 0,
			     b.fmask ? 1 : 0, a.bindless ? 1 : 0, b.bindless ? 1 : 0);
			return;
		}
		for (size_t i = 0; i < now.samplers.size(); ++i) {
			const auto& a = old.samplers[i];
			const auto& b = now.samplers[i];
			if (a == b) continue;
			LOGF("Permutation: %s %016" PRIx64 " sampler %zu root %u->%u mapping %u->%u iterations "
			     "%u->%u\n",
			     stage_name, hash, i, a.indirect_root, b.indirect_root, a.indirect_mapping_offset,
			     b.indirect_mapping_offset, a.indirect_search_iterations,
			     b.indirect_search_iterations);
			return;
		}
		LOGF("Permutation: %s %016" PRIx64 " specialization equal; push data or function code "
		     "changed\n",
		     stage_name, hash);
	}

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.profile->maintenance8)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		// Captures (RenderDoc) show the guest shader behind each module.
		SetVulkanObjectNameF(graphics.device, module, "{} {:016x}", stage_name, options.shader_hash);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		    .spirv_hash = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t)),
		};
	}

	// Adds a compiled compute permutation to the title's manifest and returns its key. Shaders
	// with calls read guest code at run time and are not recorded.
	uint64_t RecordComputeProgram(const ShaderParams& params, const ShaderComputeInputInfo& input,
	                              uint32_t push_data_cursor, const Permutation& permutation) {
		ComputeProgramRecord record;
		record.shader_hash = params.hash;
		record.code =
		    std::make_shared<const std::vector<uint32_t>>(params.code.begin(), params.code.end());
		record.user_data_count  = params.user_data_count;
		record.input            = ComputeProgramRecord::StaticInput(input);
		record.push_data_cursor = push_data_cursor;
		record.specialization   = permutation.specialization;
		record.spirv_hash       = permutation.spirv_hash;
		const auto key          = record.Key();
		compute_recorded++;
		if (!manifest->Add(record) || (list_predicts && list_predicts(key))) {
			compute_predicted++;
		}
		return key;
	}

	// Adds a compiled graphics stage permutation to the title's pipeline manifest and returns
	// its key. Shaders with calls are not recorded.
	template <typename InputInfo>
	uint64_t RecordGraphicsStage(ShaderType stage, const ShaderParams& params,
	                             const InputInfo& input, uint32_t push_data_cursor,
	                             const Permutation& permutation) {
		GraphicsStageRecord record;
		record.stage            = stage;
		record.shader_hash      = params.hash;
		record.user_data_count  = params.user_data_count;
		record.push_data_cursor = push_data_cursor;
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			record.pixel = GraphicsStageRecord::StaticInput(input);
		} else {
			record.vertex = GraphicsStageRecord::StaticInput(input);
		}
		record.specialization = permutation.specialization;
		record.spirv_hash     = permutation.spirv_hash;
		record.SetCode(
		    std::make_shared<const std::vector<uint32_t>>(params.code.begin(), params.code.end()),
		    params.back_code.empty() ? nullptr
		                             : std::make_shared<const std::vector<uint32_t>>(
		                                   params.back_code.begin(), params.back_code.end()));
		(void)pipeline_manifest->AddStage(record);
		return record.Key();
	}

	// Counts a graphics shader's first use, and whether the guest created it before.
	void NoteGraphicsShaderUse(ShaderType stage, std::span<const uint32_t> code) {
		if (created_code == nullptr || stage == ShaderType::Compute) return;
		const auto id = ShaderCodeId::Of(code);
		if (!census_seen.insert(id.hash).second) return;
		auto& counts = census[static_cast<size_t>(stage)];
		counts[0]++;
		if (created_code->Find(id) != nullptr) counts[1]++;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		GuestScalarMemory    scalar_memory;
		GuestScalarReadCache read_cache(scalar_memory);
		static_assert(GuestScalarReadCache::ChunkBytes ==
		                  ShaderRecompiler::IR::SrtReadCacheView::ChunkBytes &&
		              std::has_single_bit(GuestScalarReadCache::Entries));
		const ShaderRecompiler::IR::SrtReadCacheView read_cache_view {
		    .tags       = read_cache.Tags(),
		    .data       = read_cache.Data(),
		    .stride     = GuestScalarReadCache::Stride(),
		    .index_mask = static_cast<uint32_t>(GuestScalarReadCache::Entries - 1u),
		};
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderScalarMemory,
		    .userdata                   = &read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .image_heap                 = image_heap,
		    .read_cache                 = &read_cache_view,
		    .mapped_range_size          = Libs::LibKernel::Memory::MappedRangeSize,
		};
		ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.stage            = input_info.logical_stage;
			options.input_info.vertex = &input_info;
			options.user_data_base   = 8;
			if (options.stage == ShaderType::Mesh ||
			    options.stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size =
				    options.stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			options.stage            = ShaderType::Pixel;
			options.input_info.pixel = &input_info;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			options.stage              = ShaderType::Compute;
			options.input_info.compute = &input_info;
			runtime.workgroup_counts   = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		const auto stage           = options.stage;
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto entry = programs.find(lookup_key);
		const auto bind = [&](const Permutation& permutation) {
			input_info.stage = {.program   = &permutation.program,
			                    .resources = &entry->second.resources,
			                    .handle    = &permutation.handle};
			permutation.program.bindings.AdvancePushData(push_data_cursor);
			return permutation.handle;
		};
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			if (cached.skip_dispatch) return {};
			cached.AdoptCompiledWalker();
			Permutation* permutation = nullptr;
			if (cached.call_source) source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) {
				bool adopted = false;
				if (cached.lookahead != nullptr) {
					PublishLookahead(stage, params, cached);
					adopted =
					    lookahead_command != nullptr && AdoptLookahead(cached, params, input_info);
				}
				bool materialized = adopted;
				if (!adopted) {
					read_cache.Prefetch(cached.walk_read_backs);
					materialized = cached.Materialize(runtime, stage);
					read_cache.Prefetch({});
					if (const auto read_backs = read_cache.ReadBacks(); !read_backs.empty()) {
						cached.walk_read_backs.assign(read_backs.begin(), read_backs.end());
					}
				}
				if (!materialized) return SkipUnavailableDraw(stage, params.hash, input_info);
				permutation = cached.call_source ? cached.FindPermutation<true>(push_data_cursor)
				                                 : cached.FindPermutation<false>(push_data_cursor);
			}
			if (permutation != nullptr) return bind(*permutation);
		}

		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		options.profile     = &profile;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			programs.try_emplace(lookup_key, ShaderRecompiler::IR::ResourcePlan{}, nullptr)
			    .first->second.skip_dispatch = true;
			return {};
		}
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (!current->calls.empty()) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == programs.end()) {
			NoteGraphicsShaderUse(stage, params.code);
			auto retained = !source.calls.empty() ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			std::shared_ptr<SrtWalkerCompileQueue::Job> walker_job;
			entry = programs
			            .try_emplace(lookup_key, ExtractPlan(translated.program, walker_job),
			                         std::move(retained))
			            .first;
			entry->second.walker_job = std::move(walker_job);
			if (lookahead != nullptr && SlotOf(stage) != LookaheadSlot::None &&
			    !entry->second.call_source && entry->second.walker_job != nullptr) {
				entry->second.lookahead = std::make_shared<LookaheadPlan>(
				    walker_queue.Submit(std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
				        ShaderRecompiler::IR::ExtractResourcePlan(
				            translated.program,
				            ShaderRecompiler::IR::SrtWalkerCompilation::SimpleOnly))),
				    next_lookahead_helper++);
				PublishLookahead(stage, params, entry->second);
			}
			if (!entry->second.Materialize(runtime, stage))
				return SkipUnavailableDraw(stage, params.hash, input_info);
		} else if (source_changed) {
			entry->second.resource_plan = ExtractPlan(translated.program, entry->second.walker_job);
			if (!entry->second.Materialize(runtime, stage))
				return SkipUnavailableDraw(stage, params.hash, input_info);
			if (const auto* permutation = entry->second.FindPermutation<true>(push_data_cursor)) {
				return bind(*permutation);
			}
		}
		// One line per re-translation (never per draw), so it is always on.
		if (!entry->second.permutations.empty()) {
			LogPermutationDifference(stage_name, params.hash,
			                         entry->second.permutations.back().specialization,
			                         entry->second.specialization);
		}
		// A shader whose GPU-selected tables change with the scene gets the union of this draw's
		// candidate layout and the last module's, so that the draws both serve need no module. A
		// layout an earlier session recorded (whose module the precompile already built) is
		// taken as it is when it serves the draw, and is the one widened before any module.
		const ShaderRecompiler::IR::ResourceSpecialization* prior    = nullptr;
		bool                                                recorded = false;
		// The widest module: a draw that a narrower module served (another candidate count) must
		// not restart the growth from its layout.
		const auto width = [](const ShaderRecompiler::IR::ResourceSpecialization& layout) {
			return layout.images.size() + layout.samplers.size();
		};
		for (const auto& permutation: entry->second.permutations) {
			if (std::ranges::equal(permutation.function_code, entry->second.FunctionCode()) &&
			    (prior == nullptr || width(permutation.specialization) >= width(*prior)))
				prior = &permutation.specialization;
		}
		std::vector<std::shared_ptr<const ComputeProgramRecord>> records;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (manifest != nullptr && !entry->second.call_source) {
				records = manifest->Records();
				// This program's records, the widest first: the module of the widest one that
				// serves the draw also serves the most later draws.
				std::erase_if(
				    records, [&, state = std::vector<uint32_t> {}](const auto& record) mutable {
					    if (record->shader_hash != params.hash ||
					        record->user_data_count != params.user_data_count ||
					        record->code == nullptr || record->code->size() != params.code.size() ||
					        record->push_data_cursor != push_data_cursor)
						    return true;
					    auto input               = record->input;
					    input.host_subgroup_size = input_info.host_subgroup_size;
					    BuildStageStaticKey(input, state);
					    return state != lookup_key.static_state;
				    });
				std::ranges::stable_sort(records, std::greater {}, [&](const auto& record) {
					return width(record->specialization);
				});
				for (const auto& record: records) {
					recorded = ShaderRecompiler::IR::AdaptToSpecialization(
					    entry->second.resource_plan, record->specialization,
					    entry->second.resources, entry->second.specialization);
					if (recorded) break;
				}
				if (!records.empty() &&
				    (prior == nullptr || width(records.front()->specialization) > width(*prior)))
					prior = &records.front()->specialization;
			}
		}
		const auto images_before = entry->second.specialization.images.size();
		if (!recorded && prior != nullptr) {
			ShaderRecompiler::IR::WidenSpecialization(entry->second.resource_plan, *prior,
			                                          entry->second.resources,
			                                          entry->second.specialization);
		}
		if (recorded || entry->second.specialization.images.size() != images_before) {
			LOGF("Permutation: %s %016" PRIx64 " %s images %zu->%zu\n", stage_name, params.hash,
			     recorded ? "recorded layout" : "widened", images_before,
			     entry->second.specialization.images.size());
		}
		// Unbound buffers take the typed layouts already compiled for them, so this module also
		// serves those bound cases.
		auto specialization = entry->second.specialization;
		for (auto it = entry->second.permutations.rbegin(); it != entry->second.permutations.rend();
		     ++it) {
			if (std::ranges::equal(it->function_code, entry->second.FunctionCode()))
				ShaderRecompiler::IR::AdoptServingBuffers(entry->second.resource_plan,
				                                          specialization, it->specialization);
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), std::move(specialization), push_data_cursor,
		    entry->second.FunctionCode()));
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (manifest != nullptr && !entry->second.call_source) {
				auto& added = entry->second.permutations.back();
				added.handle.record_key =
				    RecordComputeProgram(params, input_info, push_data_cursor, added);
			}
		} else {
			if (pipeline_manifest != nullptr && !entry->second.call_source) {
				auto& added = entry->second.permutations.back();
				added.handle.record_key =
				    RecordGraphicsStage(stage, params, input_info, push_data_cursor, added);
			}
		}

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return bind(entry->second.permutations.back());
	}

	// Programs whose walks the command streams plan ahead, by the registers that name them.
	enum class LookaheadSlot : uint64_t { None, Compute, Vertex, Pixel };
	static LookaheadSlot SlotOf(ShaderType stage) {
		switch (stage) {
			case ShaderType::Compute: return LookaheadSlot::Compute;
			case ShaderType::Vertex:
			case ShaderType::Mesh: return LookaheadSlot::Vertex;
			case ShaderType::Pixel: return LookaheadSlot::Pixel;
			default: return LookaheadSlot::None;
		}
	}
	static uint64_t LookaheadKey(LookaheadSlot slot, uint64_t shader_address) {
		return (shader_address << 2u) | static_cast<uint64_t>(slot);
	}

	// The program last used at a shader address: planned walks of that address use its plan, and
	// its user data starts with the same words before the user SGPRs.
	void PublishLookahead(ShaderType stage, const ShaderParams& params, const SourceEntry& entry) {
		auto& target = lookahead_targets[LookaheadKey(SlotOf(stage), params.Base())];
		if (target.plan != entry.lookahead) target.plan = entry.lookahead;
		target.user_data_count = params.user_data_count;
		std::copy_n(params.user_data.begin(), target.prefix.size(), target.prefix.begin());
	}

	// Takes the helper's snapshot for this command when ResourceLookahead allows it.
	template <typename InputInfo>
	bool AdoptLookahead(SourceEntry& cached, const ShaderParams& params,
	                    const InputInfo& input_info) {
		LookaheadInputs inputs;
		inputs.shader_base = params.Base();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			inputs.indirect_args_address = input_info.indirect_args_address;
			inputs.workgroup_counts      = {input_info.workgroup_counts[0],
			                                input_info.workgroup_counts[1],
			                                input_info.workgroup_counts[2]};
		}
		inputs.user_data_count = params.user_data_count;
		std::copy_n(params.user_data.begin(), params.user_data_count, inputs.user_data.begin());
		if (lookahead->TryAdopt(lookahead_stream, lookahead_command, cached.lookahead.get(), inputs,
		                        cached.resources, cached.specialization) !=
		    ResourceLookahead::Outcome::Adopted) {
			return false;
		}
		cached.NoteUserDataReads();
		return true;
	}

	// Plans the walk of the program at `shader_address` with these user SGPRs, if one was used.
	void PlanLookahead(const void* stream, const void* command, LookaheadSlot slot,
	                   uint64_t shader_address, std::span<const uint32_t> user_sgprs,
	                   LookaheadInputs& inputs) {
		const auto target = lookahead_targets.find(LookaheadKey(slot, shader_address));
		if (target == lookahead_targets.end() || target->second.user_data_count < user_sgprs.size())
			return;
		const auto prefix = target->second.user_data_count - user_sgprs.size();
		if (prefix > target->second.prefix.size()) return;
		inputs.shader_base     = shader_address;
		inputs.user_data_count = target->second.user_data_count;
		std::copy_n(target->second.prefix.begin(), prefix, inputs.user_data.begin());
		std::ranges::copy(user_sgprs, inputs.user_data.begin() + static_cast<ptrdiff_t>(prefix));
		lookahead->Plan(stream, command, target->second.plan, inputs);
	}

	// The plan the GPU thread interprets right away, and a second extraction whose general walker
	// is compiled in the background (SourceEntry::AdoptCompiledWalker).
	ShaderRecompiler::IR::ResourcePlan
	ExtractPlan(const ShaderRecompiler::IR::Program&         program,
	            std::shared_ptr<SrtWalkerCompileQueue::Job>& walker_job) {
		using ShaderRecompiler::IR::SrtWalkerCompilation;
		auto plan =
		    ShaderRecompiler::IR::ExtractResourcePlan(program, SrtWalkerCompilation::SimpleOnly);
		walker_job.reset();
		if (plan.resource_tracking_complete && plan.srt_plan_complete && !plan.srt_reads.empty()) {
			walker_job = walker_queue.Submit(std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
			    ShaderRecompiler::IR::ExtractResourcePlan(program,
			                                              SrtWalkerCompilation::SimpleOnly)));
		}
		return plan;
	}

	static bool SameBacking(uint64_t address, const void* data, uint64_t size, bool mapped) {
		const auto result = Libs::LibKernel::Memory::CompareBacking(address, data, size);
		return mapped ? result == Libs::LibKernel::Memory::BackingCompare::Equal
		              : result == Libs::LibKernel::Memory::BackingCompare::Unmapped;
	}

	ProgramCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	             uint32_t lookahead_threads)
	    : graphics(graphics), image_heap(image_heap), profile {HostShaderProfile(graphics)} {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		if (lookahead_threads != 0) {
			lookahead = std::make_unique<ResourceLookahead>(
			    LookaheadMemory {.read        = Libs::LibKernel::Memory::TryReadBacking,
			                     .gpu_dirty   = Libs::LibKernel::Memory::HasGpuDirtyBytes,
			                     .same        = SameBacking,
			                     .mapped_size = Libs::LibKernel::Memory::MappedRangeSize},
			    lookahead_threads, image_heap);
		}
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	ShaderRecompiler::IR::ImageHeapSource*                      image_heap;
	const ShaderRecompiler::Spirv::Profile                      profile;
	uint64_t                                                    next_shader_id = 0;
	// Receives every compiled compute permutation; null when there is no manifest.
	ProgramManifest* manifest = nullptr;
	// Receives every compiled graphics stage permutation; null when there is no manifest.
	PipelineManifest* pipeline_manifest = nullptr;
	// Whether an imported pipeline list predicts a compute program key (may be empty).
	std::function<bool(uint64_t)> list_predicts;
	// Compute permutations recorded this session, and those a record or the list predicted.
	uint64_t compute_recorded  = 0;
	uint64_t compute_predicted = 0;
	// Graphics shaders used this session, per stage, and those the guest had created first
	// (created_code; census of what a list can resolve at creation).
	const ShaderCodeStore*       created_code = nullptr;
	std::unordered_set<uint64_t> census_seen;
	std::array<std::array<uint64_t, 2>, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1>
	    census {};
	// Last, so its thread stops before the entries holding its jobs are destroyed.
	SrtWalkerCompileQueue walker_queue;

	// Compute dispatches ahead of the GPU thread: the program last used at each shader address
	// with its registers, and the command the next GetComputeProgram belongs to.
	struct LookaheadTarget {
		std::shared_ptr<LookaheadPlan> plan;
		uint32_t                       user_data_count = 0;
		std::array<uint32_t, 8>        prefix {};
	};
	std::unique_ptr<ResourceLookahead>            lookahead;
	std::unordered_map<uint64_t, LookaheadTarget> lookahead_targets;
	const void*                                   lookahead_stream      = nullptr;
	const void*                                   lookahead_command     = nullptr;
	uint32_t                                      next_lookahead_helper = 0;
	std::chrono::steady_clock::time_point         next_lookahead_report {};
};

PrecompileOutcome PrecompileComputeProgram(GraphicContext& graphics, vk::PipelineCache driver_cache,
                                           const ComputeProgramRecord& record, uint64_t* spirv_hash,
                                           bool code_only, PipelineBuildMode mode) {
	EXIT_IF(mode != PipelineBuildMode::Precompile && mode != PipelineBuildMode::Unoptimized);
	EXIT_IF(record.code == nullptr || record.code->empty());
	// The inputs GetComputeProgram derives from the device.
	auto input               = record.input;
	input.host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
	input.lds_storage = input.lds_size_dwords * 4u >
	                    graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	// Translation reads the user-data count; the values only feed the resource walk.
	const std::vector<uint32_t> user_data(record.user_data_count, 0u);

	const auto                       profile = HostShaderProfile(graphics);
	ShaderRecompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.profile            = &profile;
	options.shader_hash        = record.shader_hash;
	options.user_data          = user_data;
	options.dump_ir            = false;
	options.early_dump         = false;
	options.emit_debug_info    = Config::GraphicsDebugDumpEnabled();
	options.dump_label         = "Precompile CS";
	options.input_info.compute = &input;
	options.wave_size          = input.wave_size;

	const auto compile = [&]() -> std::optional<ShaderRecompiler::CompileResult> {
		auto source = ShaderRecompiler::PrepareShaderSource(*record.code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			return std::nullopt;
		}
		auto       translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		const auto specialization =
		    code_only ? ShaderRecompiler::IR::CodeOnlySpecialization(translated.program)
		              : std::optional(record.specialization);
		// A recompiler change may give the shader other resources than the record was made for.
		if (!specialization.has_value() ||
		    !ShaderRecompiler::IR::SpecializationApplies(translated.program, *specialization)) {
			return std::nullopt;
		}
		return ShaderRecompiler::CompileProgram(std::move(translated), options, *specialization,
		                                        record.push_data_cursor);
	};
	std::optional<ShaderRecompiler::CompileResult> result;
	// No dispatch has used a created shader, and a recorded program may meet a recompiler that
	// no longer handles its code: either is skipped instead of ending the process.
	try {
		const Common::RecoverableFatalScope recoverable;
		result = compile();
	} catch (const Common::RecoverableFatal& fatal) {
		PipelineCacheLog("Precompile: {} CS {:016x} cannot be translated: {}",
		                 code_only ? "created" : "recorded", record.shader_hash, fatal.message);
		return PrecompileOutcome::Skipped;
	}
	if (!result.has_value() || !ValidateShaderSpirv(options.dump_label, record.shader_hash,
	                                                result->spirv, profile.maintenance8)) {
		return PrecompileOutcome::Skipped;
	}
	if (spirv_hash != nullptr) {
		*spirv_hash = XXH3_64bits(result->spirv.data(), result->spirv.size() * sizeof(uint32_t));
	}
	const auto module = CompileSPV(result->spirv, graphics.device);
	if (module == nullptr) {
		return PrecompileOutcome::Skipped;
	}
	const auto                             program = std::move(result->program).TakeCompiledInfo();
	ShaderRecompiler::IR::ResourceSnapshot resources;
	input.stage = {.program = &program, .resources = &resources};

	PipelineCache::Pipeline pipeline;
	bool                    from_driver_cache = false;
	(void)CreatePipelineInternal(graphics, pipeline, input, module, driver_cache, mode,
	                             &from_driver_cache);
	graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	graphics.device.destroyShaderModule(module, nullptr);
	return from_driver_cache ? PrecompileOutcome::Cached : PrecompileOutcome::Compiled;
}

// A graphics stage translated for the precompile, shared by the recorded pipelines that use it.
struct PipelineCache::PrecompiledStage {
	std::mutex                                              mutex;
	bool                                                    done   = false;
	bool                                                    failed = false;
	vk::ShaderModule                                        module = nullptr;
	std::optional<ShaderRecompiler::IR::CompiledShaderInfo> program;
	ShaderRecompiler::IR::ResourceSnapshot                  resources;
	ShaderProgram                                           handle;
	// The input the stage was translated with: the record's, with the device's fields.
	ShaderVertexInputInfo vertex;
	ShaderPixelInputInfo  pixel;
};

// Translates a recorded graphics stage as ProgramCache::Get would. Returns false when the stage
// cannot be rebuilt from its record (calls, a specialization this recompiler no longer derives,
// or code it cannot translate).
bool PipelineCache::TranslateRecordedStage(GraphicContext&            graphics,
                                           const GraphicsStageRecord& record,
                                           PrecompiledStage&          out) {
	static std::atomic<uint64_t>     next_id {0};
	const auto                       profile = HostShaderProfile(graphics);
	ShaderRecompiler::CompileOptions options;
	options.profile         = &profile;
	options.shader_hash     = record.shader_hash;
	options.dump_ir         = false;
	options.early_dump      = false;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	const std::vector<uint32_t> user_data(record.user_data_count, 0u);
	options.user_data = user_data;
	if (record.back_code != nullptr) {
		options.back_code = *record.back_code;
	}
	options.stage = record.stage;
	if (record.stage == ShaderType::Pixel) {
		out.pixel                = record.pixel;
		options.input_info.pixel = &out.pixel;
		options.wave_size        = out.pixel.wave_size;
		options.dump_label       = "Precompile PS";
	} else {
		out.vertex = record.vertex;
		// The fields GetGraphicsPrograms derives from the device.
		if (out.vertex.mesh.threads_num[0] != 0) {
			out.vertex.mesh.host_subgroup_size = graphics.subgroup_size;
		}
		if (out.vertex.clip_space.enabled) {
			const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
			out.vertex.clip_space.half_extent[0] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
			out.vertex.clip_space.half_extent[1] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		}
		options.input_info.vertex = &out.vertex;
		options.wave_size         = out.vertex.wave_size;
		options.user_data_base    = 8;
		if (record.stage == ShaderType::Mesh || record.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
			options.wave_size = record.stage == ShaderType::Mesh ? out.vertex.mesh.wave_size : 64u;
		}
		options.dump_label = "Precompile VS";
	}
	std::optional<ShaderRecompiler::CompileResult> result;
	try {
		const Common::RecoverableFatalScope recoverable;
		auto source = ShaderRecompiler::PrepareShaderSource(*record.code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			return false;
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		if (!ShaderRecompiler::IR::SpecializationApplies(translated.program,
		                                                 record.specialization)) {
			return false;
		}
		result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                          record.specialization, record.push_data_cursor);
	} catch (const Common::RecoverableFatal& fatal) {
		PipelineCacheLog("Precompile: recorded stage {:016x} cannot be translated: {}",
		                 record.shader_hash, fatal.message);
		return false;
	}
	if (!ValidateShaderSpirv(options.dump_label, record.shader_hash, result->spirv,
	                         profile.maintenance8)) {
		return false;
	}
	out.module = CompileSPV(result->spirv, graphics.device);
	if (out.module == nullptr) {
		return false;
	}
	out.program.emplace(std::move(result->program).TakeCompiledInfo());
	out.handle = {.id = ++next_id, .module = out.module};
	return true;
}

PrecompileOutcome PipelineCache::PrecompileGraphicsPipeline(
    const GraphicsPipelineRecord&                               record,
    std::span<const std::shared_ptr<const GraphicsStageRecord>> stages, PipelineBuildMode mode) {
	EXIT_IF(stages.size() != record.vertex_stage_count + (record.pixel_stage != 0 ? 1u : 0u));
	// Each stage once, however many pipelines use it.
	std::vector<std::shared_ptr<PrecompiledStage>> translated;
	for (const auto& stage: stages) {
		std::shared_ptr<PrecompiledStage> memo;
		{
			std::lock_guard lock(m_stage_memo_mutex);
			auto&           entry = m_stage_memo[stage->Key()];
			if (entry == nullptr) entry = std::make_shared<PrecompiledStage>();
			memo = entry;
		}
		std::lock_guard lock(memo->mutex);
		if (!memo->done) {
			memo->done   = true;
			memo->failed = !TranslateRecordedStage(m_graphics, *stage, *memo);
		}
		if (memo->failed) {
			return PrecompileOutcome::Skipped;
		}
		translated.push_back(std::move(memo));
	}
	DrawRenderState state;
	for (uint32_t i = 0; i < record.vertex_stage_count; i++) {
		const auto& stage          = *translated[i];
		state.vertex_info[i]       = stage.vertex;
		state.vertex_info[i].stage = {
		    .program = &*stage.program, .resources = &stage.resources, .handle = &stage.handle};
	}
	state.ps_active = record.pixel_stage != 0;
	if (state.ps_active) {
		const auto& stage         = *translated.back();
		state.ps_input_info       = stage.pixel;
		state.ps_input_info.stage = {
		    .program = &*stage.program, .resources = &stage.resources, .handle = &stage.handle};
	}
	state.topology                 = record.static_params.topology;
	state.primitive_restart_enable = record.static_params.primitive_restart_enable;
	Pipeline pipeline;
	bool     from_driver_cache = false;
	try {
		const Common::RecoverableFatalScope recoverable;
		(void)CreatePipelineInternal(m_graphics, pipeline, record.rendering, record.vertex_input,
		                             state, record.static_params, m_driver_cache, mode,
		                             &from_driver_cache);
	} catch (const Common::RecoverableFatal& fatal) {
		PipelineCacheLog("Precompile: recorded pipeline {:016x} cannot be built: {}", record.Key(),
		                 fatal.message);
		return PrecompileOutcome::Skipped;
	}
	m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	return from_driver_cache ? PrecompileOutcome::Cached : PrecompileOutcome::Compiled;
}

void PipelineCache::ReleasePrecompiledStages() {
	std::lock_guard lock(m_stage_memo_mutex);
	for (const auto& [key, stage]: m_stage_memo) {
		(void)key;
		if (stage->module != nullptr) {
			m_graphics.device.destroyShaderModule(stage->module, nullptr);
		}
	}
	m_stage_memo.clear();
}

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap,
                             CommandScheduler*                       scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      // The lookahead plans from the command streams, which run with a scheduler.
      m_program_cache(std::make_unique<ProgramCache>(
          graphics, image_heap,
          scheduler != nullptr ? std::min(Config::LookaheadThreads(), 8u) : 0u)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	ShaderRecompiler::Spirv::SetMaximalReconvergence(graphics.maximal_reconvergence_enabled);
	InitializeDriverCache();
	// Both modes replace or install pipeline handles while a command buffer is recording, which
	// needs the scheduler.
	m_defer_optimization = m_scheduler != nullptr && Config::AsyncPipelinesEnabled();
	m_skip_pending_draws = m_scheduler != nullptr && Config::SkipPendingDrawsEnabled();
	if (m_defer_optimization || m_skip_pending_draws) {
		// Driver compilers are multithreaded themselves; two builds at a time keep the CPU
		// available for the guest.
		const auto threads = std::clamp(std::thread::hardware_concurrency() / 8u, 1u, 2u);
		m_compile_queue    = std::make_unique<PipelineCompileQueue>(
		    threads, [device = graphics.device](uint64_t handle) {
			    device.destroyPipeline(PipelineFromCompileHandle(handle), nullptr);
		    });
	}
}

PipelineCache::~PipelineCache() {
	if (m_created != nullptr) {
		ShaderSetCreatedComputeSink({});
	}
	if (m_created_code != nullptr) {
		ShaderSetCreatedCodeSink({});
	}
	StopBackgroundCompiles();
	(void)Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::StopBackgroundCompiles() {
	if (m_precompile_queue != nullptr) {
		{
			// Created shaders stop queueing list entries.
			std::lock_guard lock(m_list_mutex);
			if (m_list_open) {
				m_list_open = false;
				(void)m_precompile_progress.EndQueueing();
			}
		}
		const auto remaining = m_precompile_queue->Remaining();
		m_precompile_queue->Stop();
		if (remaining != 0) {
			LogPrecompileStats("stopped");
		}
		m_precompile_queue.reset();
		ReleasePrecompiledStages();
	}
	if (m_compile_queue == nullptr) {
		return;
	}
	if (!m_pending_optimized.empty()) {
		LOGF("PipelineCache: dropping %zu pending pipeline builds\n", m_pending_optimized.size());
	}
	// Results that were not installed are destroyed here; their pipelines keep the unoptimized
	// handles, or none if they were never built.
	m_compile_queue->Stop();
	m_pending_optimized.clear();
	m_pending_builds.clear();
	m_skip_pending_draws = false;
}

void PipelineCache::DeferOptimization(Pipeline& pipeline, PipelineCompileQueue::Job job) {
	if (!job) {
		return;
	}
	const auto ticket = m_compile_queue->Enqueue(std::move(job));
	m_pending_optimized.emplace(ticket, &pipeline);
}

// Replaces unoptimized pipelines whose optimized builds finished. Runs on the GPU thread while
// a command buffer is being recorded: command buffers recorded so far may have bound the old
// handle, so it is destroyed once the current command buffer has completed.
void PipelineCache::InstallOptimizedPipelines() {
	if (m_precompile_dirty.load(std::memory_order_relaxed) &&
	    m_precompile_dirty.exchange(false, std::memory_order_relaxed)) {
		CheckpointDriverCache();
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_precompile_queue != nullptr && now >= m_next_timeout_check) {
		m_next_timeout_check = now + std::chrono::seconds {5};
		CheckPrecompileTimeouts();
	}
	// Sessions usually end without a save: the coverage is also logged once a minute.
	if (m_pipeline_manifest != nullptr && now >= m_next_coverage_report) {
		if (m_next_coverage_report != std::chrono::steady_clock::time_point {}) {
			LogCoverage();
		}
		m_next_coverage_report = now + std::chrono::minutes {1};
	}
	if (m_pending_optimized.empty() || !m_scheduler->Active()) {
		return;
	}
	m_completed.clear();
	m_compile_queue->TakeCompleted(m_completed);
	for (const auto& completion: m_completed) {
		const auto entry = m_pending_optimized.find(completion.ticket);
		EXIT_IF(entry == m_pending_optimized.end());
		auto& pipeline = *entry->second;
		m_pending_optimized.erase(entry);
		if (pipeline.pipeline == nullptr) {
			// A first build: its draws were skipped until now, nothing has bound a handle.
			const auto pending = m_pending_builds.find(&pipeline);
			EXIT_IF(pending == m_pending_builds.end());
			if (completion.handle == 0) {
				EXIT("PipelineCache: background pipeline build failed (VS %016" PRIx64
				     " PS %016" PRIx64 ")\n",
				     pending->second.vs_hash, pending->second.ps_hash);
			}
			pipeline.pipeline = PipelineFromCompileHandle(completion.handle);
			const auto ms     = std::chrono::duration<double, std::milli>(
			                        std::chrono::steady_clock::now() - pending->second.queued)
			                        .count();
			LOGF("PipelineCache: pipeline of VS %016" PRIx64 " PS %016" PRIx64
			     " is ready after %.0f ms; %" PRIu64 " draws were skipped\n",
			     pending->second.vs_hash, pending->second.ps_hash, ms,
			     pending->second.skipped_draws);
			m_pending_builds.erase(pending);
			continue;
		}
		if (completion.handle == 0) {
			LOGF("PipelineCache: optimized build failed; keeping unoptimized pipeline %p\n",
			     static_cast<void*>(pipeline.pipeline));
			continue;
		}
		const auto unoptimized = pipeline.pipeline;
		pipeline.pipeline      = PipelineFromCompileHandle(completion.handle);
		m_scheduler->DeferOperation([device = m_graphics.device, unoptimized] {
			device.destroyPipeline(unoptimized, nullptr);
		});
	}
	if (!m_completed.empty()) {
		CheckpointDriverCache();
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}

	const auto directory = m_driver_cache_path.parent_path();
	SetManifest(
	    std::make_unique<ProgramManifest>(ProgramManifest::PathFor(directory, title_id),
	                                      ProgramManifest::PreviousPathFor(directory, title_id)));
	const auto loaded = m_manifest->Loaded();
	PipelineCacheLog("Program manifest: {} compute programs in {}{}", loaded.records,
	                 Common::PathToString(m_manifest->Path()),
	                 loaded.rewritten ? fmt::format(" (rewritten, {} unreadable bytes dropped)",
	                                                loaded.dropped_bytes)
	                                  : std::string {});
	SetCreatedShaders(
	    std::make_unique<ProgramManifest>(ProgramManifest::CreatedPathFor(directory, title_id)));
	m_precompile_attempts = std::make_unique<PrecompileAttempts>(
	    PrecompileAttempts::PathFor(directory, title_id),
	    PrecompileAttempts::PreviousPathFor(directory, title_id));
	m_title_id = title_id;
	m_pipeline_manifest =
	    std::make_unique<PipelineManifest>(PipelineManifest::PathFor(directory, title_id));
	m_program_cache->pipeline_manifest = m_pipeline_manifest.get();
	m_program_cache->list_predicts     = [this](uint64_t key) {
		std::lock_guard lock(m_list_mutex);
		return m_list_keys.contains(key);
	};
	const auto pipelines = m_pipeline_manifest->Loaded();
	PipelineCacheLog("Pipeline manifest: {} graphics pipelines, {} stages in {}{}",
	                 pipelines.pipelines, pipelines.stages,
	                 Common::PathToString(m_pipeline_manifest->Path()),
	                 pipelines.rewritten ? fmt::format(" (rewritten, {} unreadable bytes dropped)",
	                                                   pipelines.dropped_bytes)
	                                     : std::string {});
	SetCreatedCode(
	    std::make_unique<ShaderCodeStore>(ShaderCodeStore::PathFor(directory, title_id)));
	PipelineCacheLog("Created shader code: {} shaders in {}", m_created_code->Size(),
	                 Common::PathToString(ShaderCodeStore::PathFor(directory, title_id)));
	if (m_precompile_attempts->FailedCount() + m_precompile_attempts->RetriedCount() != 0) {
		PipelineCacheLog("Precompile attempts: {} builds quarantined, {} to be retried once",
		                 m_precompile_attempts->FailedCount(),
		                 m_precompile_attempts->RetriedCount());
	}
	const auto created = m_created->Loaded();
	PipelineCacheLog("Created compute shaders: {} in {}{}", created.records,
	                 Common::PathToString(m_created->Path()),
	                 created.rewritten ? fmt::format(" (rewritten, {} unreadable bytes dropped)",
	                                                 created.dropped_bytes)
	                                   : std::string {});
}

void PipelineCache::SetManifest(std::unique_ptr<ProgramManifest> manifest) {
	m_manifest                = std::move(manifest);
	m_program_cache->manifest = m_manifest.get();
}

void PipelineCache::SetCreatedShaders(std::unique_ptr<ProgramManifest> created) {
	ShaderSetCreatedComputeSink({});
	m_created = std::move(created);
	if (m_created == nullptr) {
		return;
	}
	// Runs on the creating guest thread; the manifest serializes its own state.
	ShaderSetCreatedComputeSink([created = m_created.get()](const ShaderCreatedCompute& shader) {
		ComputeProgramRecord record;
		record.shader_hash = shader.hash;
		record.code =
		    std::make_shared<const std::vector<uint32_t>>(shader.code.begin(), shader.code.end());
		record.user_data_count = shader.user_data_count;
		record.input           = ComputeProgramRecord::StaticInput(shader.input);
		(void)created->Add(record);
	});
}

namespace {

// A precompile build that has not finished after this long is abandoned: the driver may never
// finish it (seen for optimized builds of some large compute modules on NVIDIA).
constexpr auto PrecompileBuildBudget = std::chrono::seconds {300};
// The renderer waits this long at most for a running precompile build of a program it needs,
// then builds the pipeline itself.
constexpr auto PrecompileExpediteWait = std::chrono::seconds {30};

// The attempts key of a build's unoptimized replacement.
uint64_t UnoptimizedKey(uint64_t key) {
	return XXH3_64bits_withSeed(&key, sizeof(key), 0x756e6f7074ull);
}

} // namespace

void PipelineCache::SetCreatedCode(std::unique_ptr<ShaderCodeStore> store) {
	ShaderSetCreatedCodeSink({});
	m_created_code                = std::move(store);
	m_program_cache->created_code = m_created_code.get();
	if (m_created_code == nullptr) {
		return;
	}
	// Runs on the creating guest thread; the store serializes its own state.
	ShaderSetCreatedCodeSink([this](const ShaderCreatedCode& shader) {
		static constexpr std::array<const char*, 9> Names = {"CS",  "PS",  "GS",  "HS", "GSF",
		                                                     "HSF", "GSB", "HSB", "FS"};
		if (!m_created_code->Add(shader.binary_type, shader.code)) {
			return;
		}
		{
			std::lock_guard lock(m_list_mutex);
			if (m_list_open && m_list_resolver != nullptr) {
				QueueResolved(m_list_resolver->OnCode(shader.code));
			}
		}
		// A census of the created shaders, at most every 10 s.
		static std::atomic<int64_t> next_report {0};
		const auto                  now  = std::chrono::duration_cast<std::chrono::seconds>(
		                                       std::chrono::steady_clock::now().time_since_epoch())
		                                       .count();
		auto                        next = next_report.load(std::memory_order_relaxed);
		if (now >= next && next_report.compare_exchange_strong(next, now + 10)) {
			const auto  counts = m_created_code->CountByType();
			std::string line   = "Created shaders:";
			for (size_t type = 0; type < Names.size(); type++) {
				if (counts[type] != 0) line += fmt::format(" {} {}", Names[type], counts[type]);
			}
			PipelineCacheLog("{}", line);
		}
	});
}

bool PipelineCache::EnqueuePrecompileJob(uint64_t key, std::string label, PrecompileBuild build) {
	auto* attempts = m_precompile_attempts.get();
	if (attempts != nullptr && attempts->Failed(key)) {
		// An optimized build that timed out is replaced by the unoptimized build, which the
		// renderer then also uses (GetComputePipeline).
		if (!attempts->TimedOut(key) || attempts->Failed(UnoptimizedKey(key))) {
			return false;
		}
		EnqueuePrecompileJob(UnoptimizedKey(key), label + " (unoptimized)", std::move(build),
		                     PipelineBuildMode::Unoptimized);
		return true;
	}
	EnqueuePrecompileJob(key, std::move(label), std::move(build), PipelineBuildMode::Precompile);
	return true;
}

void PipelineCache::EnqueuePrecompileJob(uint64_t key, std::string label, PrecompileBuild build,
                                         PipelineBuildMode mode) {
	auto& progress = m_precompile_progress;
	// Counted before the job can run; a key the queue rejects is uncounted again below.
	progress.Queued();
	const bool queued = m_precompile_queue->Enqueue(key, [this, key, label, build, mode] {
		// A build that ends the process or never finishes leaves an unfinished attempt, so
		// that later runs skip it (PrecompileAttempts).
		auto*      attempts = m_precompile_attempts.get();
		const auto settled  = std::make_shared<std::atomic<bool>>(false);
		const auto start    = std::chrono::steady_clock::now();
		if (attempts != nullptr) {
			attempts->Begin(key);
		}
		{
			std::lock_guard lock(m_precompile_running_mutex);
			m_precompile_running[key] = {
			    .start = start, .settled = settled, .label = label, .build = build, .mode = mode};
		}
		const auto outcome = build(mode);
		const auto ms      = std::chrono::duration_cast<std::chrono::milliseconds>(
		                         std::chrono::steady_clock::now() - start)
		                         .count();
		{
			std::lock_guard lock(m_precompile_running_mutex);
			m_precompile_running.erase(key);
		}
		if (settled->exchange(true)) {
			// Abandoned by CheckPrecompileTimeouts; it stays quarantined.
			m_precompile_abandoned.fetch_sub(1, std::memory_order_relaxed);
			PipelineCacheLog("Precompile: {} finished after {} s, past the time budget", label,
			                 ms / 1000);
			return;
		}
		if (attempts != nullptr) {
			attempts->End(key);
		}
		if (outcome == PrecompileOutcome::Compiled) {
			m_driver_cache_dirty.store(true, std::memory_order_relaxed);
			m_precompile_dirty.store(true, std::memory_order_relaxed);
		}
		if (m_precompile_progress.Finish(outcome, static_cast<uint64_t>(ms)) &&
		    !m_precompile_only) {
			LogPrecompileStats("done");
		}
	});
	if (!queued) {
		progress.Rejected();
	}
}

void PipelineCache::CheckPrecompileTimeouts() {
	const auto                                          now = std::chrono::steady_clock::now();
	std::vector<std::pair<uint64_t, RunningPrecompile>> expired;
	{
		std::lock_guard lock(m_precompile_running_mutex);
		for (const auto& [key, running]: m_precompile_running) {
			if (now - running.start >= PrecompileBuildBudget && !running.settled->exchange(true)) {
				expired.emplace_back(key, running);
			}
		}
	}
	for (auto& [key, running]: expired) {
		m_precompile_abandoned.fetch_add(1, std::memory_order_relaxed);
		if (m_precompile_attempts != nullptr) {
			m_precompile_attempts->TimeOut(key);
		}
		const bool replace = running.mode == PipelineBuildMode::Precompile &&
		                     (m_precompile_attempts == nullptr ||
		                      !m_precompile_attempts->Failed(UnoptimizedKey(key)));
		PipelineCacheLog(
		    "Precompile: {} has not finished after {} s; abandoned and quarantined{}",
		    running.label,
		    std::chrono::duration_cast<std::chrono::seconds>(now - running.start).count(),
		    replace ? ", building it unoptimized" : "");
		// Queued before the abandoned build is counted as finished, so the run stays open.
		if (replace) {
			EnqueuePrecompileJob(UnoptimizedKey(key), running.label + " (unoptimized)",
			                     running.build, PipelineBuildMode::Unoptimized);
		}
		if (m_precompile_progress.Finish(PrecompileOutcome::TimedOut, 0) && !m_precompile_only) {
			LogPrecompileStats("done");
		}
	}
}

void PipelineCache::EnqueuePrecompile(std::shared_ptr<const ComputeProgramRecord> record,
                                      bool                                        code_only) {
	const auto key = record->Key();
	auto       label =
	    fmt::format("{} CS {:016x}", code_only ? "created" : "recorded", record->shader_hash);
	(void)EnqueuePrecompileJob(
	    key, std::move(label),
	    [this, record = std::move(record), code_only](PipelineBuildMode mode) {
		    return PrecompileComputeProgram(m_graphics, m_driver_cache, *record, nullptr, code_only,
		                                    mode);
	    });
}

void PipelineCache::EnqueuePrecompile(
    std::shared_ptr<const GraphicsPipelineRecord>           record,
    std::vector<std::shared_ptr<const GraphicsStageRecord>> stages) {
	const auto key = record->Key();
	auto       label =
	    fmt::format("pipeline {:016x} (VS {:016x} PS {:016x})", key, stages.front()->shader_hash,
	                record->pixel_stage != 0 ? stages.back()->shader_hash : 0);
	(void)EnqueuePrecompileJob(
	    key, std::move(label),
	    [this, record = std::move(record), stages = std::move(stages)](PipelineBuildMode mode) {
		    return PrecompileGraphicsPipeline(*record, stages, mode);
	    });
}

void PipelineCache::QueueResolved(PipelineListResolver::Resolved resolved) {
	for (size_t i = 0; i < resolved.compute.size(); i++) {
		m_list_keys.insert(resolved.compute[i]->Key());
		EnqueuePrecompile(std::move(resolved.compute[i]), resolved.compute_code_only[i]);
	}
	for (auto& pipeline: resolved.pipelines) {
		m_list_keys.insert(pipeline.record->Key());
		auto stages = std::move(pipeline.vertex_stages);
		if (pipeline.pixel_stage != nullptr) {
			stages.push_back(std::move(pipeline.pixel_stage));
		}
		EnqueuePrecompile(std::move(pipeline.record), std::move(stages));
	}
}

void PipelineCache::ImportPipelineList() {
	const auto path = Config::ImportPipelineListPath();
	if (path.empty()) {
		return;
	}
	PipelineList::ReadStats stats;
	auto                    contents = PipelineList::Read(path, m_title_id, &stats);
	if (!stats.valid || !stats.title_matches) {
		PipelineCacheLog("Pipeline list: {} is {}; ignored", Common::PathToString(path),
		                 !stats.valid ? "not a pipeline list" : "the list of " + contents.title_id);
		return;
	}
	const auto entries =
	    contents.compute.size() + contents.stages.size() + contents.pipelines.size();
	m_list_resolver = std::make_unique<PipelineListResolver>(std::move(contents));
	// Code this emulator has: the shaders the guest created and the recorded ones, then the
	// dumped originals (--graphics-debug-dump) for what is still missing.
	const auto local = [this](const ShaderCodeId& id) -> ShaderCode {
		if (auto code = m_created_code != nullptr ? m_created_code->Find(id) : nullptr) return code;
		if (auto code =
		        m_pipeline_manifest != nullptr ? m_pipeline_manifest->FindCode(id) : nullptr)
			return code;
		return nullptr;
	};
	std::unordered_map<ShaderCodeId, ShaderCode, ShaderCodeIdHash> compute_code;
	for (const auto* manifest: {m_manifest.get(), m_created.get()}) {
		if (manifest == nullptr) continue;
		for (const auto& record: manifest->Records()) {
			compute_code.try_emplace(ShaderCodeId::Of(*record->code), record->code);
		}
	}
	auto   resolved   = m_list_resolver->Resolve([&](const ShaderCodeId& id) -> ShaderCode {
		if (auto code = local(id)) return code;
		const auto found = compute_code.find(id);
		return found != compute_code.end() ? found->second : nullptr;
	});
	size_t from_dumps = 0;
	if (m_list_resolver->Pending()) {
		// File names carry the program hash, which is the code's for all but merged stages:
		// read every dump once and offer its code.
		std::error_code error;
		const auto      folder = Config::GetShaderLogFolder() / "original";
		for (const auto& file: std::filesystem::directory_iterator(folder, error)) {
			if (!file.is_regular_file(error) || file.path().extension() != ".bin") continue;
			const auto size = file.file_size(error);
			if (error || size == 0 || size % sizeof(uint32_t) != 0 || size > (64u << 20u)) continue;
			std::vector<uint32_t> code(size / sizeof(uint32_t));
			std::ifstream         in(file.path(), std::ios::binary);
			if (!in.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size)))
				continue;
			auto more = m_list_resolver->OnCode(code);
			from_dumps += more.compute.size() + more.pipelines.size();
			resolved.compute.insert(resolved.compute.end(), more.compute.begin(),
			                        more.compute.end());
			resolved.compute_code_only.insert(resolved.compute_code_only.end(),
			                                  more.compute_code_only.begin(),
			                                  more.compute_code_only.end());
			std::ranges::move(more.pipelines, std::back_inserter(resolved.pipelines));
			if (!m_list_resolver->Pending()) break;
		}
	}
	const auto counts = m_list_resolver->GetCounts();
	PipelineCacheLog("Pipeline list: {} entries in {}{}: compute {} of {} resolved, graphics "
	                 "pipelines {} of {} ({} without their stages), {} from dumped shaders; {} "
	                 "shaders unknown so far",
	                 entries, Common::PathToString(path),
	                 stats.dropped_bytes != 0
	                     ? fmt::format(" ({} damaged bytes dropped)", stats.dropped_bytes)
	                     : std::string {},
	                 counts.compute_resolved, counts.compute, counts.pipelines_resolved,
	                 counts.pipelines, counts.pipelines_broken, from_dumps, counts.shaders_pending);
	QueueResolved(std::move(resolved));
}

void PipelineCache::QueuePrecompile(uint32_t thread_count) {
	EXIT_IF(thread_count == 0);
	if (m_precompile_queue == nullptr) {
		m_precompile_queue = std::make_unique<ProgramPrecompileQueue>(thread_count);
	}
	auto&           progress = m_precompile_progress;
	std::lock_guard lock(m_list_mutex);
	progress.BeginQueueing();
	// Recorded programs and pipelines first, in the order they were first needed.
	for (const auto& record: m_manifest->Records()) {
		EnqueuePrecompile(record, false);
	}
	const auto recorded_compute = progress.Get().queued;
	const auto recorded_pipelines =
	    m_pipeline_manifest != nullptr
	        ? m_pipeline_manifest->Pipelines()
	        : std::vector<std::shared_ptr<const GraphicsPipelineRecord>> {};
	for (const auto& pipeline: recorded_pipelines) {
		std::vector<std::shared_ptr<const GraphicsStageRecord>> stages;
		for (const auto key: pipeline->VertexStages())
			stages.push_back(m_pipeline_manifest->FindStage(key));
		if (pipeline->pixel_stage != 0)
			stages.push_back(m_pipeline_manifest->FindStage(pipeline->pixel_stage));
		if (std::ranges::find(stages, nullptr) != stages.end()) {
			continue;
		}
		EnqueuePrecompile(pipeline, std::move(stages));
	}
	const auto recorded = progress.Get().queued;
	ImportPipelineList();
	const auto listed = progress.Get().queued - recorded;
	// A created shader no session has dispatched gets the module of its code-only
	// specialization, which the first dispatches of most such shaders use.
	if (m_created != nullptr) {
		for (const auto& record: m_created->Records()) {
			if (!m_manifest->HasShader(record->shader_hash)) {
				EnqueuePrecompile(record, true);
			}
		}
	}
	const auto queued = progress.Get().queued;
	PipelineCacheLog(
	    "Precompile: {} recorded compute programs, {} recorded graphics pipelines, {} from the "
	    "pipeline list and {} created compute shaders queued on {} threads ({} quarantined "
	    "builds skipped or rebuilt unoptimized, {} retried once)",
	    recorded_compute, recorded - recorded_compute, listed, queued - recorded - listed,
	    thread_count, m_precompile_attempts != nullptr ? m_precompile_attempts->FailedCount() : 0,
	    m_precompile_attempts != nullptr ? m_precompile_attempts->RetriedCount() : 0);
	// Entries of the list whose shaders the guest has not created stay queueable while it runs.
	m_list_open = !m_precompile_only && m_list_resolver != nullptr && m_list_resolver->Pending();
	if (!m_list_open && progress.EndQueueing() && !m_precompile_only) {
		LogPrecompileStats("done");
	}
}

void PipelineCache::LogPrecompileStats(const char* when) {
	const auto counts = m_precompile_progress.Get();
	PipelineCacheLog(
	    "Precompile {}: {} pipelines already in the driver cache, {} compiled ({} s of "
	    "driver time), {} skipped, {} timed out",
	    when, counts.cached, counts.compiled, counts.compile_ms / 1000u, counts.skipped,
	    counts.timed_out);
}

void PipelineCache::StartPrecompile(uint32_t thread_count) {
	if (m_manifest == nullptr || m_driver_cache == nullptr) {
		PipelineCacheLog("Precompile: no driver pipeline cache for this title; nothing to build");
		return;
	}
	QueuePrecompile(thread_count);
}

PipelineCache::PrecompileResult PipelineCache::PrecompileAll(uint32_t thread_count,
                                                             const std::function<bool()>& pump) {
	m_precompile_only = true;
	StartPrecompile(thread_count);
	if (m_precompile_queue == nullptr) {
		return Save() ? PrecompileResult::Done : PrecompileResult::SaveFailed;
	}
	auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds {30};
	bool interrupted = false;
	while (!m_precompile_progress.Complete()) {
		if (pump && !pump()) {
			interrupted = true;
			break;
		}
		CheckPrecompileTimeouts();
		if (std::chrono::steady_clock::now() >= next_report) {
			next_report += std::chrono::seconds {30};
			const auto counts = m_precompile_progress.Get();
			PipelineCacheLog("Precompile: {} builds left", counts.queued - counts.finished);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds {250});
	}
	if (interrupted) {
		PipelineCacheLog("Precompile: interrupted (window closed)");
	}
	LogPrecompileStats(interrupted ? "stopped" : "done");
	const bool joinable = !interrupted && m_precompile_abandoned.load() == 0;
	if (joinable) {
		m_precompile_queue->Stop();
		m_precompile_queue.reset();
		ReleasePrecompiledStages();
		return Save() ? PrecompileResult::Done : PrecompileResult::SaveFailed;
	}
	// Builds still run (abandoned, or interrupted): their threads are never joined. The driver
	// cache is written without waiting for them, at most for a bounded time in case the driver
	// holds the cache.
	auto saved   = std::make_shared<std::atomic<int>>(0);
	auto pending = std::make_shared<std::atomic<bool>>(true);
	std::thread([this, saved, pending] {
		saved->store(Save() ? 1 : 2);
		pending->store(false);
	}).detach();
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds {120};
	while (pending->load() && std::chrono::steady_clock::now() < deadline) {
		if (pump) (void)pump();
		std::this_thread::sleep_for(std::chrono::milliseconds {250});
	}
	if (pending->load()) {
		PipelineCacheLog("Vulkan pipeline cache: save did not finish within 120 s");
	}
	if (interrupted) {
		return PrecompileResult::Interrupted;
	}
	return saved->load() == 1 ? PrecompileResult::Done : PrecompileResult::SaveFailed;
}

bool PipelineCache::ExportPipelineList(const std::filesystem::path& path) {
	if (m_manifest == nullptr || m_pipeline_manifest == nullptr) {
		PipelineCacheLog("Pipeline list: no manifests for this title; nothing to export");
		return false;
	}
	const auto contents = PipelineList::FromManifests(m_title_id, m_manifest.get(), m_created.get(),
	                                                  m_pipeline_manifest.get());
	if (!PipelineList::Write(path, contents)) {
		PipelineCacheLog("Pipeline list: failed to write {}", Common::PathToString(path));
		return false;
	}
	size_t code_only = 0;
	for (const auto& entry: contents.compute) {
		code_only += entry.code_only ? 1u : 0u;
	}
	std::error_code error;
	PipelineCacheLog(
	    "Pipeline list: exported {} compute programs ({} created code-only), {} graphics "
	    "pipelines and {} stages to {} ({} KiB, no shader code)",
	    contents.compute.size(), code_only, contents.pipelines.size(), contents.stages.size(),
	    Common::PathToString(path), std::filesystem::file_size(path, error) / 1024u);
	return true;
}

void PipelineCache::LogCoverage() {
	auto& programs = *m_program_cache;
	if (m_graphics_from_cache + m_graphics_compiled != 0) {
		PipelineCacheLog("Graphics pipelines this session: {} from the driver cache, {} compiled; "
		                 "{} of {} recorded were predicted (by earlier sessions or the list)",
		                 m_graphics_from_cache, m_graphics_compiled, m_graphics_predicted,
		                 m_graphics_recorded);
	}
	if (programs.compute_recorded != 0) {
		PipelineCacheLog("Compute programs this session: {} of {} recorded were predicted",
		                 programs.compute_predicted, programs.compute_recorded);
	}
	if (programs.created_code != nullptr && !programs.census_seen.empty()) {
		std::string line = "Graphics shaders used this session (created by the guest before use):";
		const std::array<std::pair<ShaderType, const char*>, 6> stages = {
		    {{ShaderType::Vertex, "VS"},
		     {ShaderType::Pixel, "PS"},
		     {ShaderType::Mesh, "GS"},
		     {ShaderType::Local, "LS"},
		     {ShaderType::TessellationControl, "HS"},
		     {ShaderType::TessellationEvaluation, "DS"}}};
		for (const auto& [stage, name]: stages) {
			const auto& counts = programs.census[static_cast<size_t>(stage)];
			if (counts[0] != 0) line += fmt::format(" {} {} ({})", name, counts[0], counts[1]);
		}
		PipelineCacheLog("{}", line);
	}
	if (m_list_resolver != nullptr) {
		const auto counts = m_list_resolver->GetCounts();
		PipelineCacheLog("Pipeline list: compute {} of {} and graphics pipelines {} of {} resolved "
		                 "this session",
		                 counts.compute_resolved, counts.compute, counts.pipelines_resolved,
		                 counts.pipelines);
	}
}

void PipelineCache::CheckpointDriverCache() {
	// A driver cache without a file (tests) is not persisted.
	if (m_driver_cache == nullptr || m_driver_cache_path.empty()) {
		return;
	}
	m_driver_cache_dirty.store(true, std::memory_order_relaxed);
	const auto now = std::chrono::steady_clock::now();
	if (now < m_next_driver_cache_save || m_driver_cache_writer.Busy()) {
		return;
	}
	// Snapshotting and writing a large cache takes seconds; pipeline creation continues
	// meanwhile, and pipelines created during the snapshot mark the cache dirty again.
	m_next_driver_cache_save = now + DriverCacheSaveInterval;
	m_driver_cache_dirty.store(false, std::memory_order_relaxed);
	m_driver_cache_writer.TryStart([this] {
		if (!WriteDriverCache()) {
			m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		}
	});
}

bool PipelineCache::Save() {
	if (m_compute_from_cache + m_compute_compiled != 0) {
		PipelineCacheLog("Compute pipelines this session: {} from the driver cache, {} compiled",
		                 m_compute_from_cache, m_compute_compiled);
	}
	LogCoverage();
	m_driver_cache_writer.Wait();
	if (m_driver_cache == nullptr || m_driver_cache_path.empty() ||
	    !m_driver_cache_dirty.exchange(false)) {
		return true;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;
	if (!WriteDriverCache()) {
		m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		return false;
	}
	return true;
}

bool PipelineCache::WriteDriverCache() {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	return true;
}

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state,
                                        uint32_t indirect_draw) {
	const auto& context           = command.GetRegisters();
	const auto& user_config       = command.GetUserConfig();
	const auto& sh                = context.GetShaderRegisters();
	const auto& vertex_regs       = command.GetShaders().GetVs();
	const auto& pixel_regs        = command.GetShaders().GetPs();
	auto&       vertex_info       = state.vertex_info;
	auto&       pixel_info        = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	const auto  write_mask          = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    color_output_mask   = object_primitive_id ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			color_output_mask |= 1u << slot;
		}
	}
	const auto& db = sh.db_shader_control;
	// Without color outputs, a pixel shader still runs for its depth or coverage side effects.
	const bool pixel_active =
	    pixel_regs.ps_regs.data_addr != 0 &&
	    (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	     db.shader_mask_export_enable || db.shader_dual_export_enable ||
	     db.shader_execute_on_noop || sh.AlphaToMask());
	state.ps_active = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_draw = indirect_draw;
	}
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	if (pixel_active) {
		if (!m_program_cache->Get(pixel_params, pixel_info, push_data_cursor)) {
			state.unavailable = true;
			return;
		}
		for (const auto& output: pixel_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		if (!m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor)) {
			state.unavailable = true;
			return;
		}
	}
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineCache::LookaheadEnabled() const {
	return m_program_cache->lookahead != nullptr;
}

void PipelineCache::BeginLookaheadRun(const void* stream) {
	m_program_cache->lookahead->BeginRun(stream);
}

void PipelineCache::PlanLookaheadDispatch(const void* stream, const void* command,
                                          const HW::ComputeShaderInfo& regs, uint32_t groups_x,
                                          uint32_t groups_y, uint32_t groups_z,
                                          bool thread_dimensions, uint64_t indirect_args_address) {
	const auto& cs = regs.cs_regs;
	// The inputs RenderExecutor::DispatchDirect/DispatchIndirect give the walk.
	LookaheadInputs inputs;
	inputs.indirect_args_address = indirect_args_address;
	if (indirect_args_address == 0) {
		if (groups_x == 0 || groups_y == 0 || groups_z == 0) return;
		inputs.workgroup_counts = {groups_x, groups_y, groups_z};
		if (thread_dimensions) {
			const std::array<uint32_t, 3> sizes {cs.num_thread_x, cs.num_thread_y, cs.num_thread_z};
			for (uint32_t axis = 0; axis < 3u; ++axis) {
				const auto size               = std::max(sizes[axis], 1u);
				const auto threads            = inputs.workgroup_counts[axis];
				inputs.workgroup_counts[axis] = threads / size + (threads % size != 0u);
			}
		}
	}
	if (cs.user_sgpr > HW::UserSgprInfo::SGPRS_MAX) return;
	m_program_cache->PlanLookahead(stream, command, ProgramCache::LookaheadSlot::Compute,
	                               cs.data_addr,
	                               std::span(regs.cs_user_sgpr.value).first(cs.user_sgpr), inputs);
}

void PipelineCache::PlanLookaheadDraw(const void* stream, const void* command,
                                      const HW::VertexShaderInfo& vs,
                                      const HW::PixelShaderInfo&  ps) {
	// GetGraphicsPrograms walks the pixel shader first.
	if (ps.ps_regs.data_addr != 0 && ps.ps_regs.rsrc2.user_sgpr <= HW::UserSgprInfo::SGPRS_MAX) {
		LookaheadInputs inputs;
		m_program_cache->PlanLookahead(
		    stream, command, ProgramCache::LookaheadSlot::Pixel, ps.ps_regs.data_addr,
		    std::span(ps.ps_user_sgpr.value).first(ps.ps_regs.rsrc2.user_sgpr), inputs);
	}
	if (vs.es_regs.data_addr != 0 && vs.gs_regs.rsrc2.user_sgpr <= HW::UserSgprInfo::SGPRS_MAX) {
		LookaheadInputs inputs;
		m_program_cache->PlanLookahead(
		    stream, command, ProgramCache::LookaheadSlot::Vertex, vs.es_regs.data_addr,
		    std::span(vs.gs_user_sgpr.value).first(vs.gs_regs.rsrc2.user_sgpr), inputs);
	}
}

void PipelineCache::KickLookahead() {
	auto& programs = *m_program_cache;
	programs.lookahead->Kick();
	const auto now = std::chrono::steady_clock::now();
	if (now < programs.next_lookahead_report) return;
	if (programs.next_lookahead_report != std::chrono::steady_clock::time_point {}) {
		using Outcome    = ResourceLookahead::Outcome;
		const auto stats = programs.lookahead->TakeStats();
		const auto count = [&](Outcome outcome) {
			return static_cast<unsigned long long>(stats.outcomes[static_cast<size_t>(outcome)]);
		};
		LOGF("Lookahead: %u helpers, %llu planned, adopted %llu, not planned %llu, changed %llu, "
		     "late %llu, failed %llu, dirty %llu, moved %llu, heap %llu (10 s)\n",
		     programs.lookahead->HelperCount(), static_cast<unsigned long long>(stats.planned),
		     count(Outcome::Adopted), count(Outcome::NotPlanned), count(Outcome::Changed),
		     count(Outcome::Late), count(Outcome::Failed), count(Outcome::Dirty),
		     count(Outcome::Moved), count(Outcome::Heap));
	} else {
		(void)programs.lookahead->TakeStats();
	}
	programs.next_lookahead_report = now + std::chrono::seconds(10);
}

void PipelineCache::SetLookaheadCommand(const void* stream, const void* command) {
	m_program_cache->lookahead_stream  = stream;
	m_program_cache->lookahead_command = command;
}

PipelineCache::Pipeline* PipelineCache::ResolvePendingBuild(Pipeline&          pipeline,
                                                            PendingDrawBlocker blocker) {
	const auto entry = m_pending_builds.find(&pipeline);
	EXIT_IF(entry == m_pending_builds.end());
	auto&      pending = entry->second;
	const auto waited  = std::chrono::steady_clock::now() - pending.queued;
	if (m_skip_pending_draws && PendingDrawMaySkip(blocker, waited, m_skip_window)) {
		pending.skipped_draws++;
		return nullptr;
	}
	if (!pending.wait_logged) {
		pending.wait_logged = true;
		LOGF("PipelineCache: waiting for the pipeline of VS %016" PRIx64 " PS %016" PRIx64
		     " (%s)\n",
		     pending.vs_hash, pending.ps_hash,
		     blocker == PendingDrawBlocker::None ? "skipped for too long"
		                                         : PendingDrawBlockerName(blocker));
	}
	m_compile_queue->Expedite(pending.ticket);
	InstallOptimizedPipelines();
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
	return &pipeline;
}

void PipelineCache::RecordGraphicsPipeline(const GraphicsPipelineKey& key,
                                           const DrawRenderState&     state) {
	GraphicsPipelineRecord record;
	const auto             vertex_stages = state.VertexStages();
	record.vertex_stage_count            = static_cast<uint32_t>(vertex_stages.size());
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		record.vertex_stages[i] = vertex_stages[i].stage.handle->record_key;
		if (record.vertex_stages[i] == 0) return;
	}
	if (state.ps_active) {
		record.pixel_stage = state.ps_input_info.stage.handle->record_key;
		if (record.pixel_stage == 0) return;
	}
	record.rendering      = key.rendering;
	record.vertex_input   = key.vertex_input;
	record.static_params  = key.static_params;
	const auto record_key = record.Key();
	m_graphics_recorded++;
	bool predicted = !m_pipeline_manifest->AddPipeline(record);
	if (!predicted) {
		std::lock_guard lock(m_list_mutex);
		predicted = m_list_keys.contains(record_key);
	}
	if (predicted) {
		m_graphics_predicted++;
	}
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state) {
	return *FindGraphicsPipeline(command, state, PendingDrawBlocker::Disabled);
}

PipelineCache::Pipeline* PipelineCache::FindGraphicsPipeline(const CommandBuffer&   command,
                                                             const DrawRenderState& state,
                                                             PendingDrawBlocker     blocker) {
	const auto* colors         = state.color_info;
	const auto& depth          = state.depth_info;
	const auto  vertex_info    = state.VertexStages();
	const auto& ps_input_info  = state.ps_input_info;
	const auto& vs_input_info  = vertex_info.front();
	const auto* vertex_program = vs_input_info.stage.handle;
	const auto* pixel_program  = state.ps_input_info.stage.handle;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(vertex_program == nullptr);
	const bool ps_active = state.ps_active;
	EXIT_IF(ps_active && pixel_program == nullptr);
	const auto color_count = state.color_count;

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program->id;
	const auto ps_id = ps_active ? pixel_program->id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		key.vertex_shader_ids[i] = vertex_info[i].stage.handle->id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_active) {
			alpha_source = ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info.ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = state.topology;
	static_params.primitive_restart_enable = state.primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info.ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	InstallOptimizedPipelines();
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		return found.pipeline != nullptr ? &found : ResolvePendingBuild(found, blocker);
	}
	if (m_pipeline_manifest != nullptr) {
		RecordGraphicsPipeline(key, state);
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program->module), ps_id,
		     ps_active ? static_cast<void*>(pixel_program->module) : nullptr);
	}

	// A draw that may be skipped leaves the whole build to the background queue.
	const auto build_mode = m_skip_pending_draws && PendingDrawMaySkip(blocker, {}, m_skip_window)
	                            ? PipelineBuildMode::Background
	                        : m_defer_optimization ? PipelineBuildMode::UnoptimizedFirst
	                                               : PipelineBuildMode::Optimized;
	auto       cached     = std::make_unique<Pipeline>();
	bool       from_driver_cache = false;
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	auto job = [&] {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		return CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, state,
		                              static_params, m_driver_cache, build_mode,
		                              &from_driver_cache);
	}();
	(from_driver_cache ? m_graphics_from_cache : m_graphics_compiled)++;
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr && !job);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	auto& created = *iter->second;
	if (created.pipeline == nullptr) {
		const PendingBuild pending {
		    .ticket        = m_compile_queue->Enqueue(std::move(job)),
		    .queued        = std::chrono::steady_clock::now(),
		    .vs_hash       = vs_input_info.stage.program->shader_hash,
		    .ps_hash       = ps_active ? ps_input_info.stage.program->shader_hash : 0,
		    .skipped_draws = 1,
		};
		m_pending_optimized.emplace(pending.ticket, &created);
		m_pending_builds.emplace(&created, pending);
		LOGF("PipelineCache: skipping draws of VS %016" PRIx64 " PS %016" PRIx64
		     " until its pipeline is ready (built in the background; the draws only change the "
		     "picture; they wait after %" PRId64 " ms)\n",
		     pending.vs_hash, pending.ps_hash,
		     static_cast<int64_t>(
		         std::chrono::duration_cast<std::chrono::milliseconds>(m_skip_window).count()));
		return nullptr;
	}
	DeferOptimization(created, std::move(job));
	CheckpointDriverCache();

	return &created;
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(input_info.stage.handle == nullptr);
	const auto& compute_program = *input_info.stage.handle;

	InstallOptimizedPipelines();
	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	// A recorded program whose precompile has not finished: build it here, or wait for the
	// worker building it, rather than compiling it twice; the build below then hits the driver
	// cache.
	if (m_precompile_queue != nullptr && compute_program.record_key != 0 &&
	    !m_precompile_queue->Expedite(compute_program.record_key, PrecompileExpediteWait)) {
		PipelineCacheLog("PipelineCache: the precompile of CS program {:016x} still runs after "
		                 "{} s; building it here",
		                 compute_program.record_key, PrecompileExpediteWait.count());
	}
	// A program whose optimized build ran past the precompile budget is never built optimized:
	// the driver may not finish that build at all.
	const bool optimized_hangs   = compute_program.record_key != 0 &&
	                               m_precompile_attempts != nullptr &&
	                               m_precompile_attempts->TimedOut(compute_program.record_key);
	auto cached            = std::make_unique<Pipeline>();
	bool from_driver_cache = false;
	auto job               = [&] {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		return CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                              m_driver_cache,
		                              optimized_hangs        ? PipelineBuildMode::Unoptimized
		                              : m_defer_optimization ? PipelineBuildMode::UnoptimizedFirst
		                                                     : PipelineBuildMode::Optimized,
		                              &from_driver_cache);
	}();
	(from_driver_cache ? m_compute_from_cache : m_compute_compiled)++;

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	DeferOptimization(*iter->second, std::move(job));
	CheckpointDriverCache();

	return *iter->second;
}
} // namespace Libs::Graphics
