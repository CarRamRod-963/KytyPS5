#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/pendingDrawPolicy.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCacheWriter.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;
class CommandScheduler;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 117);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// How CreatePipelineInternal builds a pipeline that the driver cache does not hold.
enum class PipelineBuildMode : uint8_t {
	// Optimized, on the calling thread.
	Optimized,
	// Unoptimized on the calling thread; the returned job builds the optimized pipeline.
	UnoptimizedFirst,
	// Nothing on the calling thread; the returned job builds the optimized pipeline.
	Background,
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Without a scheduler to retire replaced pipelines, every pipeline is built optimized on
	// first use.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              CommandScheduler* scheduler = nullptr);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();
	// Joins the background compiles. Their pipeline layouts reference the renderer's bindless
	// set layout, so the owner calls this before destroying it.
	void StopBackgroundCompiles();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;
		// A stage's resources could not be specialized for this draw's guest data; the draw
		// must be skipped (see ResourceSnapshot::unavailable).
		bool unavailable = false;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	// With --skip-pending-draws: like GetGraphicsPipeline, but returns null when the draw is to
	// be skipped, which only happens when `blocker` is None. A missing pipeline is then built on
	// the background queue, and the draws using it are skipped until it is ready, for at most
	// PendingDrawSkipWindow; afterwards, and for any blocked draw, the lookup waits for it.
	Pipeline* FindGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                               const RenderDepthInfo&                 depth,
	                               std::span<const ShaderVertexInputInfo> vertex_info,
	                               CommandBuffer&                         command,
	                               const ShaderPixelInputInfo*            ps_input_info,
	                               vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                               const GraphicsPrograms& programs, PendingDrawBlocker blocker);
	[[nodiscard]] bool SkipPendingDrawsEnabled() const { return m_skip_pending_draws; }
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	friend struct PipelineCacheTestAccess;
	struct ProgramCache;

	// A graphics pipeline whose first build runs on the background queue; its handle is null.
	struct PendingBuild {
		uint64_t                              ticket = 0;
		std::chrono::steady_clock::time_point queued;
		uint64_t                              vs_hash       = 0;
		uint64_t                              ps_hash       = 0;
		uint64_t                              skipped_draws = 0;
		bool                                  wait_logged   = false;
	};

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	CommandScheduler*             m_scheduler;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	// Set when pipelines were created since the last snapshot; a failed write sets it again.
	std::atomic<bool> m_driver_cache_dirty {false};
	// Writes the driver cache off the GPU thread; joined before the cache is destroyed.
	PipelineCacheWriter m_driver_cache_writer;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	// Optimized builds of pipelines that are in use unoptimized, by compile ticket. Pipelines are
	// never evicted, so the pointers stay valid.
	std::unique_ptr<PipelineCompileQueue>         m_compile_queue;
	std::unordered_map<uint64_t, Pipeline*>       m_pending_optimized;
	std::vector<PipelineCompileQueue::Completion> m_completed;
	// Missing pipelines are built unoptimized first (--async-pipelines).
	bool m_defer_optimization = false;
	// Draws that only change the picture skip pipelines that are still building
	// (--skip-pending-draws).
	bool                                              m_skip_pending_draws = false;
	std::chrono::steady_clock::duration               m_skip_window        = PendingDrawSkipWindow;
	std::unordered_map<const Pipeline*, PendingBuild> m_pending_builds;

	void InitializeDriverCache();
	void CheckpointDriverCache();
	// Snapshots the driver cache and writes it to m_driver_cache_path through a temporary file.
	// Safe on any thread while the cache lives: the driver synchronizes the cache object.
	bool WriteDriverCache();
	void DeferOptimization(Pipeline& pipeline, PipelineCompileQueue::Job job);
	void InstallOptimizedPipelines();
	// Returns the pipeline of a pending first build, or null when the draw may skip it.
	Pipeline* ResolvePendingBuild(Pipeline& pipeline, PendingDrawBlocker blocker);
};

inline uint64_t PipelineCompileHandle(vk::Pipeline pipeline) {
	return std::bit_cast<uint64_t>(static_cast<VkPipeline>(pipeline));
}

inline vk::Pipeline PipelineFromCompileHandle(uint64_t handle) {
	return std::bit_cast<VkPipeline>(handle);
}

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// Creates the pipeline layout and the pipeline as `mode` says. Unless the build is Optimized,
// the pipeline may be unoptimized or (Background) still null; the returned job then builds the
// optimized one (see PipelineCache::DeferOptimization).
PipelineCompileQueue::Job CreatePipelineInternal(
    GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
    const PipelineRenderingState& rendering, const PipelineVertexInputState& vertex_input,
    std::span<const ShaderVertexInputInfo> vertex_info, const ShaderPixelInputInfo* ps_input_info,
    const PipelineCache::GraphicsPrograms& programs, const PipelineStaticParameters& static_params,
    vk::PipelineCache driver_cache, PipelineBuildMode mode);
PipelineCompileQueue::Job
CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                       const ShaderComputeInputInfo& input_info, vk::ShaderModule compute_module,
                       vk::PipelineCache driver_cache, bool defer_optimization);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
