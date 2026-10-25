#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pendingDrawPolicy.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCacheWriter.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineState.h"
#include "graphics/host_gpu/renderer/pipeline/programManifest.h"
#include "graphics/host_gpu/renderer/pipeline/programPrecompileQueue.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandBuffer;
class CommandScheduler;

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;
	// ComputeProgramRecord::Key of a recorded compute program, GraphicsStageRecord::Key of a
	// recorded graphics stage, else 0.
	uint64_t record_key = 0;

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
	// Optimized on the calling thread unless the driver cache holds the pipeline; fills the
	// driver cache ahead of the pipeline's use (see PrecompileComputeProgram).
	Precompile,
	// Unoptimized on the calling thread and never optimized: for a pipeline whose optimized
	// build did not finish within the precompile time budget (PrecompileAttempts::TimedOut).
	// Built the way UnoptimizedFirst builds, so that a precompile of it fills the driver's own
	// disk cache for that build.
	Unoptimized,
};

// A draw's topology, shader stages (from PipelineCache::GetGraphicsPrograms) and targets.
struct DrawRenderState {
	RenderDepthInfo                      depth_info;
	RenderColorInfo                      color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t                             color_count                              = 0;
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask                                 = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable                 = false;
	bool                                 ps_active                                = false;
	// A stage's resources could not be specialized for this draw's guest data; the draw must be
	// skipped (see ResourceSnapshot::unavailable).
	bool                                 unavailable                              = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets                indirect_offsets;
	bool                                 indirect_draw = false;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
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
	// Waits for a running background save, then writes the driver cache if it changed. Returns
	// false when a write failed.
	bool Save();
	// Queues a driver-cache build of every compute program and graphics pipeline the title's
	// manifests recorded in earlier sessions or the imported list names, and of every created
	// compute shader without a recorded program (--precompile-pipelines). Needs the renderer's
	// bindless set layout. Recorded and listed builds come first, in first-seen order; the
	// builds run on `thread_count` low-priority threads while the session boots and on
	// `steady_thread_count` once the game renders or no recorded build is left (see
	// UpdatePrecompileThreads); zero keeps `thread_count`.
	void StartPrecompile(uint32_t thread_count, uint32_t steady_thread_count = 0);
	// The process is about to end on purpose: running precompile builds are recorded as
	// interrupted (PrecompileAttempts::Interrupt), not as builds that ended the process, and
	// builds that have not reached their driver call stop there.
	void EndPrecompileForExit();
	enum class PrecompileResult : uint8_t {
		Done,
		// The driver cache could not be written.
		SaveFailed,
		// The window was closed; the builds that were running are retried by the next run.
		Interrupted,
	};
	// --precompile-only: builds every recorded program's and pipeline's pipeline on
	// `thread_count` threads and waits for all of them; a build that runs past the time budget
	// is abandoned (and quarantined). Logs the outcome counts and saves the driver cache.
	// `pump` runs about four times a second on the calling thread (window messages); it returns
	// false to stop. Builds that are still running when this returns may never finish: the
	// caller ends the process without joining them.
	PrecompileResult PrecompileAll(uint32_t thread_count, const std::function<bool()>& pump);
	// --export-pipeline-list: writes the title's manifests as a shareable list (no shader code).
	bool ExportPipelineList(const std::filesystem::path& path);
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

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// An empty program skips the dispatch.
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state);
	// With --skip-pending-draws: like GetGraphicsPipeline, but returns null when the draw is to
	// be skipped, which only happens when `blocker` is None. A missing pipeline is then built on
	// the background queue, and the draws using it are skipped until it is ready, for at most
	// PendingDrawSkipWindow; afterwards, and for any blocked draw, the lookup waits for it.
	Pipeline* FindGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                               PendingDrawBlocker blocker);
	[[nodiscard]] bool SkipPendingDrawsEnabled() const { return m_skip_pending_draws; }
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);

	// Resource walks of compute dispatches the GPU thread has not reached (ResourceLookahead).
	// A command stream (one per command processor) plans the dispatches of a run of commands
	// that are all written, then kicks the helpers; the dispatch names its command before it
	// asks for its program.
	[[nodiscard]] bool LookaheadEnabled() const;
	void               BeginLookaheadRun(const void* stream);
	void PlanLookaheadDispatch(const void* stream, const void* command,
	                           const HW::ComputeShaderInfo& regs, uint32_t groups_x,
	                           uint32_t groups_y, uint32_t groups_z, bool thread_dimensions,
	                           uint64_t indirect_args_address);
	void PlanLookaheadDraw(const void* stream, const void* command, const HW::VertexShaderInfo& vs,
	                       const HW::PixelShaderInfo& ps);
	void KickLookahead();
	// The command whose programs are asked for next (null after it).
	void SetLookaheadCommand(const void* stream, const void* command);

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
	// Compute programs of this title (_PipelineCache/<title>.programs); null without a driver
	// cache file.
	std::unique_ptr<ProgramManifest>        m_manifest;
	// Compute shaders the guest created (_PipelineCache/<title>.v4.created): code and static
	// input without a specialization. Precompile builds those the manifest has no program for
	// with their code-only specialization. Null without a driver cache file.
	std::unique_ptr<ProgramManifest> m_created;
	// Precompile builds that never finished; null without a driver cache file.
	std::unique_ptr<PrecompileAttempts>     m_precompile_attempts;
	// Graphics pipelines of this title and their stages (_PipelineCache/<title>.v5.pipelines),
	// and the code of every shader the guest created (<title>.v5.created); null without a driver
	// cache file.
	std::unique_ptr<PipelineManifest> m_pipeline_manifest;
	std::unique_ptr<ShaderCodeStore>  m_created_code;
	std::string                       m_title_id;
	// The imported pipeline list's entries whose shaders the guest has not created yet
	// (--import-pipeline-list); guarded by m_list_mutex, which also guards queueing from guest
	// threads against the queue's shutdown.
	std::mutex                            m_list_mutex;
	std::unique_ptr<PipelineListResolver> m_list_resolver;
	bool                                  m_list_open = false;
	// Keys the imported list predicts (compute program and graphics pipeline records).
	std::unordered_set<uint64_t>            m_list_keys;
	std::unique_ptr<ProgramPrecompileQueue> m_precompile_queue;
	PrecompileProgress                      m_precompile_progress;
	// Set by precompile builds that added to the driver cache; the GPU thread checkpoints it.
	std::atomic<bool> m_precompile_dirty {false};
	// A precompile build that runs: when it started, and whether it has finished or timed out
	// (whichever comes first settles it). A build stops before its next driver call once
	// `cancel` is set (it timed out, or the cache stops); a driver call itself cannot be
	// interrupted.
	using PrecompileBuild =
	    std::function<PrecompileOutcome(PipelineBuildMode, const std::atomic<bool>& cancel)>;
	struct RunningPrecompile {
		std::chrono::steady_clock::time_point start;
		std::shared_ptr<std::atomic<bool>>    settled;
		std::shared_ptr<std::atomic<bool>>    cancel;
		std::string                           label;
		PrecompileBuild                       build;
		PipelineBuildMode                     mode  = PipelineBuildMode::Precompile;
		uint64_t                              order = 0;
	};
	// Shared with the precompile jobs: an abandoned job may return after the cache is gone.
	struct PrecompileRuns {
		std::mutex                                      mutex;
		std::unordered_map<uint64_t, RunningPrecompile> running;
		// Abandoned builds still in their driver call.
		std::atomic<uint64_t> abandoned {0};
	};
	std::shared_ptr<PrecompileRuns>       m_precompile_runs = std::make_shared<PrecompileRuns>();
	std::chrono::steady_clock::time_point m_next_timeout_check {};
	// Precompile threads after the boot (StartPrecompile); 0 without a precompile.
	uint32_t m_precompile_steady_threads = 0;
	// When a recorded or listed build was last queued (steady clock ticks; queued from guest
	// threads too).
	std::atomic<int64_t> m_last_recorded_enqueue {0};
	// Guest flips seen by UpdatePrecompileThreads: the window it counts, and how many windows
	// in a row had the game rendering.
	uint64_t                                        m_flip_window_start_count = 0;
	std::chrono::steady_clock::time_point           m_flip_window_start {};
	uint32_t                                        m_rendering_windows = 0;
	bool                                            m_game_rendering    = false;
	std::chrono::steady_clock::time_point           m_next_coverage_report {};
	// --precompile-only logs its outcome itself.
	bool m_precompile_only = false;
	// Translated stages of graphics pipelines being precompiled, shared by the pipelines that
	// use them; released when the precompile stops.
	struct PrecompiledStage;
	std::mutex                                                      m_stage_memo_mutex;
	std::unordered_map<uint64_t, std::shared_ptr<PrecompiledStage>> m_stage_memo;
	// Graphics pipelines created this session: from the driver cache or compiled, and whether a
	// record (local or imported) predicted them.
	uint64_t m_graphics_from_cache = 0;
	uint64_t m_graphics_compiled   = 0;
	uint64_t m_graphics_predicted  = 0;
	uint64_t m_graphics_recorded   = 0;
	// Live compute pipelines the driver cache provided or compiled.
	uint64_t m_compute_from_cache = 0;
	uint64_t m_compute_compiled   = 0;

	void InitializeDriverCache();
	void CheckpointDriverCache();
	// Snapshots the driver cache and writes it to m_driver_cache_path through a temporary file.
	// Safe on any thread while the cache lives: the driver synchronizes the cache object.
	bool WriteDriverCache();
	void DeferOptimization(Pipeline& pipeline, PipelineCompileQueue::Job job);
	void InstallOptimizedPipelines();
	void SetManifest(std::unique_ptr<ProgramManifest> manifest);
	// Records every graphics stage permutation and pipeline from now on in `manifest`.
	void SetPipelineManifest(std::unique_ptr<PipelineManifest> manifest);
	// Records every compute shader the guest creates from now on in `created`.
	void SetCreatedShaders(std::unique_ptr<ProgramManifest> created);
	// Queues the manifest's programs and the created shaders without one, on a new queue unless
	// one exists.
	void QueuePrecompile(uint32_t thread_count);
	// `position` is the record's FirstSeenPosition: recorded and listed builds run in that order,
	// before the created shaders (code_only).
	void EnqueuePrecompile(std::shared_ptr<const ComputeProgramRecord> record, bool code_only,
	                       uint64_t position);
	void EnqueuePrecompile(std::shared_ptr<const GraphicsPipelineRecord>           record,
	                       std::vector<std::shared_ptr<const GraphicsStageRecord>> stages,
	                       uint64_t                                                position);
	// Queues a build under the attempts rules: quarantined keys are skipped, and a key whose
	// optimized build timed out is built unoptimized instead. Returns whether it was queued.
	bool EnqueuePrecompileJob(uint64_t key, std::string label, PrecompileBuild build,
	                          uint64_t order);
	void EnqueuePrecompileJob(uint64_t key, std::string label, PrecompileBuild build,
	                          PipelineBuildMode mode, uint64_t order);
	// Abandons the precompile builds that ran past the time budget.
	void CheckPrecompileTimeouts();
	// Boot threads while recorded or listed builds are left (or were queued less than a while
	// ago) and the game is not rendering steadily, else the steady thread count.
	void        UpdatePrecompileThreads();
	void        QueueResolved(PipelineListResolver::Resolved resolved);
	void        ImportPipelineList();
	void        SetCreatedCode(std::unique_ptr<ShaderCodeStore> store);
	void        ReleasePrecompiledStages();
	static bool TranslateRecordedStage(GraphicContext& graphics, const GraphicsStageRecord& record,
	                                   PrecompiledStage& out);
	PrecompileOutcome
	PrecompileGraphicsPipeline(const GraphicsPipelineRecord&                               record,
	                           std::span<const std::shared_ptr<const GraphicsStageRecord>> stages,
	                           PipelineBuildMode mode, const std::atomic<bool>* cancel = nullptr);
	void LogCoverage();
	// Adds a pipeline the cache is about to create to the pipeline manifest.
	void RecordGraphicsPipeline(const GraphicsPipelineKey& key, const DrawRenderState& state);

	void LogPrecompileStats(const char* when);
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
    const DrawRenderState& state, const PipelineStaticParameters& static_params,
    vk::PipelineCache driver_cache, PipelineBuildMode mode, bool* from_driver_cache = nullptr,
    const std::atomic<bool>* cancel = nullptr);
PipelineCompileQueue::Job
CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                       const ShaderComputeInputInfo& input_info, vk::ShaderModule compute_module,
                       vk::PipelineCache driver_cache, PipelineBuildMode mode,
                       bool*                    from_driver_cache = nullptr,
                       const std::atomic<bool>* cancel            = nullptr);

// Translates a recorded compute program as PipelineCache::GetComputeProgram would and builds
// its pipeline through `driver_cache`, so that the pipeline cache finds it there on first use.
// Thread-safe; touches no guest state. `spirv_hash` receives the hash of the module.
// With `code_only`, the record's specialization is ignored and the module is built for
// ShaderRecompiler::IR::CodeOnlySpecialization of the translated program (a created shader that
// no dispatch has specialized yet).
// `mode` is Precompile, or Unoptimized for a program whose optimized build timed out. Once
// `cancel` is set, the build stops before its next driver call (and reports TimedOut).
PrecompileOutcome PrecompileComputeProgram(GraphicContext& graphics, vk::PipelineCache driver_cache,
                                           const ComputeProgramRecord& record,
                                           uint64_t* spirv_hash = nullptr, bool code_only = false,
                                           PipelineBuildMode mode = PipelineBuildMode::Precompile,
                                           const std::atomic<bool>* cancel = nullptr);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
