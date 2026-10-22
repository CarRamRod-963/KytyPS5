#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/barrierTracker.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTranslation.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/guestWriteSet.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t address = 0;
	uint32_t count   = 0;
	uint32_t stride  = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	DrawIndirectArgs indirect;
};

// An indirect draw packet whose arguments (and draw count) the GPU reads. The records have the
// Vulkan layouts of VkDrawIndirectCommand / VkDrawIndexedIndirectCommand.
struct IndirectDrawSource {
	uint64_t args_address = 0;
	uint32_t stride       = 0;
	// The draw count, or with a count address the most draws the count may select.
	uint32_t max_draw_count = 0;
	uint64_t count_address  = 0;
	bool     indexed        = false;
	// Indexed: the index buffer the arguments' first index starts in, and its size in elements.
	uint64_t index_base          = 0;
	uint32_t index_buffer_size   = 0;
	uint32_t index_type_and_size = 0;
	// The user-data slots of the vertex stage the packet writes its base vertex (or first
	// vertex) and first instance to, -1 for none.
	int32_t vertex_offset_slot   = -1;
	int32_t instance_offset_slot = -1;
	// Vertex user-data slots whose values earlier native draws left on the GPU (one bit each).
	uint32_t stale_slots = 0;
};

enum class DrawIndirectResult : uint8_t {
	// Recorded, or skipped for the state alone as the packet would be with any arguments.
	Done,
	// The draw needs its arguments on the host; nothing was recorded.
	ReadArguments,
	// Its vertex stage reads stale slots: nothing was recorded, settle them and draw again.
	SettleRegisters,
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	// Recording through the handle marks the buffer as holding work worth submitting. The
	// commands recorded through it may access any memory: pending guest dispatches are ordered
	// before them, and later dispatches after them.
	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Handle() for a helper pass that binds a pipeline of its own: BindComputePipeline and
	// BindGraphicsPipeline bind theirs again afterwards.
	[[nodiscard]] vk::CommandBuffer PipelineHandle() const;
	// For a pipeline barrier only. A barrier orders earlier submissions against the commands
	// recorded after it, so it keeps its meaning when the buffer is submitted later, and a buffer
	// holding nothing else is not worth a submission. Pending guest dispatches are ordered
	// before it.
	[[nodiscard]] vk::CommandBuffer BarrierHandle() const;
	// For commands that access no memory when they execute: binding pipelines, descriptors and
	// vertex/index buffers, push constants, dynamic state.
	[[nodiscard]] vk::CommandBuffer StateHandle() const;
	// Binds a compute pipeline unless it is still bound: guest dispatches often run one shader
	// many times in a row. A pipeline bound through PipelineHandle() is forgotten there.
	void BindComputePipeline(vk::Pipeline pipeline) const;
	// The same for graphics pipelines: guest draws often use one pipeline many times in a row.
	// A pipeline stays bound across render pass instances.
	void BindGraphicsPipeline(vk::Pipeline pipeline) const;
	// For one guest dispatch, recorded next with the returned handle. `writes`: the dispatch may
	// write memory through any path. `shared`: it uses emulator memory shared between dispatches
	// (the LDS backing, GDS). Records the barrier it needs against earlier commands.
	[[nodiscard]] vk::CommandBuffer DispatchHandle(bool writes, bool shared) const;
	// A guest request to finish all earlier work before later work starts (a partial flush,
	// a cache writeback or invalidation).
	void GuestFlush() const;
	// Orders pending guest dispatches before everything recorded next.
	void OrderDispatches() const;
	// Before a draw that reads its arguments on the GPU: makes every earlier write visible to
	// indirect-argument reads (barriers to all commands leave them out, see LaterAccess). Within
	// the render pass instances that follow such a barrier back to back only draws ran, so it
	// records nothing there; otherwise it ends the current instance for the barrier.
	void OrderIndirectReads() const;
	// Whether anything but barriers was recorded since the buffer began.
	[[nodiscard]] bool              HasCommands() const noexcept { return m_has_commands; }
	// Commands recorded through Handle() and DispatchHandle() since the buffer began.
	[[nodiscard]] uint32_t          CommandCount() const noexcept { return m_command_count; }
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;
	void RecordBarrier(BarrierTracker::Barrier barrier) const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	// Render pass instances begun, and the first one OrderIndirectReads covers.
	mutable uint64_t    m_render_passes        = 0;
	mutable uint64_t    m_indirect_reads_since = UINT64_MAX;
	mutable bool        m_has_commands = false;
	mutable uint32_t    m_command_count = 0;
	// The pipelines BindComputePipeline and BindGraphicsPipeline bound last, until
	// PipelineHandle() or the next Begin.
	mutable vk::Pipeline m_compute_pipeline  = nullptr;
	mutable vk::Pipeline m_graphics_pipeline = nullptr;
	// Kept across submissions: barriers order commands in submission order on the queue.
	mutable BarrierTracker m_barriers;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// A bound T# that cannot be placed is fatal. A T# read from guest memory on a shader's
	// behalf (a bindless key) may be stale or name a record the game never samples: as an
	// Optional source, such a T# returns a binding without an image instead.
	enum class TextureSource : uint8_t { Bound, Optional };
	void ValidateTextureUse(const ShaderRecompiler::IR::ImageResource& resource);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            TextureSource source = TextureSource::Bound);
	// Translates the bindless keys shaders flagged since the last call (once per frame).
	void ResolveBindlessImages(CommandBuffer& buffer);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	[[nodiscard]] DrawIndirectResult DrawIndirect(uint64_t submit_id, CommandBuffer& buffer,
	                                              const IndirectDrawSource& args);
	// Vertex shaders (code addresses) whose programs need indirect arguments on the host.
	std::unordered_set<uint64_t> m_indirect_readback_shaders;

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// ResolveTexture through a memo of (T#, resource traits) -> image, valid while the texture
	// cache's layout generation holds; the per-bind maintenance (touch, clear materialization)
	// still runs for a hit.
	void ResolveTextureMemo(const ShaderRecompiler::IR::ImageResource&   resource,
	                        const ShaderRecompiler::IR::DescriptorValue& value,
	                        TextureBinding&                              binding);
	struct TextureMemoKeyHash {
		size_t operator()(const TextureMemoKey& key) const noexcept {
			uint64_t hash = key.traits * 0x9E3779B97F4A7C15ull;
			for (const auto word: key.words) {
				hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
			}
			return static_cast<size_t>(hash ^ (hash >> 32u));
		}
	};
	struct TextureMemoEntry {
		ImageId                   image_id;
		TextureCache::ImageDesc   desc;
		TextureCache::LayoutStamp stamp;
		// Unique per insertion; the description of an entry never changes after it.
		uint64_t serial = 0;
		// The layout generation TextureMemoEntryValid last accepted the entry at. Every change
		// that check depends on (an image registered, unregistered or deleted, replaced or
		// expanded so that its bindings need a rebind, a stencil associated) notes a layout
		// change, so while the generation is unchanged the entry is still valid.
		uint64_t validated = UINT64_MAX;
	};
	// Whether a memo hit may return the entry's image (the checks a lookup would otherwise redo).
	[[nodiscard]] bool      TextureMemoEntryValid(const TextureMemoEntry& entry);
	static constexpr size_t TextureMemoFirstSweep = 8192;
	std::unordered_map<TextureMemoKey, TextureMemoEntry, TextureMemoKeyHash> m_texture_memo;
	size_t m_texture_memo_sweep_at = TextureMemoFirstSweep;
	uint64_t m_texture_memo_serial   = 0;
	// Advances with every erased entry: bindings keep entry pointers only within one epoch.
	uint64_t m_texture_memo_epoch = 1;
	// Render-target image lookups per color slot and for depth (TextureCache::ImageLookup).
	std::array<TextureCache::ImageLookup, RENDER_COLOR_ATTACHMENTS_MAX> m_color_target_lookups;
	TextureCache::ImageLookup                                           m_depth_target_lookup;
	// The registers (a byte copy) and other inputs a render-target lookup last decoded its
	// description from, and the lookup's generation after that FindImage. While all of them are
	// unchanged, the lookup's input is still the description they decode to, so a draw skips the
	// decode and the description compare.
	struct TargetLookupSource {
		static constexpr size_t MaxRegisterBytes = 160;

		std::array<std::byte, MaxRegisterBytes> registers {};
		uint64_t                                inputs     = 0;
		uint64_t                                generation = 0;
		bool                                    valid      = false;
		// Color targets: what the decode produces besides the description.
		uint32_t                        guest_mip_level   = 0;
		uint32_t                        guest_array_layer = 0;
		Prospero::ColorComponentMapping export_mapping;

		template <typename Registers>
		[[nodiscard]] bool Matches(const Registers& value, uint64_t other_inputs,
		                           const TextureCache::ImageLookup& lookup) const {
			static_assert(std::is_trivially_copyable_v<Registers> &&
			              sizeof(Registers) <= MaxRegisterBytes);
			return valid && inputs == other_inputs && generation == lookup.generation &&
			       std::memcmp(registers.data(), &value, sizeof(Registers)) == 0;
		}
		template <typename Registers>
		void Remember(const Registers& value, uint64_t other_inputs,
		              const TextureCache::ImageLookup& lookup) {
			static_assert(std::is_trivially_copyable_v<Registers> &&
			              sizeof(Registers) <= MaxRegisterBytes);
			std::memcpy(registers.data(), &value, sizeof(Registers));
			inputs     = other_inputs;
			generation = lookup.generation;
			valid      = true;
		}
	};
	std::array<TargetLookupSource, RENDER_COLOR_ATTACHMENTS_MAX> m_color_target_sources;
	TargetLookupSource                                           m_depth_target_source;
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	void                      PrepareBindlessHeaps(PreparedBindings& prepared);
	void                      EnsureBindlessPlaceholder();
	void                      CommitBindlessImages(const PreparedBindings& prepared,
	                                               vk::CommandBuffer       vk_buffer);
	void                      ValidateBindlessSlots();
	[[nodiscard]] uint32_t    ResolveBindlessKey(uint32_t                            heap,
	                                             const BindlessTranslation::Request& request);
	void                      FillBindlessHeap(uint32_t heap);
	void                      ResetFullBindlessTable();
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool TryConsumeComputeCopy(const ShaderComputeInputInfo& input, uint32_t group_x,
	                                         uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	GuestWriteSet                         m_commit_writes;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	// Bindless state: whether the placeholder slot (every unresolved key) holds the null
	// descriptor, the image resource each heap's T#s are decoded as, and the texture cache
	// deletions the slots were validated at.
	bool                                  m_bindless_placeholder_ready = false;
	// A heap did not fit in the translation table since the last reset.
	bool                                             m_bindless_heap_refused = false;
	std::vector<ShaderRecompiler::IR::ImageResource> m_bindless_resources;
	uint64_t                              m_bindless_deletion_epoch = 0;
	std::vector<uint32_t>                 m_bindless_feedback;
	// Slots keys resolved to, by the address of the record's T# block: a heap of new instance
	// records over the same materials fills its region from here instead of waiting for the
	// GPU to flag every key again.
	struct BindlessResolved {
		uint32_t layer = 0;
		uint32_t slot  = 0;
		ImageId  image;
	};
	std::unordered_map<uint64_t, std::vector<BindlessResolved>> m_bindless_resolved;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
