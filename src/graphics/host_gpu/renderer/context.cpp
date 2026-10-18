#include "common/assert.h"
#include "common/common.h"
#include "common/gpuTiming.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	RecordBarrier(m_barriers.BeforeCommand());
	m_has_commands = true;
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::BarrierHandle() const {
	EXIT_IF(IsInvalid());
	RecordBarrier(m_barriers.BeforeBarrier());
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::StateHandle() const {
	EXIT_IF(IsInvalid());
	m_has_commands = true;
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::DispatchHandle(bool writes) const {
	EXIT_IF(IsInvalid() || m_rendering);
	Common::GpuTiming::Sync().AddUnmergedBarriers(writes ? 2 : 1);
	RecordBarrier(m_barriers.BeforeDispatch(writes));
	m_has_commands = true;
	return m_buffer;
}

void CommandBuffer::GuestFlush() const {
	EXIT_IF(IsInvalid());
	Common::GpuTiming::Sync().AddUnmergedBarriers(1);
	const auto barrier = m_barriers.Flush();
	if (barrier != BarrierTracker::Barrier::None) {
		EndRendering();
		RecordBarrier(barrier);
	}
}

void CommandBuffer::OrderDispatches() const {
	EXIT_IF(IsInvalid());
	RecordBarrier(m_barriers.BeforeBarrier());
}

static_assert(static_cast<uint8_t>(BarrierTracker::Barrier::ComputeToCompute) - 1u ==
                  static_cast<uint8_t>(Common::GpuTiming::BarrierKind::ComputeToCompute) &&
              static_cast<uint8_t>(BarrierTracker::Barrier::AllToAll) - 1u ==
                  static_cast<uint8_t>(Common::GpuTiming::BarrierKind::AllToAll));

void CommandBuffer::RecordBarrier(BarrierTracker::Barrier barrier) const {
	using Barrier = BarrierTracker::Barrier;
	if (barrier == Barrier::None) {
		return;
	}
	// Dispatches end a render pass instance and the start of one orders them first, so no
	// tracked barrier falls inside one.
	EXIT_IF(m_rendering);
	constexpr auto ShaderAccess =
	    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	constexpr auto ComputeReads = vk::AccessFlagBits2::eShaderRead |
	                              vk::AccessFlagBits2::eShaderWrite |
	                              vk::AccessFlagBits2::eUniformRead;
	constexpr auto MemoryAccess =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::MemoryBarrier2 memory {};
	switch (barrier) {
		case Barrier::ComputeToCompute:
			memory.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			memory.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
			memory.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			memory.dstAccessMask = ComputeReads;
			break;
		case Barrier::AllToCompute:
			memory.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory.srcAccessMask = MemoryAccess;
			memory.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			memory.dstAccessMask = ComputeReads;
			break;
		case Barrier::ComputeToAll:
			memory.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			memory.srcAccessMask = ShaderAccess;
			memory.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory.dstAccessMask = MemoryAccess;
			break;
		case Barrier::AllToAll:
			memory.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
			memory.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory.dstAccessMask = MemoryAccess;
			break;
		case Barrier::None: return;
	}
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &memory;
	m_buffer.pipelineBarrier2(dependency);
	Common::GpuTiming::Sync().AddBarrier(
	    static_cast<Common::GpuTiming::BarrierKind>(static_cast<uint8_t>(barrier) - 1u));
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = m_buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_has_commands = false;
}

void CommandBuffer::End() const {
	EndRendering();
	// The deferred barrier after the last dispatches stays in this submission.
	OrderDispatches();

	auto result = m_buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	// Attachment loads and stores access memory like any other command.
	Handle().beginRendering(rendering);
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state = state;
	for (auto& color: m_render_state.color_attachments) {
		color.is_clear = false;
	}
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
