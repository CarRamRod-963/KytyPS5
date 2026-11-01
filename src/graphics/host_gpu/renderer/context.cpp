#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cmdSignature.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
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
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	// A new command buffer starts without any pipeline or dynamic state (KYTY_DIAG_DRAW_STATE_FILTER).
	InvalidateGraphicsState();
}

void CommandBuffer::InvalidateGraphicsState() const noexcept {
	m_graphics_state = GraphicsState {};
	// The record packets: the recording thread's copy is forgotten at the same place of the command stream.
	NoteGraphicsStateInvalidated(m_buffer);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

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
	if (CmdSignature::Enabled()) {
		// KYTY_TRACE_CMD_SIGNATURE: the render pass's size, layers, and each attachment's layout, load op and clear
		// value (not its view: a handle).
		std::array<uint64_t, 4 + (RENDER_COLOR_ATTACHMENTS_MAX + 1) * 4> words {};
		size_t                                                           count = 0;
		words[count++] = state.width;
		words[count++] = state.height;
		words[count++] = state.num_layers;
		words[count++] = state.num_color_attachments;
		const auto attachment_words = [&](const RenderAttachment& attachment) {
			words[count++] = static_cast<uint64_t>(attachment.image_layout) |
			                 (uint64_t {attachment.is_clear} << 32u) | (uint64_t {attachment.has_depth} << 33u) |
			                 (uint64_t {attachment.depth_clear} << 34u) | (uint64_t {attachment.has_stencil} << 35u) |
			                 (uint64_t {attachment.stencil_clear} << 36u) | (uint64_t {attachment.image_view ? 1u : 0u} << 37u);
			words[count++] = attachment.clear_value[0] | (uint64_t {attachment.clear_value[1]} << 32u);
			words[count++] = attachment.clear_value[2] | (uint64_t {attachment.clear_value[3]} << 32u);
			words[count++] = 0;
		};
		for (uint32_t i = 0; i < state.num_color_attachments; i++) {
			attachment_words(state.color_attachments[i]);
		}
		attachment_words(state.depth_stencil_attachment);
		CmdSignature::Fold(CmdSignature::Kind::BeginRendering, words.data(), count * sizeof(uint64_t));
	}

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
	FlushComputeAccess(Handle()); // compute writes before rendering reads them
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
	// KYTY_FIX_OCCLUSION_QUERIES: a host occlusion query begun in this render pass instance ends in it; slots for
	// the next ones are reset outside it.
	auto& occlusion = m_context.GetOcclusionQueries();
	occlusion.BeforeEndRendering(Handle());
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
	occlusion.AfterEndRendering(Handle());
}

} // namespace Libs::Graphics
