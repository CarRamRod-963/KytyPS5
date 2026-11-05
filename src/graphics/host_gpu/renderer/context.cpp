#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
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
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {
	m_barriers.SetSerializeDispatches(Config::SerializeDispatchesEnabled());
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	FinishCopies();
	RecordBarrier(m_barriers.BeforeCommand());
	m_has_commands = true;
	m_command_count++;
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::PipelineHandle() const {
	const auto handle   = Handle();
	m_compute_pipeline  = nullptr;
	m_graphics_pipeline = nullptr;
	return handle;
}

vk::CommandBuffer CommandBuffer::BarrierHandle() const {
	EXIT_IF(IsInvalid());
	FinishCopies();
	RecordBarrier(m_barriers.BeforeBarrier());
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::StateHandle() const {
	EXIT_IF(IsInvalid());
	m_has_commands = true;
	return m_buffer;
}

void CommandBuffer::BindComputePipeline(vk::Pipeline pipeline) const {
	if (pipeline == m_compute_pipeline) {
		return;
	}
	StateHandle().bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	m_compute_pipeline = pipeline;
}

void CommandBuffer::BindGraphicsPipeline(vk::Pipeline pipeline) const {
	if (pipeline == m_graphics_pipeline) {
		return;
	}
	StateHandle().bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	m_graphics_pipeline = pipeline;
}

vk::CommandBuffer CommandBuffer::DispatchHandle(bool writes, bool shared) const {
	EXIT_IF(IsInvalid() || m_rendering);
	Common::GpuTiming::Sync().AddUnmergedBarriers(writes ? 2 : 1);
	FinishCopies();
	RecordBarrier(m_barriers.BeforeDispatch(writes, shared));
	m_has_commands = true;
	m_command_count++;
	return m_buffer;
}

void CommandBuffer::GuestFlush() const {
	EXIT_IF(IsInvalid());
	Common::GpuTiming::Sync().AddUnmergedBarriers(1);
	FinishCopies();
	const auto barrier = m_barriers.Flush();
	if (barrier != BarrierTracker::Barrier::None) {
		EndRendering();
		RecordBarrier(barrier);
	}
}

void CommandBuffer::OrderDispatches() const {
	EXIT_IF(IsInvalid());
	FinishCopies();
	RecordBarrier(m_barriers.BeforeBarrier());
}

void CommandBuffer::OrderIndirectReads() const {
	EXIT_IF(IsInvalid());
	if (m_rendering && m_render_passes == m_indirect_reads_since) {
		return;
	}
	EndRendering();
	vk::MemoryBarrier2 memory {};
	memory.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	memory.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	memory.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect;
	memory.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &memory;
	BarrierHandle().pipelineBarrier2(dependency);
	m_indirect_reads_since = m_render_passes + 1;
}

void CommandBuffer::CopyToBuffer(vk::Buffer source, vk::Buffer destination,
                                 std::span<const vk::BufferCopy> regions,
                                 vk::DeviceAddress               source_address,
                                 vk::DeviceAddress               destination_address) const {
	EXIT_IF(IsInvalid());
	if (regions.empty()) {
		return;
	}
	EndRendering();
	RecordBarrier(m_barriers.BeforeCommand());
	// Outside a batch no later copy joins the run, so its ranges are not needed.
	m_copy_ranges.clear();
	if (m_copy_batches != 0) {
		const auto key =
		    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkBuffer>(destination)));
		for (const auto& region: regions) {
			m_copy_ranges.push_back({key, region.dstOffset, region.dstOffset + region.size});
		}
	}
	const auto barrier = m_copies.BeforeCopy(m_copy_ranges);
	if (barrier != CopyRunTracker::Barrier::None) {
		// Copies of a run may be scatter dispatches.
		constexpr auto Stages =
		    vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader;
		constexpr auto Writes =
		    vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderStorageWrite;
		vk::MemoryBarrier2 memory {};
		if (barrier == CopyRunTracker::Barrier::AllToTransfer) {
			memory.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
			memory.srcAccessMask =
			    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		} else {
			// The copies the barrier orders this one after.
			RecordScatter();
			memory.srcStageMask  = Stages;
			memory.srcAccessMask = Writes;
		}
		memory.dstStageMask  = Stages;
		memory.dstAccessMask = Writes | vk::AccessFlagBits2::eShaderStorageRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &memory;
		m_buffer.pipelineBarrier2(dependency);
	}
	// Only the copies of a batch can share a dispatch; the shader copies whole dwords.
	const bool scatter = m_copy_batches != 0 && source_address != 0 && destination_address != 0;
	if (scatter && source != m_scatter_source) {
		RecordScatter();
		m_scatter_source         = source;
		m_scatter_source_address = source_address;
	}
	m_direct_regions.clear();
	const auto pieces = m_scatter_pieces.size();
	for (const auto& region: regions) {
		constexpr auto Alignment = UploadScatter::Alignment;
		const auto     body      = region.size / Alignment * Alignment;
		if (!scatter || body == 0 || region.srcOffset % Alignment != 0 ||
		    region.dstOffset % Alignment != 0 || region.srcOffset + body > UINT32_MAX) {
			m_direct_regions.push_back(region);
			continue;
		}
		UploadScatter::AddPieces(m_scatter_pieces, destination_address + region.dstOffset,
		                         static_cast<uint32_t>(region.srcOffset),
		                         static_cast<uint32_t>(body));
		if (body != region.size) {
			m_direct_regions.emplace_back(region.srcOffset + body, region.dstOffset + body,
			                              region.size - body);
		}
	}
	if (m_scatter_pieces.size() != pieces) {
		if (!m_scatter_destinations.empty() &&
		    m_scatter_destinations.back().buffer == destination) {
			m_scatter_destinations.back().end = m_scatter_pieces.size();
		} else {
			m_scatter_destinations.push_back(
			    {destination, destination_address, m_scatter_pieces.size()});
		}
	}
	if (!m_direct_regions.empty()) {
		m_buffer.copyBuffer(source, destination, static_cast<uint32_t>(m_direct_regions.size()),
		                    m_direct_regions.data());
	}
	m_has_commands = true;
	m_command_count++;
	if (m_copy_batches == 0) {
		FinishCopies();
	}
}

void CommandBuffer::RecordScatter() const {
	if (m_scatter_pieces.empty()) {
		return;
	}
	if (!m_context.GetUploadScatter().Record(m_buffer, m_scatter_source_address,
	                                         m_scatter_pieces)) {
		// The ring is full: a copy command per destination buffer.
		std::vector<vk::BufferCopy> copies;
		size_t                      first = 0;
		for (const auto& destination: m_scatter_destinations) {
			copies.clear();
			for (size_t i = first; i < destination.end; i++) {
				const auto& piece   = m_scatter_pieces[i];
				const auto  address = static_cast<uint64_t>(piece.destination_low) |
				                      (static_cast<uint64_t>(piece.destination_high) << 32u);
				copies.emplace_back(piece.source_offset, address - destination.address, piece.size);
			}
			m_buffer.copyBuffer(m_scatter_source, destination.buffer,
			                    static_cast<uint32_t>(copies.size()), copies.data());
			first = destination.end;
		}
	} else if (m_compute_pipeline) {
		// The pipeline bound for a guest dispatch about to be recorded.
		m_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_compute_pipeline);
	}
	m_scatter_pieces.clear();
	m_scatter_destinations.clear();
}

void CommandBuffer::FinishCopies() const {
	RecordScatter();
	if (!m_copies.BeforeOther()) {
		return;
	}
	// A copy ends the render pass instance, and starting one records this barrier first.
	EXIT_IF(m_rendering);
	vk::MemoryBarrier2 memory {};
	memory.srcStageMask =
	    vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader;
	memory.srcAccessMask =
	    vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderStorageWrite;
	memory.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	memory.dstAccessMask = LaterAccess2;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &memory;
	m_buffer.pipelineBarrier2(dependency);
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
			memory.dstAccessMask = LaterAccess2;
			break;
		case Barrier::AllToAll:
			memory.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
			memory.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			memory.dstAccessMask = LaterAccess2;
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
	// The previous buffer recorded the barrier of its copies when it ended.
	(void)m_copies.BeforeOther();
	EXIT_IF(!m_scatter_pieces.empty());
	m_has_commands         = false;
	m_command_count        = 0;
	m_compute_pipeline     = nullptr;
	m_graphics_pipeline    = nullptr;
	m_indirect_reads_since = UINT64_MAX;
}

void CommandBuffer::End() const {
	EndRendering();
	// The deferred barriers after the last copies and dispatches stay in this submission.
	OrderDispatches();

	auto result = m_buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	const bool ordered = m_rendering && m_render_passes == m_indirect_reads_since;
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp       = vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp      = vk::AttachmentStoreOp::eStore;
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
	m_render_state = state;
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
	// A pass that directly follows one OrderIndirectReads covers is covered too: nothing was
	// recorded between them.
	m_render_passes++;
	if (ordered) {
		m_indirect_reads_since = m_render_passes;
	}
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering = false;
}

} // namespace Libs::Graphics
