#include "graphics/host_gpu/renderer/cache/uploadScatter.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_blit_shaders/gpu_buffer_scatter_spv.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace Libs::Graphics {

static_assert(UploadScatter::PieceBytes % 16 == 0 && UploadScatter::PieceBytes / 16 <= 256,
              "a piece is at most one 16-byte vector per invocation of a 256-wide workgroup");

UploadScatter::UploadScatter(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_ring(graphics, scheduler, MemoryUsage::Stream, RingSize) {
	SetVulkanObjectNameF(m_graphics.device, m_ring.Handle(), "Kyty.UploadScatterRing");

	vk::PipelineLayoutCreateInfo layout_info {};
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_layout),
	                     "create upload scatter pipeline layout");

	const auto                                      ring = m_ring.BufferDeviceAddress();
	const std::array<uint32_t, 2>                   ring_words {static_cast<uint32_t>(ring),
	                                                            static_cast<uint32_t>(ring >> 32u)};
	const std::array<vk::SpecializationMapEntry, 2> entries {
	    vk::SpecializationMapEntry {0, 0, sizeof(uint32_t)},
	    vk::SpecializationMapEntry {1, sizeof(uint32_t), sizeof(uint32_t)},
	};
	vk::SpecializationInfo specialization {};
	specialization.mapEntryCount = static_cast<uint32_t>(entries.size());
	specialization.pMapEntries   = entries.data();
	specialization.dataSize      = sizeof(ring_words);
	specialization.pData         = ring_words.data();

	const auto module = CompileSPV(GPU_BUFFER_SCATTER_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage               = vk::ShaderStageFlagBits::eCompute;
	stage.module              = module;
	stage.pName               = "main";
	stage.pSpecializationInfo = &specialization;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.flags  = vk::PipelineCreateFlagBits::eDispatchBase;
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create upload scatter pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Kyty.UploadScatter");
}

UploadScatter::~UploadScatter() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_layout, nullptr);
}

void UploadScatter::AddPieces(std::vector<Piece>& pieces, uint64_t destination,
                              uint32_t source_offset, uint32_t size) {
	EXIT_IF(((destination | source_offset | size) % Alignment) != 0);
	for (uint32_t done = 0; done < size; done += PieceBytes) {
		const auto address = destination + done;
		pieces.push_back({static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u),
		                  source_offset + done, std::min(PieceBytes, size - done)});
	}
}

bool UploadScatter::Record(vk::CommandBuffer command, vk::DeviceAddress source,
                           std::span<const Piece> pieces) {
	if (pieces.empty()) {
		return true;
	}
	const auto tables = (pieces.size() + MaxTablePieces - 1) / MaxTablePieces;
	// A table is a header (the source address), its pieces and the padding to the next one.
	const auto bytes =
	    (tables - 1) * Common::AlignUp((MaxTablePieces + 1) * sizeof(Piece), TableAlignment) +
	    (pieces.size() - (tables - 1) * MaxTablePieces + 1) * sizeof(Piece);
	// No wait: it would submit the command buffer before the copies are recorded.
	const auto [mapped, offset] = m_ring.Map(bytes, TableAlignment, false);
	if (mapped == nullptr) {
		return false;
	}
	EXIT_IF(offset % TableAlignment != 0);
	const Piece header {static_cast<uint32_t>(source), static_cast<uint32_t>(source >> 32u), 0, 0};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	uint64_t position = 0;
	for (uint64_t first = 0; first < pieces.size(); first += MaxTablePieces) {
		const auto count = std::min<uint64_t>(MaxTablePieces, pieces.size() - first);
		std::memcpy(mapped + position, &header, sizeof(header));
		std::memcpy(mapped + position + sizeof(Piece), pieces.data() + first,
		            count * sizeof(Piece));
		command.dispatchBase(0, static_cast<uint32_t>((offset + position) / TableAlignment), 0,
		                     static_cast<uint32_t>(count), 1, 1);
		position += Common::AlignUp((count + 1) * sizeof(Piece), TableAlignment);
	}
	m_ring.Commit();
	return true;
}

} // namespace Libs::Graphics
