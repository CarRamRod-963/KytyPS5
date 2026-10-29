#include "graphics/host_gpu/renderer/meshIndirectTasks.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/mesh_indirect_tasks_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

namespace {

// mesh_indirect_tasks.comp's push constants.
struct Conversion {
	uint32_t record_word;
	uint32_t max_count;
	uint32_t primitive_size;
	uint32_t primitive_step;
	uint32_t primitives_per_group;
	uint32_t max_groups_x;
	uint32_t max_groups_y;
	uint32_t max_groups_total;
};

constexpr uint64_t TasksSize = 3 * sizeof(uint32_t);

} // namespace

MeshIndirectTasks::MeshIndirectTasks(GraphicContext& graphics): m_graphics(graphics) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create mesh indirect tasks descriptor layout");

	const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Conversion)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	    "create mesh indirect tasks pipeline layout");

	const auto module = CompileSPV(MESH_INDIRECT_TASKS_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh indirect tasks pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Mesh Indirect Tasks");
}

MeshIndirectTasks::~MeshIndirectTasks() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
}

void MeshIndirectTasks::Record(CommandBuffer& command, const Buffer& record, uint64_t record_offset,
                               const Buffer& tasks, uint64_t tasks_offset,
                               const MeshIndirectShape& shape) {
	EXIT_IF((record_offset & 3u) != 0 || shape.primitive_step == 0 ||
	        shape.primitives_per_group == 0);
	// The record is bound from an aligned offset below it.
	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto record_base = record_offset - record_offset % alignment;
	const auto record_end  = record_offset + 2 * sizeof(uint32_t);
	EXIT_IF(record_end > record.Size() || tasks_offset % alignment != 0 ||
	        tasks_offset + TasksSize > tasks.Size());

	const auto& limits = m_graphics.mesh_shader_properties;
	Conversion  push {
	    .record_word          = static_cast<uint32_t>((record_offset - record_base) / 4u),
	    .max_count            = shape.max_count,
	    .primitive_size       = shape.primitive_size,
	    .primitive_step       = shape.primitive_step,
	    .primitives_per_group = shape.primitives_per_group,
	    .max_groups_x         = limits.maxMeshWorkGroupCount[0],
	    .max_groups_y         = limits.maxMeshWorkGroupCount[1],
	    .max_groups_total     = limits.maxMeshWorkGroupTotalCount,
	};

	const vk::DescriptorBufferInfo infos[] {
	    {record.Handle(), record_base, record_end - record_base},
	    {tasks.Handle(), tasks_offset, TasksSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	command.EndRendering();
	const auto native = command.PipelineHandle();
	// Whatever wrote the record (a guest dispatch, an upload) before the read; the indirect
	// draw's own barrier (OrderIndirectReads) then orders the tasks before the draw reads them.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	native.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	native.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	native.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                     &push);
	native.dispatch(1, 1, 1);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0, nullptr, 0,
	                       nullptr);
}

} // namespace Libs::Graphics
