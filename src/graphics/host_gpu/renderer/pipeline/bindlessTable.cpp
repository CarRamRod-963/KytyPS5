#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstring>

namespace Libs::Graphics {

namespace IR = ShaderRecompiler::IR;

BindlessTable::BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	if (!graphics.bindless_enabled) {
		return;
	}
	EXIT_IF(graphics.device == nullptr || graphics.bindless_set_layout != nullptr);
	constexpr vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eAll;
	const std::array<vk::DescriptorSetLayoutBinding, 3> bindings {{
	    {IR::BindlessImagesBinding, vk::DescriptorType::eSampledImage, IR::BindlessImageSlots,
	     stages, nullptr},
	    {IR::BindlessTranslationBinding, vk::DescriptorType::eStorageBuffer, 1, stages, nullptr},
	    {IR::BindlessFeedbackBinding, vk::DescriptorType::eStorageBuffer, 1, stages, nullptr},
	}};
	const std::array<vk::DescriptorBindingFlags, 3> binding_flags {
	    vk::DescriptorBindingFlagBits::ePartiallyBound |
	        vk::DescriptorBindingFlagBits::eUpdateAfterBind,
	    vk::DescriptorBindingFlags {}, vk::DescriptorBindingFlags {}};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = static_cast<uint32_t>(binding_flags.size());
	flags_info.pBindingFlags = binding_flags.data();
	vk::DescriptorSetLayoutCreateInfo create {};
	create.pNext        = &flags_info;
	create.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &m_layout) !=
	        vk::Result::eSuccess);
	graphics.bindless_set_layout = m_layout;
}

BindlessTable::~BindlessTable() {
	if (m_pool != nullptr) {
		m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
	}
	if (m_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
		m_graphics.bindless_set_layout = nullptr;
	}
}

void BindlessTable::Create() {
	EXIT_IF(!Enabled() || Created());
	const auto bytes = uint64_t {Entries} * sizeof(uint32_t);
	m_translation    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
	                                            vk::BufferUsageFlagBits::eStorageBuffer, bytes);
	m_feedback       = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                            vk::BufferUsageFlagBits::eStorageBuffer, bytes);
	EXIT_IF(m_translation->Mapped().empty() || m_feedback->Mapped().empty());
	std::memset(m_feedback->Mapped().data(), 0, bytes);
	m_entries.emplace(
	    std::span {reinterpret_cast<uint32_t*>(m_translation->Mapped().data()), Entries},
	    IR::BindlessImageSlots);
	FlushTranslation();
	if (!m_feedback->IsCoherent()) {
		m_feedback->Flush(0, bytes);
	}

	const std::array<vk::DescriptorPoolSize, 2> sizes {{
	    {vk::DescriptorType::eSampledImage, IR::BindlessImageSlots},
	    {vk::DescriptorType::eStorageBuffer, 2},
	}};
	vk::DescriptorPoolCreateInfo pool {};
	pool.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool.maxSets       = 1;
	pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&pool, nullptr, &m_pool) !=
	        vk::Result::eSuccess);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &m_layout;
	EXIT_IF(m_graphics.device.allocateDescriptorSets(&allocate, &m_set) != vk::Result::eSuccess);

	const std::array<vk::DescriptorBufferInfo, 2> buffers {{
	    {m_translation->Handle(), 0, bytes},
	    {m_feedback->Handle(), 0, bytes},
	}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = i == 0 ? IR::BindlessTranslationBinding : IR::BindlessFeedbackBinding;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &buffers[i];
	}
	m_graphics.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(),
	                                       0, nullptr);
}

vk::DescriptorSet BindlessTable::Set() {
	if (!Created()) {
		Create();
	}
	return m_set;
}

BindlessTranslation& BindlessTable::Translation() {
	if (!Created()) {
		Create();
	}
	return *m_entries;
}

void BindlessTable::WriteSlot(uint32_t slot, vk::ImageView view, vk::ImageLayout layout) {
	// A null view is a native null descriptor (nullDescriptor).
	EXIT_IF(!Created() || slot >= IR::BindlessImageSlots);
	const vk::DescriptorImageInfo image {nullptr, view, layout};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = IR::BindlessImagesBinding;
	write.dstArrayElement = slot;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampledImage;
	write.pImageInfo      = &image;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
}

void BindlessTable::MakeFeedbackVisible(vk::CommandBuffer command_buffer) {
	EXIT_IF(!Created());
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllGraphics |
	                       vk::PipelineStageFlagBits2::eComputeShader;
	barrier.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command_buffer.pipelineBarrier2(dependency);
}

void BindlessTable::TakeFeedback(std::vector<uint32_t>& entries, uint32_t limit) {
	EXIT_IF(!Created());
	const auto used = m_entries->UsedEntries();
	if (!m_feedback->IsCoherent()) {
		m_feedback->Invalidate(0, uint64_t {used} * sizeof(uint32_t));
	}
	auto* flags = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	// Resume behind the last entry taken so a busy region cannot starve the ones after it.
	if (m_feedback_cursor >= used) {
		m_feedback_cursor = BindlessTranslation::FirstEntry;
	}
	const auto start = m_feedback_cursor;
	for (uint32_t scanned = 0; scanned < used && entries.size() < limit; ++scanned) {
		const auto entry = start + scanned < used ? start + scanned : start + scanned - used;
		if (flags[entry] != 0) {
			flags[entry] = 0;
			entries.push_back(entry);
			m_feedback_cursor = entry + 1;
		}
	}
	if (!m_feedback->IsCoherent()) {
		m_feedback->Flush(0, uint64_t {used} * sizeof(uint32_t));
	}
}

void BindlessTable::FlushTranslation() {
	EXIT_IF(!m_translation);
	if (!m_translation->IsCoherent()) {
		m_translation->Flush(0, uint64_t {m_entries->UsedEntries()} * sizeof(uint32_t));
	}
}

} // namespace Libs::Graphics
