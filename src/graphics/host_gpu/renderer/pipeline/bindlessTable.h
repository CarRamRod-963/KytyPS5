#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTranslation.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class CommandScheduler;

// Descriptor set 1 of programs with bindless images (IR::BindlessDescriptorSet): the image
// array keys translate to, the translation entries and the feedback words shaders flag for
// pending keys. The set is bound once per draw and updated after bind as keys resolve.
class BindlessTable {
public:
	static constexpr uint32_t Entries = 1u << 22;

	BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler);
	~BindlessTable();
	KYTY_CLASS_NO_COPY(BindlessTable);

	[[nodiscard]] bool                    Enabled() const noexcept { return m_layout != nullptr; }
	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] vk::DescriptorSet       Set();
	[[nodiscard]] BindlessTranslation&    Translation();
	[[nodiscard]] bool                    Created() const noexcept { return m_set != nullptr; }

	void WriteSlot(uint32_t slot, vk::ImageView view, vk::ImageLayout layout);
	// Makes the GPU's shader writes to the feedback words visible to the host.
	void MakeFeedbackVisible(vk::CommandBuffer command_buffer);
	// The entries shaders flagged, up to `limit`, cleared as they are taken.
	void TakeFeedback(std::vector<uint32_t>& entries, uint32_t limit);
	void FlushTranslation();
	// Drops every heap (BindlessTranslation::Reset) and the feedback flags of their entries.
	// The caller must have waited for every submission that reads the table.
	void Reset();

private:
	void Create();

	GraphicContext&                    m_graphics;
	CommandScheduler&                  m_scheduler;
	vk::DescriptorSetLayout            m_layout = nullptr;
	vk::DescriptorPool                 m_pool   = nullptr;
	vk::DescriptorSet                  m_set    = nullptr;
	std::unique_ptr<Buffer>            m_translation;
	std::unique_ptr<Buffer>            m_feedback;
	std::optional<BindlessTranslation> m_entries;
	uint32_t                           m_feedback_cursor = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
