#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECTTASKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECTTASKS_H_

#include "common/abi.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
struct GraphicContext;

// The groups of a mesh draw that emulates a guest indirect draw: the input primitives the mesh
// shader turns into its output, as the guest's draw names them (ShaderMeshInputInfo).
struct MeshIndirectShape {
	uint32_t max_count            = UINT32_MAX; // the index buffer's size in indices
	uint32_t primitive_size       = 0;
	uint32_t primitive_step       = 0;
	uint32_t primitives_per_group = 0;
};

// Converts a GPU-written indirect draw record into a VkDrawMeshTasksIndirectCommandEXT on the
// GPU, so the host never reads the record (one invocation of mesh_indirect_tasks.comp).
class MeshIndirectTasks {
public:
	explicit MeshIndirectTasks(GraphicContext& graphics);
	~MeshIndirectTasks();
	KYTY_CLASS_NO_COPY(MeshIndirectTasks);

	// Records the conversion outside rendering. The record must be dword aligned; the tasks
	// offset must satisfy minStorageBufferOffsetAlignment. The tasks are ready for an indirect
	// draw recorded after it.
	void Record(CommandBuffer& command, const Buffer& record, uint64_t record_offset,
	            const Buffer& tasks, uint64_t tasks_offset, const MeshIndirectShape& shape);

private:
	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECTTASKS_H_
