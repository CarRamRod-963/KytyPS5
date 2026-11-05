#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADSCATTER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADSCATTER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Copies many ranges of one source buffer into destination buffers with one compute dispatch
// (gpu_buffer_scatter.comp) instead of a copy command per destination buffer: the back-to-back
// uploads of a CommandBuffer::BatchCopies run, thousands per frame in Saros, cost a driver call
// each. Each range is split into pieces of at most PieceBytes, one workgroup each. The pieces go
// into a ring of their own, which the shader reads at an address fixed in the pipeline; a
// dispatch passes its table's place in the ring as its base workgroup, so it binds no
// descriptors and pushes no constants and leaves the state bound for other dispatches alone.
class UploadScatter {
public:
	struct Piece {
		uint32_t destination_low  = 0;
		uint32_t destination_high = 0;
		// Byte offset in the source buffer.
		uint32_t source_offset = 0;
		uint32_t size          = 0;
	};
	static_assert(sizeof(Piece) == 16);

	static constexpr uint32_t PieceBytes = 4096;
	// Addresses, offsets and sizes the shader copies are multiples of this.
	static constexpr uint32_t Alignment = 4;

	UploadScatter(GraphicContext& graphics, CommandScheduler& scheduler);
	~UploadScatter();
	KYTY_CLASS_NO_COPY(UploadScatter);

	// Appends the pieces that copy `size` bytes at `source_offset` of the source buffer to
	// `destination`; all three are multiples of Alignment.
	static void AddPieces(std::vector<Piece>& pieces, uint64_t destination, uint32_t source_offset,
	                      uint32_t size);

	// Records the dispatches that copy `pieces` from the buffer at `source`; they are ordered
	// like copy commands, by barriers the caller records for the compute shader stage. Binds
	// its own compute pipeline. Records nothing and returns false when the pieces do not fit in
	// the ring without a wait.
	[[nodiscard]] bool Record(vk::CommandBuffer command, vk::DeviceAddress source,
	                          std::span<const Piece> pieces);

private:
	// Pieces per table: the dispatch size every device supports.
	static constexpr uint64_t MaxTablePieces = 65535;
	// Tables start at multiples of this; a table's index in the ring is the base workgroup Y.
	static constexpr uint64_t TableAlignment = 256;
	static constexpr uint64_t RingSize       = 4 * 1024 * 1024;
	static_assert(RingSize / TableAlignment <= 65535);

	GraphicContext&    m_graphics;
	StreamBuffer       m_ring;
	vk::PipelineLayout m_layout   = nullptr;
	vk::Pipeline       m_pipeline = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADSCATTER_H_
