#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <atomic>
#include <compare>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
class TileManager;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool storage       = false;
	bool render_target = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
};

class Image final {
public:
	// `memory_class` is the residency class of the backing; depth images are always targets.
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info,
	      MemoryPriority::Class memory_class = MemoryPriority::Class::Cache);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	// Depth/stencil images cannot be storage images, so a shader that writes a stencil plane
	// writes an R8_UINT color copy of it instead. `plane` is the plane's guest layout (its
	// stencil association) and `range` its subresources in this image. Returns the copy holding
	// the plane's current values: for the plane in the stencil aspect, the aspect is copied
	// into it unless it already holds newer values; a plane saved by SelectStencilPlane is
	// returned as saved.
	[[nodiscard]] Image& StencilPlane(const ImageInfo& plane, const ImageSubresourceRange& range);
	// The guest binds one depth image with different stencil planes over time, but the image has
	// one stencil aspect. Binding `plane` where another plane in other memory held the same layers
	// saves that plane's values into its copy and restores `plane`'s saved values, so each plane
	// keeps its own contents. Planes in overlapping memory or on other layers share the aspect.
	void SelectStencilPlane(const ImageInfo& plane, const ImageSubresourceRange& range);
	// Whether `plane` and `range` describe a stencil plane of this image that a color copy can
	// hold (single-sampled, the image's extent, whole mip chains).
	[[nodiscard]] bool IsValidStencilPlane(const ImageInfo&             plane,
	                                       const ImageSubresourceRange& range) const noexcept;
	// The copy of `plane` now receives shader writes. For the plane in the stencil aspect it
	// holds the newer values: the next access to this image through any other path copies them
	// back into the stencil aspect first.
	void WriteStencilPlane(const GuestRange& plane, std::optional<ImageSubresourceRange> range,
	                       vk::CommandBuffer command_buffer);
	// Whether the image holds the values of `plane`, in its stencil aspect or in a saved copy.
	[[nodiscard]] bool HoldsStencilPlane(const GuestRange& plane) const noexcept;
	// The copy holding a plane that is not in the stencil aspect, if one was saved.
	[[nodiscard]] Image* SavedStencilPlane(const GuestRange& plane) noexcept;
	// The stencil plane copy that owns `view`, if any.
	[[nodiscard]] Image* StencilPlaneCopyOf(vk::ImageView view) noexcept;
	[[nodiscard]] bool StencilPlaneIsNewer() const noexcept { return m_stencil_plane_newer; }
	[[nodiscard]] bool HasStencilAspect() const noexcept {
		return static_cast<bool>(FullAspectMask(backing.format) &
		                         vk::ImageAspectFlagBits::eStencil);
	}

	void MarkCpuDirty() {
		m_cpu_dirty       = true;
		m_maybe_cpu_dirty = false;
		HeapChanged();
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!IsCpuDirty()) {
			m_maybe_cpu_hash  = HashGuestEdges();
			m_maybe_cpu_dirty = true;
			HeapChanged();
		}
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty       = false;
		m_maybe_cpu_dirty = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept { m_gpu_modified = false; }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		m_buffer_modified = true;
		HeapChanged();
	}
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : Common::AlignUp(info.data.size, 1024);
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;
	[[nodiscard]] vk::ImageLayout ReadLayout() const noexcept {
		return info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
		                      : vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	// Resident image heaps sample an image without rediscovering it until one changes its
	// contents, tracking, identity or layout.
	void HeapChanged() const noexcept {
		if (heap_refs.load(std::memory_order_relaxed) != 0u) {
			heap_epoch.fetch_add(1u, std::memory_order_relaxed);
		}
	}

	static inline std::atomic<uint64_t> heap_epoch {0};

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	// Stencil associations that refer to this depth image.
	std::vector<ImageId> stencil_ids;
	// The current stencil plane's mapping into the depth image, also retained by its association.
	ImageSubresourceRange stencil_subresources;
	uint64_t         tick_accessed_last = 0;
	size_t           lru_id             = 0;
	// Resident image heap elements that select this image.
	std::atomic<uint32_t> heap_refs {0};
	// The garbage-collection tick and backing of the last TextureCache::TouchImage: binding an
	// image again within the tick has nothing left to record.
	uint64_t                      touched_tick   = UINT64_MAX;
	const MemoryPriority::Object* touched_memory = nullptr;
	// TextureCache event sequence numbers: the last native GPU write to the image and the last
	// time its guest color metadata was read for a clear code. Native rendering leaves the guest
	// metadata untouched, so a code read before a later native write describes an older surface.
	uint64_t         native_write_sequence  = 0;
	uint64_t         metadata_read_sequence = 0;
	uint64_t         metadata_write_sequence = 0;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	struct StencilPlaneCopy {
		ImageInfo               info;
		ImageSubresourceRange   range;
		std::unique_ptr<Image>  color;
		std::unique_ptr<Buffer> staging;
		// The copy holds the plane's values and the stencil aspect does not.
		bool saved = false;

		[[nodiscard]] bool Describes(const ImageInfo&             plane,
		                             const ImageSubresourceRange& plane_range) const noexcept;
	};
	[[nodiscard]] StencilPlaneCopy* FindStencilPlaneCopy(const GuestRange& plane) noexcept;
	[[nodiscard]] StencilPlaneCopy& ObtainStencilPlaneCopy(const ImageInfo&             plane,
	                                                       const ImageSubresourceRange& range);
	void ValidateStencilPlane(const ImageInfo& plane, const ImageSubresourceRange& range) const;

	// Copies the stencil plane's values back into the stencil aspect if they are newer.
	void ReturnStencilPlane();
	// Moves the stencil values between the stencil aspect and a color copy of a plane through
	// the copy's staging buffer.
	void CopyStencilPlane(StencilPlaneCopy& copy, bool to_plane);

	// The last read-only transition that went through the per-subresource states. Repeating it
	// changes nothing and emits no barrier, so it is answered without walking the subresources
	// while the state it left behind is still the image's state.
	struct SettledTransition {
		vk::Image               image = nullptr;
		vk::ImageLayout         layout {};
		vk::AccessFlags2        access {};
		vk::PipelineStageFlags2 stage {};
		ImageSubresourceRange   range;
		VulkanImageState        state;
		size_t                  subresources = 0;
	};
	std::optional<SettledTransition> m_settled;

	std::vector<StencilPlaneCopy> m_stencil_planes;
	// The plane whose values the stencil aspect holds, and its subresources in this image.
	std::optional<ImageInfo> m_stencil_owner;
	ImageSubresourceRange    m_stencil_owner_range;
	// The owner's copy holds newer values than the stencil aspect.
	bool m_stencil_plane_newer = false;

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash  = 0;
	bool              m_cpu_dirty       = false;
	bool              m_maybe_cpu_dirty = false;
	bool              m_gpu_modified    = false;
	bool              m_buffer_modified = false;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
