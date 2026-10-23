#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

// A T# and the resource traits that decide how it binds: the key of the texture memo.
struct TextureMemoKey {
	std::array<uint32_t, 8> words;
	uint64_t                traits;
	bool                    operator==(const TextureMemoKey&) const = default;
};

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
	// The texture memo entry whose description `desc` holds (RenderExecutor::ResolveTextureMemo),
	// 0 for none: a hit on that entry leaves the description as it is instead of copying it.
	uint64_t memo_serial = 0;
	// The memo entry this binding last used and its key, valid while the memo has erased nothing
	// since memo_epoch: a binding refilled with the same T# reaches the entry without a lookup.
	const void*    memo_entry = nullptr;
	TextureMemoKey memo_key {};
	uint64_t       memo_epoch = 0;
};

// A stage's texture bindings. Shrinking keeps the bindings past the new size: a stage that
// alternates between programs with many and few images (GPU-selected image tables have
// hundreds of candidates) refills them, with their memo entries and description storage,
// instead of destroying and value-constructing them again for every draw.
class TextureBindings {
public:
	void resize(size_t size) {
		if (size > m_storage.size()) {
			m_storage.resize(size);
		}
		m_size = size;
	}
	void push_back(TextureBinding binding) {
		if (m_size < m_storage.size()) {
			m_storage[m_size] = std::move(binding);
		} else {
			m_storage.push_back(std::move(binding));
		}
		m_size++;
	}
	[[nodiscard]] size_t          size() const noexcept { return m_size; }
	[[nodiscard]] bool            empty() const noexcept { return m_size == 0; }
	[[nodiscard]] TextureBinding& operator[](size_t index) noexcept {
		EXIT_IF(index >= m_size);
		return m_storage[index];
	}
	[[nodiscard]] const TextureBinding& operator[](size_t index) const noexcept {
		EXIT_IF(index >= m_size);
		return m_storage[index];
	}
	[[nodiscard]] TextureBinding&       back() noexcept { return (*this)[m_size - 1]; }
	[[nodiscard]] TextureBinding*       data() noexcept { return m_storage.data(); }
	[[nodiscard]] const TextureBinding* data() const noexcept { return m_storage.data(); }
	[[nodiscard]] TextureBinding*       begin() noexcept { return m_storage.data(); }
	[[nodiscard]] TextureBinding*       end() noexcept { return m_storage.data() + m_size; }
	[[nodiscard]] const TextureBinding* begin() const noexcept { return m_storage.data(); }
	[[nodiscard]] const TextureBinding* end() const noexcept { return m_storage.data() + m_size; }

private:
	std::vector<TextureBinding> m_storage;
	size_t                      m_size = 0;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	TextureBindings                       images;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
	// The bindless heaps the stage samples through (BindlessTranslation) and its flattened SRT
	// with their regions patched in; empty when the stage has no bindless images.
	std::vector<uint32_t>                 bindless_heaps;
	std::vector<uint32_t>                 bindless_srt;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
