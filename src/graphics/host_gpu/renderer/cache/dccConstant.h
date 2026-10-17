#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCCONSTANT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCCONSTANT_H_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>

namespace Libs::Graphics::DccConstant {

// DCC key 0x10 is the GFX9-GFX10.3 "clear to single" code (radeonsi's GFX9_DCC_CLEAR_SINGLE): the
// 256-byte block it describes holds one value, stored in the block's first pixel; the rest of the
// block's bytes are not read. A fast clear writes only the keys and that pixel of every block
// (Ghost of Yotei resets its motion-vector target this way every frame).
inline constexpr uint8_t  Key        = 0x10;
inline constexpr uint64_t BlockBytes = 256;

[[nodiscard]] constexpr bool SupportsElementBytes(uint32_t element_bytes) noexcept {
	return element_bytes != 0 && element_bytes <= 16 && std::has_single_bit(element_bytes);
}

// Whether every key of a metadata slice is the single-value code. Key bytes are swizzled
// relative to the blocks they describe, so only a uniform slice can be decoded.
[[nodiscard]] inline bool IsConstantSlice(std::span<const uint8_t> keys) noexcept {
	return !keys.empty() &&
	       std::all_of(keys.begin(), keys.end(), [](uint8_t key) { return key == Key; });
}

// Decompresses single-value blocks in place: every element of each 256-byte block becomes a copy
// of the block's first element. The GPU expansion before an image upload does the same.
[[nodiscard]] inline bool ExpandBlocks(std::span<uint8_t> data, uint32_t element_bytes) noexcept {
	if (!SupportsElementBytes(element_bytes) || data.size() % BlockBytes != 0) {
		return false;
	}
	for (size_t block = 0; block < data.size(); block += BlockBytes) {
		for (size_t offset = element_bytes; offset < BlockBytes; offset += element_bytes) {
			std::memcpy(data.data() + block + offset, data.data() + block, element_bytes);
		}
	}
	return true;
}

} // namespace Libs::Graphics::DccConstant

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCCONSTANT_H_
