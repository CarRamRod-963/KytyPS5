#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCRETILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCRETILE_H_

#include <array>
#include <cstdint>
#include <span>

namespace Libs::Graphics::DccRetile {

// The SDK's DCC retile compute shader copies DCC keys from one metadata layout into another. Each
// invocation of its 8x8 workgroups reads eight key bytes (k8UInt elements of the source V#) and
// writes them as four k8_8UInt elements of the destination V#. The guest runs it after rendering
// to a DCC target, and the texture cache reads the destination back to look for fast-clear codes.
// Locate is the shader's address arithmetic in instruction order (names follow its VGPRs), so the
// copy can run on the CPU when both sides are current in guest memory.

inline constexpr uint32_t GroupWidth  = 8;
inline constexpr uint32_t GroupHeight = 8;

struct Invocation {
	// Source element indices; stores[i] writes loads[2 * i] and loads[2 * i + 1].
	std::array<uint32_t, 8> loads {};
	// Destination element indices.
	std::array<uint32_t, 4> stores {};
};

[[nodiscard]] constexpr uint32_t Bfe23(uint32_t value, uint32_t offset) noexcept {
	return (value >> offset) & 0x7fffffu;
}

// x and y are the invocation's global coordinates (workgroup id * 8 + local id); pitch is the
// shader's first user data word.
[[nodiscard]] constexpr Invocation Locate(uint32_t pitch, uint32_t x, uint32_t y) noexcept {
	uint32_t v10 = y;
	uint32_t v4  = x;
	uint32_t v9  = v10 << 3u;
	uint32_t v0  = Bfe23(v10, 6u);
	uint32_t v7  = (v4 << 6u) + 8u;
	uint32_t v8  = v10 << 4u;
	uint32_t v13 = Bfe23(v4, 3u);
	uint32_t v11 = v9 >> 1u;
	uint32_t v2  = v9 >> 2u;
	uint32_t v12 = pitch * v0;
	uint32_t v3  = v10 << 5u;
	uint32_t v6  = v10 << 6u;
	uint32_t v15 = v11 & 8u;
	uint32_t v14 = v2 & 32u;
	v0           = v4 << 10u;
	uint32_t v1  = v10 << 8u;
	uint32_t v5  = v7 >> 9u;
	v15          = (2u & v2) | v15;
	v14          = (10u & v10) | v14;
	v10          = v7 >> 3u;
	v2           = (v13 + v12) << 12u;
	v5           = (v5 + v12) << 12u;
	v13          = (0x80u & v11) | v14;
	v14          = (32u & v9) | v15;
	v11          = v7 >> 4u;
	v9           = (17u & v10) | 0x300u;
	v12          = (17u & v10) | 0x900u;
	uint32_t v16 = (17u & v10) | 0x100u;
	v7           = v7 >> 2u;
	v1           = (0xf00u & v1) | v13;
	v13          = (0x80u & v8) | v14;
	v8           = (17u & v10) | 0xb00u;
	v10          = v4 << 6u;
	v9           = (4u & v11) | v9;
	v14          = (4u & v11) | v12;
	v12          = (0x200u & v3) | v13;
	v15          = (4u & v11) | v8;
	v13          = v10 >> 3u;
	v11          = (4u & v11) | v16;
	v16          = (64u & v7) | v9;
	v14          = (64u & v7) | v14;
	v15          = (64u & v7) | v15;
	v8           = (0x800u & v6) | v12;
	v3           = v4 << 7u;
	v12          = (16u & v13) | 0xa00u;
	v4           = v4 << 8u;
	uint32_t v17 = (0x400u & v0) | v15;
	v6           = (64u & v7) | v11;
	v7           = v10 >> 4u;
	v11          = (16u & v13) | 0x800u;
	v9           = (0x400u & v0) | v6;
	v15          = 4u & v7;
	v6           = (0x400u & v0) | v14;
	v14          = (0x400u & v0) | v16;
	v16          = (16u & v13) | 0x200u;
	v9           = (v1 ^ v9) + v5;
	uint32_t v18 = (16u & v13) | v15;
	v13          = v10 >> 2u;
	v7           = (v1 ^ v14) + v5;
	v14          = 64u & v10;
	v10          = (64u & v10) | v8;
	v6           = (v1 ^ v6) + v5;
	v18          = (64u & v13) | v18;
	v5           = (v1 ^ v17) + v5;
	v17          = 64u & v13;
	v10          = (0x100u & v3) | v10;
	v13          = 16u | v8 | v14;
	uint32_t v19 = v16 | v15 | v17;
	v11          = v11 | v15 | v17;
	v12          = v12 | v15 | v17;
	v15          = 4u | v8 | v14;
	v10          = (0x400u & v4) | v10;
	v17          = (0x400u & v0) | v18;
	v16          = (0x400u & v0) | v11;
	v12          = (0x400u & v0) | v12;
	v0           = (0x400u & v0) | v19;
	v18          = 20u | v8 | v14;
	v14          = (0x100u & v3) | v15;
	v11          = (v1 ^ v17) + v2;
	v8           = (v1 ^ v16) + v2;
	v0           = (v1 ^ v0) + v2;
	v12          = (v1 ^ v12) + v2;
	v1           = (0x100u & v3) | v13;
	v13          = (0x100u & v3) | v18;

	Invocation result;
	// BUFFER_LOAD_FORMAT_X destinations: v3 <- [v9], v0 <- [v0], v5 <- [v5].
	const uint32_t load_v3 = v9;
	const uint32_t load_v0 = v0;
	const uint32_t load_v5 = v5;
	v1                     = (0x400u & v4) | v1;
	v9                     = (0x400u & v4) | v13;
	v4                     = (0x400u & v4) | v14;
	v13                    = v2 + v1;
	v14                    = v2 + v9;
	v9                     = v2 + v4;
	// v1 <- [v7], v4 <- [v12].
	const uint32_t load_v1 = v7;
	const uint32_t load_v4 = v12;
	v7                     = v2 + v10;
	v10                    = v13 >> 1u;
	v9                     = v9 >> 1u;
	v12                    = v7 >> 1u;
	// v2 <- [v11], v7 <- [v6], v6 <- [v8].
	const uint32_t load_v2 = v11;
	const uint32_t load_v7 = v6;
	const uint32_t load_v6 = v8;
	v11                    = v14 >> 1u;
	// BUFFER_STORE_FORMAT_XY: [v12] <- (v2, v3), [v9] <- (v0, v1), [v10] <- (v6, v7),
	// [v11] <- (v4, v5).
	result.loads  = {load_v2, load_v3, load_v0, load_v1, load_v6, load_v7, load_v4, load_v5};
	result.stores = {v12, v9, v10, v11};
	return result;
}

// Runs the shader over groups_x * groups_y workgroups with the guest's bounds: loads past the
// source's element count read 0 and stores past the destination's are dropped, like structured
// buffer accesses whose index exceeds num_records.
inline void Run(uint32_t pitch, uint32_t groups_x, uint32_t groups_y,
                std::span<const uint8_t> source, std::span<uint8_t> destination) noexcept {
	const auto destination_elements = destination.size() / 2u;
	for (uint32_t y = 0; y < groups_y * GroupHeight; ++y) {
		for (uint32_t x = 0; x < groups_x * GroupWidth; ++x) {
			const auto             invocation = Locate(pitch, x, y);
			std::array<uint8_t, 8> keys {};
			for (size_t i = 0; i < keys.size(); ++i) {
				const auto index = invocation.loads[i];
				keys[i]          = index < source.size() ? source[index] : uint8_t {0};
			}
			for (size_t i = 0; i < invocation.stores.size(); ++i) {
				const auto index = invocation.stores[i];
				if (index < destination_elements) {
					destination[size_t {index} * 2u]      = keys[i * 2u];
					destination[size_t {index} * 2u + 1u] = keys[i * 2u + 1u];
				}
			}
		}
	}
}

} // namespace Libs::Graphics::DccRetile

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCRETILE_H_
