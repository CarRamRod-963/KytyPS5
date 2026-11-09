#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer Input { uint data[]; } input_buffer;
layout(set = 0, binding = 0, std430) readonly buffer Input8 { uint8_t data[]; } input_buffer8;
layout(set = 0, binding = 0, std430) readonly buffer Input16 { uint16_t data[]; } input_buffer16;
#if defined(TILER_IMAGE_3D)
// Detiles straight into a volume image through an unsigned view of its texel size.
#define TILER_IMAGE
layout(set = 0, binding = 1) uniform writeonly uimage3D output_image;
#elif defined(TILER_IMAGE_2D)
// Detiles straight into the layers of a 2D image through an unsigned view of its texel size.
#define TILER_IMAGE
layout(set = 0, binding = 1) uniform writeonly uimage2DArray output_image;
#else
layout(set = 0, binding = 1, std430) buffer Output { uint data[]; } output_buffer;
layout(set = 0, binding = 1, std430) buffer Output8 { uint8_t data[]; } output_buffer8;
layout(set = 0, binding = 1, std430) buffer Output16 { uint16_t data[]; } output_buffer16;
#endif

struct Mip {
	uint src_base;
	uint dst_base;
	uint width;
	uint height;
	uint pitch_bytes;
	uint slice_bytes;
	uint tiled_slice_bytes;
	uint blocks_per_row;
	uint surface_z;
	uint tail_x;
	uint tail_y;
	uint num_texels;
};

layout(set = 0, binding = 2, std140) uniform TilingInfo {
	uint num_texels;
	uint color_transform;
	Mip mips[16];
} info;

#include "gpu_tiler_color.inc"

#ifdef TILER_IMAGE
// The texel's bytes, as the copy from a linear buffer would place them: one unsigned component
// per 32 bits (one for 8- and 16-bit texels).
void store_element(uint src, uvec3 p) {
	uvec4 value = uvec4(0u);
	if (ELEMENT_BYTES == 1u) {
		value.x = uint(input_buffer8.data[src]);
	} else if (ELEMENT_BYTES == 2u) {
		value.x = uint(input_buffer16.data[src >> 1]);
	} else {
		value.x = input_buffer.data[src >> 2];
		if (ELEMENT_BYTES >= 8u) value.y = input_buffer.data[(src >> 2) + 1u];
		if (ELEMENT_BYTES >= 16u) {
			value.z = input_buffer.data[(src >> 2) + 2u];
			value.w = input_buffer.data[(src >> 2) + 3u];
		}
	}
	imageStore(output_image, ivec3(p), value);
}
#else
void copy_element(uint src, uint dst) {
	if ((ELEMENT_BYTES == 4u || ELEMENT_BYTES == 8u) && info.color_transform != 0u) {
		uvec2 value = uvec2(input_buffer.data[src >> 2], 0u);
		if (ELEMENT_BYTES == 8u) value.y = input_buffer.data[(src >> 2) + 1u];
		value = transform_color(value, info.color_transform, TILE == 0u);
		output_buffer.data[dst >> 2] = value.x;
		if (ELEMENT_BYTES == 8u) output_buffer.data[(dst >> 2) + 1u] = value.y;
	} else if (ELEMENT_BYTES >= 4) {
		for (uint i = 0; i < ELEMENT_BYTES; i += 4) {
			output_buffer.data[(dst + i) >> 2] = input_buffer.data[(src + i) >> 2];
		}
	} else if (ELEMENT_BYTES == 2) {
		output_buffer16.data[dst >> 1] = input_buffer16.data[src >> 1];
	} else {
		output_buffer8.data[dst] = input_buffer8.data[src];
	}
}
#endif

void main() {
	uint group = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
	// Reject padding groups before multiplying: the active range may approach 4 GiB.
	if (group > (info.num_texels - 1u) / 64u) {
		return;
	}
	uint texel = group * 64u + gl_LocalInvocationID.x;
	if (texel >= info.num_texels) {
		return;
	}
	uint mip = 0u;
	uint mip_texels = info.mips[mip].num_texels;
	while (texel >= mip_texels) {
		texel -= mip_texels;
		mip_texels = info.mips[++mip].num_texels;
	}
	Mip params = info.mips[mip];
	uint row = texel / params.width;
	uint z = row / params.height;
	uvec3 p = uvec3(texel - row * params.width, row - z * params.height, z);

	uvec3 extent = block_extent();
	uvec3 block = p / extent;
	uvec3 swizzle = p + uvec3(params.tail_x, params.tail_y, params.surface_z);

	uint block_index = block.y * params.blocks_per_row + block.x;
	uint tiled = block.z * params.tiled_slice_bytes + block_index * BLOCK_BYTES +
	             block_offset(swizzle);
#ifdef TILER_IMAGE
	// One level at its origin: the texel lands at its coordinate (the layer or depth in z).
	store_element(params.src_base + tiled, p);
#else
	uint linear = p.z * params.slice_bytes + p.y * params.pitch_bytes +
	              p.x * ELEMENT_BYTES;
	if (TILE == 0) {
		copy_element(params.src_base + tiled, params.dst_base + linear);
	} else {
		copy_element(params.src_base + linear, params.dst_base + tiled);
	}
#endif
}
