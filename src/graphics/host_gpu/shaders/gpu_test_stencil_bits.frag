#version 450

// Runs behind a stencil test that compares one bit: every fragment that passes sets that bit
// in its pixel's word, so one draw per bit reads the whole stencil value back.
layout(early_fragment_tests) in;

layout(set = 0, binding = 0, std430) buffer Output {
	uint value[];
}
output_data;

layout(push_constant) uniform Push {
	uint width;
	uint bit;
}
push;

void main() {
	const uvec2 pixel = uvec2(gl_FragCoord.xy);
	atomicOr(output_data.value[pixel.y * push.width + pixel.x], 1u << push.bit);
}
