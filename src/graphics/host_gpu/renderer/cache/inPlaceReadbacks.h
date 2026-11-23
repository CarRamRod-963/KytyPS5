#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEREADBACKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEREADBACKS_H_

#include "common/emulatorConfig.h"

#include <cstdint>
#include <span>

// Whether a readback of GPU-written bytes reads a host-readable buffer's mapping (in place) or
// copies the bytes on the GPU into host memory first. Reads of device-local memory through the
// mapping (resizable BAR) run at ~1.5 GB/s on some hosts and at tens of MB/s on others, while a
// GPU copy costs a submission and a wait whatever the host: which one is cheaper depends on the
// host and on the size. A probe at startup times both at a few sizes (BufferCache::
// ProbeInPlaceReadbacks); this decides from those times, without Vulkan.
namespace Libs::Graphics::InPlaceReadbacks {

// The best of a few repetitions of each path for one size.
struct Sample {
	uint64_t bytes       = 0;
	int64_t  in_place_ns = 0; // Reading the bytes through the mapping.
	int64_t  copy_ns     = 0; // Recording a GPU copy, submitting it, waiting, reading host memory.
};

struct Decision {
	// The largest readback read in place; 0 = none (every readback is a GPU copy).
	uint64_t    max_bytes = 0;
	const char* reason    = "";
};

// Off: never in place. On: up to `limit` bytes. Auto: in place up to the size where reading in
// place stops being cheaper than a GPU copy. `samples` are sorted by size; between two sizes both
// costs are taken as linear in the size, so the limit is where the two lines cross; past the
// largest size the limit is `limit`. Without samples (the probe found no host-readable memory
// for its buffer) Auto reads nothing in place: game buffers placed in such memory earlier would
// otherwise be read at an unknown speed, and a GPU copy is never slow.
[[nodiscard]] inline Decision Decide(Config::InPlaceReadbacks mode, std::span<const Sample> samples,
                                     uint64_t limit) {
	if (mode == Config::InPlaceReadbacks::Off) {
		return {0, "off"};
	}
	if (mode == Config::InPlaceReadbacks::On) {
		return {limit, "on"};
	}
	if (samples.empty()) {
		return {0, "auto, not measured: no host-readable memory for the probe"};
	}
	const auto margin = [](const Sample& sample) { return sample.copy_ns - sample.in_place_ns; };
	if (margin(samples.front()) < 0) {
		return {0, "auto, slower than a GPU copy"};
	}
	for (size_t i = 1; i < samples.size(); i++) {
		const auto& below = samples[i - 1];
		const auto& above = samples[i];
		if (above.bytes > limit) {
			break;
		}
		if (margin(above) < 0) {
			// margin(below) >= 0 > margin(above): the lines cross between the two sizes.
			const auto gain  = static_cast<double>(margin(below));
			const auto drop  = gain - static_cast<double>(margin(above));
			const auto cross = static_cast<double>(below.bytes) +
			                   static_cast<double>(above.bytes - below.bytes) * gain / drop;
			return {static_cast<uint64_t>(cross), "auto, up to where a GPU copy is faster"};
		}
	}
	return {limit, "auto, faster than a GPU copy"};
}

} // namespace Libs::Graphics::InPlaceReadbacks

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEREADBACKS_H_ */
