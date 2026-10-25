#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEFILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEFILE_H_

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace Libs::Graphics::DriverCacheFile {

// The file of a title's Vulkan pipeline cache: the signature line, the XXH3-64 hash of the
// payload, then the payload (vkGetPipelineCacheData). Sizes are 64-bit: a driver cache can pass
// 4 GiB (Ghost of Yotei's NVIDIA cache does). Vulkan-free so that it can be tested without a
// device.

// Reads and writes go through the stream in pieces of at most this many bytes.
inline constexpr uint64_t DefaultChunkSize = uint64_t {1} << 30u;
// --precompile-only rebuilds a cache file larger than this from the pipelines it builds: the
// driver keeps every pipeline it ever saw, also those of shader code no build emits anymore.
inline constexpr uint64_t RebuildThreshold = uint64_t {3} << 30u;

enum class LoadStatus {
	Missing,
	Loaded,
	// Too short for the signature and the hash.
	InvalidSize,
	ReadFailed,
	// Another driver or device wrote it.
	SignatureMismatch,
	// The payload does not match its hash (a torn or damaged file).
	DataMismatch,
};

struct Loaded {
	LoadStatus           status    = LoadStatus::Missing;
	uint64_t             file_size = 0;
	std::vector<uint8_t> payload;
};

[[nodiscard]] Loaded Load(const std::filesystem::path& path, std::string_view signature,
                          uint64_t chunk_size = DefaultChunkSize);

// Writes the file through `<path>.tmp` and renames it over `path`, so that a failed or
// interrupted write leaves the previous file. Creates the directory.
[[nodiscard]] bool Store(const std::filesystem::path& path, std::string_view signature,
                         const uint8_t* payload, uint64_t payload_size,
                         uint64_t chunk_size = DefaultChunkSize);

// Whether --precompile-only starts from an empty driver cache instead of the file's
// (`requested` = --rebuild-pipeline-cache).
[[nodiscard]] constexpr bool RebuildOnPrecompile(bool requested, uint64_t file_size) {
	return requested || file_size > RebuildThreshold;
}

} // namespace Libs::Graphics::DriverCacheFile

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEFILE_H_ */
