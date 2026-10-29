#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEHISTORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEHISTORY_H_

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// The pipelines a title's driver cache file was given, by create-info fingerprint
// (_PipelineCache/<title>.v1.built, written with the file), and the stale misses sessions found.
// A cache probe that misses one of those pipelines finds a stale entry: the driver keeps an entry
// it does not provide, and building the pipeline again does not replace it (NVIDIA writes entries
// it took from its own disk cache without the compression dictionary they need, and they fail
// to load in a process whose disk cache lacks it). Such a pipeline compiles in every session
// until the cache is rebuilt, which --precompile-only does once a stale miss was seen.
// Thread-safe; Vulkan-free.
class DriverCacheHistory {
public:
	[[nodiscard]] static std::filesystem::path PathFor(const std::filesystem::path& directory,
	                                                   const std::string&           title_id);
	// The file's signature: the driver cache's signature and the hash of the cache data the
	// history describes (DriverCacheFile::Loaded::payload_hash), so that another driver or a
	// cache file written without the history (or restored) starts a new history.
	[[nodiscard]] static std::string Signature(std::string_view driver_cache_signature,
	                                           uint64_t         cache_payload_hash);

	// Replaces the history with the file's; a missing, damaged or foreign file leaves it empty.
	// Returns whether the file was read.
	bool Load(const std::filesystem::path& path, std::string_view signature);
	// Writes `fingerprints` (a Snapshot) and the stale misses through a temporary file.
	[[nodiscard]] static bool Store(const std::filesystem::path& path, std::string_view signature,
	                                const std::vector<uint64_t>& fingerprints,
	                                uint64_t                     stale_count);

	// The cache starts empty: nothing was given to it.
	void Clear();
	// A build through the cache finished: the driver compiled the pipeline into it or provided
	// it. Zero (no fingerprint) is ignored.
	void Built(uint64_t fingerprint);
	// A cache probe missed. Returns whether the cache was given the pipeline before (a stale
	// entry), and counts it.
	bool Missed(uint64_t fingerprint);

	// Stale misses of this process and of the sessions since the file was rebuilt.
	[[nodiscard]] uint64_t StaleCount() const;
	[[nodiscard]] uint64_t SessionStaleCount() const { return m_stale.load(); }
	[[nodiscard]] size_t   Size() const;
	// The fingerprints, sorted. Taken before the cache's data, it names only pipelines the data
	// holds.
	[[nodiscard]] std::vector<uint64_t> Snapshot() const;

private:
	mutable std::mutex           m_mutex;
	std::unordered_set<uint64_t> m_built;
	uint64_t                     m_loaded_stale = 0;
	std::atomic<uint64_t>        m_stale {0};
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEHISTORY_H_ */
