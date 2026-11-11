#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVSIMPLIFIEDSTORE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVSIMPLIFIEDSTORE_H_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// SimplifyForDriver results remembered across sessions. Every session translates each program at
// its first use, and the driver simplification is most of a large module's emission time (Saros:
// ~3.2 of ~3.9 s per session), while its output depends only on the module it is given (and the
// simplifier's options): a store maps the hash of an emitted module to its simplified words.
//
// The file starts with a header naming the build that wrote it; a file of another build (whose
// simplifier may differ) is replaced. Records are appended by a writer thread a few seconds after
// they are inserted, so a session that is killed keeps what it wrote until then; a torn last record
// is dropped when the file is opened. Vulkan-free so that it can be tested without a device.
class SimplifiedStore {
public:
	struct Key {
		uint64_t low                          = 0;
		uint64_t high                         = 0;
		bool     operator==(const Key&) const = default;
	};

	// The key of an emitted module; `variant` tells apart the simplifier options in effect.
	[[nodiscard]] static Key MakeKey(std::span<const uint32_t> words, uint64_t variant);

	// Opens (or creates) the file at `path` for the build `build`. A path that cannot be written
	// leaves a store that only remembers in memory.
	SimplifiedStore(std::filesystem::path path, std::string build,
	                std::chrono::milliseconds flush_interval = std::chrono::seconds {2});
	~SimplifiedStore();
	SimplifiedStore(const SimplifiedStore&)            = delete;
	SimplifiedStore& operator=(const SimplifiedStore&) = delete;

	// Copies the simplified words of `key` into `words`; false if none are known.
	[[nodiscard]] bool Find(const Key& key, std::vector<uint32_t>& words) const;
	// Remembers the simplified words of `key`; the writer thread appends them to the file.
	void Insert(const Key& key, std::vector<uint32_t> words);
	// Appends every record not written yet (the writer thread calls it, the destructor too).
	void Flush();

	[[nodiscard]] size_t Size() const;
	// Records the open read from the file.
	[[nodiscard]] size_t Loaded() const { return m_loaded; }

private:
	struct KeyHash {
		size_t operator()(const Key& key) const { return static_cast<size_t>(key.low); }
	};

	void Open();
	void WriterThread(std::stop_token stop);

	std::filesystem::path                                   m_path;
	std::string                                             m_build;
	std::chrono::milliseconds                               m_flush_interval;
	mutable std::mutex                                      m_mutex;
	std::unordered_map<Key, std::vector<uint32_t>, KeyHash> m_records;
	std::vector<Key>                                        m_pending;
	size_t                                                  m_loaded = 0;
	std::mutex                                              m_file_mutex;
	std::ofstream                                           m_file;
	std::condition_variable_any                             m_wake;
	std::jthread                                            m_writer;
};

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVSIMPLIFIEDSTORE_H_ */
