#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MANIFESTFORMAT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MANIFESTFORMAT_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <type_traits>
#include <vector>

// The building blocks of the pipeline cache's persisted records: field-by-field little-endian
// encoding, and files of checksummed entries. Every stored struct is written field by field, never as raw
// bytes, so that a struct that grows does not change a format by accident.
namespace Libs::Graphics::ManifestFormat {

class Writer {
public:
	explicit Writer(std::vector<uint8_t>& out): m_out(out) {}
	template <typename T>
	void Put(T value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto offset = m_out.size();
		m_out.resize(offset + sizeof(T));
		std::memcpy(m_out.data() + offset, &value, sizeof(T));
	}
	void PutBool(bool value) { Put<uint8_t>(value ? 1u : 0u); }
	void PutWords(std::span<const uint32_t> words) {
		const auto offset = m_out.size();
		m_out.resize(offset + words.size_bytes());
		if (!words.empty()) {
			std::memcpy(m_out.data() + offset, words.data(), words.size_bytes());
		}
	}
	template <typename Enum>
	void PutEnum(Enum value) {
		Put(static_cast<uint32_t>(value));
	}

private:
	std::vector<uint8_t>& m_out;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> data): m_data(data) {}
	template <typename T>
	bool Get(T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (m_data.size() - m_offset < sizeof(T)) {
			return false;
		}
		std::memcpy(&value, m_data.data() + m_offset, sizeof(T));
		m_offset += sizeof(T);
		return true;
	}
	bool GetBool(bool& value) {
		uint8_t raw = 0;
		if (!Get(raw) || raw > 1u) {
			return false;
		}
		value = raw != 0u;
		return true;
	}
	bool GetWords(std::vector<uint32_t>& words, uint32_t count) {
		if ((m_data.size() - m_offset) / sizeof(uint32_t) < count) {
			return false;
		}
		words.resize(count);
		if (count != 0u) {
			std::memcpy(words.data(), m_data.data() + m_offset, count * sizeof(uint32_t));
		}
		m_offset += count * sizeof(uint32_t);
		return true;
	}
	// Reads an enum stored as u32 whose valid values are 0..max.
	template <typename Enum>
	bool GetEnum(Enum& value, uint32_t max = UINT32_MAX) {
		uint32_t raw = 0;
		if (!Get(raw) || raw > max) {
			return false;
		}
		value = static_cast<Enum>(raw);
		return true;
	}
	[[nodiscard]] bool AtEnd() const { return m_offset == m_data.size(); }

private:
	std::span<const uint8_t> m_data;
	size_t                   m_offset = 0;
};

// Bounds that no real record reaches; a larger count is corruption.
constexpr uint32_t MaxResources   = 1u << 16u;
constexpr uint32_t MaxPayloadSize = 64u << 20u;

// The fields a specialization stored in a given format had: key-map header flags and image heap
// flags were added to the program manifest in its versions 3 and 4.
struct SpecializationFields {
	bool mapping_header = true;
	bool image_heap     = true;
};

void PutSpecialization(Writer&                                             w,
                       const ShaderRecompiler::IR::ResourceSpecialization& specialization);
bool GetSpecialization(Reader& r, ShaderRecompiler::IR::ResourceSpecialization& specialization,
                       SpecializationFields fields = {});

// A file of entries: an 8-byte magic, a u32 format version, then entries of a u32 payload size,
// the payload's XXH3-64 and the payload.
using Magic = std::array<char, 8>;

std::vector<uint8_t> EntryFileHeader(const Magic& magic, uint32_t version);
// Appends the framed payload to `out`.
void FrameEntry(std::vector<uint8_t>& out, std::span<const uint8_t> payload);

struct EntryFileRead {
	// The file exists and starts with the magic; `version` is then its format version.
	bool     valid   = false;
	uint32_t version = 0;
	size_t   size    = 0;
	// Bytes up to the end of the last entry that was intact and accepted.
	size_t             intact = 0;
	[[nodiscard]] bool Clean() const { return valid && intact == size; }
};
// Reads a file of entries and passes each intact payload to `parse` in file order, stopping at
// the first torn, corrupt or rejected (parse returned false) entry. A missing file reads as
// empty and invalid.
EntryFileRead ReadEntryFile(const std::filesystem::path& path, const Magic& magic,
                            const std::function<bool(std::span<const uint8_t>, uint32_t)>& parse);
// Replaces `path` by a file with the given entries, through a temporary file.
bool WriteEntryFile(const std::filesystem::path& path, const Magic& magic, uint32_t version,
                    std::span<const std::vector<uint8_t>> payloads);
// Opens `path` for appending entries; each one is flushed, so a session that ends in a crash
// keeps what it appended. Not thread-safe.
class EntryAppender {
public:
	bool Open(const std::filesystem::path& path) {
		m_file.open(path, std::ios::binary | std::ios::app);
		return m_file.is_open();
	}
	[[nodiscard]] bool IsOpen() const { return m_file.is_open(); }
	void               Append(std::span<const uint8_t> payload);

private:
	std::ofstream        m_file;
	std::vector<uint8_t> m_buffer;
};

} // namespace Libs::Graphics::ManifestFormat

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MANIFESTFORMAT_H_
