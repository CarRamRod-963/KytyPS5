#include "graphics/shader/recompiler/backend/spirv/SpirvSimplifiedStore.h"

#include <array>
#include <cstring>
#include <system_error>
#include <xxhash.h>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

constexpr std::array<char, 8> Magic = {'K', 'y', 'S', 'p', 'v', 'S', 'i', '1'};
// A larger file starts over: it is read whole at every start, and a title's modules of one build
// take ~100 MB at most.
constexpr uintmax_t MaxFileSize = uintmax_t {512} << 20u;

struct RecordHeader {
	uint64_t low      = 0;
	uint64_t high     = 0;
	uint64_t count    = 0;
	uint64_t checksum = 0;
};
static_assert(sizeof(RecordHeader) == 32);

uint64_t Checksum(const uint32_t* words, uint64_t count) {
	return XXH3_64bits(words, count * sizeof(uint32_t));
}

} // namespace

SimplifiedStore::Key SimplifiedStore::MakeKey(std::span<const uint32_t> words, uint64_t variant) {
	const auto hash = XXH3_128bits_withSeed(words.data(), words.size_bytes(), variant);
	return {hash.low64, hash.high64};
}

SimplifiedStore::SimplifiedStore(std::filesystem::path path, std::string build,
                                 std::chrono::milliseconds flush_interval)
    : m_path(std::move(path)), m_build(std::move(build)), m_flush_interval(flush_interval) {
	Open();
	if (m_file.is_open()) {
		m_writer = std::jthread([this](std::stop_token stop) { WriterThread(stop); });
	}
}

SimplifiedStore::~SimplifiedStore() {
	if (m_writer.joinable()) {
		m_writer.request_stop();
		m_wake.notify_all();
		m_writer.join();
	}
	Flush();
}

void SimplifiedStore::Open() {
	std::error_code error;
	const auto      size      = std::filesystem::exists(m_path, error)
	                                ? std::filesystem::file_size(m_path, error)
	                                : uintmax_t {0};
	uint64_t        valid_end = 0;
	if (!error && size > 0 && size <= MaxFileSize) {
		std::ifstream                  in(m_path, std::ios::binary);
		std::array<char, Magic.size()> magic {};
		uint32_t                       build_size = 0;
		std::string                    build;
		if (in.read(magic.data(), magic.size()) && magic == Magic &&
		    in.read(reinterpret_cast<char*>(&build_size), sizeof(build_size)) &&
		    build_size == m_build.size()) {
			build.resize(build_size);
			if (in.read(build.data(), build_size) && build == m_build) {
				valid_end = magic.size() + sizeof(build_size) + build_size;
			}
		}
		// Records up to the first one that is incomplete or damaged (an interrupted append).
		while (valid_end != 0) {
			RecordHeader header;
			if (!in.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
			    header.count > (size - valid_end - sizeof(header)) / sizeof(uint32_t)) {
				break;
			}
			std::vector<uint32_t> words(header.count);
			if (!in.read(reinterpret_cast<char*>(words.data()),
			             static_cast<std::streamsize>(header.count * sizeof(uint32_t))) ||
			    Checksum(words.data(), header.count) != header.checksum) {
				break;
			}
			m_records.try_emplace(Key {header.low, header.high}, std::move(words));
			valid_end += sizeof(header) + header.count * sizeof(uint32_t);
		}
	}
	m_loaded = m_records.size();
	if (valid_end != 0) {
		if (valid_end != size) {
			std::filesystem::resize_file(m_path, valid_end, error);
		}
		if (!error) {
			m_file.open(m_path, std::ios::binary | std::ios::app);
		}
		return;
	}
	// Missing, another build's or unreadable: start a new file.
	std::filesystem::create_directories(m_path.parent_path(), error);
	m_file.open(m_path, std::ios::binary | std::ios::trunc);
	const auto build_size = static_cast<uint32_t>(m_build.size());
	if (!m_file.write(Magic.data(), Magic.size()) ||
	    !m_file.write(reinterpret_cast<const char*>(&build_size), sizeof(build_size)) ||
	    !m_file.write(m_build.data(), build_size) || !m_file.flush()) {
		m_file.close();
	}
}

bool SimplifiedStore::Find(const Key& key, std::vector<uint32_t>& words) const {
	std::lock_guard lock(m_mutex);
	const auto      it = m_records.find(key);
	if (it == m_records.end()) {
		return false;
	}
	words = it->second;
	return true;
}

void SimplifiedStore::Insert(const Key& key, std::vector<uint32_t> words) {
	{
		std::lock_guard lock(m_mutex);
		if (!m_records.try_emplace(key, std::move(words)).second || !m_file.is_open()) {
			return;
		}
		m_pending.push_back(key);
	}
	m_wake.notify_all();
}

void SimplifiedStore::Flush() {
	std::lock_guard                                           file_lock(m_file_mutex);
	std::vector<std::pair<Key, const std::vector<uint32_t>*>> records;
	{
		std::lock_guard lock(m_mutex);
		// Records are never changed or erased once inserted, and map nodes do not move.
		for (const auto& key: m_pending) {
			records.emplace_back(key, &m_records.at(key));
		}
		m_pending.clear();
	}
	if (records.empty() || !m_file.is_open()) {
		return;
	}
	for (const auto& [key, words]: records) {
		const RecordHeader header {key.low, key.high, words->size(),
		                           Checksum(words->data(), words->size())};
		m_file.write(reinterpret_cast<const char*>(&header), sizeof(header));
		m_file.write(reinterpret_cast<const char*>(words->data()),
		             static_cast<std::streamsize>(words->size() * sizeof(uint32_t)));
	}
	if (!m_file.flush()) {
		m_file.close();
	}
}

size_t SimplifiedStore::Size() const {
	std::lock_guard lock(m_mutex);
	return m_records.size();
}

void SimplifiedStore::WriterThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		{
			std::unique_lock lock(m_mutex);
			if (!m_wake.wait(lock, stop, [this] { return !m_pending.empty(); })) {
				return;
			}
			// Gather what arrives close together (a burst of new programs) into one append.
			(void)m_wake.wait_for(lock, stop, m_flush_interval, [] { return false; });
		}
		Flush();
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
