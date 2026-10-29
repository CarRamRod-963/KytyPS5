#include "graphics/host_gpu/renderer/pipeline/driverCacheHistory.h"

#include "graphics/host_gpu/renderer/pipeline/driverCacheFile.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace Libs::Graphics {

// The file's payload: the stale miss count, then the fingerprints (64-bit words).

std::filesystem::path DriverCacheHistory::PathFor(const std::filesystem::path& directory,
                                                  const std::string&           title_id) {
	return directory / (title_id + ".v1.built");
}

std::string DriverCacheHistory::Signature(std::string_view driver_cache_signature,
                                          uint64_t         cache_payload_hash) {
	char prefix[32];
	std::snprintf(prefix, sizeof(prefix), "KytyBuilt1:%016" PRIx64 ":", cache_payload_hash);
	return prefix + std::string(driver_cache_signature);
}

bool DriverCacheHistory::Load(const std::filesystem::path& path, std::string_view signature) {
	Clear();
	const auto loaded = DriverCacheFile::Load(path, signature);
	if (loaded.status != DriverCacheFile::LoadStatus::Loaded || loaded.payload.empty() ||
	    loaded.payload.size() % sizeof(uint64_t) != 0) {
		return false;
	}
	const auto      words = loaded.payload.size() / sizeof(uint64_t);
	std::lock_guard lock(m_mutex);
	std::memcpy(&m_loaded_stale, loaded.payload.data(), sizeof(uint64_t));
	m_built.reserve(words - 1);
	for (size_t i = 1; i < words; i++) {
		uint64_t fingerprint = 0;
		std::memcpy(&fingerprint, loaded.payload.data() + i * sizeof(uint64_t),
		            sizeof(fingerprint));
		if (fingerprint != 0) {
			m_built.insert(fingerprint);
		}
	}
	return true;
}

bool DriverCacheHistory::Store(const std::filesystem::path& path, std::string_view signature,
                               const std::vector<uint64_t>& fingerprints, uint64_t stale_count) {
	std::vector<uint64_t> words;
	words.reserve(fingerprints.size() + 1);
	words.push_back(stale_count);
	words.insert(words.end(), fingerprints.begin(), fingerprints.end());
	return DriverCacheFile::Store(path, signature, reinterpret_cast<const uint8_t*>(words.data()),
	                              words.size() * sizeof(uint64_t));
}

void DriverCacheHistory::Clear() {
	std::lock_guard lock(m_mutex);
	m_built.clear();
	m_loaded_stale = 0;
	m_stale.store(0);
}

void DriverCacheHistory::Built(uint64_t fingerprint) {
	if (fingerprint == 0) {
		return;
	}
	std::lock_guard lock(m_mutex);
	m_built.insert(fingerprint);
}

bool DriverCacheHistory::Missed(uint64_t fingerprint) {
	if (fingerprint == 0) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	if (!m_built.contains(fingerprint)) {
		return false;
	}
	m_stale.fetch_add(1);
	return true;
}

uint64_t DriverCacheHistory::StaleCount() const {
	std::lock_guard lock(m_mutex);
	return m_loaded_stale + m_stale.load();
}

size_t DriverCacheHistory::Size() const {
	std::lock_guard lock(m_mutex);
	return m_built.size();
}

std::vector<uint64_t> DriverCacheHistory::Snapshot() const {
	std::vector<uint64_t> fingerprints;
	{
		std::lock_guard lock(m_mutex);
		fingerprints.assign(m_built.begin(), m_built.end());
	}
	std::ranges::sort(fingerprints);
	return fingerprints;
}

} // namespace Libs::Graphics
