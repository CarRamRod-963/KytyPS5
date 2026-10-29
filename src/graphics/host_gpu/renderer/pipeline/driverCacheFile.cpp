#include "graphics/host_gpu/renderer/pipeline/driverCacheFile.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <system_error>
#include <xxhash.h>

namespace Libs::Graphics::DriverCacheFile {

namespace {

bool ReadAll(std::ifstream& file, char* data, uint64_t size, uint64_t chunk_size) {
	for (uint64_t done = 0; done < size;) {
		const auto piece = std::min(size - done, chunk_size);
		if (!file.read(data + done, static_cast<std::streamsize>(piece))) {
			return false;
		}
		done += piece;
	}
	return true;
}

bool WriteAll(std::ofstream& file, const char* data, uint64_t size, uint64_t chunk_size) {
	for (uint64_t done = 0; done < size;) {
		const auto piece = std::min(size - done, chunk_size);
		if (!file.write(data + done, static_cast<std::streamsize>(piece))) {
			return false;
		}
		done += piece;
	}
	return true;
}

} // namespace

Loaded Load(const std::filesystem::path& path, std::string_view signature, uint64_t chunk_size) {
	Loaded          loaded;
	std::error_code error;
	const auto      size = std::filesystem::file_size(path, error);
	if (error) {
		loaded.status =
		    std::filesystem::exists(path, error) ? LoadStatus::ReadFailed : LoadStatus::Missing;
		return loaded;
	}
	loaded.file_size      = size;
	const uint64_t header = signature.size() + sizeof(uint64_t);
	chunk_size            = std::max<uint64_t>(chunk_size, 1);
	if (size < header) {
		loaded.status = LoadStatus::InvalidSize;
		return loaded;
	}
	std::ifstream file(path, std::ios::binary);
	std::string   cached_signature(signature.size(), '\0');
	uint64_t      hash = 0;
	if (!file || !ReadAll(file, cached_signature.data(), cached_signature.size(), chunk_size) ||
	    !ReadAll(file, reinterpret_cast<char*>(&hash), sizeof(hash), chunk_size)) {
		loaded.status = LoadStatus::ReadFailed;
		return loaded;
	}
	if (cached_signature != signature) {
		loaded.status = LoadStatus::SignatureMismatch;
		return loaded;
	}
	loaded.payload.resize(size - header);
	if (!ReadAll(file, reinterpret_cast<char*>(loaded.payload.data()), loaded.payload.size(),
	             chunk_size)) {
		loaded.payload = {};
		loaded.status  = LoadStatus::ReadFailed;
		return loaded;
	}
	if (XXH3_64bits(loaded.payload.data(), loaded.payload.size()) != hash) {
		loaded.payload = {};
		loaded.status  = LoadStatus::DataMismatch;
		return loaded;
	}
	loaded.status       = LoadStatus::Loaded;
	loaded.payload_hash = hash;
	return loaded;
}

bool Store(const std::filesystem::path& path, std::string_view signature, const uint8_t* payload,
           uint64_t payload_size, uint64_t chunk_size, uint64_t* payload_hash) {
	std::error_code error;
	if (path.has_parent_path()) {
		std::filesystem::create_directories(path.parent_path(), error);
		if (error) {
			return false;
		}
	}
	chunk_size      = std::max<uint64_t>(chunk_size, 1);
	const auto hash = XXH3_64bits(payload, payload_size);
	if (payload_hash != nullptr) {
		*payload_hash = hash;
	}
	auto temp = path;
	temp += ".tmp";
	bool written = false;
	{
		std::ofstream file(temp, std::ios::binary | std::ios::trunc);
		written = file && WriteAll(file, signature.data(), signature.size(), chunk_size) &&
		          WriteAll(file, reinterpret_cast<const char*>(&hash), sizeof(hash), chunk_size) &&
		          WriteAll(file, reinterpret_cast<const char*>(payload), payload_size, chunk_size);
		file.close();
		written = written && !file.fail();
	}
	if (written) {
		std::filesystem::rename(temp, path, error);
		written = !error;
	}
	if (!written) {
		std::filesystem::remove(temp, error);
	}
	return written;
}

} // namespace Libs::Graphics::DriverCacheFile
