#include "graphics/host_gpu/renderer/pipeline/driverCacheFile.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace DriverCacheFile = Libs::Graphics::DriverCacheFile;
using DriverCacheFile::LoadStatus;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "DriverCacheFileTests: failed: %s\n", text);
    std::abort();
  }
}

std::filesystem::path TempPath(const char *name) {
  auto dir =
      std::filesystem::temp_directory_path() / "kyty_driver_cache_file_tests";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir / name;
}

std::vector<uint8_t> Payload(uint64_t size, uint32_t seed) {
  std::vector<uint8_t> payload(size);
  for (uint64_t i = 0; i < size; i++) {
    payload[i] = static_cast<uint8_t>((i * 131u + seed) >> 3u);
  }
  return payload;
}

constexpr char Signature[] = "KytyPC2:000010de:00002684:00000000:0123\n";

void TestRoundTrip() {
  const auto path = TempPath("cache.bin");
  Check(DriverCacheFile::Load(path, Signature).status == LoadStatus::Missing,
        "a missing file loads as missing");
  // Chunks smaller than every part of the file and the payload, and one
  // larger.
  for (const uint64_t chunk : {uint64_t{1}, uint64_t{7}, uint64_t{4096},
                               DriverCacheFile::DefaultChunkSize}) {
    const auto payload = Payload(10'000 + chunk, static_cast<uint32_t>(chunk));
    Check(DriverCacheFile::Store(path, Signature, payload.data(),
                                 payload.size(), chunk),
          "the file is stored");
    Check(!std::filesystem::exists(path.string() + ".tmp"),
          "the temporary file is renamed");
    Check(std::filesystem::file_size(path) ==
              sizeof(Signature) - 1 + sizeof(uint64_t) + payload.size(),
          "the file holds the signature, the hash and the payload");
    const auto loaded = DriverCacheFile::Load(path, Signature, chunk);
    Check(loaded.status == LoadStatus::Loaded, "the file loads");
    Check(loaded.file_size == std::filesystem::file_size(path),
          "the file size is reported");
    Check(loaded.payload == payload, "the payload round-trips");
  }
  Check(DriverCacheFile::Store(path, Signature, nullptr, 0),
        "an empty payload is stored");
  const auto empty = DriverCacheFile::Load(path, Signature);
  Check(empty.status == LoadStatus::Loaded && empty.payload.empty(),
        "an empty payload loads");
}

void TestCreatesDirectory() {
  const auto path = TempPath("unused").parent_path() / "a" / "b" / "cache.bin";
  const auto payload = Payload(100, 1);
  Check(DriverCacheFile::Store(path, Signature, payload.data(), payload.size()),
        "the directory is created");
  Check(DriverCacheFile::Load(path, Signature).payload == payload,
        "the file in the new directory loads");
}

void TestRejects() {
  const auto path = TempPath("cache.bin");
  const auto payload = Payload(1000, 2);
  Check(DriverCacheFile::Store(path, Signature, payload.data(), payload.size()),
        "the file is stored");
  std::string other = Signature;
  other[10] ^= 1;
  const auto foreign = DriverCacheFile::Load(path, other);
  Check(foreign.status == LoadStatus::SignatureMismatch &&
            foreign.payload.empty(),
        "another driver's file is rejected");

  // A flipped payload byte.
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(static_cast<std::streamoff>(sizeof(Signature) - 1 +
                                           sizeof(uint64_t) + 500));
    file.put('\x5a');
  }
  const auto damaged = DriverCacheFile::Load(path, Signature);
  Check(damaged.status == LoadStatus::DataMismatch && damaged.payload.empty(),
        "a damaged payload is rejected");

  // Shorter than the signature and the hash.
  std::filesystem::resize_file(path, sizeof(Signature) + 2);
  Check(DriverCacheFile::Load(path, Signature).status ==
            LoadStatus::InvalidSize,
        "a truncated header is rejected");
  // A truncated payload no longer matches its hash.
  Check(DriverCacheFile::Store(path, Signature, payload.data(), payload.size()),
        "the file is stored again");
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  Check(DriverCacheFile::Load(path, Signature).status ==
            LoadStatus::DataMismatch,
        "a truncated payload is rejected");
}

void TestFailedStoreKeepsFile() {
  const auto path = TempPath("cache.bin");
  const auto payload = Payload(1000, 3);
  Check(DriverCacheFile::Store(path, Signature, payload.data(), payload.size()),
        "the file is stored");
  // The temporary file cannot be created where a directory has its name.
  std::filesystem::create_directories(path.string() + ".tmp");
  const auto other = Payload(2000, 4);
  Check(!DriverCacheFile::Store(path, Signature, other.data(), other.size()),
        "a store that cannot write its temporary file fails");
  Check(DriverCacheFile::Load(path, Signature).payload == payload,
        "a failed store leaves the previous file");
}

} // namespace

int main() {
  TestRoundTrip();
  TestCreatesDirectory();
  TestRejects();
  TestFailedStoreKeepsFile();
  std::filesystem::remove_all(std::filesystem::temp_directory_path() /
                              "kyty_driver_cache_file_tests");
  std::printf("DriverCacheFileTests: all tests passed\n");
  return 0;
}
