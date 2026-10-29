#include "graphics/host_gpu/renderer/pipeline/driverCacheFile.h"
#include "graphics/host_gpu/renderer/pipeline/driverCacheHistory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Libs::Graphics::DriverCacheHistory;
namespace DriverCacheFile = Libs::Graphics::DriverCacheFile;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "DriverCacheHistoryTests: failed: %s\n", text);
    std::abort();
  }
}

std::filesystem::path TempDir() {
  auto dir = std::filesystem::temp_directory_path() /
             "kyty_driver_cache_history_tests";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

constexpr char DriverSignature[] = "KytyPC2:000010de:00002684:00000000:0123\n";

void TestBuiltAndMissed() {
  DriverCacheHistory history;
  Check(!history.Missed(0x1234), "a pipeline never given to the cache is not "
                                 "stale");
  Check(history.StaleCount() == 0, "nothing is stale yet");
  history.Built(0x1234);
  history.Built(0x1234);
  history.Built(0);
  Check(history.Size() == 1, "a pipeline is remembered once, zero never");
  Check(history.Missed(0x1234), "a miss of a pipeline the cache was given is "
                                "stale");
  Check(history.Missed(0x1234), "every such miss counts");
  Check(!history.Missed(0x5678), "another pipeline is not stale");
  Check(!history.Missed(0), "zero is never stale");
  Check(history.StaleCount() == 2, "two stale misses");
  history.Clear();
  Check(history.Size() == 0 && history.StaleCount() == 0,
        "Clear forgets the pipelines and the stale misses");
  Check(!history.Missed(0x1234), "a cleared history has nothing stale");
}

void TestSnapshotIsSorted() {
  DriverCacheHistory history;
  for (const uint64_t fingerprint : {uint64_t{9}, uint64_t{3}, uint64_t{7}}) {
    history.Built(fingerprint);
  }
  Check(history.Snapshot() == std::vector<uint64_t>{3, 7, 9},
        "the snapshot holds every fingerprint, sorted");
}

void TestRoundTrip() {
  const auto dir = TempDir();
  const auto path = DriverCacheHistory::PathFor(dir, "PPSA00000");
  Check(path.filename() == "PPSA00000.v1.built", "the file is named after the "
                                                 "title");
  const auto signature = DriverCacheHistory::Signature(DriverSignature, 0xabc);
  DriverCacheHistory history;
  Check(!history.Load(path, signature), "a missing file does not load");
  history.Built(0x1111);
  history.Built(0x2222);
  Check(DriverCacheHistory::Store(path, signature, history.Snapshot(), 3),
        "the history is stored");
  Check(!std::filesystem::exists(path.string() + ".tmp"),
        "the temporary file is renamed");

  DriverCacheHistory loaded;
  loaded.Built(0x9999);
  Check(loaded.Load(path, signature), "the history loads");
  Check(loaded.Snapshot() == std::vector<uint64_t>{0x1111, 0x2222},
        "the loaded history replaces the previous one");
  Check(loaded.StaleCount() == 3 && loaded.SessionStaleCount() == 0,
        "the stale misses of earlier sessions load");
  Check(loaded.Missed(0x2222), "a loaded pipeline is stale when missed");
  Check(loaded.StaleCount() == 4 && loaded.SessionStaleCount() == 1,
        "this session's stale misses add to them");
  loaded.Clear();
  Check(loaded.StaleCount() == 0, "Clear (a rebuilt cache) forgets them");

  // Another cache file (another payload hash) or another driver: a new history.
  DriverCacheHistory other;
  Check(
      !other.Load(path, DriverCacheHistory::Signature(DriverSignature, 0xabd)),
      "the history of another cache file does not load");
  Check(other.Size() == 0, "a history that did not load is empty");
  Check(!other.Load(path,
                    DriverCacheHistory::Signature(
                        "KytyPC2:000010de:00002684:00000001:0123\n", 0xabc)),
        "the history of another driver does not load");

  // A damaged file is ignored.
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(-1, std::ios::end);
    file.put('\x7f');
  }
  Check(!other.Load(path, signature) && other.Size() == 0,
        "a damaged history does not load");
}

void TestCacheFilePayloadHash() {
  const auto dir = TempDir();
  const auto path = dir / "cache.bin";
  const std::vector<uint8_t> payload{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t hash = 0;
  Check(DriverCacheFile::Store(path, DriverSignature, payload.data(),
                               payload.size(),
                               DriverCacheFile::DefaultChunkSize, &hash),
        "the cache file is stored");
  const auto loaded = DriverCacheFile::Load(path, DriverSignature);
  Check(loaded.status == DriverCacheFile::LoadStatus::Loaded,
        "the cache file loads");
  Check(hash != 0 && loaded.payload_hash == hash,
        "the stored and the loaded payload hash agree");
}

} // namespace

int main() {
  TestBuiltAndMissed();
  TestSnapshotIsSorted();
  TestRoundTrip();
  TestCacheFilePayloadHash();
  std::printf("DriverCacheHistoryTests: ok\n");
  return 0;
}
