#include "graphics/host_gpu/renderer/pipeline/scalarReadCache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace {

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "ScalarReadCacheTests: failed: %s\n", message);
    std::abort();
  }
}

// Guest memory as the GPU thread sees it: the backing, and a "GPU copy" holding
// the bytes of recorded GPU writes until a readback publishes them to the
// backing.
struct FakeGuestMemory {
  static constexpr uint64_t Base = 0x10000;
  static constexpr uint64_t Size = 0x1000;

  std::array<uint8_t, Size> backing{};
  std::array<uint8_t, Size> gpu{};
  std::vector<std::pair<uint64_t, uint64_t>> dirty;
  // Every range asked for, and the number of calls (one wait each).
  std::vector<std::pair<uint64_t, uint64_t>> readbacks;
  uint64_t readback_calls = 0;
  // GPU writes whose asynchronous download completes: the next readback
  // publishes them (writes their bytes and clears their dirty mark) as well.
  std::vector<std::pair<uint64_t, uint64_t>> downloaded;
  uint64_t reads = 0;
  uint64_t sequence = 0;

  static bool Overlaps(uint64_t a, uint64_t a_size, uint64_t b,
                       uint64_t b_size) {
    return a < b + b_size && b < a + a_size;
  }

  // A recorded GPU write: the backing keeps its old bytes.
  void GpuWrite(uint64_t address, uint32_t value) {
    std::memcpy(gpu.data() + (address - Base), &value, sizeof(value));
    dirty.emplace_back(address, sizeof(value));
    ++sequence;
  }
  void CpuWrite(uint64_t address, uint32_t value) {
    std::memcpy(backing.data() + (address - Base), &value, sizeof(value));
    std::memcpy(gpu.data() + (address - Base), &value, sizeof(value));
  }

  bool Read(uint64_t address, void *data, uint64_t size) {
    ++reads;
    if (address < Base || address + size > Base + Size)
      return false;
    std::memcpy(data, backing.data() + (address - Base), size);
    return true;
  }
  bool HasGpuDirtyBytes(uint64_t address, uint64_t size) const {
    for (const auto &[start, bytes] : dirty) {
      if (Overlaps(start, bytes, address, size))
        return true;
    }
    return false;
  }
  void
  ReadGpuDirtyBytes(std::span<const Libs::Graphics::ScalarReadRange> ranges) {
    ++readback_calls;
    readbacks.insert(readbacks.end(), ranges.begin(), ranges.end());
    const auto requested = [&](uint64_t start, uint64_t bytes) {
      for (const auto &[address, size] : ranges) {
        if (Overlaps(start, bytes, address, size))
          return true;
      }
      return false;
    };
    std::vector<std::pair<uint64_t, uint64_t>> remaining;
    const auto published = [&](uint64_t start, uint64_t bytes) {
      for (const auto &[download, download_bytes] : downloaded) {
        if (Overlaps(start, bytes, download, download_bytes))
          return true;
      }
      return false;
    };
    for (const auto &[start, bytes] : dirty) {
      if (!requested(start, bytes) && !published(start, bytes)) {
        remaining.emplace_back(start, bytes);
        continue;
      }
      std::memcpy(backing.data() + (start - Base), gpu.data() + (start - Base),
                  bytes);
    }
    dirty = std::move(remaining);
    downloaded.clear();
  }
  uint64_t GpuWriteSequence() const { return sequence; }
};

using Cache = Libs::Graphics::ScalarReadCache<FakeGuestMemory>;

uint32_t ReadWord(Cache &cache, uint64_t address) {
  uint32_t value = 0xdeadbeefu;
  cache.Read(address, {&value, 1});
  return value;
}

void TestCleanReadsDoNotSynchronize() {
  FakeGuestMemory memory;
  memory.CpuWrite(FakeGuestMemory::Base + 0x40, 7);
  memory.CpuWrite(FakeGuestMemory::Base + 0x44, 8);
  Cache cache(memory);
  Check(ReadWord(cache, FakeGuestMemory::Base + 0x40) == 7, "clean word");
  Check(ReadWord(cache, FakeGuestMemory::Base + 0x44) == 8, "clean neighbour");
  Check(memory.readbacks.empty(), "a clean range was read back");
  Check(memory.reads == 1,
        "two words of one chunk took more than one backing read");
}

void TestDirtyChunkIsReadBackFirst() {
  FakeGuestMemory memory;
  const auto count = FakeGuestMemory::Base + 0x5c4;
  memory.GpuWrite(count, 3669);
  Cache cache(memory);
  Check(ReadWord(cache, count) == 3669,
        "a GPU-written count read stale backing bytes");
  Check(memory.readbacks.size() == 1 &&
            memory.readbacks[0] == std::pair<uint64_t, uint64_t>{count, 4},
        "the dirty word was not read back exactly once before it was served");
  Check(ReadWord(cache, count) == 3669 && memory.readbacks.size() == 1,
        "a synchronized chunk was read back again");

  // A dirty range in another chunk does not make clean chunks pay.
  memory.GpuWrite(FakeGuestMemory::Base + 0x800, 1);
  Cache second(memory);
  (void)ReadWord(second, FakeGuestMemory::Base + 0x100);
  Check(memory.readbacks.size() == 1,
        "a clean chunk was read back for a dirty neighbour");
}

// Ghost of Yotei writes counters on the GPU next to SRT data the walks read; a
// chunk that holds a GPU-dirty word costs a readback only for that word.
void TestOnlyReadDirtyBytesAreReadBack() {
  FakeGuestMemory memory;
  const auto chunk = FakeGuestMemory::Base + 0x300;
  memory.CpuWrite(chunk + 0x10, 11);
  memory.CpuWrite(chunk + 0x14, 12);
  memory.CpuWrite(chunk + 0x40, 5);
  memory.GpuWrite(chunk + 0x40, 6);
  Cache cache(memory);
  Check(ReadWord(cache, chunk + 0x10) == 11 &&
            ReadWord(cache, chunk + 0x14) == 12,
        "clean words of a dirty chunk");
  Check(memory.readbacks.empty(),
        "clean words of a chunk were read back for a dirty neighbour");
  Check(ReadWord(cache, chunk + 0x40) == 6,
        "a GPU-written word of a cached chunk read stale bytes");
  Check(memory.readbacks.size() == 1 &&
            memory.readbacks[0] ==
                std::pair<uint64_t, uint64_t>{chunk + 0x40, 4},
        "the dirty word was not read back alone");
  std::array<uint32_t, 2> pair{};
  cache.Read(chunk + 0x3c, pair);
  Check(pair == std::array<uint32_t, 2>{0u, 6u} && memory.readbacks.size() == 1,
        "a synchronized word was read back again or lost");
}

void TestCachedChunkDroppedAfterGpuWrite() {
  FakeGuestMemory memory;
  const auto count = FakeGuestMemory::Base + 0x5e4;
  memory.CpuWrite(count, 0);
  Cache cache(memory);
  Check(ReadWord(cache, count) == 0, "initial count");
  memory.GpuWrite(count, 1885);
  Check(ReadWord(cache, count) == 1885,
        "a chunk cached before a GPU write was served after it");
  Check(memory.readbacks.size() == 1, "the rewritten chunk was not read back");
  const auto reads = memory.reads;
  Check(ReadWord(cache, count + 4) == 0 && memory.reads == reads,
        "an unchanged sequence dropped the refilled chunk");
}

// A readback also publishes completed asynchronous downloads. A chunk filled
// while one of their words was still GPU-dirty holds the old value; once the
// word is published it is clean, and the chunk must not serve it.
void TestPublishedWordOfCachedChunk() {
  FakeGuestMemory memory;
  const auto chunk = FakeGuestMemory::Base + 0x400;
  const auto other = FakeGuestMemory::Base + 0x700;
  memory.CpuWrite(chunk + 0x20, 1);
  memory.CpuWrite(chunk + 0x30, 9);
  memory.GpuWrite(chunk + 0x20, 2);
  Cache cache(memory);
  Check(ReadWord(cache, chunk + 0x30) == 9, "clean word of a dirty chunk");
  memory.GpuWrite(other, 5);
  // The other word's write recorded later dropped the chunks; refill them.
  Check(ReadWord(cache, chunk + 0x30) == 9, "refilled clean word");
  memory.downloaded.emplace_back(chunk + 0x20, 4);
  Check(ReadWord(cache, other) == 5, "a GPU-written word of another chunk");
  Check(!memory.HasGpuDirtyBytes(chunk + 0x20, 4),
        "the download was not published by the readback");
  Check(ReadWord(cache, chunk + 0x20) == 2,
        "a chunk served a published word from before its publication");
}

// A walk that reads two GPU-written tables waits twice; the next walk of the
// same program, given the ranges the first read back, waits once.
void TestWalkReadBacksShareOneWait() {
  FakeGuestMemory memory;
  const auto first = FakeGuestMemory::Base + 0x100;
  const auto second = FakeGuestMemory::Base + 0x600;
  memory.GpuWrite(first, 1);
  memory.GpuWrite(second, 2);
  std::vector<Libs::Graphics::ScalarReadRange> history;
  {
    Cache walk(memory);
    walk.Prefetch(history);
    Check(ReadWord(walk, first) == 1 && ReadWord(walk, second) == 2,
          "first walk values");
    Check(memory.readback_calls == 2, "the first walk did not wait per table");
    const auto read_backs = walk.ReadBacks();
    history.assign(read_backs.begin(), read_backs.end());
  }
  Check(history == std::vector<Libs::Graphics::ScalarReadRange>{{first, 4},
                                                                {second, 4}},
        "the walk did not report the ranges it read back");
  memory.GpuWrite(first, 3);
  memory.GpuWrite(second, 4);
  {
    Cache walk(memory);
    walk.Prefetch(history);
    Check(ReadWord(walk, first) == 3 && ReadWord(walk, second) == 4,
          "a prefetched walk read stale values");
    Check(memory.readback_calls == 3,
          "the prefetched walk waited more than once");
  }
  // Nothing dirty: the hint costs nothing.
  {
    Cache walk(memory);
    walk.Prefetch(history);
    Check(ReadWord(walk, first) == 3 && ReadWord(walk, second) == 4 &&
              memory.readback_calls == 3 && walk.ReadBacks().empty(),
          "a clean walk read back its hint");
  }
  // A walk that reads only one of the tables still waits once and serves
  // exact values.
  memory.GpuWrite(second, 5);
  memory.GpuWrite(first, 6);
  {
    Cache walk(memory);
    walk.Prefetch(history);
    Check(ReadWord(walk, first) == 6 && memory.readback_calls == 4,
          "a walk reading one hinted table");
  }
  Check(!memory.HasGpuDirtyBytes(second, 4),
        "the hinted table was not read back in the same round trip");
}

void TestUncachedAndUnmappedReads() {
  FakeGuestMemory memory;
  // A read that ends past the mapping falls back to an exact read and returns
  // zeros.
  Cache cache(memory);
  std::array<uint32_t, 2> values{1, 2};
  cache.Read(FakeGuestMemory::Base + FakeGuestMemory::Size - 4, values);
  Check(values == std::array<uint32_t, 2>{},
        "an unmapped tail did not read as zeros");
  // A read wider than a chunk synchronizes its own range.
  const auto wide = FakeGuestMemory::Base + 0x200;
  memory.GpuWrite(wide + 0x90, 42);
  std::array<uint32_t, 40> words{};
  cache.Read(wide, words);
  Check(words[0x90 / 4] == 42, "a wide read missed a GPU write");
  Check(!memory.readbacks.empty() &&
            memory.readbacks.back() ==
                std::pair<uint64_t, uint64_t>{wide, sizeof(words)},
        "a wide read did not read back its own range");
}

// Readers that look chunks up themselves (the compiled resource walker) may use
// only tagged entries: chunks read without GPU-dirty bytes since the last GPU
// write.
void TestTagsServeOnlyCleanChunks() {
  FakeGuestMemory memory;
  const auto clean = FakeGuestMemory::Base + 0x180;
  const auto mixed = FakeGuestMemory::Base + 0x200;
  memory.CpuWrite(clean + 8, 41);
  memory.CpuWrite(mixed, 42);
  memory.GpuWrite(mixed + 0x40, 43);
  Cache cache(memory);
  const auto index = [](uint64_t address) {
    return (address / Cache::ChunkBytes) % Cache::Entries;
  };
  const auto tag = [&](uint64_t address) {
    return cache.Tags()[index(address)];
  };
  for (size_t entry = 0; entry < Cache::Entries; ++entry)
    Check(cache.Tags()[entry] == UINT64_MAX, "a new cache has a tagged entry");
  Check(ReadWord(cache, clean + 8) == 41 && tag(clean) == clean,
        "a clean chunk was not tagged");
  uint32_t word = 0;
  std::memcpy(&word, cache.Data() + index(clean) * Cache::Stride() + 8,
              sizeof(word));
  Check(word == 41, "a tagged entry does not hold the chunk's bytes");
  Check(ReadWord(cache, mixed) == 42 && tag(mixed) == UINT64_MAX,
        "a chunk with GPU-dirty bytes was tagged");
  memory.GpuWrite(FakeGuestMemory::Base + 0x900, 1);
  (void)ReadWord(cache, FakeGuestMemory::Base + 0x300);
  Check(tag(clean) == UINT64_MAX,
        "a chunk filled before a GPU write stayed tagged");
}

} // namespace

int main() {
  TestCleanReadsDoNotSynchronize();
  TestDirtyChunkIsReadBackFirst();
  TestOnlyReadDirtyBytesAreReadBack();
  TestCachedChunkDroppedAfterGpuWrite();
  TestPublishedWordOfCachedChunk();
  TestWalkReadBacksShareOneWait();
  TestUncachedAndUnmappedReads();
  TestTagsServeOnlyCleanChunks();
  std::printf("ScalarReadCacheTests: ok\n");
  return 0;
}
