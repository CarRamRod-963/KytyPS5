#include "graphics/host_gpu/renderer/pipeline/scalarReadCache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  std::vector<std::pair<uint64_t, uint64_t>> readbacks;
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
  void ReadGpuDirtyBytes(uint64_t address, uint64_t size) {
    readbacks.emplace_back(address, size);
    std::vector<std::pair<uint64_t, uint64_t>> remaining;
    for (const auto &[start, bytes] : dirty) {
      if (!Overlaps(start, bytes, address, size)) {
        remaining.emplace_back(start, bytes);
        continue;
      }
      std::memcpy(backing.data() + (start - Base), gpu.data() + (start - Base),
                  bytes);
    }
    dirty = std::move(remaining);
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
            memory.readbacks[0] ==
                std::pair<uint64_t, uint64_t>{count & ~uint64_t{127}, 128},
        "the dirty chunk was not read back exactly once before it was filled");
  Check(ReadWord(cache, count) == 3669 && memory.readbacks.size() == 1,
        "a synchronized chunk was read back again");

  // A dirty range in another chunk does not make clean chunks pay.
  memory.GpuWrite(FakeGuestMemory::Base + 0x800, 1);
  Cache second(memory);
  (void)ReadWord(second, FakeGuestMemory::Base + 0x100);
  Check(memory.readbacks.size() == 1,
        "a clean chunk was read back for a dirty neighbour");
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

} // namespace

int main() {
  TestCleanReadsDoNotSynchronize();
  TestDirtyChunkIsReadBackFirst();
  TestCachedChunkDroppedAfterGpuWrite();
  TestUncachedAndUnmappedReads();
  std::printf("ScalarReadCacheTests: ok\n");
  return 0;
}
