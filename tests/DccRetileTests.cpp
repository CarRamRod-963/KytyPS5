#include "graphics/host_gpu/renderer/cache/dccRetile.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {

namespace DccRetile = Libs::Graphics::DccRetile;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "DccRetileTests: failed: %s\n", message);
    std::abort();
  }
}

// The dispatch Ghost of Yotei issues for its 3840x2160 DCC targets: 8x40 workgroups, pitch 8,
// 0x28000 key bytes on both sides. The shader moves every key exactly once.
void TestFullTargetIsPermutation() {
  constexpr uint32_t groups_x = 8;
  constexpr uint32_t groups_y = 40;
  constexpr uint32_t bytes = 0x28000;
  std::vector<uint32_t> loaded(bytes, 0);
  std::vector<uint32_t> stored(bytes / 2, 0);
  for (uint32_t y = 0; y < groups_y * DccRetile::GroupHeight; ++y) {
    for (uint32_t x = 0; x < groups_x * DccRetile::GroupWidth; ++x) {
      const auto invocation = DccRetile::Locate(8, x, y);
      for (const auto index : invocation.loads) {
        Check(index < bytes, "a load left the source");
        loaded[index]++;
      }
      for (const auto index : invocation.stores) {
        Check(index < bytes / 2, "a store left the destination");
        stored[index]++;
      }
    }
  }
  for (const auto count : loaded) {
    Check(count == 1, "a source key was not read exactly once");
  }
  for (const auto count : stored) {
    Check(count == 1, "a destination element was not written exactly once");
  }

  std::vector<uint8_t> source(bytes);
  for (uint32_t i = 0; i < bytes; ++i) {
    source[i] = static_cast<uint8_t>(i * 131u + (i >> 9u));
  }
  std::vector<uint8_t> destination(bytes, 0xcd);
  DccRetile::Run(8, groups_x, groups_y, source, destination);
  std::vector<uint32_t> histogram_source(256, 0);
  std::vector<uint32_t> histogram_destination(256, 0);
  for (uint32_t i = 0; i < bytes; ++i) {
    histogram_source[source[i]]++;
    histogram_destination[destination[i]]++;
  }
  Check(histogram_source == histogram_destination,
        "the copy did not move every key byte exactly once");
  const auto first = DccRetile::Locate(8, 0, 0);
  Check(destination[first.stores[0] * 2] == source[first.loads[0]] &&
            destination[first.stores[0] * 2 + 1] == source[first.loads[1]],
        "a store did not take its loads in register order");
}

// Out-of-range loads read zero and out-of-range stores are dropped, as with num_records bounds.
void TestBounds() {
  constexpr uint32_t bytes = 0x28000;
  std::vector<uint8_t> source(bytes);
  for (uint32_t i = 0; i < bytes; ++i) {
    source[i] = static_cast<uint8_t>(1u + i % 251u);
  }
  // Half the source records: keys past them read as zero.
  std::vector<uint8_t> destination(bytes, 0xcd);
  DccRetile::Run(8, 8, 40, std::span(source).first(bytes / 2), destination);
  uint32_t zeros = 0;
  for (uint32_t y = 0; y < 40 * DccRetile::GroupHeight; ++y) {
    for (uint32_t x = 0; x < 8 * DccRetile::GroupWidth; ++x) {
      const auto invocation = DccRetile::Locate(8, x, y);
      for (size_t i = 0; i < 8; ++i) {
        const auto load = invocation.loads[i];
        const auto byte = destination[invocation.stores[i / 2] * 2 + i % 2];
        Check(byte == (load < bytes / 2 ? source[load] : 0u),
              "a key did not come from its load, or an out-of-range load was not zero");
        zeros += byte == 0;
      }
    }
  }
  Check(zeros != 0, "the short source left every load in range");
  // Half the destination records, followed by canaries: stores past the records are dropped.
  std::vector<uint8_t> canaries(bytes, 0xcd);
  DccRetile::Run(8, 8, 40, source, std::span(canaries).first(bytes / 2));
  for (uint32_t i = bytes / 2; i < bytes; ++i) {
    Check(canaries[i] == 0xcd, "a store past the destination records was not dropped");
  }
  std::vector<uint8_t> full(bytes, 0xcd);
  DccRetile::Run(8, 8, 40, source, full);
  for (uint32_t i = 0; i < bytes / 2; ++i) {
    Check(canaries[i] == full[i], "a shorter destination changed the in-range result");
  }
}

void TestEmptyDispatch() {
  const std::vector<uint8_t> source(64, 1);
  std::vector<uint8_t> destination(64, 7);
  DccRetile::Run(8, 0, 4, source, destination);
  for (const auto byte : destination) {
    Check(byte == 7, "an empty dispatch wrote the destination");
  }
}

} // namespace

int main() {
  TestFullTargetIsPermutation();
  TestBounds();
  TestEmptyDispatch();
  std::printf("DccRetileTests: ok\n");
  return 0;
}
