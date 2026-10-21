#include "graphics/host_gpu/filteredRangeSet.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::FilteredRangeSet;
using Libs::Graphics::RangeSet;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "FilteredRangeSetTests: failed: %s\n", message);
    std::abort();
  }
}

struct Random {
  uint64_t state;
  uint64_t Next() {
    state ^= state << 13u;
    state ^= state >> 7u;
    state ^= state << 17u;
    return state;
  }
  uint64_t Below(uint64_t bound) { return Next() % bound; }
};

uint64_t RandomSize(Random &random) {
  switch (random.Below(4)) {
  case 0: return 1 + random.Below(256);                     // a few dwords
  case 1: return 1 + random.Below(uint64_t{1} << 20u);      // up to a fine-filter range
  case 2: return 1 + random.Below(uint64_t{1} << 28u);      // coarse granules
  default: return uint64_t{1} << (12u + random.Below(30)); // up to 2 TiB: saturates
  }
}

// Every answer equals the plain set's, through adds, subtracts, rebuilds and clears.
void TestMatchesRangeSet() {
  Random random{0x1234567887654321ull};
  FilteredRangeSet filtered;
  RangeSet plain;
  constexpr uint64_t span = uint64_t{1} << 40u;
  for (uint32_t step = 0; step < 200000; ++step) {
    const auto address = 0x10000000ull + random.Below(span);
    const auto size = RandomSize(random);
    const auto op = random.Below(16);
    if (op < 3) {
      filtered.Add(address, size);
      plain.Add(address, size);
    } else if (op < 7) {
      filtered.Subtract(address, size);
      plain.Subtract(address, size);
    } else if (op == 7 && random.Below(64) == 0) {
      filtered.Clear();
      plain.Clear();
    } else {
      Check(filtered.Intersects(address, size) == plain.Intersects(address, size),
            "Intersects disagreed with the plain set");
      Check(filtered.Contains(address, size) == plain.Contains(address, size),
            "Contains disagreed with the plain set");
      std::vector<std::pair<uint64_t, uint64_t>> a;
      std::vector<std::pair<uint64_t, uint64_t>> b;
      filtered.ForEachInRange(address, size, [&](uint64_t x, uint64_t y) { a.emplace_back(x, y); });
      plain.ForEachInRange(address, size, [&](uint64_t x, uint64_t y) { b.emplace_back(x, y); });
      Check(a == b, "ForEachInRange disagreed with the plain set");
    }
    Check(filtered.Empty() == plain.Empty(), "Empty disagreed with the plain set");
  }
}

// Small dirty ranges among many disjoint queries: queries next to them stay negative, queries
// touching a byte of them are positive, and a subtracted range stops intersecting.
void TestNearbyQueries() {
  FilteredRangeSet set;
  set.Add(0x500000100ull, 4);
  set.Add(0x600000000ull, uint64_t{1} << 30u); // coarse granules
  Check(set.Intersects(0x500000100ull, 1) && set.Intersects(0x5000000f0ull, 0x11),
        "a query touching a small range missed it");
  Check(!set.Intersects(0x500000104ull, 0x80) && !set.Intersects(0x5000000c0ull, 0x40),
        "a query next to a small range intersected it");
  Check(set.Intersects(0x63fffff80ull, 0x80) && !set.Intersects(0x640000000ull, 0x80),
        "a query at the end of a large range was misjudged");
  set.Subtract(0x500000100ull, 4);
  Check(!set.Intersects(0x500000100ull, 1), "a subtracted range still intersected");
  set.Subtract(0x600000000ull, uint64_t{1} << 30u);
  Check(set.Empty() && !set.Intersects(0x600000000ull, 1), "an emptied set still intersected");
}

} // namespace

int main() {
  TestMatchesRangeSet();
  TestNearbyQueries();
  std::printf("FilteredRangeSetTests: ok\n");
  return 0;
}
