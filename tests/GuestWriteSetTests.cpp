#include "graphics/host_gpu/renderer/pipeline/guestWriteSet.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::GuestWriteSet;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "GuestWriteSetTests: failed: %s\n", message);
    std::abort();
  }
}

// The reference the set replaces: every read against every written range.
bool BruteOverlap(const GuestWriteSet::Range *ranges, int count,
                  uint64_t address, uint64_t size) {
  for (int i = 0; i < count; ++i) {
    const auto &range = ranges[i];
    if (address < range.address + range.size &&
        range.address < address + size) {
      return true;
    }
  }
  return false;
}

void TestEmpty() {
  GuestWriteSet set;
  Check(set.Empty(), "a new set has ranges");
  Check(set.FindOverlap(0, UINT64_MAX) == nullptr,
        "an empty set found an overlap");
  Check(set.Add(0x1000, 0, GuestWriteSet::Kind::Image) && set.Empty(),
        "an empty range was added");
  Check(!set.Add(UINT64_MAX - 0xf, 0x20, GuestWriteSet::Kind::Buffer),
        "a wrapping range was accepted");
}

void TestBoundaries() {
  GuestWriteSet set;
  Check(set.Add(0x1000, 0x100, GuestWriteSet::Kind::Image),
        "image range refused");
  Check(set.Add(0x8000, 0x10, GuestWriteSet::Kind::Buffer),
        "buffer range refused");
  Check(set.FindOverlap(0x0f00, 0x100) == nullptr,
        "a read ending at a write overlapped");
  Check(set.FindOverlap(0x1100, 0x10) == nullptr,
        "a read starting at a write's end overlapped");
  Check(set.FindOverlap(0x2000, 0x1000) == nullptr,
        "a read between two writes (inside their bound) overlapped");
  const auto *image = set.FindOverlap(0x10ff, 1);
  Check(image != nullptr && image->kind == GuestWriteSet::Kind::Image,
        "the last byte of an image write was missed");
  const auto *buffer = set.FindOverlap(0x7000, 0x1001);
  Check(buffer != nullptr && buffer->kind == GuestWriteSet::Kind::Buffer,
        "the first byte of a buffer write was missed");
  Check(set.FindOverlap(0x8010, 0x100) == nullptr,
        "a read past the bound overlapped");
  const auto *spanning = set.FindOverlap(0, 0x10000);
  Check(spanning != nullptr && spanning->kind == GuestWriteSet::Kind::Image,
        "a read covering both writes did not report the first added one");
}

void TestClear() {
  GuestWriteSet set;
  Check(set.Add(0x1000, 0x100, GuestWriteSet::Kind::Image), "range refused");
  set.Clear();
  Check(set.Empty() && set.FindOverlap(0x1000, 0x100) == nullptr,
        "a cleared set kept a range");
  Check(set.Add(0x5000, 0x100, GuestWriteSet::Kind::Buffer), "range refused");
  Check(set.FindOverlap(0x1000, 0x100) == nullptr,
        "a cleared set kept its old bound");
}

void TestAgainstBruteForce() {
  uint64_t state = 0x243f6a8885a308d3ull;
  auto next = [&state] {
    state ^= state << 13u;
    state ^= state >> 7u;
    state ^= state << 17u;
    return state;
  };
  for (int round = 0; round < 2000; ++round) {
    GuestWriteSet set;
    GuestWriteSet::Range ranges[8];
    const int count = static_cast<int>(next() % 8u);
    for (int i = 0; i < count; ++i) {
      ranges[i] = {next() % 0x10000u, 1u + next() % 0x800u,
                   (next() & 1u) != 0 ? GuestWriteSet::Kind::Image
                                      : GuestWriteSet::Kind::Buffer};
      Check(set.Add(ranges[i].address, ranges[i].size, ranges[i].kind),
            "range refused");
    }
    for (int read = 0; read < 64; ++read) {
      const uint64_t address = next() % 0x11000u;
      const uint64_t size = 1u + next() % 0x400u;
      const auto *found = set.FindOverlap(address, size);
      Check((found != nullptr) == BruteOverlap(ranges, count, address, size),
            "the set disagrees with the pairwise check");
      if (found != nullptr) {
        // The first overlapping range in insertion order, as the pairwise loop
        // reports.
        for (int i = 0; i < count; ++i) {
          if (BruteOverlap(&ranges[i], 1, address, size)) {
            Check(found->address == ranges[i].address &&
                      found->size == ranges[i].size,
                  "the set reported another range than the first overlapping "
                  "one");
            break;
          }
        }
      }
    }
  }
}

} // namespace

int main() {
  TestEmpty();
  TestBoundaries();
  TestClear();
  TestAgainstBruteForce();
  std::puts("GuestWriteSetTests: all cases passed");
  return 0;
}
