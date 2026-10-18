#include "graphics/host_gpu/renderer/cache/layoutChangeLog.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Libs::Graphics::LayoutChangeLog;

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "LayoutChangeLogTests: failed: %s\n", text);
    std::exit(1);
  }
}

constexpr uint64_t MiB = uint64_t{1} << 20u;

void TestRecentChanges() {
  LayoutChangeLog log;
  const auto before = log.Now();
  Check(!log.ChangedSince(before, 0x1000'0000, MiB),
        "an empty log reported a change");
  log.Note(0x1000'0000 + MiB, MiB);
  Check(!log.ChangedSince(before, 0x1000'0000, MiB),
        "an adjacent change overlapped");
  Check(log.ChangedSince(before, 0x1000'0000, MiB + 1u),
        "an overlapping change was missed");
  const auto after = log.Now();
  Check(!log.ChangedSince(after, 0x1000'0000 + MiB, MiB),
        "a change before the stamp was seen");
  log.Note(0x2000'0000, 0);
  Check(log.Now().sequence == after.sequence,
        "an empty change advanced the log");
}

// The old log forgot everything after 4096 changes, so every stamp, also of
// memory nothing touched, read as changed and every memoized lookup missed at
// once.
void TestUnrelatedChangesKeepStamps() {
  LayoutChangeLog log;
  const auto stamp = log.Now();
  for (uint32_t change = 0; change < 10u * LayoutChangeLog::MaxRecent;
       ++change) {
    log.Note(0x4000'0000 + uint64_t{change % 64u} * 4u * MiB, 4u * MiB);
  }
  Check(log.RecentCount() <= LayoutChangeLog::MaxRecent,
        "the recent list grew without bound");
  Check(!log.ChangedSince(stamp, 0x1000'0000, 16u * MiB),
        "changes elsewhere invalidated an old stamp");
  Check(log.ChangedSince(stamp, 0x4000'0000 + 8u * MiB, 1u),
        "a folded change was missed by an old stamp");
  Check(log.ChangedSince(stamp, 0x3000'0000, 0x1000'0000 + 1u),
        "a folded change was missed by a range spanning many granules");
}

// Against an exact model: a change after the stamp that overlaps the range is
// always reported; a report without one comes only from a folded change in a
// granule the range touches.
void TestAgainstExactModel() {
  struct Change {
    uint64_t sequence, address, size;
  };
  std::mt19937_64 random(7);
  LayoutChangeLog log;
  std::vector<Change> all;
  const auto granule = [](uint64_t address) {
    return address >> LayoutChangeLog::GranuleBits;
  };
  for (uint32_t step = 0; step < 6u * LayoutChangeLog::MaxRecent; ++step) {
    const auto address = 0x1'0000'0000ull + (random() % 4096u) * 0x10000u;
    const auto size = 0x1000u + random() % (8u * MiB);
    log.Note(address, size);
    all.push_back({log.Now().sequence, address, size});
    if (step % 7u != 0u)
      continue;
    const LayoutChangeLog::Stamp stamp{random() % (log.Now().sequence + 1u)};
    const auto query = 0x1'0000'0000ull + (random() % 4096u) * 0x10000u;
    const auto bytes = 0x100u + random() % (4u * MiB);
    bool exact = false;
    bool nearby = false;
    for (const auto &change : all) {
      if (change.sequence <= stamp.sequence)
        continue;
      exact |= query < change.address + change.size &&
               change.address < query + bytes;
      nearby |= granule(query) <= granule(change.address + change.size - 1u) &&
                granule(change.address) <= granule(query + bytes - 1u);
    }
    const bool reported = log.ChangedSince(stamp, query, bytes);
    Check(!exact || reported, "a change overlapping the range was missed");
    Check(!reported || nearby,
          "a change was reported with none in the range's granules");
  }
}

} // namespace

int main() {
  TestRecentChanges();
  TestUnrelatedChangesKeepStamps();
  TestAgainstExactModel();
  std::printf("LayoutChangeLogTests: all cases passed\n");
  return 0;
}
