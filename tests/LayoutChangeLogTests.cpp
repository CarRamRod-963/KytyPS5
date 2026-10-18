#include "graphics/host_gpu/renderer/cache/layoutChangeLog.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
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

// The log before the granule index: every query walked all recent changes after
// its stamp. The indexed log must answer exactly as it did.
class WalkingLog {
public:
  void Note(uint64_t address, uint64_t size) {
    if (size == 0)
      return;
    if (m_recent.size() >= LayoutChangeLog::MaxRecent) {
      const auto count = LayoutChangeLog::MaxRecent / 2u;
      for (size_t index = 0; index < count; ++index) {
        const auto &change = m_recent[index];
        for (auto granule = change.address >> LayoutChangeLog::GranuleBits;
             granule <= (change.address + change.size - 1u) >>
             LayoutChangeLog::GranuleBits;
             ++granule) {
          auto &sequence = m_granules[granule];
          sequence = std::max(sequence, change.sequence);
        }
        m_folded_through = change.sequence;
      }
      m_recent.erase(m_recent.begin(),
                     m_recent.begin() + static_cast<std::ptrdiff_t>(count));
    }
    m_recent.push_back({++m_sequence, address, size});
  }

  bool ChangedSince(uint64_t stamp, uint64_t address, uint64_t size) const {
    if (stamp >= m_sequence || size == 0)
      return false;
    for (auto it = m_recent.rbegin(); it != m_recent.rend(); ++it) {
      if (it->sequence <= stamp)
        return false;
      if (address < it->address + it->size && it->address < address + size)
        return true;
    }
    if (stamp >= m_folded_through)
      return false;
    for (auto granule = address >> LayoutChangeLog::GranuleBits;
         granule <= (address + size - 1u) >> LayoutChangeLog::GranuleBits;
         ++granule) {
      const auto found = m_granules.find(granule);
      if (found != m_granules.end() && found->second > stamp)
        return true;
    }
    return false;
  }

private:
  struct Change {
    uint64_t sequence, address, size;
  };
  std::vector<Change> m_recent;
  std::unordered_map<uint64_t, uint64_t> m_granules;
  uint64_t m_sequence = 0;
  uint64_t m_folded_through = 0;
};

// Identical answers to the walking log: small and wide changes (some spanning
// more than WideGranules granules), queries from one byte to a few hundred
// MiB, stamps old and new, across many folds.
void TestSameAnswersAsWalkingLog() {
  std::mt19937_64 random(11);
  LayoutChangeLog log;
  WalkingLog walking;
  uint32_t changed = 0;
  uint32_t unchanged = 0;
  const auto pick_size = [&](uint32_t kind) -> uint64_t {
    switch (kind % 8u) {
    case 0:
      return 1u + random() % 256u;
    case 1:
      return 0x1000u + random() % (64u * MiB);
    case 2:
      return 0x1000u + random() % (400u * MiB);
    default:
      return 0x1000u + random() % (4u * MiB);
    }
  };
  for (uint32_t step = 0; step < 12u * LayoutChangeLog::MaxRecent; ++step) {
    // Mostly a hot window (transient targets), sometimes anywhere in 16 GiB.
    const auto address = random() % 4u == 0u
                             ? 0x1'0000'0000ull + random() % (16ull << 30u)
                             : 0x2'0000'0000ull + random() % (32u * MiB);
    const auto size = pick_size(static_cast<uint32_t>(random()));
    log.Note(address, size);
    walking.Note(address, size);
    for (uint32_t query = 0; query < 3u; ++query) {
      const auto now = log.Now().sequence;
      const auto age =
          random() % 3u == 0u ? random() % (now + 1u) : random() % 64u;
      const LayoutChangeLog::Stamp stamp{now > age ? now - age : 0u};
      const auto where = random() % 2u == 0u
                             ? 0x1'0000'0000ull + random() % (16ull << 30u)
                             : 0x2'0000'0000ull + random() % (40u * MiB);
      const auto bytes = pick_size(static_cast<uint32_t>(random()));
      const bool reported = log.ChangedSince(stamp, where, bytes);
      Check(reported == walking.ChangedSince(stamp.sequence, where, bytes),
            "the indexed log answered unlike the walking log");
      (reported ? changed : unchanged)++;
    }
  }
  Check(changed > 1000u && unchanged > 1000u,
        "the comparison did not exercise both answers");
}

} // namespace

int main() {
  TestRecentChanges();
  TestUnrelatedChangesKeepStamps();
  TestAgainstExactModel();
  TestSameAnswersAsWalkingLog();
  std::printf("LayoutChangeLogTests: all cases passed\n");
  return 0;
}
