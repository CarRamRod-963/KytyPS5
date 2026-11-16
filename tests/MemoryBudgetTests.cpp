#include "graphics/host_gpu/memoryBudget.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::MemoryBudget::BufferMarks;
using Libs::Graphics::MemoryBudget::CacheLimit;
using Libs::Graphics::MemoryBudget::CollectionMarks;
using Libs::Graphics::MemoryBudget::DriverHeadroom;
using Libs::Graphics::MemoryBudget::DriverUsage;
using Libs::Graphics::MemoryBudget::GiB;
using Libs::Graphics::MemoryBudget::HeapState;
using Libs::Graphics::MemoryBudget::LiveMarks;
using Libs::Graphics::MemoryBudget::LostAccessReport;
using Libs::Graphics::MemoryBudget::MiB;
using Libs::Graphics::MemoryBudget::Monitor;
using Libs::Graphics::MemoryBudget::TextureMarks;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryBudgetTests: failed: %s\n", text);
    std::abort();
  }
}

void TestLimit() {
  Check(DriverUsage(
            {.budget = 10 * GiB, .usage = 3 * GiB, .blocks = 2 * GiB}) == GiB,
        "driver usage is not usage minus VMA blocks");
  Check(DriverUsage({.budget = 10 * GiB, .usage = GiB, .blocks = 2 * GiB}) == 0,
        "a stale usage below the blocks does not clamp to zero");

  Check(DriverHeadroom(2 * GiB) == 256 * MiB,
        "small budgets do not keep 256 MiB");
  Check(DriverHeadroom(8 * GiB) == 512 * MiB,
        "headroom is not a sixteenth of the budget");
  Check(DriverHeadroom(64 * GiB) == GiB, "headroom is not capped at 1 GiB");

  // Heap 0 of a 16 GB card in the open world: 15.38 GB budget, 15.06 GB
  // used, 14.26 GB of it in VMA blocks. The driver's 0.8 GB and the headroom
  // leave 13.6 GB to the caches.
  const HeapState game{.budget = 15378141184ull,
                       .usage = 15063334912ull,
                       .blocks = 14263754240ull};
  const auto limit = CacheLimit(game);
  Check(limit == game.budget - (game.usage - game.blocks) -
                     DriverHeadroom(game.budget),
        "the limit is not budget minus driver usage minus headroom");
  Check(limit < game.blocks, "the open-world heap is not over the limit");
  Check(game.blocks - limit ==
            game.usage + DriverHeadroom(game.budget) - game.budget,
        "the excess is not the missing headroom");

  // A driver that owns more than the budget leaves nothing to the caches.
  Check(CacheLimit({.budget = 4 * GiB, .usage = 5 * GiB, .blocks = GiB}) == 0,
        "the limit underflows");
}

// The marks the caches computed once at start before they followed the budget.
CollectionMarks ReferenceTextureMarks(int64_t budget) {
  constexpr int64_t gib = 1024ll * 1024 * 1024;
  const auto threshold = std::min<int64_t>(budget, 8 * gib);
  return {
      .trigger =
          static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0)),
      .pressure = static_cast<uint64_t>(std::max<int64_t>(
          std::min(budget - 6 * threshold / 10, budget - gib), gib + gib / 2)),
      .critical = static_cast<uint64_t>(std::max<int64_t>(
          std::min(budget - 2 * threshold / 10, budget - gib / 2), 3 * gib))};
}

CollectionMarks ReferenceBufferMarks(int64_t budget) {
  constexpr int64_t gib = 1024ll * 1024 * 1024;
  const auto threshold = std::min<int64_t>(budget, 8 * gib);
  const auto trigger = static_cast<uint64_t>(std::max<int64_t>(
      std::min(budget - 6 * threshold / 10, budget - gib), gib));
  return {
      .trigger = trigger,
      .pressure = trigger,
      .critical = static_cast<uint64_t>(std::max<int64_t>(
          std::min(budget - 2 * threshold / 10, budget - gib / 2), 2 * gib))};
}

void TestMarks() {
  for (uint64_t budget = 0; budget <= 40 * GiB; budget += 37 * MiB + 13) {
    Check(TextureMarks(budget) ==
              ReferenceTextureMarks(static_cast<int64_t>(budget)),
          "texture marks differ from the formula at start");
    Check(BufferMarks(budget) ==
              ReferenceBufferMarks(static_cast<int64_t>(budget)),
          "buffer marks differ from the formula at start");
  }

  // A 12 GB card: 11,172 MiB of budget at start, 10,251 MiB later in the
  // session (both minus the 1 GiB reserve of GetTotalMemoryBudget).
  constexpr uint64_t s = 1000;
  LiveMarks marks;
  Check(marks.Generation() == 0, "marks without a budget have a generation");
  Check(marks.Update(10148 * MiB, 0), "the first budget sets no marks");
  Check(marks.Generation() == 1, "the first marks did not start a generation");
  Check(marks.Texture() == TextureMarks(10148 * MiB),
        "the texture marks do not follow the first budget");
  Check(marks.Texture().pressure / MiB == 5232 &&
            marks.Texture().critical / MiB == 8509,
        "a 12 GB card's marks moved");
  Check(!marks.Update(10148 * MiB + 511 * MiB, s),
        "a change below the hysteresis moves the marks");
  Check(!marks.Update(10148 * MiB - 511 * MiB, 2 * s),
        "a drop below the hysteresis moves the marks");
  Check(marks.Budget() == 10148 * MiB, "the hysteresis moved the budget");
  Check(marks.Update(9227 * MiB, 3 * s),
        "a budget drop is not followed at once");
  Check(marks.Texture() == TextureMarks(9227 * MiB) &&
            marks.Buffer() == BufferMarks(9227 * MiB) &&
            marks.Generation() == 2,
        "the marks do not follow the dropped budget");
  Check(marks.Texture().critical < TextureMarks(10148 * MiB).critical,
        "a smaller budget does not lower the critical mark");
  Check(!marks.Update(9227 * MiB + 300 * MiB, 4 * s),
        "a fluctuation moves the marks");

  // A rise is followed only once it held for RiseHoldMs, to the lowest
  // budget seen meanwhile.
  constexpr uint64_t hold = LiveMarks::RiseHoldMs;
  Check(!marks.Update(9227 * MiB + 900 * MiB, 10 * s),
        "a budget rise is followed at once");
  Check(!marks.Update(9227 * MiB + 600 * MiB, 10 * s + hold / 2),
        "a budget rise is followed before it held");
  Check(marks.Update(9227 * MiB + 900 * MiB, 10 * s + hold),
        "a budget rise that held is not followed");
  Check(marks.Budget() == 9227 * MiB + 600 * MiB && marks.Generation() == 3,
        "the held rise did not move the marks to its lowest budget");

  // A budget that flaps (a 16 GB card: 14,185 <-> ~13,300 MiB about every
  // minute) keeps the lower marks: a rise that falls back before it held
  // restarts the hold.
  LiveMarks flap;
  Check(flap.Update(14185 * MiB, 0) && flap.Update(13287 * MiB, 60 * s),
        "the flapping budget's first drop is not followed");
  for (uint64_t i = 0; i < 6; i++) {
    const uint64_t t = 70 * s + i * 50 * s;
    Check(!flap.Update(14185 * MiB, t) &&
              !flap.Update(14185 * MiB, t + hold - 1),
          "a flapping budget's rise is followed before it held");
    Check(!flap.Update(13406 * MiB, t + hold),
          "a fall-back inside the hysteresis moves the marks");
  }
  Check(flap.Budget() == 13287 * MiB && flap.Generation() == 2,
        "the flapping budget moved the marks");
  Check(!flap.Update(14185 * MiB, 1000 * s) &&
            flap.Update(14185 * MiB, 1000 * s + hold),
        "a rise that held after the flapping is not followed");
  Check(flap.Texture() == TextureMarks(14185 * MiB), "the marks are not back");
}

void TestLostAccessReport() {
  constexpr uint64_t s = 1000;
  LostAccessReport report;
  Check(!report.Summary(0), "an empty report summarizes");
  for (uint64_t i = 0; i < LostAccessReport::FirstReports; i++) {
    Check(report.Note(), "one of the first accesses is not logged on its own");
  }
  Check(!report.Summary(s), "accesses logged on their own are summarized");
  for (int i = 0; i < 1000; i++) {
    Check(!report.Note(), "a later access is logged on its own");
  }
  Check(report.Summary(2 * s).accesses == 1000,
        "the first summary is not due at once");
  Check(!report.Note(), "a later access is logged on its own");
  Check(!report.Summary(2 * s + LostAccessReport::PeriodMs - 1),
        "summaries are not limited to one per period");
  Check(report.Summary(2 * s + LostAccessReport::PeriodMs).accesses == 1,
        "the access after a summary is not summarized a period later");
  Check(!report.Summary(100 * s), "a summary repeats");
  Check(report.Total() == LostAccessReport::FirstReports + 1001,
        "the total is not counted");

  // Overflows of the fault report area are summarized like accesses, also on
  // their own (the dropped reports hold no lost access to count).
  report.NoteOverflow();
  report.NoteOverflow();
  const auto overflowed = report.Summary(200 * s);
  Check(overflowed && overflowed.accesses == 0 && overflowed.overflows == 2,
        "overflows of the report area are not summarized");
  report.NoteOverflow();
  Check(!report.Summary(200 * s + LostAccessReport::PeriodMs - 1),
        "overflow summaries are not limited to one per period");
  Check(report.Summary(200 * s + LostAccessReport::PeriodMs).overflows == 1,
        "an overflow after a summary is not summarized a period later");
  Check(report.Overflows() == 3 &&
            report.Total() == LostAccessReport::FirstReports + 1001,
        "the overflow total is not counted apart from the accesses");
}

void TestMonitor() {
  constexpr uint64_t s = 1000;
  const auto device = [](uint64_t driver) {
    return HeapState{
        .budget = 15 * GiB, .usage = 12 * GiB + driver, .blocks = 12 * GiB};
  };
  const auto host = [](uint64_t driver) {
    return HeapState{
        .budget = 32 * GiB, .usage = 608 * MiB + driver, .blocks = 608 * MiB};
  };

  Monitor monitor;
  Check(monitor.Sample(device(GiB), host(128 * MiB), 5 * s) ==
            Monitor::Report::First,
        "the first sample is not reported");
  Check(monitor.Sample(device(4 * GiB), host(5 * GiB), 14 * s) ==
            Monitor::Report::None,
        "a sample within the period is evaluated");
  Check(monitor.Sample(device(GiB + 63 * MiB), host(128 * MiB), 16 * s) ==
            Monitor::Report::None,
        "growth below the threshold is reported");
  Check(monitor.Sample(device(GiB + 64 * MiB), host(128 * MiB), 26 * s) ==
            Monitor::Report::DriverGrew,
        "driver growth is not reported");
  Check(monitor.Sample(device(GiB + 64 * MiB), host(128 * MiB), 36 * s) ==
            Monitor::Report::None,
        "an unchanged heap is reported again");

  // A demotion moves the driver's allocation from the device heap into the host
  // heap.
  Check(monitor.Sample(device(GiB / 4), host(128 * MiB + 3 * GiB), 46 * s) ==
            Monitor::Report::Demoted,
        "host-heap growth is not reported as a demotion");

  // Shrinking lowers the reference, so later growth is measured from the low
  // point.
  Check(monitor.Sample(device(GiB / 8), host(128 * MiB), 56 * s) ==
            Monitor::Report::None,
        "shrinking is reported");
  Check(monitor.Sample(device(GiB / 8 + 64 * MiB), host(128 * MiB), 66 * s) ==
            Monitor::Report::DriverGrew,
        "growth after a shrink is measured from the old reference");

  // Memory at low residency priority is reported when it moves by 512 MiB
  // either way.
  Check(monitor.Sample(device(GiB / 8 + 64 * MiB), host(128 * MiB), 76 * s,
                       511 * MiB) == Monitor::Report::None,
        "a small priority change is reported");
  Check(monitor.Sample(device(GiB / 8 + 64 * MiB), host(128 * MiB), 86 * s,
                       2 * GiB) == Monitor::Report::Priorities,
        "aging to low priority is not reported");
  Check(monitor.Sample(device(GiB / 8 + 64 * MiB), host(128 * MiB), 96 * s,
                       GiB) == Monitor::Report::Priorities,
        "promotion out of low priority is not reported");

  // Usage above the budget is reported every period, and once more when it is
  // back within the budget.
  const auto over = [](uint64_t usage) {
    return HeapState{.budget = 10 * GiB, .usage = usage, .blocks = usage - GiB};
  };
  Monitor paging;
  Check(paging.Sample(over(9 * GiB), host(0), 0) == Monitor::Report::First,
        "the first sample is not reported");
  Check(paging.Sample(over(10 * GiB), host(0), 10 * s) == Monitor::Report::None,
        "usage at the budget is reported as over it");
  Check(paging.Sample(over(10 * GiB + MiB), host(0), 20 * s) ==
            Monitor::Report::OverBudget,
        "usage over the budget is not reported");
  Check(paging.Sample(over(12 * GiB), host(0), 25 * s) == Monitor::Report::None,
        "over-budget reports are not limited to one per period");
  Check(paging.Sample(over(10 * GiB + MiB), host(0), 30 * s) ==
            Monitor::Report::OverBudget,
        "usage that stays over the budget is not reported again");
  Check(paging.Sample(over(9 * GiB), host(0), 40 * s) ==
            Monitor::Report::WithinBudget,
        "the return within the budget is not reported");
  Check(paging.Sample(over(9 * GiB), host(0), 50 * s) == Monitor::Report::None,
        "usage within the budget is reported again");
}

} // namespace

int main() {
  TestLimit();
  TestMarks();
  TestMonitor();
  TestLostAccessReport();
  std::printf("MemoryBudgetTests: all tests passed\n");
  return 0;
}
