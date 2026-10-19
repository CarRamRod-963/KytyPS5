#include "graphics/host_gpu/memoryBudget.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::MemoryBudget::CacheLimit;
using Libs::Graphics::MemoryBudget::DriverHeadroom;
using Libs::Graphics::MemoryBudget::DriverUsage;
using Libs::Graphics::MemoryBudget::GiB;
using Libs::Graphics::MemoryBudget::HeapState;
using Libs::Graphics::MemoryBudget::MiB;
using Libs::Graphics::MemoryBudget::Monitor;

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
}

} // namespace

int main() {
  TestLimit();
  TestMonitor();
  std::printf("MemoryBudgetTests: all tests passed\n");
  return 0;
}
