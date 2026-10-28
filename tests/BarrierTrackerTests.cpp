#include "graphics/host_gpu/renderer/barrierTracker.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using Libs::Graphics::BarrierTracker;
using Libs::Graphics::CopyRunTracker;
using Barrier = BarrierTracker::Barrier;
using CopyBarrier = CopyRunTracker::Barrier;
using CopyRange = CopyRunTracker::Range;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "BarrierTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

void TestSerializedDispatchSequences() {
  BarrierTracker tracker;
  tracker.SetSerializeDispatches(true);
  Check(tracker.BeforeDispatch(true) == Barrier::AllToCompute,
        "the first writing dispatch waits for everything recorded before");
  Check(tracker.BeforeDispatch(true) == Barrier::ComputeToCompute,
        "write after write between dispatches");
  Check(tracker.BeforeDispatch(false) == Barrier::ComputeToCompute,
        "read after write between dispatches");
  Check(tracker.BeforeDispatch(false) == Barrier::None,
        "two reading dispatches need no barrier");
  Check(tracker.BeforeDispatch(true) == Barrier::ComputeToCompute,
        "write after read between dispatches");
  Check(tracker.HasPendingDispatches(), "dispatches wait for their barrier");
  Check(tracker.BeforeCommand() == Barrier::ComputeToAll,
        "another command waits for the dispatches");
  Check(!tracker.HasPendingDispatches(), "the barrier is recorded once");
  Check(tracker.BeforeCommand() == Barrier::None,
        "commands other than dispatches keep their own barriers");
  Check(tracker.BeforeDispatch(false) == Barrier::None,
        "a reading dispatch after another command keeps the old order");
  Check(tracker.BeforeDispatch(true) == Barrier::AllToCompute,
        "a writing dispatch after another command waits for everything");
}

void TestDispatchRuns() {
  BarrierTracker tracker;
  Check(tracker.Flush() == Barrier::AllToAll,
        "a flush with unknown earlier work is recorded");
  Check(tracker.BeforeDispatch(true) == Barrier::None,
        "the flush ordered everything before the first dispatch");
  Check(tracker.BeforeDispatch(true) == Barrier::None,
        "dispatches without a guest flush between them may overlap");
  Check(tracker.BeforeDispatch(false) == Barrier::None,
        "a read without a guest flush may overlap the writes");
  Check(tracker.Flush() == Barrier::None,
        "a flush after dispatches only is left to the next dispatch");
  Check(tracker.BeforeDispatch(false) == Barrier::ComputeToCompute,
        "a guest flush orders the next dispatch after the earlier writes");
  Check(tracker.BeforeDispatch(false) == Barrier::None,
        "two reading dispatches need no barrier");
  Check(tracker.BeforeDispatch(true, true) == Barrier::None,
        "a dispatch using shared emulator memory may overlap the others");
  Check(tracker.BeforeDispatch(false, true) == Barrier::ComputeToCompute,
        "two dispatches using shared emulator memory are ordered");
  Check(tracker.BeforeDispatch(true) == Barrier::None,
        "a dispatch without shared memory may overlap them");
  Check(tracker.BeforeCommand() == Barrier::ComputeToAll,
        "another command waits for the dispatches");
}

void TestFlushes() {
  BarrierTracker tracker;
  Check(tracker.Flush() == Barrier::AllToAll,
        "a flush with unknown earlier work is recorded");
  Check(tracker.Flush() == Barrier::None, "a repeated flush is dropped");
  Check(tracker.BeforeDispatch(true) == Barrier::None,
        "a flush already ordered everything before the dispatch");
  Check(tracker.Flush() == Barrier::None,
        "a flush after dispatches only is left to the dispatch barriers");
  Check(tracker.BeforeDispatch(false) == Barrier::ComputeToCompute,
        "the dropped flush still orders the dispatches");
  Check(tracker.BeforeBarrier() == Barrier::ComputeToAll,
        "a barrier recorded elsewhere comes after the dispatches");
  Check(tracker.BeforeBarrier() == Barrier::None,
        "a barrier is no work of its own");
  Check(tracker.Flush() == Barrier::None,
        "nothing but dispatches since the flush");
  Check(tracker.BeforeCommand() == Barrier::None, "nothing pending");
  Check(tracker.Flush() == Barrier::AllToAll,
        "a flush after another command is recorded");
  Check(tracker.BeforeDispatch(true) == Barrier::None,
        "the recorded flush covers the next dispatch");
}

// Execution-dependency model with two stages: guest compute and everything
// else. Item j is ordered after item i when a chain of barriers between them
// starts with a barrier whose source scope contains i's stage and ends with one
// whose destination scope contains j's. The barriers' access masks always cover
// the accesses of their stages, so an execution chain is also a memory
// dependency chain.
constexpr uint8_t Compute = 1;
constexpr uint8_t Other = 2;
constexpr uint8_t All = Compute | Other;

struct Item {
  bool barrier = false;
  // A dispatch using emulator memory shared between dispatches.
  bool shared = false;
  uint32_t flushes = 0; // Guest flushes recorded before the command.
  uint8_t stage = 0; // A command's stage.
  bool writes = false;
  uint8_t source = 0; // A barrier's scopes.
  uint8_t destination = 0;
  uint32_t command = 0; // Index of the command in the generated sequence.
};

void AddBarrier(std::vector<Item> &items, Barrier barrier) {
  switch (barrier) {
  case Barrier::None:
    return;
  case Barrier::ComputeToCompute:
    items.push_back(
        {.barrier = true, .source = Compute, .destination = Compute});
    return;
  case Barrier::AllToCompute:
    items.push_back({.barrier = true, .source = All, .destination = Compute});
    return;
  case Barrier::ComputeToAll:
    items.push_back({.barrier = true, .source = Compute, .destination = All});
    return;
  case Barrier::AllToAll:
    items.push_back({.barrier = true, .source = All, .destination = All});
    return;
  }
}

// ordered[i][j]: command j is ordered after command i.
std::vector<std::vector<bool>> Ordered(const std::vector<Item> &items,
                                       uint32_t commands) {
  std::vector<std::vector<bool>> ordered(commands,
                                         std::vector<bool>(commands, false));
  for (size_t i = 0; i < items.size(); i++) {
    if (items[i].barrier) {
      continue;
    }
    uint8_t reached = 0;
    for (size_t j = i + 1; j < items.size(); j++) {
      const auto &item = items[j];
      if (item.barrier) {
        if ((item.source & (items[i].stage | reached)) != 0) {
          reached |= item.destination;
        }
      } else if ((item.stage & reached) != 0) {
        ordered[items[i].command][item.command] = true;
      }
    }
  }
  return ordered;
}

enum class Op {
  WriteDispatch,
  ReadDispatch,
  Command,
  Flush,
  ForeignBarrier,
  End
};

void TestAgainstPerDispatchBarriers(bool serialize) {
  std::mt19937 random(serialize ? 1234 : 4321);
  uint64_t legacy_barriers = 0;
  uint64_t tracked_barriers = 0;
  for (int round = 0; round < 3000; round++) {
    const auto length = 1 + random() % 24;
    std::vector<Item> legacy;
    std::vector<Item> tracked;
    std::vector<Item> commands;
    BarrierTracker tracker;
    tracker.SetSerializeDispatches(serialize);
    uint32_t flushes = 0;
    // Unknown work before the sequence.
    const Item before{.stage = Other, .writes = true, .command = 0};
    legacy.push_back(before);
    tracked.push_back(before);
    commands.push_back(before);
    for (uint32_t i = 0; i < length; i++) {
      const auto op = static_cast<Op>(random() % 6);
      const auto command = static_cast<uint32_t>(commands.size());
      switch (op) {
      case Op::WriteDispatch:
      case Op::ReadDispatch: {
        const bool writes = op == Op::WriteDispatch;
        const bool shared = random() % 4 == 0;
        const Item item{.shared = shared,
                        .flushes = flushes,
                        .stage = Compute,
                        .writes = writes,
                        .command = command};
        if (writes) {
          AddBarrier(legacy, Barrier::AllToCompute);
        }
        legacy.push_back(item);
        AddBarrier(legacy, Barrier::ComputeToAll);
        AddBarrier(tracked, tracker.BeforeDispatch(writes, shared));
        tracked.push_back(item);
        commands.push_back(item);
        break;
      }
      case Op::Command: {
        const Item item{.stage = Other, .writes = true, .command = command};
        legacy.push_back(item);
        AddBarrier(tracked, tracker.BeforeCommand());
        tracked.push_back(item);
        commands.push_back(item);
        break;
      }
      case Op::Flush:
        flushes++;
        AddBarrier(legacy, Barrier::AllToAll);
        AddBarrier(tracked, tracker.Flush());
        break;
      case Op::ForeignBarrier:
      case Op::End:
        AddBarrier(tracked, tracker.BeforeBarrier());
        break;
      }
    }
    // Unknown work after the sequence.
    const Item after{.stage = Other,
                     .writes = true,
                     .command = static_cast<uint32_t>(commands.size())};
    legacy.push_back(after);
    AddBarrier(tracked, tracker.BeforeCommand());
    tracked.push_back(after);
    commands.push_back(after);

    const auto count = static_cast<uint32_t>(commands.size());
    const auto old_order = Ordered(legacy, count);
    const auto new_order = Ordered(tracked, count);
    for (uint32_t i = 0; i < count; i++) {
      for (uint32_t j = i + 1; j < count; j++) {
        const bool dispatches =
            commands[i].stage == Compute && commands[j].stage == Compute;
        const bool read_after_read =
            dispatches && !commands[i].writes && !commands[j].writes;
        // Without a guest flush between them the guest does not order two
        // dispatches either, unless both use emulator memory shared between
        // dispatches.
        const bool guest_unordered =
            !serialize && dispatches &&
            commands[i].flushes == commands[j].flushes &&
            !(commands[i].shared && commands[j].shared);
        if (old_order[i][j] && !read_after_read && !guest_unordered &&
            !new_order[i][j]) {
          std::fprintf(stderr, "round %d: commands %u -> %u lost their order\n",
                       round, i, j);
          Check(false,
                "every hazard the per-dispatch barriers ordered stays ordered");
        }
      }
    }
    for (const auto &item : legacy) {
      legacy_barriers += item.barrier ? 1 : 0;
    }
    for (const auto &item : tracked) {
      tracked_barriers += item.barrier ? 1 : 0;
    }
  }
  Check(tracked_barriers < legacy_barriers,
        "fewer barriers than one per dispatch side");
  std::printf("BarrierTrackerTests: %llu barriers instead of %llu\n",
              static_cast<unsigned long long>(tracked_barriers),
              static_cast<unsigned long long>(legacy_barriers));
}

void TestCopyRuns() {
  CopyRunTracker tracker;
  Check(!tracker.BeforeOther(), "no copies, no barrier");
  const CopyRange first[] = {{1, 0, 16}, {1, 64, 80}};
  Check(tracker.BeforeCopy(first) == CopyBarrier::AllToTransfer,
        "the first copy waits for everything recorded before");
  Check(tracker.HasPendingCopies(), "copies wait for their barrier");
  const CopyRange other_buffer[] = {{2, 0, 16}};
  Check(tracker.BeforeCopy(other_buffer) == CopyBarrier::None,
        "a copy into another buffer joins the run");
  const CopyRange disjoint[] = {{1, 16, 64}, {1, 80, 96}};
  Check(tracker.BeforeCopy(disjoint) == CopyBarrier::None,
        "a copy of adjacent bytes joins the run");
  const CopyRange overlap[] = {{1, 90, 100}};
  Check(tracker.BeforeCopy(overlap) == CopyBarrier::TransferToTransfer,
        "a copy over bytes the run wrote waits for that copy");
  const CopyRange after_barrier[] = {{1, 0, 90}};
  Check(tracker.BeforeCopy(after_barrier) == CopyBarrier::None,
        "the transfer barrier ordered the earlier copies");
  Check(tracker.BeforeCopy(after_barrier) == CopyBarrier::TransferToTransfer,
        "the same bytes twice are ordered");
  Check(tracker.BeforeOther(), "another command waits for the copies");
  Check(!tracker.HasPendingCopies() && !tracker.BeforeOther(),
        "the barrier after the copies is recorded once");
  Check(tracker.BeforeCopy(after_barrier) == CopyBarrier::AllToTransfer,
        "a copy after another command waits for it");
  const CopyRange empty[] = {{1, 8, 8}};
  Check(tracker.BeforeCopy(empty) == CopyBarrier::None,
        "an empty range writes nothing");
  Check(tracker.BeforeOther(), "the run ends");

  // Ranges merged from several copies still find overlaps inside them.
  const CopyRange low[] = {{3, 0, 100}};
  const CopyRange high[] = {{3, 100, 200}};
  const CopyRange inner[] = {{3, 150, 160}};
  Check(tracker.BeforeCopy(low) == CopyBarrier::AllToTransfer, "new run");
  Check(tracker.BeforeCopy(high) == CopyBarrier::None, "adjacent bytes");
  Check(tracker.BeforeCopy(inner) == CopyBarrier::TransferToTransfer,
        "bytes inside a merged range overlap");
  (void)tracker.BeforeOther();
}

// Random sequences of copies and other commands against a model that keeps
// every range: each copy must be ordered after every earlier command of another
// kind and after every earlier copy that wrote one of its bytes, and before
// every later command of another kind.
void TestCopyRunsAgainstModel() {
  std::mt19937 random(7);
  uint64_t copies = 0;
  uint64_t barriers = 0;
  for (int round = 0; round < 2000; round++) {
    CopyRunTracker tracker;
    // Copies recorded since the last barrier before a copy (all of their
    // ranges), and whether copies wait for their barrier to later commands.
    std::vector<CopyRange> unordered;
    bool pending = false;
    const int count = 1 + static_cast<int>(random() % 24);
    for (int i = 0; i < count; i++) {
      if (random() % 5 == 0) {
        const bool barrier = tracker.BeforeOther();
        Check(barrier == pending,
              "copies are ordered before the next other command");
        pending = false;
        unordered.clear();
        barriers += barrier ? 1 : 0;
        continue;
      }
      std::vector<CopyRange> ranges;
      const int regions = 1 + static_cast<int>(random() % 3);
      for (int r = 0; r < regions; r++) {
        const uint64_t buffer = 1 + random() % 3;
        const uint64_t begin = random() % 64;
        const uint64_t size = random() % 12;
        // The regions of one copy are disjoint.
        bool overlaps_own = false;
        for (const auto &own : ranges) {
          overlaps_own |= own.buffer == buffer && own.begin < begin + size &&
                          begin < own.end;
        }
        if (!overlaps_own) {
          ranges.push_back({buffer, begin, begin + size});
        }
      }
      bool hazard = false;
      for (const auto &range : ranges) {
        for (const auto &earlier : unordered) {
          hazard |= range.begin < range.end && earlier.begin < earlier.end &&
                    earlier.buffer == range.buffer &&
                    earlier.begin < range.end && range.begin < earlier.end;
        }
      }
      const auto barrier = tracker.BeforeCopy(ranges);
      copies++;
      if (!pending) {
        Check(barrier == CopyBarrier::AllToTransfer,
              "the first copy after another command waits for it");
      } else if (hazard) {
        Check(barrier == CopyBarrier::TransferToTransfer,
              "a copy over bytes of an unordered copy waits for it");
      } else {
        Check(barrier == CopyBarrier::None,
              "an independent copy records no barrier");
      }
      if (barrier != CopyBarrier::None) {
        unordered.clear();
        barriers++;
      }
      unordered.insert(unordered.end(), ranges.begin(), ranges.end());
      pending = true;
    }
    barriers += tracker.BeforeOther() ? 1 : 0;
  }
  Check(barriers < copies * 2, "fewer barriers than two per copy");
  std::printf("BarrierTrackerTests: %llu copy barriers instead of %llu\n",
              static_cast<unsigned long long>(barriers),
              static_cast<unsigned long long>(copies * 2));
}

} // namespace

int main() {
  TestSerializedDispatchSequences();
  TestDispatchRuns();
  TestFlushes();
  TestAgainstPerDispatchBarriers(true);
  TestAgainstPerDispatchBarriers(false);
  TestCopyRuns();
  TestCopyRunsAgainstModel();
  std::printf("BarrierTrackerTests: ok\n");
  return 0;
}
