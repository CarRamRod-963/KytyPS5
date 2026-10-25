#include "graphics/host_gpu/renderer/pipeline/bindlessTranslation.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::BindlessTranslation;
constexpr auto Pending = Libs::Graphics::ShaderRecompiler::IR::BindlessPending;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "BindlessTranslationTests: failed: %s\n", text);
    std::abort();
  }
}

BindlessTranslation::HeapKey Key(uint64_t base) {
  return {base, 8, 0, 64, 0x170, 4};
}

void TestHeapsAndKeys() {
  std::vector<uint32_t> entries(64, 0);
  BindlessTranslation table(entries, 4);
  Check(entries[0] == BindlessTranslation::PlaceholderSlot &&
            entries[1] == Pending,
        "entry 0 lands on the placeholder, the rest are pending");

  const auto heap = table.FindOrCreateHeap(Key(0x1000), 3);
  Check(heap != BindlessTranslation::NoHeap &&
            table.GetHeap(heap).region == 1 &&
            table.GetHeap(heap).Entries() == 12,
        "a heap takes a region of records * layers entries");
  Check(table.FindOrCreateHeap(Key(0x1000), 2) == heap,
        "a smaller use shares the heap");

  BindlessTranslation::Request request;
  Check(table.Decode(1 + 2 * 4 + 3, request) && request.heap == heap &&
            request.record == 2 && request.layer == 3,
        "an entry decodes to its record and layer");
  Check(!table.Decode(0, request) && !table.Decode(13, request),
        "the sentinel and entries outside any heap decode to nothing");

  const auto slot = table.AllocateSlot(Common::SlotId{7}, 70, 0x10);
  Check(slot != BindlessTranslation::NoSlot &&
            slot != BindlessTranslation::PlaceholderSlot,
        "a slot is allocated beside the placeholder");
  table.Translate(11, slot);
  Check(entries[11] == slot && !table.Decode(11, request),
        "a translated entry is not pending");
  Check(table.FindSlot(Common::SlotId{7}, 70, 0x10) == slot &&
            table.FindSlot(Common::SlotId{7}, 70, 0x20) ==
                BindlessTranslation::NoSlot,
        "a slot is found by image and view");
  // The texture cache gives a deleted image's id to the next image it
  // inserts, and the driver may give a destroyed view's handle to the
  // next view.
  Check(table.FindSlot(Common::SlotId{7}, 71, 0x10) ==
                BindlessTranslation::NoSlot &&
            table.GetSlot(slot).serial == 70,
        "an image that reuses the id of the slot's image does not find "
        "the slot");
  Check(table.GetHeap(heap).slots.size() == 1 &&
            table.GetHeap(heap).slots[0] == slot,
        "the heap lists the slot its keys resolved to");
  table.Translate(5, BindlessTranslation::PlaceholderSlot);
  Check(entries[5] == 0 && table.GetHeap(heap).slots.size() == 1,
        "the placeholder is not listed as a heap slot");

  const auto grown = table.FindOrCreateHeap(Key(0x1000), 5);
  Check(grown == heap && table.GetHeap(heap).region == 13 &&
            table.GetHeap(heap).records == 5,
        "a larger use moves the heap to a new region");
  Check(entries[11] == slot && entries[13 + 10] == slot && entries[13 + 4] == 0,
        "the old region keeps its translations and the new one copies them");
  Check(!table.Decode(1, request) && table.Decode(13 + 1, request) &&
            request.record == 0,
        "entries of the old region no longer decode");
  Check(table.GetSlot(slot).entries.size() == 1 &&
            table.GetSlot(slot).entries[0] == 13 + 10,
        "the slot's entries follow the heap");

  table.FreeSlot(slot);
  Check(entries[13 + 10] == Pending && table.GetHeap(heap).slots.empty() &&
            !table.GetSlot(slot).InUse(),
        "freeing a slot makes its entries pending again");
  Check(table.AllocateSlot(Common::SlotId{8}, 80, 0) == slot,
        "a freed slot is reused");
  Check(table.AllocateSlot(Common::SlotId{9}, 90, 0) !=
                BindlessTranslation::NoSlot &&
            table.AllocateSlot(Common::SlotId{10}, 100, 0) !=
                BindlessTranslation::NoSlot &&
            table.AllocateSlot(Common::SlotId{11}, 110, 0) ==
                BindlessTranslation::NoSlot,
        "slots run out at the capacity");
  Check(table.FindOrCreateHeap(Key(0x2000), 20) == BindlessTranslation::NoHeap,
        "a heap that does not fit is refused");
  Check(table.FindOrCreateHeap(Key(0x2000), 5) != BindlessTranslation::NoHeap &&
            table.UsedEntries() == 53,
        "regions are appended");
}

// Heaps keyed by per-frame addresses fill the append-only table; a reset (with
// no GPU work in flight) must make room again without forgetting which images
// the slots hold.
void TestReset() {
  std::vector<uint32_t> entries(32, 0);
  BindlessTranslation table(entries, 4);
  uint64_t base = 0x1000;
  uint32_t heaps = 0;
  while (table.FindOrCreateHeap(Key(base), 2) != BindlessTranslation::NoHeap) {
    base += 0x1000;
    ++heaps;
  }
  Check(heaps == 3 && table.UsedEntries() == 25 && table.Capacity() == 32,
        "per-frame heaps fill the table");
  const auto slot = table.AllocateSlot(Common::SlotId{7}, 70, 0x10);
  table.Translate(9 + 1, slot);
  Check(table.GetSlot(slot).entries.size() == 1, "the slot lists its entry");

  table.Reset();
  Check(table.UsedEntries() == BindlessTranslation::FirstEntry,
        "a reset frees every region");
  bool pending = true;
  for (uint32_t entry = BindlessTranslation::FirstEntry; entry < entries.size();
       ++entry) {
    pending &= entries[entry] == Pending;
  }
  Check(pending && entries[0] == BindlessTranslation::PlaceholderSlot,
        "a reset makes every entry pending and keeps the placeholder entry");
  BindlessTranslation::Request request;
  Check(!table.Decode(10, request), "entries of dropped heaps do not decode");
  Check(table.GetSlot(slot).InUse() && table.GetSlot(slot).entries.empty() &&
            table.FindSlot(Common::SlotId{7}, 70, 0x10) == slot,
        "slots keep their images and forget the dropped entries");

  const auto heap = table.FindOrCreateHeap(Key(base), 2);
  Check(heap == 0 &&
            table.GetHeap(heap).region == BindlessTranslation::FirstEntry,
        "the next heap starts at the first entry again");
  table.Translate(BindlessTranslation::FirstEntry, slot);
  table.FreeSlot(slot);
  Check(entries[BindlessTranslation::FirstEntry] == Pending &&
            entries[10] == Pending,
        "freeing a slot after a reset touches only its new entries");
}

} // namespace

int main() {
  TestHeapsAndKeys();
  TestReset();
  std::printf("BindlessTranslationTests: all tests passed\n");
  return 0;
}
