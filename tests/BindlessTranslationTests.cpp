#include "graphics/host_gpu/renderer/pipeline/bindlessTranslation.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::BindlessTranslation;
constexpr auto Pending = Libs::Graphics::ShaderRecompiler::IR::BindlessPending;

void Check(bool value, const char* text) {
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
	BindlessTranslation   table(entries, 4);
	Check(entries[0] == BindlessTranslation::PlaceholderSlot && entries[1] == Pending,
	      "entry 0 lands on the placeholder, the rest are pending");

	const auto heap = table.FindOrCreateHeap(Key(0x1000), 3);
	Check(heap != BindlessTranslation::NoHeap && table.GetHeap(heap).region == 1 &&
	          table.GetHeap(heap).Entries() == 12,
	      "a heap takes a region of records * layers entries");
	Check(table.FindOrCreateHeap(Key(0x1000), 2) == heap, "a smaller use shares the heap");

	BindlessTranslation::Request request;
	Check(table.Decode(1 + 2 * 4 + 3, request) && request.heap == heap && request.record == 2 &&
	          request.layer == 3,
	      "an entry decodes to its record and layer");
	Check(!table.Decode(0, request) && !table.Decode(13, request),
	      "the sentinel and entries outside any heap decode to nothing");

	const auto slot = table.AllocateSlot(Common::SlotId {7}, 0x10);
	Check(slot != BindlessTranslation::NoSlot && slot != BindlessTranslation::PlaceholderSlot,
	      "a slot is allocated beside the placeholder");
	table.Translate(11, slot);
	Check(entries[11] == slot && !table.Decode(11, request), "a translated entry is not pending");
	Check(table.FindSlot(Common::SlotId {7}, 0x10) == slot &&
	          table.FindSlot(Common::SlotId {7}, 0x20) == BindlessTranslation::NoSlot,
	      "a slot is found by image and view");
	Check(table.GetHeap(heap).slots.size() == 1 && table.GetHeap(heap).slots[0] == slot,
	      "the heap lists the slot its keys resolved to");
	table.Translate(5, BindlessTranslation::PlaceholderSlot);
	Check(entries[5] == 0 && table.GetHeap(heap).slots.size() == 1,
	      "the placeholder is not listed as a heap slot");

	const auto grown = table.FindOrCreateHeap(Key(0x1000), 5);
	Check(grown == heap && table.GetHeap(heap).region == 13 && table.GetHeap(heap).records == 5,
	      "a larger use moves the heap to a new region");
	Check(entries[11] == slot && entries[13 + 10] == slot && entries[13 + 4] == 0,
	      "the old region keeps its translations and the new one copies them");
	Check(!table.Decode(1, request) && table.Decode(13 + 1, request) && request.record == 0,
	      "entries of the old region no longer decode");
	Check(table.GetSlot(slot).entries.size() == 1 && table.GetSlot(slot).entries[0] == 13 + 10,
	      "the slot's entries follow the heap");

	table.FreeSlot(slot);
	Check(entries[13 + 10] == Pending && table.GetHeap(heap).slots.empty() &&
	          !table.GetSlot(slot).InUse(),
	      "freeing a slot makes its entries pending again");
	Check(table.AllocateSlot(Common::SlotId {8}, 0) == slot, "a freed slot is reused");
	Check(table.AllocateSlot(Common::SlotId {9}, 0) != BindlessTranslation::NoSlot &&
	          table.AllocateSlot(Common::SlotId {10}, 0) != BindlessTranslation::NoSlot &&
	          table.AllocateSlot(Common::SlotId {11}, 0) == BindlessTranslation::NoSlot,
	      "slots run out at the capacity");
	Check(table.FindOrCreateHeap(Key(0x2000), 20) == BindlessTranslation::NoHeap,
	      "a heap that does not fit is refused");
	Check(table.FindOrCreateHeap(Key(0x2000), 5) != BindlessTranslation::NoHeap &&
	          table.UsedEntries() == 53,
	      "regions are appended");
}

} // namespace

int main() {
	TestHeapsAndKeys();
	std::printf("BindlessTranslationTests: all tests passed\n");
	return 0;
}
