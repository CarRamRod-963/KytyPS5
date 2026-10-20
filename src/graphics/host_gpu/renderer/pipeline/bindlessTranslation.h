#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTRANSLATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTRANSLATION_H_

#include "common/common.h"
#include "common/slotVector.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

// The key -> slot translation of the bindless image table (IR::BindlessDescriptorSet): one
// region of entries per heap of guest records a shader reaches through per-lane pointers, one
// slot per host image view. An entry is IR::BindlessPending until the renderer resolves the
// key a shader flagged; the entries live in the storage the GPU reads (a mapped buffer), so a
// translation is visible to the next submission. Regions are append-only: nothing a submitted
// frame may read is given another meaning. Heaps are keyed by guest address, and a game that
// builds its records in per-frame memory creates new heaps every frame, so the owner resets
// the table (Reset) once no submitted work can read it any more.
class BindlessTranslation {
public:
	// Entry 0 is where out-of-range keys land; slot 0 holds the placeholder image.
	static constexpr uint32_t FirstEntry      = 1;
	static constexpr uint32_t PlaceholderSlot = 0;
	static constexpr uint32_t NoHeap          = UINT32_MAX;
	static constexpr uint32_t NoSlot          = UINT32_MAX;

	struct HeapKey {
		uint64_t base           = 0;
		uint32_t record_stride  = 0;
		uint32_t pointer_offset = 0;
		uint32_t table_offset   = 0;
		uint32_t layer_stride   = 0;
		uint32_t layer_count    = 1;

		bool operator==(const HeapKey&) const = default;
	};

	struct Heap {
		HeapKey               key;
		uint32_t              records = 0;
		uint32_t              region  = 0;
		std::vector<uint32_t> slots;

		[[nodiscard]] uint32_t Entries() const { return records * key.layer_count; }
	};

	struct Slot {
		Common::SlotId        image;
		uint64_t              view = 0;
		std::vector<uint32_t> entries;

		[[nodiscard]] bool InUse() const { return static_cast<bool>(image); }
	};

	struct Request {
		uint32_t heap   = NoHeap;
		uint32_t record = 0;
		uint32_t layer  = 0;
	};

	BindlessTranslation(std::span<uint32_t> entries, uint32_t slot_capacity);

	// The heap of `key` with room for `records` records, or NoHeap when no region fits. A heap
	// that grows moves to a new region and keeps its translations.
	[[nodiscard]] uint32_t    FindOrCreateHeap(const HeapKey& key, uint32_t records);
	[[nodiscard]] const Heap& GetHeap(uint32_t heap) const { return m_heaps[heap]; }
	[[nodiscard]] uint32_t    UsedEntries() const { return m_used_entries; }
	[[nodiscard]] uint32_t    Capacity() const { return static_cast<uint32_t>(m_entries.size()); }

	// The key a flagged entry stands for; false when the entry is pending in no live region.
	[[nodiscard]] bool Decode(uint32_t entry, Request& request) const;

	[[nodiscard]] uint32_t    FindSlot(Common::SlotId image, uint64_t view) const;
	[[nodiscard]] uint32_t    AllocateSlot(Common::SlotId image, uint64_t view);
	[[nodiscard]] const Slot& GetSlot(uint32_t slot) const { return m_slots[slot]; }
	[[nodiscard]] uint32_t    SlotCount() const { return static_cast<uint32_t>(m_slots.size()); }
	void                      Translate(uint32_t entry, uint32_t slot);
	// The slot's entries become pending again (its image is gone).
	void FreeSlot(uint32_t slot);
	// Drops every heap and region: all entries become pending and the next heap starts at
	// FirstEntry again. Slots keep their images. Only valid while no GPU work reads the entries.
	void Reset();

private:
	struct Region {
		uint32_t first = 0;
		uint32_t count = 0;
		uint32_t heap  = NoHeap;
	};

	[[nodiscard]] const Region* FindRegion(uint32_t entry) const;

	std::span<uint32_t>   m_entries;
	uint32_t              m_used_entries = FirstEntry;
	std::vector<Heap>     m_heaps;
	std::vector<Region>   m_regions;
	std::vector<Slot>     m_slots;
	std::vector<uint32_t> m_free_slots;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTRANSLATION_H_
