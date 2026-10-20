#include "graphics/host_gpu/renderer/pipeline/bindlessTranslation.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>

namespace Libs::Graphics {

BindlessTranslation::BindlessTranslation(std::span<uint32_t> entries, uint32_t slot_capacity)
    : m_entries(entries), m_slots(slot_capacity) {
	EXIT_IF(entries.size() <= FirstEntry || slot_capacity <= PlaceholderSlot + 1);
	std::ranges::fill(m_entries, ShaderRecompiler::IR::BindlessPending);
	m_entries[0] = PlaceholderSlot;
	m_slots[PlaceholderSlot].image = Common::SlotId {0};
	m_free_slots.reserve(slot_capacity - 1);
	for (uint32_t slot = slot_capacity - 1; slot > PlaceholderSlot; --slot) {
		m_free_slots.push_back(slot);
	}
}

uint32_t BindlessTranslation::FindOrCreateHeap(const HeapKey& key, uint32_t records) {
	EXIT_IF(key.layer_count == 0);
	const auto found = std::ranges::find(m_heaps, key, &Heap::key);
	const auto heap  = found == m_heaps.end() ? NoHeap
	                                          : static_cast<uint32_t>(found - m_heaps.begin());
	if (heap != NoHeap && records <= found->records) {
		return heap;
	}
	const auto count = static_cast<uint64_t>(records) * key.layer_count;
	if (count > m_entries.size() - m_used_entries) {
		return NoHeap;
	}
	const auto region = m_used_entries;
	m_used_entries += static_cast<uint32_t>(count);
	if (heap == NoHeap) {
		const auto created = static_cast<uint32_t>(m_heaps.size());
		m_heaps.push_back({key, records, region, {}});
		m_regions.push_back({region, static_cast<uint32_t>(count), created});
		return created;
	}
	// The old region keeps its translations for submissions still reading it.
	auto& grown = *found;
	std::ranges::copy(m_entries.subspan(grown.region, grown.Entries()),
	                  m_entries.begin() + region);
	const auto shift = region - grown.region;
	for (const auto slot: grown.slots) {
		for (auto& entry: m_slots[slot].entries) {
			if (entry >= grown.region && entry < grown.region + grown.Entries()) {
				entry += shift;
			}
		}
	}
	for (auto& old: m_regions) {
		if (old.heap == heap) {
			old.heap = NoHeap;
		}
	}
	grown.region  = region;
	grown.records = records;
	m_regions.push_back({region, static_cast<uint32_t>(count), heap});
	return heap;
}

const BindlessTranslation::Region* BindlessTranslation::FindRegion(uint32_t entry) const {
	// Regions are appended in entry order.
	const auto next = std::ranges::upper_bound(m_regions, entry, {}, &Region::first);
	if (next == m_regions.begin()) {
		return nullptr;
	}
	const auto& region = *std::prev(next);
	return entry < region.first + region.count ? &region : nullptr;
}

bool BindlessTranslation::Decode(uint32_t entry, Request& request) const {
	if (entry < FirstEntry || entry >= m_used_entries ||
	    m_entries[entry] != ShaderRecompiler::IR::BindlessPending) {
		return false;
	}
	const auto* region = FindRegion(entry);
	if (region == nullptr || region->heap == NoHeap) {
		return false;
	}
	const auto& heap = m_heaps[region->heap];
	const auto  key  = entry - region->first;
	request          = {region->heap, key / heap.key.layer_count, key % heap.key.layer_count};
	return true;
}

uint32_t BindlessTranslation::FindSlot(Common::SlotId image, uint64_t view) const {
	for (uint32_t slot = PlaceholderSlot + 1; slot < m_slots.size(); ++slot) {
		if (m_slots[slot].image == image && m_slots[slot].view == view) {
			return slot;
		}
	}
	return NoSlot;
}

uint32_t BindlessTranslation::AllocateSlot(Common::SlotId image, uint64_t view) {
	EXIT_IF(!image);
	if (m_free_slots.empty()) {
		return NoSlot;
	}
	const auto slot = m_free_slots.back();
	m_free_slots.pop_back();
	m_slots[slot] = {image, view, {}};
	return slot;
}

void BindlessTranslation::Translate(uint32_t entry, uint32_t slot) {
	EXIT_IF(entry < FirstEntry || entry >= m_used_entries || slot >= m_slots.size() ||
	        !m_slots[slot].InUse());
	const auto* region = FindRegion(entry);
	EXIT_IF(region == nullptr || region->heap == NoHeap);
	m_entries[entry] = slot;
	if (slot == PlaceholderSlot) {
		return;
	}
	m_slots[slot].entries.push_back(entry);
	auto& slots = m_heaps[region->heap].slots;
	if (std::ranges::find(slots, slot) == slots.end()) {
		slots.push_back(slot);
	}
}

void BindlessTranslation::Reset() {
	std::fill(m_entries.begin() + FirstEntry, m_entries.begin() + m_used_entries,
	          ShaderRecompiler::IR::BindlessPending);
	m_used_entries = FirstEntry;
	m_heaps.clear();
	m_regions.clear();
	for (auto& slot: m_slots) {
		slot.entries.clear();
	}
}

void BindlessTranslation::FreeSlot(uint32_t slot) {
	EXIT_IF(slot == PlaceholderSlot || slot >= m_slots.size() || !m_slots[slot].InUse());
	for (const auto entry: m_slots[slot].entries) {
		m_entries[entry] = ShaderRecompiler::IR::BindlessPending;
	}
	for (auto& heap: m_heaps) {
		std::erase(heap.slots, slot);
	}
	m_slots[slot] = {};
	m_free_slots.push_back(slot);
}

} // namespace Libs::Graphics
