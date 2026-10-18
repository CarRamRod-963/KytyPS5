#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_DESCRIPTORINTERNER_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_DESCRIPTORINTERNER_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Finds a descriptor value among the distinct values of one candidate list, the way a linear
// std::find over the list does, in expected constant time. The caller owns the list and appends
// each value Find did not find; the interner only holds indices into it. A short list is
// searched linearly; once it grows past LinearLimit the interner hashes it. Slots carry the
// generation of the list they belong to, so starting a list costs nothing and a warm interner
// allocates nothing.
class DescriptorInterner {
public:
	static constexpr uint32_t NotFound    = UINT32_MAX;
	static constexpr size_t   LinearLimit = 8;

	// Starts a new, empty list.
	void Reset() noexcept {
		m_hashed = false;
		if (++m_generation == 0u) {
			std::ranges::fill(m_slots, Slot {});
			m_generation = 1u;
		}
	}

	// The index of value in list, which holds exactly the values appended since Reset, or
	// NotFound.
	[[nodiscard]] uint32_t Find(std::span<const DescriptorValue> list,
	                            const DescriptorValue&           value) const noexcept {
		if (!m_hashed) {
			const auto found = std::find(list.begin(), list.end(), value);
			return found == list.end() ? NotFound : static_cast<uint32_t>(found - list.begin());
		}
		const auto mask = m_slots.size() - 1u;
		for (auto slot = Hash(value) & mask;; slot = (slot + 1u) & mask) {
			const auto& entry = m_slots[slot];
			if (entry.generation != m_generation) return NotFound;
			if (list[entry.index] == value) return entry.index;
		}
	}

	// Records the value just appended at the end of list (one Find did not find).
	void Added(std::span<const DescriptorValue> list) {
		if (!m_hashed) {
			if (list.size() <= LinearLimit) return;
			Rebuild(list, std::bit_ceil(list.size() * 4u));
			return;
		}
		if (list.size() * 2u > m_slots.size()) {
			Rebuild(list, m_slots.size() * 2u);
			return;
		}
		Insert(list, static_cast<uint32_t>(list.size() - 1u));
	}

private:
	struct Slot {
		uint32_t generation = 0;
		uint32_t index      = 0;
	};

	[[nodiscard]] static size_t Hash(const DescriptorValue& value) noexcept {
		uint64_t hash = (value.dword_count + 1ull) * 0x9e3779b97f4a7c15ull;
		for (const auto dword: value.dwords) {
			hash = (hash ^ dword) * 0xff51afd7ed558ccdull;
			hash ^= hash >> 32u;
		}
		return static_cast<size_t>(hash);
	}

	void Insert(std::span<const DescriptorValue> list, uint32_t index) noexcept {
		const auto mask = m_slots.size() - 1u;
		auto       slot = Hash(list[index]) & mask;
		while (m_slots[slot].generation == m_generation)
			slot = (slot + 1u) & mask;
		m_slots[slot] = {m_generation, index};
	}

	void Rebuild(std::span<const DescriptorValue> list, size_t capacity) {
		if (capacity > m_slots.size()) {
			m_slots.assign(capacity, Slot {});
			m_generation = 1u;
		} else if (++m_generation == 0u) {
			std::ranges::fill(m_slots, Slot {});
			m_generation = 1u;
		}
		m_hashed = true;
		for (uint32_t index = 0; index < list.size(); ++index)
			Insert(list, index);
	}

	std::vector<Slot> m_slots;
	uint32_t          m_generation = 0;
	bool              m_hashed     = false;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_DESCRIPTORINTERNER_H_
