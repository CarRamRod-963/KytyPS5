#ifndef KYTY_COMMON_RANGE_DIRECTORY_H_
#define KYTY_COMMON_RANGE_DIRECTORY_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace Common {

// Address ranges of the objects of a cache, by slot index, readable from any thread without a
// lock: the owner of the cache (one writer thread) sets a slot when it registers an object and
// clears it when it unregisters it, and a fault report on another thread lists the objects over
// an address range. Each slot is a sequence lock: odd while it is written, so a reader skips a
// slot it caught being changed (and counts it). Setting or clearing a slot is a few relaxed
// stores; slots at Capacity and above are not kept.
class RangeDirectory {
public:
	static constexpr size_t Capacity = size_t {1} << 16u;

	struct Range {
		uint64_t base  = 0;
		uint64_t size  = 0;
		uint32_t flags = 0;
	};

	RangeDirectory(): m_slots(std::make_unique<Slot[]>(Capacity)) {}

	void Set(uint32_t index, uint64_t base, uint64_t size, uint32_t flags = 0) noexcept {
		if (index >= Capacity) {
			return;
		}
		auto& slot = m_slots[index];
		Begin(slot);
		slot.base.store(base, std::memory_order_relaxed);
		slot.size.store(size, std::memory_order_relaxed);
		slot.flags.store(flags, std::memory_order_relaxed);
		End(slot);
		auto used = m_used.load(std::memory_order_relaxed);
		while (used <= index &&
		       !m_used.compare_exchange_weak(used, index + 1, std::memory_order_release)) {
		}
	}

	void Clear(uint32_t index) noexcept {
		if (index >= Capacity) {
			return;
		}
		auto& slot = m_slots[index];
		Begin(slot);
		slot.size.store(0, std::memory_order_relaxed);
		End(slot);
	}

	// Sets or clears flag bits of a kept slot.
	void SetFlags(uint32_t index, uint32_t mask, bool set) noexcept {
		if (index >= Capacity) {
			return;
		}
		auto&      slot  = m_slots[index];
		const auto flags = slot.flags.load(std::memory_order_relaxed);
		Begin(slot);
		slot.flags.store(set ? (flags | mask) : (flags & ~mask), std::memory_order_relaxed);
		End(slot);
	}

	// Calls f(index, range) for every kept range that overlaps [begin, end), in slot order, and
	// returns how many slots were being written and left out.
	template <typename F>
	size_t ForEachOverlap(uint64_t begin, uint64_t end, F&& f) const noexcept {
		size_t     busy = 0;
		const auto used = m_used.load(std::memory_order_acquire);
		for (uint32_t index = 0; index < used; index++) {
			const auto& slot     = m_slots[index];
			const auto  sequence = slot.sequence.load(std::memory_order_acquire);
			if ((sequence & 1u) != 0) {
				busy++;
				continue;
			}
			Range range;
			range.base  = slot.base.load(std::memory_order_relaxed);
			range.size  = slot.size.load(std::memory_order_relaxed);
			range.flags = slot.flags.load(std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_acquire);
			if (slot.sequence.load(std::memory_order_relaxed) != sequence) {
				busy++;
				continue;
			}
			if (range.size != 0 && range.base < end && begin < range.base + range.size) {
				f(index, range);
			}
		}
		return busy;
	}

private:
	struct Slot {
		std::atomic<uint32_t> sequence {0};
		std::atomic<uint32_t> flags {0};
		std::atomic<uint64_t> base {0};
		std::atomic<uint64_t> size {0};
	};

	static void Begin(Slot& slot) noexcept {
		slot.sequence.store(slot.sequence.load(std::memory_order_relaxed) + 1u,
		                    std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
	}
	static void End(Slot& slot) noexcept {
		slot.sequence.store(slot.sequence.load(std::memory_order_relaxed) + 1u,
		                    std::memory_order_release);
	}

	std::unique_ptr<Slot[]> m_slots;
	std::atomic<uint32_t>   m_used {0};
};

} // namespace Common

#endif /* KYTY_COMMON_RANGE_DIRECTORY_H_ */
