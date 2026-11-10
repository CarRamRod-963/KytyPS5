#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PENDINGLABELS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PENDINGLABELS_H_

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

// End-of-pipe labels recorded by the command streams and not yet stored to guest memory. A
// label is stored when the submission before it completes, so a command stream that waits for
// one of these labels (WAIT_REG_MEM) waits for GPU work the emulator recorded itself. Such a wait
// can instead continue behind a barrier, because every Vulkan command recorded later executes
// after that work; the wait is then elided.
//
// What the barrier cannot cover is a host access to the label's own bytes: until the label is
// stored they still hold the old value. The GPU thread's reads of guest memory for commands after
// an elided wait (uploads, resource walks) see the elided labels overlaid (PatchElided); a packet
// that reads or writes the label itself, or a guest memory write on the GPU thread, first waits
// until the labels are stored. The list stays short: a label leaves it once it is stored.
class PendingLabels {
public:
	struct Label {
		uint64_t id      = 0;
		uint64_t address = 0;
		uint64_t value   = 0;
		uint64_t tick    = 0; // The submission the label completes with.
		uint32_t size    = 0; // 4 or 8 bytes.
		bool     elided  = false;
		bool     stored  = false; // Stored early by a settling access.
		bool     known   = true;  // A timestamp's value is known only once its submission ends.
	};

	// Returns the id that Stored() takes once the label is in guest memory. The submission it
	// completes with follows with SetTick(). A label whose value is not `known` yet gets it with
	// SetValue() before it is stored; no wait is elided on it until then.
	uint64_t Add(uint64_t address, uint32_t size, uint64_t value, bool known = true) {
		std::lock_guard lock(m_mutex);
		const auto      id = ++m_last_id;
		m_labels.push_back({id, address, value, 0, size, false, false, known});
		return id;
	}

	void SetValue(uint64_t id, uint64_t value) {
		std::lock_guard lock(m_mutex);
		if (const auto it = std::ranges::find(m_labels, id, &Label::id); it != m_labels.end()) {
			it->value = value;
			it->known = true;
		}
	}

	void SetTick(uint64_t id, uint64_t tick) {
		std::lock_guard lock(m_mutex);
		if (const auto it = std::ranges::find(m_labels, id, &Label::id); it != m_labels.end()) {
			it->tick = tick;
		}
	}

	void Stored(uint64_t id) {
		std::lock_guard lock(m_mutex);
		const auto      it = std::ranges::find(m_labels, id, &Label::id);
		if (it == m_labels.end()) {
			return;
		}
		if (it->elided) {
			m_elided.fetch_sub(1, std::memory_order_release);
		}
		m_labels.erase(it);
	}

	// Elides a wait for `size` bytes at `address` when the latest label that overlaps them covers
	// them all and its value satisfies `condition(value)`. The labels recorded later win: they are
	// stored later.
	template <typename Condition>
	[[nodiscard]] bool Elide(uint64_t address, uint32_t size, Condition&& condition) {
		std::lock_guard lock(m_mutex);
		auto*           label = Satisfying(address, size, condition);
		if (label == nullptr) {
			return false;
		}
		if (!label->elided) {
			label->elided = true;
			m_elided.fetch_add(1, std::memory_order_release);
		}
		return true;
	}

	// Whether Elide would find a label for the wait, without eliding it.
	template <typename Condition>
	[[nodiscard]] bool Satisfies(uint64_t address, uint32_t size, Condition&& condition) {
		std::lock_guard lock(m_mutex);
		return Satisfying(address, size, condition) != nullptr;
	}

	// Overlays the elided labels onto `size` bytes read from `address`, in recording order: what
	// the memory holds once they are stored.
	void PatchElided(uint64_t address, uint64_t size, void* data) const {
		std::lock_guard lock(m_mutex);
		for (const auto& label: m_labels) {
			if (!label.elided || !Overlaps(label, address, size)) {
				continue;
			}
			const auto begin = std::max(address, label.address);
			const auto end   = std::min(address + size, label.address + label.size);
			std::memcpy(static_cast<uint8_t*>(data) + (begin - address),
			            reinterpret_cast<const uint8_t*>(&label.value) + (begin - label.address),
			            end - begin);
		}
	}

	[[nodiscard]] bool HasElided() const noexcept {
		return m_elided.load(std::memory_order_acquire) != 0;
	}

	// The latest submission an elided label overlapping the range completes with; 0 if none.
	[[nodiscard]] uint64_t ElidedTick(uint64_t address, uint64_t size) const {
		if (!HasElided()) {
			return 0;
		}
		std::lock_guard lock(m_mutex);
		uint64_t        tick = 0;
		for (const auto& label: m_labels) {
			if (label.elided && Overlaps(label, address, size)) {
				tick = std::max(tick, label.tick);
			}
		}
		return tick;
	}

	// Calls `store(label)`, in recording order, for every label overlapping the range whose
	// submission is at most `completed` and that is not stored yet, and marks it stored. For
	// labels whose regular store is queued behind the caller; that store then skips the write
	// (ShouldStore), so a later label is never overwritten by an earlier one.
	template <typename Store>
	void StoreCompleted(uint64_t address, uint64_t size, uint64_t completed, Store&& store) {
		std::lock_guard lock(m_mutex);
		for (auto& label: m_labels) {
			if (!label.stored && label.tick <= completed && Overlaps(label, address, size)) {
				store(label);
				label.stored = true;
				if (label.elided) {
					label.elided = false;
					m_elided.fetch_sub(1, std::memory_order_release);
				}
			}
		}
	}

	// Whether the regular store of the label must still write it.
	[[nodiscard]] bool ShouldStore(uint64_t id) const {
		std::lock_guard lock(m_mutex);
		const auto      it = std::ranges::find(m_labels, id, &Label::id);
		return it == m_labels.end() || !it->stored;
	}

	[[nodiscard]] size_t Size() const {
		std::lock_guard lock(m_mutex);
		return m_labels.size();
	}

	// A label the GPU stored to guest memory, in the guest store epoch current before the store
	// (see BdaSyncQueue). Replaces the record of the same bytes.
	void NoteStoredValue(uint64_t address, uint32_t size, uint64_t value, uint64_t epoch) {
		std::lock_guard lock(m_stored_mutex);
		m_stored[StoredSlot(address)] = {address, value, epoch, size};
	}

	// Whether `size` bytes at `address`, read as `observed`, are the value of the GPU's last label
	// store there, made in store epoch `epoch` or later. A wait of a submission of `epoch` that
	// observes such a value orders no guest store: on the console it may pass on the GPU's store,
	// which a guest store made before the submission (to reset the label, say) precedes. A value
	// other than the stored one drops the record: the bytes were written since.
	[[nodiscard]] bool StoredSince(uint64_t address, uint32_t size, uint64_t observed,
	                               uint64_t epoch) {
		std::lock_guard lock(m_stored_mutex);
		auto&           record = m_stored[StoredSlot(address)];
		if (record.address != address) {
			return false;
		}
		if (record.size != size || record.value != observed) {
			record = {};
			return false;
		}
		return record.epoch >= epoch;
	}

	// Forgets the label stores of unmapped memory.
	void ForgetStored(uint64_t address, uint64_t size) {
		std::lock_guard lock(m_stored_mutex);
		const auto      first = address > 7 ? address - 7 : 0;
		for (auto& record: m_stored) {
			if (record.address >= first && record.address < address + size) {
				record = {};
			}
		}
	}

	// The label stores recorded at the moment; at most STORED_CAPACITY.
	[[nodiscard]] size_t StoredRecords() const {
		std::lock_guard lock(m_stored_mutex);
		return static_cast<size_t>(std::ranges::count_if(
		    m_stored, [](const StoredValue& record) { return record.address != 0; }));
	}

	static constexpr size_t STORED_CAPACITY = size_t {1} << 16u;

private:
	static bool Overlaps(const Label& label, uint64_t address, uint64_t size) noexcept {
		return address < label.address + label.size && label.address < address + size;
	}

	// The latest label overlapping the bytes when it covers them all and its value satisfies
	// `condition(value)`, else null. The labels recorded later win: they are stored later.
	template <typename Condition>
	Label* Satisfying(uint64_t address, uint32_t size, Condition& condition) {
		for (auto it = m_labels.rbegin(); it != m_labels.rend(); ++it) {
			if (!Overlaps(*it, address, size)) {
				continue;
			}
			if (!it->known || address < it->address || address + size > it->address + it->size) {
				return nullptr;
			}
			uint64_t value = 0;
			std::memcpy(&value,
			            reinterpret_cast<const uint8_t*>(&it->value) + (address - it->address),
			            size);
			return condition(value) ? &*it : nullptr;
		}
		return nullptr;
	}

	// The GPU's last label store per address, direct-mapped by address: one record per slot, a
	// later store to another address in the same slot replaces it. Losing a record only makes a
	// wait that observes its value order the guest stores, as without a record. A map of every
	// address ever stored grew for the whole session (Saros stores ~900 labels per frame at
	// moving addresses) and cost ~0.75 us per store on the GPU thread.
	struct StoredValue {
		uint64_t address = 0; // 0: no record.
		uint64_t value   = 0;
		uint64_t epoch   = 0;
		uint32_t size    = 0;
	};

	static size_t StoredSlot(uint64_t address) noexcept {
		constexpr uint32_t bits = std::countr_zero(STORED_CAPACITY);
		return static_cast<size_t>((address * 0x9e3779b97f4a7c15ull) >> (64u - bits));
	}

	mutable std::mutex              m_mutex;
	std::vector<Label>              m_labels;
	mutable std::mutex              m_stored_mutex;
	std::vector<StoredValue>        m_stored  = std::vector<StoredValue>(STORED_CAPACITY);
	uint64_t                        m_last_id = 0;
	std::atomic<uint32_t>           m_elided {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PENDINGLABELS_H_
