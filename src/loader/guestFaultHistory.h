#ifndef KYTY_LOADER_GUEST_FAULT_HISTORY_H_
#define KYTY_LOADER_GUEST_FAULT_HISTORY_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace Loader {

// What the exception handler resolved and resumed. Each one makes Windows write an exception
// frame below the thread's stack pointer: on a guest thread that frame overwrites whatever the
// interrupted function kept in its SysV red zone.
enum class HandledFaultKind : uint8_t {
	Read,                // a read of a page whose bytes are on the GPU
	Write,               // a write to a page the memory tracker protects
	Execute,             //
	EmulatedInstruction, // an instruction the host CPU lacks (INSERTQ, MONITORX, MWAITX, ...)
	SkippedLoad,         // a load of a compiled resource walk that was skipped
};

struct HandledFault {
	uint64_t         address = 0; // the faulting data address (0 for instructions)
	uint64_t         pc      = 0;
	uint64_t         rsp     = 0;
	uint64_t         time_ns = 0; // monotonic
	HandledFaultKind kind    = HandledFaultKind::Read;
	// Stores the emulator had made into guest memory when the fault was resolved (the store ring's
	// count): a read fault's readback is among the stores numbered after it.
	uint64_t stores = 0;
};

// The last faults one thread had resolved, for the report of a fatal fault on the same thread.
// Recording is a few stores into thread-local memory: no lock, no allocation.
class HandledFaultRing {
public:
	static constexpr size_t Capacity = 8;

	void Record(const HandledFault& fault) {
		m_entries[m_count % Capacity] = fault;
		m_count++;
	}

	// Faults recorded since the thread started (the ring keeps the last Capacity of them).
	[[nodiscard]] uint64_t Count() const { return m_count; }
	[[nodiscard]] size_t   Size() const { return m_count < Capacity ? m_count : Capacity; }

	// i = 0 is the oldest fault still kept, Size() - 1 the newest.
	[[nodiscard]] const HandledFault& At(size_t i) const {
		return m_entries[(m_count - Size() + i) % Capacity];
	}

private:
	std::array<HandledFault, Capacity> m_entries {};
	uint64_t                           m_count = 0;
};

// Address ranges of the loaded guest modules, readable without a lock from an exception handler
// (the runtime linker's program list is behind a mutex the faulting thread may hold). Slots are
// written once and never reused; an unloaded module only clears its size.
class GuestModuleMap {
public:
	static constexpr size_t Capacity = 64;
	static constexpr size_t NameSize = 48;

	struct Location {
		std::string_view module;
		uint64_t         offset = 0;
	};

	// Writers (module loads and unloads) are serialized by the caller.
	void Add(uint64_t base, uint64_t size, std::string_view name) {
		const auto index = m_used.load(std::memory_order_relaxed);
		if (index >= Capacity || size == 0) {
			return;
		}
		auto&      slot   = m_slots[index];
		const auto length = name.size() < NameSize - 1 ? name.size() : NameSize - 1;
		std::memcpy(slot.name.data(), name.data(), length);
		slot.name[length] = '\0';
		slot.base.store(base, std::memory_order_relaxed);
		slot.size.store(size, std::memory_order_release);
		m_used.store(index + 1, std::memory_order_release);
	}

	void Remove(uint64_t base) {
		const auto used = m_used.load(std::memory_order_relaxed);
		for (size_t i = 0; i < used; i++) {
			if (m_slots[i].base.load(std::memory_order_relaxed) == base) {
				m_slots[i].size.store(0, std::memory_order_release);
			}
		}
	}

	void Clear() {
		const auto used = m_used.load(std::memory_order_relaxed);
		for (size_t i = 0; i < used; i++) {
			m_slots[i].size.store(0, std::memory_order_release);
		}
	}

	// The module holding `address`, or an empty name.
	[[nodiscard]] Location Find(uint64_t address) const {
		const auto used = m_used.load(std::memory_order_acquire);
		for (size_t i = 0; i < used; i++) {
			const auto& slot = m_slots[i];
			const auto  size = slot.size.load(std::memory_order_acquire);
			const auto  base = slot.base.load(std::memory_order_relaxed);
			if (size != 0 && address >= base && address - base < size) {
				return {std::string_view(slot.name.data()), address - base};
			}
		}
		return {};
	}

private:
	struct Slot {
		std::atomic<uint64_t>      base {0};
		std::atomic<uint64_t>      size {0};
		std::array<char, NameSize> name {};
	};
	std::array<Slot, Capacity> m_slots {};
	std::atomic<size_t>        m_used {0};
};

// Host thread ids and names of the running guest threads, for a fault report that names what the
// other guest threads were doing. Readable without a lock; a name read while its thread renames
// itself may come out mixed (the report says it is a best-effort snapshot).
class GuestThreadRegistry {
public:
	static constexpr size_t Capacity = 512;
	static constexpr size_t NameSize = 32;

	// Returns false when every slot is taken (the thread is then left out of reports).
	bool Add(uint64_t host_thread_id, std::string_view name) {
		if (host_thread_id == 0) {
			return false;
		}
		for (auto& slot: m_slots) {
			uint64_t expected = 0;
			if (slot.id.load(std::memory_order_relaxed) == 0 &&
			    slot.id.compare_exchange_strong(expected, host_thread_id,
			                                    std::memory_order_acq_rel)) {
				SetName(&slot, name);
				return true;
			}
		}
		return false;
	}

	void Remove(uint64_t host_thread_id) {
		if (auto* slot = Find(host_thread_id); host_thread_id != 0 && slot != nullptr) {
			slot->id.store(0, std::memory_order_release);
		}
	}

	void Rename(uint64_t host_thread_id, std::string_view name) {
		if (auto* slot = Find(host_thread_id); host_thread_id != 0 && slot != nullptr) {
			SetName(slot, name);
		}
	}

	// Calls f(host_thread_id, name) for every registered thread.
	template <typename F>
	void ForEach(F&& f) const {
		for (const auto& slot: m_slots) {
			const auto id = slot.id.load(std::memory_order_acquire);
			if (id == 0) {
				continue;
			}
			std::array<char, NameSize> name {};
			size_t                     length = 0;
			while (length + 1 < NameSize) {
				name[length] = slot.name[length].load(std::memory_order_relaxed);
				if (name[length] == '\0') {
					break;
				}
				length++;
			}
			f(id, std::string_view(name.data(), length));
		}
	}

private:
	struct Slot {
		std::atomic<uint64_t>                   id {0};
		std::array<std::atomic<char>, NameSize> name {};
	};

	Slot* Find(uint64_t host_thread_id) {
		for (auto& slot: m_slots) {
			if (slot.id.load(std::memory_order_acquire) == host_thread_id) {
				return &slot;
			}
		}
		return nullptr;
	}

	static void SetName(Slot* slot, std::string_view name) {
		const size_t length = name.size() < NameSize - 1 ? name.size() : NameSize - 1;
		for (size_t i = 0; i < NameSize; i++) {
			slot->name[i].store(i < length ? name[i] : '\0', std::memory_order_relaxed);
		}
	}

	std::array<Slot, Capacity> m_slots {};
};

// The process's registry, filled by the guest thread library.
GuestThreadRegistry& GuestThreads();

// Lets one thread write the report of a fatal fault. A thread that faults while another one
// writes it waits for the report before it goes on to end the process: its way out is much
// shorter than the report (memory queries, a stop of every guest thread, a write of up to some
// hundred KiB), and two threads that fault on one broken object within milliseconds would
// otherwise leave a log without any report. The reporting thread itself (a fault inside the
// report) waits for nothing.
class FaultReportGate {
public:
	// True for the thread that writes the report (the first caller).
	[[nodiscard]] bool Begin() {
		if (m_busy.exchange(true, std::memory_order_acquire)) {
			return false;
		}
		Reporter() = this;
		return true;
	}

	void Finish() { m_done.store(true, std::memory_order_release); }

	// For a caller Begin turned away: returns once the report is written, after `timeout_ms` at
	// most, at once on the reporting thread. `sleep_ms(1)` waits a millisecond. True when the
	// report was written.
	template <typename Sleep>
	bool WaitForReport(uint32_t timeout_ms, Sleep&& sleep_ms) const {
		if (Reporter() == this) {
			return m_done.load(std::memory_order_acquire);
		}
		for (uint32_t waited = 0; !m_done.load(std::memory_order_acquire); waited++) {
			if (waited >= timeout_ms) {
				return false;
			}
			sleep_ms(1);
		}
		return true;
	}

private:
	static const FaultReportGate*& Reporter() {
		thread_local const FaultReportGate* reporter = nullptr;
		return reporter;
	}

	std::atomic<bool> m_busy {false};
	std::atomic<bool> m_done {false};
};

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_FAULT_HISTORY_H_ */
