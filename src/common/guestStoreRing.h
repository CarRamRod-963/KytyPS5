#ifndef KYTY_COMMON_GUEST_STORE_RING_H_
#define KYTY_COMMON_GUEST_STORE_RING_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Common {

// Why the emulator stored bytes into guest memory on its own: the guest did not ask for this store
// at this moment (a readback, a write-back, a label landing late, published metadata keys, ...).
enum class GuestStorePath : uint8_t {
	Backing,              // a backing write without a more specific path
	BufferReadback,       // GPU-written buffer bytes read back (fault, walk, CP or store reads)
	InPlaceReadback,      // the same, read from host-visible device memory
	GuestThreadReadback,  // the same, copied by the guest thread that waited for it
	AsyncReadback,        // a hot page's download published when its submission completed
	ImageWriteBack,       // a GPU-written image copied back into its memory
	MetadataClearPublish, // expanded color-metadata keys of an applied fast clear
	DccConstantPublish,   // expanded keys of a decompressed constant DCC slice
	Label,                // an end-of-pipe label or counter stored at completion
	GpuThreadLabel,       // the same, stored by the GPU thread it was handed to
	ElidedLabel,          // a label whose store an earlier read had elided
	OcclusionCounters,    // z-pass counters of an occlusion query dump
	Count,
};

inline const char* GuestStorePathName(GuestStorePath path) {
	switch (path) {
		case GuestStorePath::Backing: return "backing write";
		case GuestStorePath::BufferReadback: return "buffer readback";
		case GuestStorePath::InPlaceReadback: return "in-place readback";
		case GuestStorePath::GuestThreadReadback: return "guest-thread readback";
		case GuestStorePath::AsyncReadback: return "async readback";
		case GuestStorePath::ImageWriteBack: return "image write-back";
		case GuestStorePath::MetadataClearPublish: return "metadata clear publish";
		case GuestStorePath::DccConstantPublish: return "DCC constant publish";
		case GuestStorePath::Label: return "label";
		case GuestStorePath::GpuThreadLabel: return "GPU-thread label";
		case GuestStorePath::ElidedLabel: return "elided label";
		case GuestStorePath::OcclusionCounters: return "occlusion counters";
		case GuestStorePath::Count: break;
	}
	return "?";
}

// One store, as a fault report reads it back.
struct GuestStore {
	uint64_t       sequence = 0; // 1 for the first store recorded
	uint64_t       tsc      = 0; // time stamp counter when it was recorded
	uint64_t       address  = 0;
	uint64_t       size     = 0;
	uint64_t       first    = 0; // the first (up to) 8 bytes stored, little-endian
	uint32_t       thread   = 0; // host thread id
	GuestStorePath path     = GuestStorePath::Backing;
};

// The last Capacity stores the emulator made into guest memory without the guest asking for them
// at that moment. A fatal guest fault report prints the ones near the faulting data, so a crash
// over bytes the emulator wrote names the writer. Recording is one relaxed fetch_add and a few
// relaxed stores into the slot it picks: no lock, no allocation. A reader (the fault report)
// takes a slot only when its sequence number is the same before and after the copy.
class GuestStoreRing {
public:
	static constexpr size_t Capacity = 4096;

	void Record(GuestStorePath path, uint64_t address, uint64_t size, const void* data,
	            uint32_t thread, uint64_t tsc) noexcept {
		const auto sequence = m_next.fetch_add(1, std::memory_order_relaxed) + 1;
		auto&      slot     = m_slots[sequence % Capacity];
		uint64_t   first    = 0;
		if (data != nullptr) {
			std::memcpy(&first, data, size < sizeof(first) ? size : sizeof(first));
		}
		// 0 marks the slot as being written; a reader skips it.
		slot.sequence.store(0, std::memory_order_relaxed);
		std::atomic_signal_fence(std::memory_order_seq_cst);
		slot.tsc.store(tsc, std::memory_order_relaxed);
		slot.address.store(address, std::memory_order_relaxed);
		slot.size.store(size, std::memory_order_relaxed);
		slot.first.store(first, std::memory_order_relaxed);
		slot.info.store(uint64_t {thread} | (uint64_t {static_cast<uint8_t>(path)} << 32u),
		                std::memory_order_relaxed);
		slot.sequence.store(sequence, std::memory_order_release);
	}

	// Stores recorded since the start.
	[[nodiscard]] uint64_t Count() const { return m_next.load(std::memory_order_acquire); }

	// Copies the kept stores, oldest first, into out (at most out_size) and returns how many. A
	// store being written meanwhile, or overwritten by a newer one, is left out.
	size_t Snapshot(GuestStore* out, size_t out_size) const {
		const auto newest = Count();
		const auto kept   = newest < Capacity ? newest : Capacity;
		size_t     count  = 0;
		for (uint64_t sequence = newest - kept + 1; sequence <= newest && count < out_size;
		     sequence++) {
			const auto& slot = m_slots[sequence % Capacity];
			if (slot.sequence.load(std::memory_order_acquire) != sequence) {
				continue;
			}
			GuestStore store;
			store.sequence  = sequence;
			store.tsc       = slot.tsc.load(std::memory_order_relaxed);
			store.address   = slot.address.load(std::memory_order_relaxed);
			store.size      = slot.size.load(std::memory_order_relaxed);
			store.first     = slot.first.load(std::memory_order_relaxed);
			const auto info = slot.info.load(std::memory_order_relaxed);
			store.thread    = static_cast<uint32_t>(info);
			const auto path = static_cast<uint8_t>(info >> 32u);
			store.path      = path < static_cast<uint8_t>(GuestStorePath::Count)
			                      ? static_cast<GuestStorePath>(path)
			                      : GuestStorePath::Backing;
			std::atomic_thread_fence(std::memory_order_acquire);
			if (slot.sequence.load(std::memory_order_relaxed) != sequence) {
				continue;
			}
			out[count++] = store;
		}
		return count;
	}

private:
	struct Slot {
		std::atomic<uint64_t> sequence {0};
		std::atomic<uint64_t> tsc {0};
		std::atomic<uint64_t> address {0};
		std::atomic<uint64_t> size {0};
		std::atomic<uint64_t> first {0};
		std::atomic<uint64_t> info {0}; // thread | path << 32
	};
	// Writers on different cores take the counter and then different slots.
	alignas(64) std::atomic<uint64_t> m_next {0};
	alignas(64) std::array<Slot, Capacity> m_slots {};
};

// Whether [address, address + size) and [begin, end) share a byte.
inline bool GuestStoreOverlaps(const GuestStore& store, uint64_t begin, uint64_t end) {
	const auto store_end = store.address + store.size;
	return store.address < end && begin < (store.size == 0 ? store.address + 1 : store_end);
}

// The process's ring and the path the calling thread's backing writes are recorded under.
GuestStoreRing& GuestStores();
GuestStorePath  CurrentGuestStorePath();

// Records a store of the calling thread under the path its scope names (or `path` when given).
void RecordGuestStore(uint64_t address, uint64_t size, const void* data);
void RecordGuestStore(GuestStorePath path, uint64_t address, uint64_t size, const void* data);

// Names the path of the backing writes the calling thread makes while it is alive.
class GuestStoreScope {
public:
	explicit GuestStoreScope(GuestStorePath path);
	~GuestStoreScope();
	GuestStoreScope(const GuestStoreScope&)            = delete;
	GuestStoreScope& operator=(const GuestStoreScope&) = delete;

private:
	GuestStorePath m_previous;
};

// The time stamp counter and how many of its ticks pass per microsecond (measured between the
// first use and the call).
uint64_t GuestStoreClock();
uint64_t GuestStoreClockTicksPerUs();

} // namespace Common

#endif /* KYTY_COMMON_GUEST_STORE_RING_H_ */
