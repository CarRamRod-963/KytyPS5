#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RECORDINGSTREAM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RECORDINGSTREAM_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

// A single-producer stream of deferred calls with a thread of its own that runs them in order.
// The producer copies each call, its arguments and everything they point to into the stream
// (Enqueue); the stream's thread runs the calls in the order they were queued. Vulkan-free: the
// Vulkan recording thread (vulkanRecording.h) builds on it.
//
// Calls are queued into chunks of a fixed ring. The producer fills one chunk and hands it over
// when it is full or on Publish; the stream's thread runs the handed-over chunks and gives them
// back. Arguments too large for a chunk are copied into blocks the chunk owns until its calls
// ran. Only the producer may call Enqueue, Publish, Pending and Drain.
namespace Libs::Graphics::Recording {

constexpr size_t   ChunkBytes = 64 * 1024;
constexpr uint32_t ChunkCount = 64;
// A chunk is handed over after this many calls, so the stream's thread starts early.
constexpr uint32_t ChunkMaxCalls = 128;
// Arguments larger than this are copied into a block of their own.
constexpr size_t LargeCopyBytes = ChunkBytes / 4;

namespace Detail {

constexpr uint32_t End = std::numeric_limits<uint32_t>::max();

struct Header {
	void (*run)(void*)     = nullptr;
	void (*destroy)(void*) = nullptr;
	uint32_t payload       = 0;
	uint32_t next          = End;
};

struct Chunk {
	alignas(64) std::array<std::byte, ChunkBytes> data;
	size_t                                    used  = 0;
	uint32_t                                  first = End;
	uint32_t                                  last  = End;
	uint32_t                                  calls = 0;
	std::vector<std::unique_ptr<std::byte[]>> blocks;
};

} // namespace Detail

// Copies one call and what its arguments point to into the chunk being filled. When anything does
// not fit, the writer turns invalid and the call is not queued (its copies are dropped).
class Writer {
public:
	explicit Writer(Detail::Chunk& chunk)
	    : m_chunk(chunk), m_used(chunk.used), m_blocks(chunk.blocks.size()) {}
	~Writer() {
		if (!Queued()) {
			m_chunk.blocks.resize(m_blocks);
		}
	}
	Writer(const Writer&)            = delete;
	Writer& operator=(const Writer&) = delete;

	// False once something could not be copied; the call then goes nowhere.
	[[nodiscard]] bool Valid() const noexcept { return m_valid; }
	// The call did not fit into the rest of the chunk (it may fit into an empty one).
	[[nodiscard]] bool Overflow() const noexcept { return m_overflow; }
	[[nodiscard]] bool Queued() const noexcept { return m_valid && m_call != Detail::End; }

	// Makes the call invalid: for arguments a call cannot copy.
	void Fail() noexcept { m_valid = false; }

	// A copy of `count` elements at `source`, which lives until the call ran. Null for no
	// elements.
	template <typename T>
	[[nodiscard]] T* Copy(const T* source, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (source == nullptr || count == 0 || !m_valid) {
			return nullptr;
		}
		if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
			Fail();
			return nullptr;
		}
		const size_t bytes  = count * sizeof(T);
		auto*        target = static_cast<T*>(Allocate(bytes, alignof(T)));
		if (target != nullptr) {
			std::memcpy(target, source, bytes);
		}
		return target;
	}

	// Copies `bytes` untyped bytes (as for a void pointer argument).
	[[nodiscard]] void* CopyBytes(const void* source, size_t bytes) {
		return Copy(static_cast<const std::byte*>(source), bytes);
	}

	// The call itself: `function()` runs on the stream's thread, and is destroyed after it ran.
	// Only one call per writer.
	template <typename Function>
	void Call(Function&& function) {
		using Stored = std::decay_t<Function>;
		static_assert(std::is_nothrow_destructible_v<Stored>);
		static_assert(alignof(Stored) <= alignof(std::max_align_t));
		if (!m_valid || m_call != Detail::End) {
			Fail();
			return;
		}
		auto* header =
		    static_cast<Detail::Header*>(Inline(sizeof(Detail::Header), alignof(Detail::Header)));
		void* payload = Inline(sizeof(Stored), alignof(Stored));
		if (header == nullptr || payload == nullptr) {
			Fail();
			return;
		}
		new (payload) Stored(std::forward<Function>(function));
		auto* base = m_chunk.data.data();
		new (header) Detail::Header {
		    .run     = [](void* pointer) { (*static_cast<Stored*>(pointer))(); },
		    .destroy = nullptr,
		    .payload = static_cast<uint32_t>(static_cast<std::byte*>(payload) - base),
		};
		if constexpr (!std::is_trivially_destructible_v<Stored>) {
			header->destroy = [](void* pointer) { static_cast<Stored*>(pointer)->~Stored(); };
		}
		m_call = static_cast<uint32_t>(reinterpret_cast<std::byte*>(header) - base);
	}

private:
	friend class Stream;

	void* Allocate(size_t bytes, size_t alignment) {
		if (bytes > LargeCopyBytes) {
			auto  block = std::make_unique_for_overwrite<std::byte[]>(bytes);
			auto* data  = block.get();
			m_chunk.blocks.push_back(std::move(block));
			return data;
		}
		return Inline(bytes, alignment);
	}

	void* Inline(size_t bytes, size_t alignment) {
		if (!m_valid) {
			return nullptr;
		}
		const size_t offset = (m_used + alignment - 1) & ~(alignment - 1);
		if (bytes > ChunkBytes || offset > ChunkBytes - bytes) {
			m_valid    = false;
			m_overflow = true;
			return nullptr;
		}
		m_used = offset + bytes;
		return m_chunk.data.data() + offset;
	}

	Detail::Chunk& m_chunk;
	size_t         m_used;
	size_t         m_blocks;
	uint32_t       m_call     = Detail::End;
	bool           m_valid    = true;
	bool           m_overflow = false;
};

class Stream {
public:
	// `on_start` runs first on the stream's thread (a name, a priority). The thread spins `spins`
	// pauses for new calls before it blocks: a blocked thread costs the producer a wake-up.
	explicit Stream(std::function<void()> on_start = {}, uint32_t spins = 4000)
	    : m_chunks(std::make_unique<std::array<Detail::Chunk, ChunkCount>>()), m_spins(spins) {
		m_thread = std::thread([this, on_start = std::move(on_start)] {
			if (on_start) {
				on_start();
			}
			Run();
		});
	}

	// Runs every queued call, then ends the thread.
	~Stream() {
		Drain();
		m_stopping.store(true, std::memory_order_release);
		// An atomic wait ends on a changed value, not on a notification alone.
		m_published.fetch_add(1, std::memory_order_seq_cst);
		m_published.notify_one();
		m_thread.join();
	}

	Stream(const Stream&)            = delete;
	Stream& operator=(const Stream&) = delete;

	// Queues the call `capture(writer)` describes. False when nothing was queued: the call does
	// not fit into a chunk or cannot be copied; the caller then runs it itself after a Drain.
	template <typename Capture>
	bool Enqueue(Capture&& capture) {
		for (int attempt = 0; attempt < 2; ++attempt) {
			WaitForRoom();
			auto& chunk  = Filling();
			bool  queued = false;
			bool  retry  = false;
			{
				Writer writer(chunk);
				capture(writer);
				if (writer.Queued()) {
					auto* base = chunk.data.data();
					if (chunk.last != Detail::End) {
						reinterpret_cast<Detail::Header*>(base + chunk.last)->next = writer.m_call;
					} else {
						chunk.first = writer.m_call;
					}
					chunk.last = writer.m_call;
					chunk.used = writer.m_used;
					++chunk.calls;
					queued = true;
				} else {
					// A call that did not fit into the rest of a chunk gets an empty one.
					retry = writer.Overflow() && chunk.calls != 0;
				}
			}
			if (queued) {
				m_queued_calls.fetch_add(1, std::memory_order_relaxed);
				if (chunk.calls >= ChunkMaxCalls || chunk.used >= ChunkBytes * 3 / 4) {
					Publish();
				}
				return true;
			}
			if (!retry) {
				break;
			}
			Publish();
		}
		return false;
	}

	// Hands the chunk being filled to the stream's thread (calls other threads wait for).
	void Publish() {
		auto& chunk = Filling();
		if (chunk.calls == 0) {
			return;
		}
		m_published.store(++m_sequence, std::memory_order_seq_cst);
		// The thread is woken only when it said it blocks (both sides seq_cst).
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			m_published.notify_one();
		}
	}

	// Queued calls have not all run yet.
	[[nodiscard]] bool Pending() const {
		return m_completed.load(std::memory_order_acquire) != m_sequence ||
		       (*m_chunks)[m_sequence % ChunkCount].calls != 0;
	}

	// Returns once every queued call ran.
	void Drain() {
		Publish();
		for (auto done = m_completed.load(std::memory_order_acquire); done != m_sequence;
		     done      = m_completed.load(std::memory_order_acquire)) {
			m_completed.wait(done, std::memory_order_acquire);
		}
	}

	// Calls queued since the start (statistics).
	[[nodiscard]] uint64_t QueuedCalls() const noexcept {
		return m_queued_calls.load(std::memory_order_relaxed);
	}
	// Times the producer waited for a chunk to be given back (statistics).
	[[nodiscard]] uint64_t RoomWaits() const noexcept {
		return m_room_waits.load(std::memory_order_relaxed);
	}
	[[nodiscard]] std::thread::id ThreadId() const noexcept { return m_thread.get_id(); }

private:
	Detail::Chunk& Filling() { return (*m_chunks)[m_sequence % ChunkCount]; }

	void WaitForRoom() {
		auto done = m_completed.load(std::memory_order_acquire);
		if (m_sequence - done < ChunkCount) {
			return;
		}
		m_room_waits.fetch_add(1, std::memory_order_relaxed);
		for (; m_sequence - done >= ChunkCount;
		     done = m_completed.load(std::memory_order_acquire)) {
			m_completed.wait(done, std::memory_order_acquire);
		}
	}

	static void Pause() {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#else
		std::this_thread::yield();
#endif
	}

	void Run() {
		uint64_t current = 0;
		for (;;) {
			for (auto ready = m_published.load(std::memory_order_acquire); current == ready;
			     ready      = m_published.load(std::memory_order_acquire)) {
				for (uint32_t spin = 0; spin < m_spins && ready == current; ++spin) {
					Pause();
					ready = m_published.load(std::memory_order_acquire);
				}
				if (ready != current) {
					break;
				}
				m_sleeping.store(true, std::memory_order_seq_cst);
				if (m_published.load(std::memory_order_seq_cst) == current) {
					m_published.wait(current, std::memory_order_seq_cst);
				}
				m_sleeping.store(false, std::memory_order_relaxed);
			}
			if (m_stopping.load(std::memory_order_acquire)) {
				return;
			}
			auto& chunk = (*m_chunks)[current % ChunkCount];
			auto* base  = chunk.data.data();
			for (auto offset = chunk.first; offset != Detail::End;) {
				const auto* header = reinterpret_cast<const Detail::Header*>(base + offset);
				header->run(base + header->payload);
				if (header->destroy != nullptr) {
					header->destroy(base + header->payload);
				}
				offset = header->next;
			}
			chunk.blocks.clear();
			chunk.used  = 0;
			chunk.calls = 0;
			chunk.first = chunk.last = Detail::End;
			m_completed.store(++current, std::memory_order_release);
			m_completed.notify_all();
		}
	}

	std::unique_ptr<std::array<Detail::Chunk, ChunkCount>> m_chunks;
	uint32_t                                               m_spins;
	// The producer's: chunks handed over so far (the one it fills is the next).
	uint64_t m_sequence = 0;
	alignas(64) std::atomic<uint64_t> m_published {0};
	alignas(64) std::atomic<uint64_t> m_completed {0};
	alignas(64) std::atomic<bool> m_sleeping {false};
	std::atomic<bool>     m_stopping {false};
	std::atomic<uint64_t> m_queued_calls {0};
	std::atomic<uint64_t> m_room_waits {0};
	std::thread           m_thread;
};

} // namespace Libs::Graphics::Recording

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RECORDINGSTREAM_H_
