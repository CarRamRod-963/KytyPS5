#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCACHE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

namespace Libs::Graphics {

// Chunks of guest memory read while one program's resources are evaluated. The walk reads the
// SRT and its descriptors one DWORD at a time, and every backing read takes the address space
// lock and a mapping lookup; the chunks turn the dozens of reads of one evaluation into a few.
//
// The walk evaluates scalar loads on the host at record time, so a value it reads must include
// every GPU write recorded before the command, exactly as the shader's own load would. Bytes
// written by recorded or completed GPU work and not yet published to the backing are GPU-dirty.
// A chunk that holds any is marked when it is filled, and a read from a marked chunk reads back
// the GPU-dirty bytes among exactly the bytes it returns: a GPU-written word next to the
// descriptors a walk reads costs no readback unless the walk reads that word. A GPU write
// recorded after a chunk was filled drops every chunk, and a readback drops every marked chunk.
// Clean memory costs one dirty-range query per chunk.
//
// Every readback waits for the GPU work recorded before it. The ranges a walk read back are
// handed to the next walk of the same program (Prefetch), whose first readback reads back their
// GPU-dirty bytes in the same round trip: a walk that reads several GPU-written tables waits
// once, not once per table, as long as it reads the tables the previous walk read.
//
// The backend provides, on the thread that evaluates:
//   bool     Read(uint64_t address, void* data, uint64_t size);  // backing; false if unmapped
//   bool     HasGpuDirtyBytes(uint64_t address, uint64_t size);
//   void     ReadGpuDirtyBytes(std::span<const ScalarReadRange> ranges); // one readback of all
//   uint64_t GpuWriteSequence();                                  // advances per recorded write
using ScalarReadRange = std::pair<uint64_t, uint64_t>; // address, size

template <typename Backend>
class ScalarReadCache {
public:
	static constexpr uint64_t ChunkBytes   = 128;
	static constexpr size_t   Entries      = 32;
	static constexpr size_t   MaxReadBacks = 16;

	explicit ScalarReadCache(Backend& backend): m_backend(backend) {}

	// Ranges whose GPU-dirty bytes the first readback of this walk also reads back. The span
	// must stay valid until that readback or the end of the walk; at most MaxReadBacks are used.
	void Prefetch(std::span<const ScalarReadRange> ranges) {
		m_prefetch = ranges.first(std::min(ranges.size(), MaxReadBacks));
	}
	// The direct-mapped entries for readers that look chunks up themselves: entry i holds the
	// chunk at Tags()[i] in its ChunkBytes at Data() + i * Stride(). A chunk that is not tagged
	// (absent, or holding GPU-dirty bytes) must be read through Read. Only Read changes them.
	[[nodiscard]] const uint64_t* Tags() const { return m_tags.data(); }
	[[nodiscard]] const uint8_t*  Data() const {
		return reinterpret_cast<const uint8_t*>(m_entries[0].words);
	}
	[[nodiscard]] static constexpr uint32_t Stride() { return sizeof(Entry); }

	// The ranges this walk read back (the first MaxReadBacks), for the next walk's Prefetch.
	[[nodiscard]] std::span<const ScalarReadRange> ReadBacks() const {
		return std::span(m_read_backs).first(m_read_back_count);
	}

	// Unmapped guest memory reads as zeros: the walk evaluates loads the shader may never
	// execute, e.g. through a null SRT pointer.
	void Read(uint64_t address, std::span<uint32_t> values) {
		const auto bytes = values.size_bytes();
		if (const auto sequence = m_backend.GpuWriteSequence(); sequence != m_sequence) {
			for (auto& entry: m_entries) {
				entry.base = UINT64_MAX;
			}
			m_tags.fill(UINT64_MAX);
			m_sequence = sequence;
		}
		const auto base = address & ~(ChunkBytes - 1u);
		if (bytes <= ChunkBytes && address - base + bytes <= ChunkBytes) {
			auto& entry = m_entries[Index(base)];
			if (entry.base != base) {
				Fill(entry, base);
			}
			if (entry.base == base && entry.dirty && m_backend.HasGpuDirtyBytes(address, bytes)) {
				ReadBack(address, bytes);
				Fill(entry, base);
			}
			if (entry.base == base) {
				std::memcpy(values.data(),
				            reinterpret_cast<const uint8_t*>(entry.words) + (address - base),
				            bytes);
				return;
			}
			// A chunk that crosses unmapped memory: read exactly what was asked for.
		}
		if (bytes != 0 && m_backend.HasGpuDirtyBytes(address, bytes)) {
			ReadBack(address, bytes);
		}
		if (!m_backend.Read(address, values.data(), bytes)) {
			std::ranges::fill(values, 0u);
		}
	}

private:
	static constexpr std::array<uint64_t, Entries> MakeEmptyTags() {
		std::array<uint64_t, Entries> tags {};
		tags.fill(UINT64_MAX);
		return tags;
	}

	struct Entry {
		uint64_t base  = UINT64_MAX;
		bool     dirty = false;
		uint32_t words[ChunkBytes / sizeof(uint32_t)];
	};

	static size_t Index(uint64_t base) { return (base / ChunkBytes) % Entries; }

	void Fill(Entry& entry, uint64_t base) {
		entry.base  = m_backend.Read(base, entry.words, ChunkBytes) ? base : UINT64_MAX;
		entry.dirty = entry.base == base && m_backend.HasGpuDirtyBytes(base, ChunkBytes);
		m_tags[Index(base)] = entry.dirty ? UINT64_MAX : entry.base;
	}

	// A readback also publishes completed asynchronous downloads, which turns GPU-dirty bytes
	// clean without a readback of their own: a chunk filled while it held GPU-dirty bytes may
	// hold their old values under a clean mark now, so every such chunk is dropped.
	void ReadBack(uint64_t address, uint64_t size) {
		std::array<ScalarReadRange, MaxReadBacks + 1> batch {{{address, size}}};
		size_t                                        count = 1;
		for (const auto& range: m_prefetch) {
			batch[count++] = range;
		}
		m_prefetch = {};
		m_backend.ReadGpuDirtyBytes(std::span<const ScalarReadRange>(batch.data(), count));
		if (m_read_back_count < MaxReadBacks) {
			m_read_backs[m_read_back_count++] = {address, size};
		}
		for (auto& entry: m_entries) {
			if (entry.dirty) {
				entry.base  = UINT64_MAX;
				entry.dirty = false;
			}
		}
		// Dirty entries were never tagged.
	}

	Backend&                                  m_backend;
	uint64_t                                  m_sequence = 0;
	std::array<Entry, Entries>                m_entries;
	std::array<uint64_t, Entries>             m_tags = MakeEmptyTags();
	std::span<const ScalarReadRange>          m_prefetch;
	std::array<ScalarReadRange, MaxReadBacks> m_read_backs {};
	size_t                                    m_read_back_count = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCACHE_H_ */
