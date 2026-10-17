#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCACHE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>

namespace Libs::Graphics {

// Chunks of guest memory read while one program's resources are evaluated. The walk reads the
// SRT and its descriptors one DWORD at a time, and every backing read takes the address space
// lock and a mapping lookup; the chunks turn the dozens of reads of one evaluation into a few.
//
// The walk evaluates scalar loads on the host at record time, so a value it reads must include
// every GPU write recorded before the command, exactly as the shader's own load would. A chunk
// whose range holds GPU-dirty bytes (written by recorded or completed GPU work, not yet published
// to the backing) is read back before it is filled, and a GPU write recorded after a chunk was
// filled drops every chunk. Clean memory costs one dirty-range query per chunk.
//
// The backend provides, on the thread that evaluates:
//   bool     Read(uint64_t address, void* data, uint64_t size);  // backing; false if unmapped
//   bool     HasGpuDirtyBytes(uint64_t address, uint64_t size);
//   void     ReadGpuDirtyBytes(uint64_t address, uint64_t size); // readback into the backing
//   uint64_t GpuWriteSequence();                                  // advances per recorded write
template <typename Backend>
class ScalarReadCache {
public:
	static constexpr uint64_t ChunkBytes = 128;
	static constexpr size_t   Entries    = 32;

	explicit ScalarReadCache(Backend& backend): m_backend(backend) {}

	// Unmapped guest memory reads as zeros: the walk evaluates loads the shader may never
	// execute, e.g. through a null SRT pointer.
	void Read(uint64_t address, std::span<uint32_t> values) {
		const auto bytes = values.size_bytes();
		if (const auto sequence = m_backend.GpuWriteSequence(); sequence != m_sequence) {
			for (auto& entry: m_entries) {
				entry.base = UINT64_MAX;
			}
			m_sequence = sequence;
		}
		const auto base = address & ~(ChunkBytes - 1u);
		if (bytes <= ChunkBytes && address - base + bytes <= ChunkBytes) {
			auto& entry = m_entries[(base / ChunkBytes) % Entries];
			if (entry.base != base) {
				Synchronize(base, ChunkBytes);
				entry.base = m_backend.Read(base, entry.words, ChunkBytes) ? base : UINT64_MAX;
			}
			if (entry.base == base) {
				std::memcpy(values.data(),
				            reinterpret_cast<const uint8_t*>(entry.words) + (address - base),
				            bytes);
				return;
			}
			// A chunk that crosses unmapped memory: read exactly what was asked for.
		}
		Synchronize(address, bytes);
		if (!m_backend.Read(address, values.data(), bytes)) {
			std::ranges::fill(values, 0u);
		}
	}

private:
	struct Entry {
		uint64_t base = UINT64_MAX;
		uint32_t words[ChunkBytes / sizeof(uint32_t)];
	};

	void Synchronize(uint64_t address, uint64_t size) {
		if (size != 0 && m_backend.HasGpuDirtyBytes(address, size)) {
			m_backend.ReadGpuDirtyBytes(address, size);
		}
	}

	Backend&                   m_backend;
	uint64_t                   m_sequence = 0;
	std::array<Entry, Entries> m_entries;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCACHE_H_ */
