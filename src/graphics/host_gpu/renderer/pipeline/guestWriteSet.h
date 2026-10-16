#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTWRITESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTWRITESET_H_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// The guest ranges a draw or dispatch writes (storage images, attachments and their metadata,
// written buffers), collected once per commit and tested against each guest read its resource
// specialization depends on. The ranges' common bound rejects a read outside all of them with
// one comparison; a read inside the bound is tested against every range in insertion order.
class GuestWriteSet {
public:
	enum class Kind : uint8_t { Image, Buffer };

	struct Range {
		uint64_t address = 0;
		uint64_t size    = 0;
		Kind     kind    = Kind::Image;
	};

	// A range is valid when it is not empty and does not wrap the address space.
	[[nodiscard]] static constexpr bool ValidRange(uint64_t address, uint64_t size) noexcept {
		return size != 0 && address <= UINT64_MAX - size;
	}

	void Clear() noexcept {
		m_ranges.clear();
		m_begin = UINT64_MAX;
		m_end   = 0;
	}

	// Empty ranges write nothing and are ignored; a wrapping range is refused (false).
	[[nodiscard]] bool Add(uint64_t address, uint64_t size, Kind kind) {
		if (size == 0) return true;
		if (!ValidRange(address, size)) return false;
		m_ranges.push_back({address, size, kind});
		m_begin = std::min(m_begin, address);
		m_end   = std::max(m_end, address + size);
		return true;
	}

	[[nodiscard]] bool Empty() const noexcept { return m_ranges.empty(); }

	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }

	// The first added range that overlaps a valid range [address, address + size), or nullptr.
	[[nodiscard]] const Range* FindOverlap(uint64_t address, uint64_t size) const noexcept {
		if (m_ranges.empty() || address >= m_end || m_begin >= address + size) {
			return nullptr;
		}
		for (const auto& range: m_ranges) {
			if (address < range.address + range.size && range.address < address + size) {
				return &range;
			}
		}
		return nullptr;
	}

private:
	std::vector<Range> m_ranges;
	uint64_t           m_begin = UINT64_MAX;
	uint64_t           m_end   = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTWRITESET_H_
