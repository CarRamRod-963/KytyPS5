#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSREPORT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSREPORT_H_

#include <array>
#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

// Why a bindless key resolved to the placeholder (RenderExecutor::ResolveBindlessKey).
enum class BindlessPlaceholderReason : uint32_t {
	// The record's pointer is unreadable or zero.
	NoPointer,
	// The T# behind the pointer is unreadable.
	UnreadableTsharp,
	NullTsharp,
	// Its layer, level or LOD range is outside the image.
	UnsupportedTsharp,
	// No image could be found or created for it.
	NotBindable,
	DepthOrEmptyImage,
	NoFreeSlot,
	Count,
};

constexpr std::array<const char*, static_cast<size_t>(BindlessPlaceholderReason::Count)>
    BindlessPlaceholderReasonNames = {"no pointer",     "unreadable T#", "null T#",
                                      "unsupported T#", "not bindable",  "depth or empty image",
                                      "no free slot"};

// When the bindless report line is due: in the first frame, every 5 s for the first minute after
// it (loads and respawns resolve most keys then), every 30 s later, and in the frame after a
// marker key press.
class BindlessReportSchedule {
public:
	using Clock = std::chrono::steady_clock;

	static constexpr auto EarlyPeriod = std::chrono::seconds(5);
	static constexpr auto EarlySpan   = std::chrono::seconds(60);
	static constexpr auto Period      = std::chrono::seconds(30);

	// Whether the frame at `now` reports; `marker` is a marker pressed since the previous frame.
	[[nodiscard]] bool Due(Clock::time_point now, bool marker) {
		if (!m_started) {
			m_started = true;
			m_first   = now;
			m_last    = now;
			return true;
		}
		const auto period = now - m_first < EarlySpan ? EarlyPeriod : Period;
		if (!marker && now - m_last < period) {
			return false;
		}
		m_last = now;
		return true;
	}

private:
	bool              m_started = false;
	Clock::time_point m_first;
	Clock::time_point m_last;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSREPORT_H_
