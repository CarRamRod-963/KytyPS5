#ifndef KYTY_COMMON_DIAGNOSTIC_MARKER_H_
#define KYTY_COMMON_DIAGNOSTIC_MARKER_H_

#include <atomic>
#include <cstdint>

namespace Common {

// A marker key press (ScrollLock or F9 in the game window): numbered marks a player sets in the log
// while something looks wrong, and again when it looks right, so that the reports printed right
// after each mark can be compared. The window thread requests a mark, the GPU thread takes it
// once a frame.
class DiagnosticMarker {
public:
	// The number of the new mark (1, 2, ...).
	static uint32_t Request() {
		const auto number = s_count.fetch_add(1, std::memory_order_relaxed) + 1u;
		s_pending.store(number, std::memory_order_release);
		return number;
	}

	// The newest mark requested since the last call, or 0.
	static uint32_t TakePending() {
		if (s_pending.load(std::memory_order_relaxed) == 0) {
			return 0;
		}
		return s_pending.exchange(0, std::memory_order_acquire);
	}

private:
	static inline std::atomic<uint32_t> s_count {0};
	static inline std::atomic<uint32_t> s_pending {0};
};

} // namespace Common

#endif // KYTY_COMMON_DIAGNOSTIC_MARKER_H_
