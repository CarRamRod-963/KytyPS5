#ifndef KYTY_COMMON_FAULT_REPORT_LINES_H_
#define KYTY_COMMON_FAULT_REPORT_LINES_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <fmt/format.h>
#include <string_view>
#include <utility>

namespace Common {

// Lines a fatal fault report takes from code it cannot see (the GPU caches). Called from the
// exception handler: an implementation must not allocate, and a writer must not wait for a lock
// the faulting thread may hold.
class FaultReportLines {
public:
	static constexpr size_t LineSize = 192;

	virtual ~FaultReportLines() = default;

	// Returns false once the report takes no more lines (the caller may stop).
	virtual bool Line(std::string_view line) = 0;

	// Formats one line (cut at LineSize) and passes it on.
	template <typename... Args>
	bool Format(fmt::format_string<Args...> format, Args&&... args) {
		std::array<char, LineSize> text {};
		const auto                 result =
		    fmt::format_to_n(text.data(), text.size(), format, std::forward<Args>(args)...);
		return Line({text.data(), std::min(result.size, text.size())});
	}
};

} // namespace Common

#endif /* KYTY_COMMON_FAULT_REPORT_LINES_H_ */
