#ifndef KYTY_LOADER_GUEST_FAULT_REPORT_H_
#define KYTY_LOADER_GUEST_FAULT_REPORT_H_

#include "common/hostException.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <span>
#include <string_view>
#include <utility>

namespace Loader {

// Fixed-size text of a fatal fault report. The report is written from the exception handler on
// the faulting thread, which may have faulted inside the host heap: formatting must not allocate.
// Text past the capacity is dropped.
class GuestFaultText {
public:
	static constexpr size_t Capacity = 8192;

	template <typename... Args>
	void Append(fmt::format_string<Args...> format, Args&&... args) {
		const size_t room = Capacity - m_size;
		const auto   result =
		    fmt::format_to_n(m_data.data() + m_size, room, format, std::forward<Args>(args)...);
		m_truncated = m_truncated || result.size > room;
		m_size += std::min(result.size, room);
	}

	[[nodiscard]] std::string_view View() const { return {m_data.data(), m_size}; }
	[[nodiscard]] bool             Truncated() const { return m_truncated; }

private:
	std::array<char, Capacity> m_data {};
	size_t                     m_size      = 0;
	bool                       m_truncated = false;
};

// What the fatal report of an unhandled host exception can tell about the faulting thread. The
// report goes to the console and the log file: testers usually send only the log, and a
// console-only report is lost with the window.
struct GuestFaultReport {
	std::string_view                            thread;
	const Common::HostException::ExceptionInfo* info = nullptr;
	std::span<const uint8_t>                    code;         // pc-48 .. pc+48, or empty
	std::span<const uint64_t>                   stack;        // words from rsp, or empty
	std::span<const uint64_t>                   guest_frames; // return addresses, rbp chain
};

inline constexpr size_t GuestFaultCodeBefore = 48;

inline void FormatGuestFaultReport(const GuestFaultReport& report, GuestFaultText* out) {
	out->Append("--- Guest fault context ---\nthread: {}\n", report.thread);
	if (const auto* info = report.info; info != nullptr) {
		out->Append("pc=0x{:016x} fault address=0x{:016x}\n", info->exception_address,
		            info->access_violation_vaddr);
		out->Append("rax={:016x} rbx={:016x} rcx={:016x} rdx={:016x}\n"
		            "rsi={:016x} rdi={:016x} rbp={:016x} rsp={:016x}\n"
		            "r8 ={:016x} r9 ={:016x} r10={:016x} r11={:016x}\n"
		            "r12={:016x} r13={:016x} r14={:016x} r15={:016x}\n",
		            info->rax, info->rbx, info->rcx, info->rdx, info->rsi, info->rdi, info->rbp,
		            info->rsp, info->r8, info->r9, info->r10, info->r11, info->r12, info->r13,
		            info->r14, info->r15);
	}
	if (report.code.size() > GuestFaultCodeBefore) {
		out->Append("code (pc-{} .. pc+{}, fault at byte {}):", GuestFaultCodeBefore,
		            report.code.size() - GuestFaultCodeBefore, GuestFaultCodeBefore);
		for (size_t i = 0; i < report.code.size(); i++) {
			out->Append("{}{:02x}", (i % 16 == 0) ? "\n " : " ", report.code[i]);
		}
		out->Append("\n");
	}
	if (!report.stack.empty()) {
		out->Append("stack:");
		for (size_t i = 0; i < report.stack.size(); i++) {
			out->Append("{} {:016x}", (i % 4 == 0) ? "\n" : "", report.stack[i]);
		}
		out->Append("\n");
	}
	if (!report.guest_frames.empty()) {
		out->Append("guest frames (rbp chain):\n");
		for (size_t i = 0; i < report.guest_frames.size(); i++) {
			out->Append("  {:2} 0x{:016x}\n", i, report.guest_frames[i]);
		}
	}
}

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_FAULT_REPORT_H_ */
