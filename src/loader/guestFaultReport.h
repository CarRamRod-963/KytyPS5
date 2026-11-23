#ifndef KYTY_LOADER_GUEST_FAULT_REPORT_H_
#define KYTY_LOADER_GUEST_FAULT_REPORT_H_

#include "common/hostException.h"
#include "loader/guestFaultHistory.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace Loader {

// Fixed-size text of a fatal fault report. The report is written from the exception handler on
// the faulting thread, which may have faulted inside the host heap: formatting must not allocate.
// Text past the capacity is dropped.
template <size_t N>
class BasicGuestFaultText {
public:
	static constexpr size_t Capacity = N;

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

using GuestFaultText = BasicGuestFaultText<8192>;

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

// Bytes around an address a register, a stack slot or a pointer in another dump held, copied by
// the handler from whatever of them was readable. Rows are 16 bytes; row i is valid when bit i of
// readable_rows is set, a row that could not be read prints as such.
struct GuestFaultMemory {
	std::string_view         name;   // "rax", "[rbp-0x68]", ...
	std::string_view         prefix; // in front of each row's offset: "", "rsp", "rbp"
	uint64_t                 value = 0;
	uint64_t                 start = 0; // address of bytes[0]
	std::span<const uint8_t> bytes;     // a multiple of 16 bytes, at most 64 rows
	uint64_t                 readable_rows = 0;
	bool                     red_zone      = false; // mark the 128 bytes below value
};

// Window sizes. A pointer register shows 16 bytes before and 0x110 from its address (a structure
// header and the fields after it); a pointer found in the first 0x40 bytes of such a dump, or in
// the frame below rbp, shows 0x110 bytes once more (one level of indirection, at most
// GuestFaultIndirectMax dumps). The stack pointer shows 256 bytes either side, so the 128 bytes
// below it (the SysV red zone, which every exception frame of the thread overwrites on Windows)
// can be told from the frame above; rbp shows the locals below it.
inline constexpr uint64_t GuestFaultPointerBefore = 16;
inline constexpr uint64_t GuestFaultPointerAfter  = 0x110;
inline constexpr uint64_t GuestFaultPointerScan   = 0x40;
inline constexpr uint64_t GuestFaultIndirectAfter = 0x110;
inline constexpr size_t   GuestFaultIndirectMax   = 32;
inline constexpr uint64_t GuestFaultStackBefore   = 256;
inline constexpr uint64_t GuestFaultStackAfter    = 256;
inline constexpr uint64_t GuestFaultFrameBefore   = 0x100;
inline constexpr uint64_t GuestFaultFrameAfter    = 0x20;
inline constexpr uint64_t GuestFaultRedZone       = 128;

// Where another guest thread was when the fault was reported.
struct GuestThreadState {
	std::string_view name;
	uint64_t         host_thread_id = 0;
	uint64_t         rip            = 0;
	uint64_t         rsp            = 0;
	bool             read           = false; // false: the thread could not be suspended or read
	std::string_view host_module;            // module of a rip outside the guest modules
	uint64_t         host_offset = 0;
};

// The second part of a fatal report: memory at the registers, the faults the thread had resolved
// before it and where the other guest threads were. It is written after the first part, from a
// buffer of its own, so the lines of the first part keep their order.
struct GuestFaultMemoryReport {
	const Common::HostException::ExceptionInfo* info = nullptr;
	std::span<const GuestFaultMemory>           registers; // one per register, in report order
	std::span<const GuestFaultMemory>           stack;     // rsp, rbp
	std::span<const GuestFaultMemory>           indirect;  // pointers found in the dumps above
	const HandledFaultRing*                     handled = nullptr;
	const GuestModuleMap*                       modules = nullptr;
	uint64_t                                    now_ns = 0; // monotonic, when the fault is reported
	std::span<const GuestThreadState>           threads;
	size_t                                      threads_total = 0; // other guest threads registered
};

using GuestFaultMemoryText = BasicGuestFaultText<192 * 1024>;

template <typename Text>
inline void AppendModuleLocation(const GuestModuleMap* modules, uint64_t address, Text* out) {
	if (modules == nullptr) {
		return;
	}
	if (const auto location = modules->Find(address); !location.module.empty()) {
		out->Append(" ({}+0x{:x})", location.module, location.offset);
	}
}

template <typename Text>
inline void AppendMemoryRows(const GuestFaultMemory& window, Text* out) {
	const size_t rows = std::min<size_t>(window.bytes.size() / 16, 64);
	for (size_t row = 0; row < rows; row++) {
		const uint64_t address = window.start + row * 16;
		// The row's offset from the value the window is around.
		const bool     below  = address < window.value;
		const uint64_t offset = below ? window.value - address : address - window.value;
		out->Append("  {}{}0x{:03x} {:016x}:", window.prefix, below ? '-' : '+', offset, address);
		if (((window.readable_rows >> row) & 1u) == 0) {
			out->Append(" (not readable)\n");
			continue;
		}
		for (size_t word = 0; word < 2; word++) {
			uint64_t value = 0;
			std::memcpy(&value, window.bytes.data() + row * 16 + word * 8, sizeof(value));
			out->Append(" {:016x}", value);
		}
		out->Append("{}\n",
		            window.red_zone && below && offset <= GuestFaultRedZone ? "  red zone" : "");
	}
}

template <typename Text>
inline void AppendMemoryWindow(const GuestFaultMemory& window, const GuestModuleMap* modules,
                               Text* out) {
	out->Append("{}=0x{:016x}", window.name, window.value);
	AppendModuleLocation(modules, window.value, out);
	out->Append(":\n");
	AppendMemoryRows(window, out);
}

inline const char* HandledFaultKindName(HandledFaultKind kind) {
	switch (kind) {
		case HandledFaultKind::Read: return "read";
		case HandledFaultKind::Write: return "write";
		case HandledFaultKind::Execute: return "execute";
		case HandledFaultKind::EmulatedInstruction: return "emulated instruction";
		case HandledFaultKind::SkippedLoad: return "skipped load";
	}
	return "?";
}

template <typename Text>
inline void FormatGuestFaultMemory(const GuestFaultMemoryReport& report, Text* out) {
	const auto* info = report.info;
	out->Append("--- Guest memory at the registers ---\n");
	if (info != nullptr) {
		out->Append("pc=0x{:016x}", info->exception_address);
		AppendModuleLocation(report.modules, info->exception_address, out);
		out->Append("\n");
	}
	for (size_t i = 0; i < report.registers.size(); i++) {
		const auto& window = report.registers[i];
		if (window.readable_rows == 0) {
			continue;
		}
		// Registers holding the same address share one dump.
		const GuestFaultMemory* same = nullptr;
		for (size_t j = 0; j < i && same == nullptr; j++) {
			const auto& earlier = report.registers[j];
			if (earlier.readable_rows != 0 && earlier.value == window.value) {
				same = &earlier;
			}
		}
		if (same != nullptr) {
			out->Append("{}=0x{:016x}: as {}\n", window.name, window.value, same->name);
			continue;
		}
		AppendMemoryWindow(window, report.modules, out);
	}
	for (const auto& window: report.stack) {
		if (window.readable_rows == 0) {
			continue;
		}
		const uint64_t below = window.value - window.start;
		out->Append("{}=0x{:016x}: {} bytes below to {} above", window.name, window.value, below,
		            window.start + window.bytes.size() - window.value);
		out->Append("{}\n", window.red_zone ? ", the 128 bytes below rsp are the red zone" : "");
		AppendMemoryRows(window, out);
	}
	bool indirect_header = false;
	for (const auto& window: report.indirect) {
		if (window.readable_rows == 0) {
			continue;
		}
		if (!indirect_header) {
			out->Append("--- Memory the pointers in those dumps point to ---\n");
			indirect_header = true;
		}
		AppendMemoryWindow(window, report.modules, out);
	}
	if (const auto* ring = report.handled; ring != nullptr && ring->Count() != 0) {
		out->Append("--- Faults this thread had resolved before (newest last, {} in total) ---\n",
		            ring->Count());
		for (size_t i = 0; i < ring->Size(); i++) {
			const auto&    fault = ring->At(i);
			const uint64_t age = report.now_ns >= fault.time_ns ? report.now_ns - fault.time_ns : 0;
			out->Append("  {} us before: {}", age / 1000, HandledFaultKindName(fault.kind));
			if (fault.kind != HandledFaultKind::EmulatedInstruction) {
				out->Append(" 0x{:016x}", fault.address);
			}
			out->Append(" pc=0x{:016x}", fault.pc);
			AppendModuleLocation(report.modules, fault.pc, out);
			out->Append(" rsp=0x{:016x}", fault.rsp);
			if (info != nullptr) {
				const bool     below = fault.rsp < info->rsp;
				const uint64_t delta = below ? info->rsp - fault.rsp : fault.rsp - info->rsp;
				out->Append(" (fault rsp{}0x{:x})", below ? '-' : '+', delta);
			}
			out->Append("\n");
		}
	}
	if (report.threads_total != 0) {
		out->Append("--- Other guest threads (suspended together for a moment, best effort; {} of "
		            "{}) ---\n",
		            report.threads.size(), report.threads_total);
		for (const auto& thread: report.threads) {
			out->Append("  {} (host thread {}):", thread.name, thread.host_thread_id);
			if (!thread.read) {
				out->Append(" not read\n");
				continue;
			}
			out->Append(" rip=0x{:016x}", thread.rip);
			if (!thread.host_module.empty()) {
				out->Append(" ({}+0x{:x})", thread.host_module, thread.host_offset);
			} else {
				AppendModuleLocation(report.modules, thread.rip, out);
			}
			out->Append(" rsp=0x{:016x}\n", thread.rsp);
		}
	}
}

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_FAULT_REPORT_H_ */
