#ifndef KYTY_LOADER_GUEST_FAULT_REPORT_H_
#define KYTY_LOADER_GUEST_FAULT_REPORT_H_

#include "common/faultReportLines.h"
#include "common/guestStoreRing.h"
#include "common/hostException.h"
#include "loader/guestFaultHistory.h"
#include "loader/guestInstructionPatcher.h"

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
	std::optional<GuestCopyLocation>            copy;         // pc inside a red-zone copy
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
	if (const auto& copy = report.copy; copy.has_value()) {
		// The pc above is the copy's: name the guest code it runs, as module offsets.
		out->Append("pc is in the red-zone copy of guest function 0x{:x} (module+0x{:x}, copy at "
		            "0x{:x})",
		            copy->function, copy->function - copy->module, copy->copy);
		if (copy->instruction != 0) {
			out->Append(", at its instruction 0x{:x} (module+0x{:x})", copy->instruction,
			            copy->instruction - copy->module);
		}
		out->Append("\n");
		if (report.info != nullptr && copy->lowered) {
			out->Append("rsp is 128 bytes below the guest rsp there: guest rsp=0x{:016x}\n",
			            report.info->rsp + 128);
		}
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
// the frame below rbp, shows 0x110 bytes once more, and so does a pointer in the first 0x40 bytes
// of those (two levels of indirection, at most GuestFaultIndirectLevel1 dumps on the first level
// and GuestFaultIndirectMax in all). The stack pointer shows 256 bytes either side, so the 128
// bytes below it (the SysV red zone, which every exception frame of the thread overwrites on
// Windows) can be told from the frame above; rbp shows the locals below it.
inline constexpr uint64_t GuestFaultPointerBefore = 16;
inline constexpr uint64_t GuestFaultPointerAfter  = 0x110;
inline constexpr uint64_t GuestFaultPointerScan   = 0x40;
inline constexpr uint64_t GuestFaultIndirectAfter = 0x110;
inline constexpr size_t   GuestFaultIndirectLevel1 = 32;
inline constexpr size_t   GuestFaultIndirectMax    = 64;
inline constexpr size_t   GuestFaultIndirectName   = 48;
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
	// For a thread in host code: the first return address into a guest module found on its stack
	// above rsp (the guest code that called into the emulator, best effort), or 0.
	uint64_t guest_caller = 0;
};

// Why a range's stores are listed (GuestFaultInterest::reasons is a mask of these).
enum GuestFaultInterestReason : uint8_t {
	GuestFaultNearFault         = 1u << 0u, // the granule of the faulting address
	GuestFaultNearResolvedFault = 1u << 1u, // the granule of a fault the thread had resolved
	GuestFaultNearRegister      = 1u << 2u, // the granule of a register value
	GuestFaultNearDump          = 1u << 3u, // a dumped range
	GuestFaultNearDumpPointer   = 1u << 4u, // the granule of a pointer anywhere in a dump
};

// A range whose stores the report lists: around the faulting address, the faults the thread had
// resolved, the registers' pointers, the pointers in the dumped memory and the dumps themselves.
struct GuestFaultInterest {
	uint64_t begin   = 0;
	uint64_t end     = 0;
	uint8_t  reasons = 0;
};

// The ranges CollectGuestFaultInterest kept (sorted, disjoint) and how many it had to leave out
// because the table was full.
struct GuestFaultInterestSet {
	size_t count   = 0;
	size_t dropped = 0;
};

// Granule around an address whose emulator stores are listed.
inline constexpr uint64_t GuestFaultStoreGranule = 0x10000;
// A register or a pointer in a dump marks its granule and the memory after it up to the end of the
// guest mapping holding it, at most this far from the granule: the array it points to.
inline constexpr uint64_t GuestFaultPointerReach = 0x100000;
// Ranges a report keeps at most: far more than the granules of its dumps' pointers usually are
// (one granule holds many of them).
inline constexpr size_t GuestFaultInterestMax = 1024;
// Stores listed: near the fault (the newest), and the last ones overall.
inline constexpr size_t GuestFaultStoresNear = 96;
inline constexpr size_t GuestFaultStoresLast = 32;
// Stores listed since the oldest fault the thread had resolved: the first and the last half of
// this many when there were more.
inline constexpr size_t GuestFaultStoresWindow = 256;

// Calls f(value, window, offset) for the qwords of the first `scan` bytes of each window: the first
// qword of every window first, then the second, ...; a window whose value an earlier one has is
// skipped.
template <typename F>
inline void ForEachDumpedPointer(std::span<const GuestFaultMemory> windows, uint64_t scan, F&& f) {
	for (uint64_t offset = 0; offset < scan; offset += 8) {
		for (size_t i = 0; i < windows.size(); i++) {
			const auto& window = windows[i];
			if (window.readable_rows == 0) {
				continue;
			}
			bool same = false;
			for (size_t j = 0; j < i && !same; j++) {
				same = windows[j].readable_rows != 0 && windows[j].value == window.value;
			}
			const uint64_t address = window.value + offset;
			if (same || address < window.start ||
			    address + 8 > window.start + window.bytes.size()) {
				continue;
			}
			const uint64_t at = address - window.start;
			if (((window.readable_rows >> (at / 16)) & 1u) == 0 ||
			    ((window.readable_rows >> ((at + 7) / 16)) & 1u) == 0) {
				continue;
			}
			uint64_t value = 0;
			std::memcpy(&value, window.bytes.data() + at, sizeof(value));
			f(value, window, offset);
		}
	}
}

// The qword at `address` when a dump holds all of it.
inline bool ReadDumpedWord(const GuestFaultMemory& window, uint64_t address, uint64_t* value) {
	if (window.readable_rows == 0 || address < window.start ||
	    address + 8 > window.start + window.bytes.size()) {
		return false;
	}
	const uint64_t at = address - window.start;
	if (((window.readable_rows >> (at / 16)) & 1u) == 0 ||
	    ((window.readable_rows >> ((at + 7) / 16)) & 1u) == 0) {
		return false;
	}
	std::memcpy(value, window.bytes.data() + at, sizeof(*value));
	return true;
}

inline bool GuestFaultWindowCovers(const GuestFaultMemory& window, uint64_t address) {
	return window.readable_rows != 0 && address >= window.start &&
	       address - window.start < window.bytes.size();
}

// The memory pointers in the dumps point to: those in the first GuestFaultPointerScan bytes of the
// register dumps (ForEachDumpedPointer order) and in the frame below rbp (stack[1], nearest first),
// at most GuestFaultIndirectLevel1, then those in the first GuestFaultPointerScan bytes of these
// dumps, up to out.size() in all. A target some dump already holds is not dumped again, so a cycle
// ends. accept(target) leaves targets out (not a pointer, on the stack, in a module);
// dump(name, target, i) returns the window of a target (readable_rows 0 when nothing is readable).
// Names ("[rax+0x8]", "[[rax+0x8]+0x10]") go to names[i] of the i-th dump.
template <typename Accept, typename Dump>
inline size_t CollectGuestFaultIndirect(std::span<const GuestFaultMemory> registers,
                                        std::span<const GuestFaultMemory> stack,
                                        std::span<GuestFaultMemory>       out,
                                        std::span<std::array<char, GuestFaultIndirectName>> names,
                                        Accept&& accept, Dump&& dump) {
	size_t count = 0;
	size_t limit = std::min(out.size(), GuestFaultIndirectLevel1);
	auto   add   = [&](uint64_t target, std::string_view base, char sign, uint64_t offset) {
		if (count >= limit || count >= names.size() || !accept(target)) {
			return;
		}
		for (const auto& w: registers) {
			if (GuestFaultWindowCovers(w, target)) {
				return;
			}
		}
		for (const auto& w: stack) {
			if (GuestFaultWindowCovers(w, target)) {
				return;
			}
		}
		for (size_t i = 0; i < count; i++) {
			if (GuestFaultWindowCovers(out[i], target)) {
				return;
			}
		}
		auto&      name = names[count];
		const auto written =
		    fmt::format_to_n(name.data(), name.size() - 1, "[{}{}0x{:x}]", base, sign, offset);
		name[std::min(written.size, name.size() - 1)] = '\0';
		auto window = dump(std::string_view(name.data()), target, count);
		if (window.readable_rows != 0) {
			out[count++] = window;
		}
	};
	ForEachDumpedPointer(registers, GuestFaultPointerScan,
	                     [&](uint64_t value, const GuestFaultMemory& window, uint64_t offset) {
		                     add(value, window.name, '+', offset);
	                     });
	if (stack.size() > 1) {
		const auto& frame = stack[1];
		for (uint64_t offset = 8; offset <= GuestFaultFrameBefore; offset += 8) {
			uint64_t value = 0;
			if (frame.value >= offset && ReadDumpedWord(frame, frame.value - offset, &value)) {
				add(value, "rbp", '-', offset);
			}
		}
	}
	const size_t level1 = count;
	limit               = out.size();
	ForEachDumpedPointer(std::span<const GuestFaultMemory>(out.data(), level1),
	                     GuestFaultPointerScan,
	                     [&](uint64_t value, const GuestFaultMemory& window, uint64_t offset) {
		                     add(value, window.name, '+', offset);
	                     });
	return count;
}

// Sorts out[0, count) by address and merges ranges that touch (their reasons combine). Returns
// the new count.
inline size_t MergeGuestFaultInterest(std::span<GuestFaultInterest> out, size_t count) {
	if (count == 0) {
		return 0;
	}
	std::sort(
	    out.begin(), out.begin() + static_cast<ptrdiff_t>(count),
	    [](const GuestFaultInterest& a, const GuestFaultInterest& b) { return a.begin < b.begin; });
	size_t merged = 0;
	for (size_t i = 1; i < count; i++) {
		auto& last = out[merged];
		if (out[i].begin <= last.end) {
			last.end = std::max(last.end, out[i].end);
			last.reasons |= out[i].reasons;
		} else {
			out[++merged] = out[i];
		}
	}
	return merged + 1;
}

// The interest ranges of a report, in this order of priority: the 64 KiB granule around the fault
// address; around the data address of each fault the thread had resolved (a page the thread wrote
// just before, the bytes it then read); around each register value; the dumped ranges (registers,
// stack, indirect); around every qword in the readable rows of any dump that may be a pointer (a
// field anywhere in a structure, a local in the frame). The result is sorted and disjoint. When the
// table is full it is merged; what still does not fit is counted in `dropped` (the lowest priority
// goes first). No allocation: the handler of a fatal fault calls it.
//
// mapping_end(address) returns the end of the guest mapping that holds `address` (0 when it does
// not know): a register or a dump pointer marks [granule, min(mapping end, granule +
// GuestFaultPointerReach)), at least its granule.
template <typename MappingEnd>
inline GuestFaultInterestSet CollectGuestFaultInterest(
    uint64_t fault_address, std::span<const GuestFaultMemory> registers,
    std::span<const GuestFaultMemory> stack, std::span<const GuestFaultMemory> indirect,
    const HandledFaultRing* handled, std::span<GuestFaultInterest> out, MappingEnd&& mapping_end) {
	GuestFaultInterestSet set;
	auto                  add = [&](uint64_t begin, uint64_t end, uint8_t reason) {
		if (end <= begin) {
			return;
		}
		if (set.count == out.size()) {
			set.count = MergeGuestFaultInterest(out, set.count);
		}
		if (set.count == out.size()) {
			set.dropped++;
			return;
		}
		out[set.count++] = {begin, end, reason};
	};
	auto granule = [&](uint64_t value, uint8_t reason) {
		if (value < 0x10000 || value > 0x00007fffffffffffull) {
			return;
		}
		const auto begin = value & ~(GuestFaultStoreGranule - 1);
		add(begin, begin + GuestFaultStoreGranule, reason);
	};
	auto pointer = [&](uint64_t value, uint8_t reason) {
		if (value < 0x10000 || value > 0x00007fffffffffffull) {
			return;
		}
		const auto begin = value & ~(GuestFaultStoreGranule - 1);
		const auto end   = std::clamp<uint64_t>(mapping_end(value), begin + GuestFaultStoreGranule,
		                                        begin + GuestFaultPointerReach);
		add(begin, end, reason);
	};
	auto dump = [&](const GuestFaultMemory& window) {
		if (window.readable_rows != 0) {
			add(window.start, window.start + window.bytes.size(), GuestFaultNearDump);
		}
	};
	auto pointers = [&](const GuestFaultMemory& window) {
		const size_t rows = std::min<size_t>(window.bytes.size() / 16, 64);
		for (size_t row = 0; row < rows; row++) {
			if (((window.readable_rows >> row) & 1u) == 0) {
				continue;
			}
			for (size_t word = 0; word < 2; word++) {
				uint64_t value = 0;
				std::memcpy(&value, window.bytes.data() + row * 16 + word * 8, sizeof(value));
				pointer(value, GuestFaultNearDumpPointer);
			}
		}
	};
	granule(fault_address, GuestFaultNearFault);
	if (handled != nullptr) {
		// Newest first: when the table runs out the oldest faults are the ones left out.
		for (size_t i = handled->Size(); i > 0; i--) {
			const auto& fault = handled->At(i - 1);
			if (fault.kind != HandledFaultKind::EmulatedInstruction) {
				granule(fault.address, GuestFaultNearResolvedFault);
			}
		}
	}
	for (const auto& window: registers) {
		pointer(window.value, GuestFaultNearRegister);
	}
	for (const auto& window: registers) {
		dump(window);
	}
	for (const auto& window: stack) {
		dump(window);
	}
	for (const auto& window: indirect) {
		if (window.readable_rows != 0) {
			pointer(window.value, GuestFaultNearDumpPointer);
			dump(window);
		}
	}
	for (const auto& window: registers) {
		pointers(window);
	}
	for (const auto& window: stack) {
		pointers(window);
	}
	for (const auto& window: indirect) {
		pointers(window);
	}
	set.count = MergeGuestFaultInterest(out, set.count);
	return set;
}

inline GuestFaultInterestSet CollectGuestFaultInterest(uint64_t fault_address,
                                                       std::span<const GuestFaultMemory> registers,
                                                       std::span<const GuestFaultMemory> stack,
                                                       std::span<const GuestFaultMemory> indirect,
                                                       const HandledFaultRing*           handled,
                                                       std::span<GuestFaultInterest>     out) {
	return CollectGuestFaultInterest(fault_address, registers, stack, indirect, handled, out,
	                                 [](uint64_t) { return uint64_t {0}; });
}

// The end (exclusive) of the run of resolved faults that starts at entry i: one kind, pc, rsp and
// store count, data addresses on consecutive 4 KiB pages in one direction (a loop zeroing or
// filling memory the tracker protects takes one such fault per page).
inline size_t HandledFaultRunEnd(const HandledFaultRing& ring, size_t i) {
	const auto& first = ring.At(i);
	if (first.kind == HandledFaultKind::EmulatedInstruction) {
		return i + 1;
	}
	int64_t step = 0;
	size_t  end  = i + 1;
	for (; end < ring.Size(); end++) {
		const auto& previous = ring.At(end - 1);
		const auto& fault    = ring.At(end);
		const auto delta = static_cast<int64_t>((fault.address >> 12u) - (previous.address >> 12u));
		if (fault.kind != first.kind || fault.pc != first.pc || fault.rsp != first.rsp ||
		    fault.stores != first.stores || (delta != 1 && delta != -1) ||
		    (step != 0 && delta != step)) {
			break;
		}
		step = delta;
	}
	return end;
}

// One line the GPU caches gave about an interest range.
struct GuestFaultLine {
	std::array<char, Common::FaultReportLines::LineSize> text {};
	size_t                                               size = 0;

	[[nodiscard]] std::string_view View() const { return {text.data(), size}; }
};

// Keeps lines in fixed storage and counts those that did not fit.
class GuestFaultLineCollector final: public Common::FaultReportLines {
public:
	explicit GuestFaultLineCollector(std::span<GuestFaultLine> storage): m_storage(storage) {}

	bool Line(std::string_view line) override {
		if (m_count == m_storage.size()) {
			m_dropped++;
			return false;
		}
		auto& out = m_storage[m_count++];
		out.size  = std::min(line.size(), out.text.size());
		std::memcpy(out.text.data(), line.data(), out.size);
		return true;
	}

	[[nodiscard]] size_t                          Count() const { return m_count; }
	[[nodiscard]] size_t                          Dropped() const { return m_dropped; }
	[[nodiscard]] std::span<const GuestFaultLine> Lines() const { return m_storage.first(m_count); }

private:
	std::span<GuestFaultLine> m_storage;
	size_t                    m_count   = 0;
	size_t                    m_dropped = 0;
};

// Interest ranges whose GPU cache state a report lists, at most.
inline constexpr size_t GuestFaultCacheRanges = 32;
inline constexpr size_t GuestFaultCacheLines  = 64;

// For up to max_ranges interest ranges, most telling first (those around the fault or a resolved
// fault, then a register's, then the pointers in the dumps; ranges that are only dumps, the stack,
// are left out): a line naming the range, then what describe(begin, end, lines) says about it.
// Returns how many ranges it described.
template <typename Describe>
inline size_t CollectGuestFaultCacheLines(std::span<const GuestFaultInterest> ranges,
                                          size_t max_ranges, Describe&& describe,
                                          GuestFaultLineCollector* lines) {
	static constexpr uint8_t passes[] = {
	    GuestFaultNearFault | GuestFaultNearResolvedFault,
	    GuestFaultNearRegister,
	    GuestFaultNearDumpPointer,
	};
	size_t  described = 0;
	uint8_t done      = 0;
	for (const auto pass: passes) {
		for (const auto& range: ranges) {
			if ((range.reasons & pass) == 0 || (range.reasons & done) != 0) {
				continue;
			}
			if (described == max_ranges ||
			    !lines->Format("  0x{:016x}-0x{:016x}:", range.begin, range.end)) {
				return described;
			}
			described++;
			const auto before = lines->Count();
			describe(range.begin, range.end, static_cast<Common::FaultReportLines*>(lines));
			if (lines->Count() == before && !lines->Line("    nothing cached")) {
				return described;
			}
		}
		done |= pass;
	}
	return described;
}

// The reasons of the sorted, disjoint ranges a store overlaps (0: none).
inline uint8_t GuestStoreInterest(std::span<const GuestFaultInterest> ranges,
                                  const Common::GuestStore&           store) {
	const uint64_t end = store.size == 0 ? store.address + 1 : store.address + store.size;
	auto           it  = std::upper_bound(
	    ranges.begin(), ranges.end(), store.address,
	    [](uint64_t address, const GuestFaultInterest& range) { return address < range.end; });
	uint8_t reasons = 0;
	for (; it != ranges.end() && it->begin < end; ++it) {
		reasons |= it->reasons;
	}
	return reasons;
}

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
	// The emulator's own stores into guest memory the ring kept (oldest first), how many it
	// recorded in all, the time stamp counter when the fault is reported and its rate, and the
	// ranges whose stores are listed in full.
	std::span<const Common::GuestStore> stores;
	uint64_t                            stores_total       = 0;
	uint64_t                            store_tsc_now      = 0;
	uint64_t                            store_ticks_per_us = 0;
	std::span<const GuestFaultInterest> interest; // sorted, disjoint
	size_t                              interest_dropped = 0;
	// What the GPU caches held over the first cache_ranges interest ranges, and how many lines
	// did not fit.
	std::span<const GuestFaultLine> cache_lines;
	size_t                          cache_ranges        = 0;
	size_t                          cache_lines_dropped = 0;
};

// Room for every dump, every thread and the store lists at their limits (TestMemoryReportFits).
using GuestFaultMemoryText = BasicGuestFaultText<384 * 1024>;

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
inline void AppendGuestStore(const GuestFaultMemoryReport& report, const Common::GuestStore& store,
                             Text* out) {
	const uint64_t ticks = report.store_tsc_now >= store.tsc ? report.store_tsc_now - store.tsc : 0;
	out->Append("  #{} ", store.sequence);
	if (report.store_ticks_per_us != 0) {
		out->Append("{} us before", ticks / report.store_ticks_per_us);
	} else {
		out->Append("{} ticks before", ticks);
	}
	out->Append(": {} 0x{:016x} size 0x{:x} first {:016x} thread {}",
	            Common::GuestStorePathName(store.path), store.address, store.size, store.first,
	            store.thread);
	if (store.source != 0) {
		out->Append(" from {} 0x{:016x}", Common::GuestStoreSourceName(store.path), store.source);
	}
	if (const auto reasons = GuestStoreInterest(report.interest, store); reasons != 0) {
		static constexpr std::pair<uint8_t, const char*> names[] = {
		    {GuestFaultNearFault, "fault address"},
		    {GuestFaultNearResolvedFault, "resolved fault"},
		    {GuestFaultNearRegister, "register"},
		    {GuestFaultNearDump, "dump"},
		    {GuestFaultNearDumpPointer, "pointer in a dump"},
		};
		const char* separator = " (near: ";
		for (const auto& [bit, name]: names) {
			if ((reasons & bit) != 0) {
				out->Append("{}{}", separator, name);
				separator = ", ";
			}
		}
		out->Append(")");
	}
	out->Append("\n");
}

// Entries [first, end) of the ring, one run (HandledFaultRunEnd), as a mark among the stores.
template <typename Text>
inline void AppendResolvedFaultMark(const HandledFaultRing& ring, size_t first, size_t end,
                                    const GuestModuleMap* modules, Text* out) {
	const auto& fault = ring.At(first);
	if (end - first > 1) {
		out->Append("  -- {} resolved {} faults 0x{:016x} .. 0x{:016x}", end - first,
		            HandledFaultKindName(fault.kind), fault.address, ring.At(end - 1).address);
	} else {
		out->Append("  -- resolved {} fault", HandledFaultKindName(fault.kind));
		if (fault.kind != HandledFaultKind::EmulatedInstruction) {
			out->Append(" 0x{:016x}", fault.address);
		}
	}
	out->Append(" pc=0x{:016x}", fault.pc);
	AppendModuleLocation(modules, fault.pc, out);
	out->Append(" --\n");
}

// Every store between the oldest fault the thread had resolved and the fault (at most
// GuestFaultStoresWindow, the first and the last half of them when there were more), with the
// resolved faults at their place among them.
template <typename Text>
inline void AppendGuestStoreWindow(const GuestFaultMemoryReport& report, Text* out) {
	const auto* ring = report.handled;
	if (ring == nullptr || ring->Size() == 0) {
		return;
	}
	// Stores numbered after the oldest fault's count were made after it.
	const uint64_t after = ring->At(0).stores;
	size_t         first = report.stores.size();
	for (size_t i = 0; i < report.stores.size(); i++) {
		if (report.stores[i].sequence > after) {
			first = i;
			break;
		}
	}
	const size_t   in_window = report.stores.size() - first;
	const uint64_t made      = report.stores_total > after ? report.stores_total - after : 0;
	out->Append("--- The emulator's stores since the oldest of those resolved faults ({} made, {} "
	            "kept) ---\n",
	            made, in_window);
	if (made > in_window) {
		out->Append("  ({} of them are no longer in the ring)\n", made - in_window);
	}
	constexpr size_t half = GuestFaultStoresWindow / 2;
	size_t           next = 0; // the next resolved fault to place
	for (size_t i = first; i < report.stores.size(); i++) {
		const auto& store = report.stores[i];
		while (next < ring->Size() && ring->At(next).stores < store.sequence) {
			const auto end = HandledFaultRunEnd(*ring, next);
			AppendResolvedFaultMark(*ring, next, end, report.modules, out);
			next = end;
		}
		const size_t index = i - first;
		if (in_window > GuestFaultStoresWindow && index >= half && index < in_window - half) {
			if (index == half) {
				out->Append("  ... {} stores left out ...\n", in_window - 2 * half);
			}
			continue;
		}
		AppendGuestStore(report, store, out);
	}
	while (next < ring->Size()) {
		const auto end = HandledFaultRunEnd(*ring, next);
		AppendResolvedFaultMark(*ring, next, end, report.modules, out);
		next = end;
	}
}

// The emulator's own stores into guest memory: those over the report's interest ranges (the newest
// GuestFaultStoresNear of them), those since the thread's oldest resolved fault, then the last
// GuestFaultStoresLast of all.
template <typename Text>
inline void AppendGuestStores(const GuestFaultMemoryReport& report, Text* out) {
	if (report.stores_total == 0) {
		return;
	}
	size_t matched = 0;
	for (const auto& store: report.stores) {
		matched += GuestStoreInterest(report.interest, store) != 0 ? 1 : 0;
	}
	out->Append("--- The emulator's own stores into guest memory near this fault ({} of the last "
	            "{}; {} recorded; {} ranges",
	            matched, report.stores.size(), report.stores_total, report.interest.size());
	if (report.interest_dropped != 0) {
		out->Append(", {} more left out", report.interest_dropped);
	}
	out->Append(") ---\n");
	size_t skip = matched > GuestFaultStoresNear ? matched - GuestFaultStoresNear : 0;
	for (const auto& store: report.stores) {
		if (GuestStoreInterest(report.interest, store) == 0) {
			continue;
		}
		if (skip != 0) {
			skip--;
			continue;
		}
		AppendGuestStore(report, store, out);
	}
	if (report.cache_ranges != 0) {
		out->Append("--- GPU caches over {} of those ranges (buffers, tracked pages, images; read "
		            "without waiting for a lock) ---\n",
		            report.cache_ranges);
		for (const auto& line: report.cache_lines) {
			out->Append("{}\n", line.View());
		}
		if (report.cache_lines_dropped != 0) {
			out->Append("  ({} more lines left out)\n", report.cache_lines_dropped);
		}
	}
	AppendGuestStoreWindow(report, out);
	const auto last = std::min(report.stores.size(), GuestFaultStoresLast);
	out->Append("--- The emulator's last {} stores into guest memory ---\n", last);
	for (size_t i = report.stores.size() - last; i < report.stores.size(); i++) {
		AppendGuestStore(report, report.stores[i], out);
	}
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
		for (size_t i = 0; i < ring->Size();) {
			const auto     end   = HandledFaultRunEnd(*ring, i);
			const auto&    fault = ring->At(i);
			const uint64_t age = report.now_ns >= fault.time_ns ? report.now_ns - fault.time_ns : 0;
			if (end - i > 1) {
				const auto&    last = ring->At(end - 1);
				const uint64_t last_age =
				    report.now_ns >= last.time_ns ? report.now_ns - last.time_ns : 0;
				out->Append("  {}..{} us before: {} {} faults 0x{:016x} .. 0x{:016x}", age / 1000,
				            last_age / 1000, end - i, HandledFaultKindName(fault.kind),
				            fault.address, last.address);
			} else {
				out->Append("  {} us before: {}", age / 1000, HandledFaultKindName(fault.kind));
				if (fault.kind != HandledFaultKind::EmulatedInstruction) {
					out->Append(" 0x{:016x}", fault.address);
				}
			}
			i = end;
			out->Append(" pc=0x{:016x}", fault.pc);
			AppendModuleLocation(report.modules, fault.pc, out);
			out->Append(" rsp=0x{:016x}", fault.rsp);
			if (info != nullptr) {
				const bool     below = fault.rsp < info->rsp;
				const uint64_t delta = below ? info->rsp - fault.rsp : fault.rsp - info->rsp;
				out->Append(" (fault rsp{}0x{:x})", below ? '-' : '+', delta);
			}
			if (fault.stores != 0) {
				out->Append(" after store #{}", fault.stores);
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
			out->Append(" rsp=0x{:016x}", thread.rsp);
			if (thread.guest_caller != 0) {
				out->Append(" guest caller 0x{:016x}", thread.guest_caller);
				AppendModuleLocation(report.modules, thread.guest_caller, out);
			}
			out->Append("\n");
		}
	}
	AppendGuestStores(report, out);
}

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_FAULT_REPORT_H_ */
