#include "loader/guestFaultReport.h"

#include "common/rangeDirectory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "GuestFaultReportTests: failed: %s\n", text);
    std::exit(1);
  }
}

bool Contains(const std::string &text, const char *part) {
  return text.find(part) != std::string::npos;
}

void TestFullReport() {
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x9001ff0fcull;
  info.access_violation_vaddr = 0x30cbf40000ull;
  info.rdx = 0x30cbf00000ull;
  info.rsi = 0x3fff0ull;
  info.rsp = 0x7ec1c4000ull;
  info.r15 = 0x1234ull;

  std::array<uint8_t, 2 * Loader::GuestFaultCodeBefore> code{};
  for (size_t i = 0; i < code.size(); i++) {
    code[i] = static_cast<uint8_t>(i);
  }
  const std::array<uint64_t, 8> stack{0x11,           0x22, 0x33, 0x44,
                                      0x900200123ull, 0,    0,    0};
  const std::array<uint64_t, 2> frames{0x900200123ull, 0x900300456ull};

  Loader::GuestFaultReport report;
  report.thread = "GameThread";
  report.info = &info;
  report.code = code;
  report.stack = stack;
  report.guest_frames = frames;
  Loader::GuestFaultText out;
  Loader::FormatGuestFaultReport(report, &out);
  const std::string text(out.View());

  Check(Contains(text, "--- Guest fault context ---\nthread: GameThread\n"),
        "header or thread");
  Check(Contains(text,
                 "pc=0x00000009001ff0fc fault address=0x00000030cbf40000\n"),
        "pc and fault address");
  Check(Contains(text, "rdx=00000030cbf00000"), "rdx");
  Check(Contains(text, "rsi=000000000003fff0"), "rsi");
  Check(Contains(text, "r15=0000000000001234\n"), "r15 ends the register file");
  Check(Contains(text, "code (pc-48 .. pc+48, fault at byte 48):\n 00 01 02"),
        "code start");
  Check(Contains(text, " 5f\n"), "code end");
  Check(Contains(text,
                 "stack:\n 0000000000000011 0000000000000022 0000000000000033 "
                 "0000000000000044\n 0000000900200123"),
        "stack rows of four words");
  Check(Contains(text, "guest frames (rbp chain):\n   0 0x0000000900200123\n   "
                       "1 0x0000000900300456\n"),
        "guest frames");
  Check(text.ends_with("   1 0x0000000900300456\n"), "frames end the report");
  Check(!out.Truncated(), "a full report fits");
}

void TestMissingParts() {
  Loader::GuestFaultReport report;
  report.thread = "(host thread)";
  Loader::GuestFaultText out;
  Loader::FormatGuestFaultReport(report, &out);
  Check(out.View() == "--- Guest fault context ---\nthread: (host thread)\n",
        "unreadable parts must leave no empty sections");
}

void TestTruncation() {
  // Text past the fixed capacity is dropped, never written past the buffer.
  Loader::GuestFaultText out;
  const std::string line(1000, 'x');
  for (int i = 0; i < 20; i++) {
    out.Append("{}\n", line);
  }
  Check(out.Truncated(), "overflow is reported");
  Check(out.View().size() == Loader::GuestFaultText::Capacity,
        "the text fills the capacity exactly");
  Check(out.View().substr(0, 1001) == line + "\n", "the first lines are kept");
  out.Append("more");
  Check(out.View().size() == Loader::GuestFaultText::Capacity,
        "a full buffer takes no more text");
}

// A window whose bytes count up: row i holds the bytes 16*i .. 16*i+15.
Loader::GuestFaultMemory MakeWindow(std::string_view name, uint64_t value,
                                    uint64_t before, std::span<uint8_t> bytes,
                                    uint64_t readable_rows) {
  for (size_t i = 0; i < bytes.size(); i++) {
    bytes[i] = static_cast<uint8_t>(i);
  }
  Loader::GuestFaultMemory window;
  window.name = name;
  window.value = value;
  window.start = value - before;
  window.bytes = bytes;
  window.readable_rows = readable_rows;
  return window;
}

constexpr size_t PointerBytes =
    Loader::GuestFaultPointerBefore + Loader::GuestFaultPointerAfter;
constexpr uint64_t PointerRows = (uint64_t{1} << (PointerBytes / 16)) - 1;

void TestMemoryWindows() {
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x900e7c3d1ull;
  info.rsp = 0x7ec1c4000ull;
  info.rbp = 0x7ec1c4100ull;

  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x4000000ull, "eboot.bin");

  std::array<std::array<uint8_t, PointerBytes>, 5> bytes{};
  std::array<uint8_t, 512> stack_bytes{};
  std::array<uint8_t, 0x120> frame_bytes{};
  const std::array<Loader::GuestFaultMemory, 5> registers{
      // Readable: 16 bytes before and 0x110 from the pointer.
      MakeWindow("rbx", 0x30cbf00020ull, 16, bytes[0], PointerRows),
      // Nothing readable: no lines at all.
      MakeWindow("rcx", 0x30cc000000ull, 16, bytes[1], 0),
      // Partly readable: the second row is not.
      MakeWindow("rdi", 0x30cc100010ull, 16, bytes[2], PointerRows & ~2ull),
      // The same address as rbx: no second dump.
      MakeWindow("r8", 0x30cbf00020ull, 16, bytes[3], PointerRows),
      // A pointer into the module.
      MakeWindow("r9", 0x900200010ull, 16, bytes[4], PointerRows),
  };
  std::array<Loader::GuestFaultMemory, 2> stack{
      MakeWindow("rsp", info.rsp, 256, stack_bytes, 0x7fffffffull),
      MakeWindow("rbp", info.rbp, 0x100, frame_bytes, 0x3ffffull)};
  stack[0].prefix = "rsp";
  stack[0].red_zone = true;
  stack[1].prefix = "rbp";
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.registers = registers;
  report.stack = stack;
  report.modules = &modules;
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  const std::string text(out.View());

  Check(text.starts_with("--- Guest memory at the registers ---\n"
                         "pc=0x0000000900e7c3d1 (eboot.bin+0xe7c3d1)\n"),
        "memory header and the module offset of the pc");
  Check(Contains(text, "rbx=0x00000030cbf00020:\n"
                       "  -0x010 00000030cbf00010: 0706050403020100 "
                       "0f0e0d0c0b0a0908\n"
                       "  +0x000 00000030cbf00020: 1716151413121110 "
                       "1f1e1d1c1b1a1918\n"),
        "pointer rows as little-endian words, offsets from the pointer");
  Check(Contains(text, "  +0x100 00000030cbf00120: 1716151413121110 "
                       "1f1e1d1c1b1a1918\nrdi="),
        "0x110 bytes from the pointer end the window");
  Check(!Contains(text, "rcx="), "an unreadable pointer prints nothing");
  Check(Contains(text, "  +0x000 00000030cc100010: (not readable)\n"),
        "an unreadable row is named");
  Check(Contains(text, "  +0x010 00000030cc100020: 2726252423222120 "),
        "the rows after an unreadable one are dumped");
  Check(Contains(text, "r8=0x00000030cbf00020: as rbx\n"),
        "an address dumped before is not dumped again");
  Check(Contains(text, "r9=0x0000000900200010 (eboot.bin+0x200010):\n"),
        "a pointer into a module gets its offset");
  Check(Contains(text, "rsp=0x00000007ec1c4000: 256 bytes below to 256 above, "
                       "the 128 bytes below rsp are the red zone\n"),
        "stack window header");
  Check(Contains(text, "  rsp-0x100 00000007ec1c3f00: 0706050403020100 "
                       "0f0e0d0c0b0a0908\n"),
        "the stack window starts 256 bytes below rsp, outside the red zone");
  Check(Contains(text, "  rsp-0x090 00000007ec1c3f70: 7776757473727170 "
                       "7f7e7d7c7b7a7978\n"),
        "the row below the red zone is not marked");
  Check(Contains(text, "  rsp-0x080 00000007ec1c3f80: 8786858483828180 "
                       "8f8e8d8c8b8a8988  red zone\n"),
        "the red zone starts 128 bytes below rsp");
  Check(Contains(text, "  rsp-0x010 00000007ec1c3ff0: f7f6f5f4f3f2f1f0 "
                       "fffefdfcfbfaf9f8  red zone\n"),
        "the red zone ends at rsp");
  Check(Contains(text, "  rsp+0x000 00000007ec1c4000: 0706050403020100 "
                       "0f0e0d0c0b0a0908\n"),
        "rsp itself is not in the red zone");
  Check(Contains(text, "  rsp+0x0f0 00000007ec1c40f0: (not readable)\n"
                       "rbp=0x00000007ec1c4100: 256 bytes below to 32 above\n"
                       "  rbp-0x100 00000007ec1c4000: 0706050403020100 "),
        "the frame window follows the stack window, without a red zone");
  Check(text.ends_with("  rbp+0x010 00000007ec1c4110: 1716151413121110 "
                       "1f1e1d1c1b1a1918\n"),
        "the frame window ends 32 bytes above rbp, no red zone mark");
  Check(!Contains(text, "point to ---"), "no section without indirect dumps");
  Check(!Contains(text, "resolved before"),
        "no section for a thread without resolved faults");
  Check(!Contains(text, "Other guest threads"),
        "no section without other guest threads");
  Check(!out.Truncated(), "the memory part fits");
}

void TestIndirectAndThreads() {
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x900e7c3d1ull;
  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x4000000ull, "eboot.bin");
  std::array<uint8_t, 0x110> bytes{};
  const std::array<Loader::GuestFaultMemory, 1> indirect{
      MakeWindow("[r11+0x0]", 0x30d0000000ull, 0, bytes, 0x1ffff)};
  const std::array<Loader::GuestThreadState, 3> threads{
      Loader::GuestThreadState{.name = "Background Worker #2",
                               .host_thread_id = 1234,
                               .rip = 0x900e6f000ull,
                               .rsp = 0x7ec000000ull,
                               .read = true},
      Loader::GuestThreadState{.name = "AudioThread",
                               .host_thread_id = 1235,
                               .rip = 0x7ff600001000ull,
                               .rsp = 0x7ed000000ull,
                               .read = true,
                               .host_module = "kyty_emulator.exe",
                               .host_offset = 0x1000},
      Loader::GuestThreadState{.name = "RenderThread", .host_thread_id = 1236}};
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.indirect = indirect;
  report.modules = &modules;
  report.threads = threads;
  report.threads_total = 4;
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  const std::string text(out.View());
  Check(Contains(text, "--- Memory the pointers in those dumps point to ---\n"
                       "[r11+0x0]=0x00000030d0000000:\n"
                       "  +0x000 00000030d0000000: 0706050403020100 "),
        "indirect dump header and first row");
  Check(Contains(text, "  +0x100 00000030d0000100: 0706050403020100 "),
        "indirect dumps show 0x110 bytes");
  Check(Contains(text, "--- Other guest threads (suspended together for a "
                       "moment, best effort; 3 of 4) ---\n"
                       "  Background Worker #2 (host thread 1234): "
                       "rip=0x0000000900e6f000 (eboot.bin+0xe6f000) "
                       "rsp=0x00000007ec000000\n"),
        "a guest thread in guest code");
  Check(Contains(text,
                 "  AudioThread (host thread 1235): rip=0x00007ff600001000 "
                 "(kyty_emulator.exe+0x1000) rsp=0x00000007ed000000\n"),
        "a guest thread in host code");
  Check(text.ends_with("  RenderThread (host thread 1236): not read\n"),
        "a thread that could not be read");
}

void TestModuleMap() {
  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x1000ull, "eboot.bin");
  modules.Add(0x900001000ull, 0x800ull, "eboot.bin patch area");
  Check(modules.Find(0x900000fffull).module == "eboot.bin", "end of a module");
  Check(modules.Find(0x900001000ull).module == "eboot.bin patch area" &&
            modules.Find(0x900001010ull).offset == 0x10,
        "the next range starts at the end of the first");
  Check(modules.Find(0x900001800ull).module.empty(), "past every module");
  Check(modules.Find(0x8ffffffffull).module.empty(), "below every module");
  modules.Remove(0x900000000ull);
  Check(modules.Find(0x900000010ull).module.empty(),
        "an unloaded module is not found");
  const std::string long_name(100, 'm');
  modules.Add(0x910000000ull, 0x10ull, long_name);
  Check(modules.Find(0x910000000ull).module.size() ==
            Loader::GuestModuleMap::NameSize - 1,
        "long names are cut, not overrun");
  modules.Clear();
  Check(modules.Find(0x900001010ull).module.empty(), "cleared");
}

// A thread that faults while another writes the report waits for that report
// (or the time limit); the reporting thread waits for nothing.
void TestFaultReportGate() {
  {
    Loader::FaultReportGate gate;
    Check(gate.Begin(), "the first thread writes the report");
    Check(!gate.Begin(),
          "a second fault on the reporting thread is turned away");
    uint32_t slept = 0;
    Check(!gate.WaitForReport(100, [&](uint32_t ms) { slept += ms; }) &&
              slept == 0,
          "the reporting thread does not wait for its own report");
    std::atomic<bool> other_returned{false};
    std::atomic<bool> other_saw_report{false};
    std::thread other([&] {
      Check(!gate.Begin(), "a second thread is turned away");
      other_saw_report = gate.WaitForReport(10000, [](uint32_t ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
      });
      other_returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Check(!other_returned,
          "a second thread goes on before the report is written");
    gate.Finish();
    other.join();
    Check(other_saw_report, "the second thread saw the finished report");
  }
  {
    Loader::FaultReportGate gate;
    Check(gate.Begin(), "the first thread writes the report");
    std::thread other([&] {
      uint32_t slept = 0;
      Check(!gate.Begin(), "a second thread is turned away");
      Check(!gate.WaitForReport(25, [&](uint32_t ms) { slept += ms; }) &&
                slept == 25,
            "a report that never ends holds the second thread for the limit "
            "only");
    });
    other.join();
  }
}

void TestThreadRegistry() {
  // Static: the registry is large.
  static Loader::GuestThreadRegistry registry;
  auto collect = [] {
    std::string text;
    registry.ForEach([&](uint64_t id, std::string_view name) {
      text += std::to_string(id) + "=" + std::string(name) + ";";
    });
    return text;
  };
  Check(!registry.Add(0, "x"), "id 0 is not a thread");
  Check(registry.Add(7, "Main") && registry.Add(8, "Worker"), "two threads");
  Check(collect() == "7=Main;8=Worker;", "registered threads in slot order");
  registry.Rename(8, std::string(100, 'w'));
  Check(collect() ==
            "7=Main;8=" +
                std::string(Loader::GuestThreadRegistry::NameSize - 1, 'w') +
                ";",
        "a long name is cut");
  registry.Rename(0, "nobody");
  registry.Remove(0);
  registry.Remove(7);
  Check(collect() ==
            "8=" + std::string(Loader::GuestThreadRegistry::NameSize - 1, 'w') +
                ";",
        "a removed thread is gone; id 0 touches no free slot");
  Check(registry.Add(9, "Next") && collect().starts_with("9=Next;"),
        "a free slot is reused");
  for (uint64_t id = 100; id < 100 + Loader::GuestThreadRegistry::Capacity;
       id++) {
    registry.Add(id, "t");
  }
  Check(!registry.Add(5000, "full"), "a full registry refuses a thread");
}

void TestHandledFaultRing() {
  Loader::HandledFaultRing ring;
  Check(ring.Count() == 0 && ring.Size() == 0, "empty ring");
  for (uint64_t i = 1; i <= 3; i++) {
    ring.Record({.address = i, .pc = 0x100 + i, .rsp = 0, .time_ns = i * 1000});
  }
  Check(ring.Size() == 3 && ring.At(0).address == 1 && ring.At(2).address == 3,
        "oldest first before the ring wraps");
  constexpr uint64_t total = Loader::HandledFaultRing::Capacity + 12;
  for (uint64_t i = 4; i <= total; i++) {
    ring.Record({.address = i, .pc = 0x100 + i, .rsp = 0, .time_ns = i * 1000});
  }
  Check(ring.Count() == total, "every fault is counted");
  Check(ring.Size() == 64 && Loader::HandledFaultRing::Capacity == 64,
        "the ring keeps 64");
  for (size_t i = 0; i < ring.Size(); i++) {
    Check(ring.At(i).address == 13 + i, "the last 64 faults, oldest first");
  }
}

void TestHandledFaultReport() {
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x900e7c3d1ull;
  info.rsp = 0x7ec1c4000ull;
  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x4000000ull, "eboot.bin");
  modules.Add(0x904000000ull, 0x800000ull, "eboot.bin patch area");

  Loader::HandledFaultRing ring;
  using Kind = Loader::HandledFaultKind;
  constexpr uint64_t n = Loader::HandledFaultRing::Capacity + 2;
  for (uint64_t i = 0; i < n; i++) {
    ring.Record({.address = i == n - 2 ? 0 : 0x30cbf00000ull + i * 0x1000,
                 .pc = i == n - 1 ? 0x904000123ull : 0x900e7c000ull + i,
                 .rsp = i == n - 1 ? 0x7ec1c3fc0ull : 0x7ec1c4040ull,
                 .time_ns = 1'000'000 + i * 1000,
                 .kind = i == 0       ? Kind::Read
                         : i == n - 2 ? Kind::EmulatedInstruction
                                      : Kind::Write});
  }
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.handled = &ring;
  report.modules = &modules;
  report.now_ns = 1'000'000 + n * 1000;
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  const std::string text(out.View());
  Check(Contains(text, "--- Faults this thread had resolved before (newest "
                       "last, 66 in total) ---\n"
                       "  64 us before: write 0x00000030cbf02000 "
                       "pc=0x0000000900e7c002 (eboot.bin+0xe7c002) "
                       "rsp=0x00000007ec1c4040 (fault rsp+0x40)\n"),
        "the oldest kept fault first, with its age and module offset");
  Check(Contains(text, "  2 us before: emulated instruction "
                       "pc=0x0000000900e7c040 (eboot.bin+0xe7c040) "),
        "an emulated instruction has no data address");
  Check(text.ends_with("  1 us before: write 0x00000030cbf41000 "
                       "pc=0x0000000904000123 (eboot.bin patch area+0x123) "
                       "rsp=0x00000007ec1c3fc0 (fault rsp-0x40)\n"),
        "the newest fault last; a pc in the patch area is named");
  Check(!Contains(text, "read 0x"), "faults older than the ring are gone");

  // A loop over protected pages: one line for the run, in either direction;
  // a fault at another pc or after other stores starts a new line.
  Loader::HandledFaultRing runs;
  for (uint64_t i = 0; i < 5; i++) {
    runs.Record({.address = 0x332b574000ull + i * 0x1000,
                 .pc = 0x8000105aeull,
                 .rsp = 0x7ec1c4000ull,
                 .time_ns = 1'000'000 + i * 1000,
                 .kind = Kind::Write,
                 .stores = 7});
  }
  for (uint64_t i = 0; i < 3; i++) {
    runs.Record({.address = 0x1e018eafe8ull - i * 0x1000,
                 .pc = 0x900ff81c9ull,
                 .rsp = 0x7ec1c4000ull,
                 .time_ns = 1'010'000 + i * 1000,
                 .kind = Kind::Write,
                 .stores = 9});
  }
  runs.Record({.address = 0x1e018e7fe8ull,
               .pc = 0x900ff81c9ull,
               .rsp = 0x7ec1c4000ull,
               .time_ns = 1'020'000,
               .kind = Kind::Write,
               .stores = 10});
  report.handled = &runs;
  report.now_ns = 1'100'000;
  Loader::GuestFaultMemoryText run_out;
  Loader::FormatGuestFaultMemory(report, &run_out);
  const std::string run_text(run_out.View());
  Check(Contains(
            run_text,
            "(newest last, 9 in total) ---\n"
            "  100..96 us before: 5 write faults 0x000000332b574000 .. "
            "0x000000332b578000 pc=0x00000008000105ae "
            "rsp=0x00000007ec1c4000 (fault rsp+0x0) after store #7\n"
            "  90..88 us before: 3 write faults 0x0000001e018eafe8 .. "
            "0x0000001e018e8fe8 pc=0x0000000900ff81c9 (eboot.bin+0xff81c9) "
            "rsp=0x00000007ec1c4000 (fault rsp+0x0) after store #9\n"
            "  80 us before: write 0x0000001e018e7fe8 pc=0x0000000900ff81c9 "),
        "runs of page faults are one line each");

  // A run keeps one direction: back and forth over two pages is two lines.
  Loader::HandledFaultRing zigzag;
  for (const uint64_t page : {0x10ull, 0x11ull, 0x10ull}) {
    zigzag.Record({.address = page << 12u,
                   .pc = 0x900ff81c9ull,
                   .rsp = 0x7ec1c4000ull,
                   .time_ns = 1'000'000,
                   .kind = Kind::Write,
                   .stores = 3});
  }
  Check(Loader::HandledFaultRunEnd(zigzag, 0) == 2 &&
            Loader::HandledFaultRunEnd(zigzag, 2) == 3,
        "a run ends where its direction turns");
}

void TestMemoryReportFits() {
  // Every register, every pointer and every thread readable: the memory part
  // still fits its buffer.
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x9000000ffull;
  info.rsp = 0x7ec1c4000ull;
  info.rbp = 0x7ec1c4100ull;
  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x4000000ull,
              std::string(Loader::GuestModuleMap::NameSize, 'x'));
  static std::array<std::array<uint8_t, PointerBytes>, 16> bytes{};
  static std::array<Loader::GuestFaultMemory, 16> registers{};
  static const char *const names[16] = {
      "fault address", "rax", "rbx", "rcx", "rdx", "rsi",
      "rdi",           "rbp", "r8",  "r9",  "r10", "r11",
      "r12",           "r13", "r14", "r15"};
  for (size_t i = 0; i < registers.size(); i++) {
    registers[i] = MakeWindow(names[i], 0x900100000ull + i * 0x1000, 16,
                              bytes[i], PointerRows);
  }
  static std::array<std::array<uint8_t, 0x110>, Loader::GuestFaultIndirectMax>
      indirect_bytes{};
  static std::array<Loader::GuestFaultMemory, Loader::GuestFaultIndirectMax>
      indirect{};
  for (size_t i = 0; i < indirect.size(); i++) {
    indirect[i] = MakeWindow("[rbp-0x100]", 0x7ff700000000ull + i * 0x1000, 0,
                             indirect_bytes[i], 0x1ffff);
  }
  static std::array<uint8_t, 512> stack_bytes{};
  static std::array<uint8_t, 0x120> frame_bytes{};
  std::array<Loader::GuestFaultMemory, 2> stack{
      MakeWindow("rsp", info.rsp, 256, stack_bytes, 0xffffffffull),
      MakeWindow("rbp", info.rbp, 0x100, frame_bytes, 0x3ffffull)};
  stack[0].prefix = "rsp";
  stack[0].red_zone = true;
  stack[1].prefix = "rbp";
  Loader::HandledFaultRing ring;
  for (uint64_t i = 0; i < 100; i++) {
    ring.Record({.address = ~0ull,
                 .pc = 0x900000000ull + i,
                 .rsp = 0,
                 .time_ns = 0,
                 .kind = Loader::HandledFaultKind::SkippedLoad});
  }
  static std::array<Loader::GuestThreadState,
                    Loader::GuestThreadRegistry::Capacity>
      threads{};
  const std::string thread_name(Loader::GuestThreadRegistry::NameSize - 1, 't');
  for (auto &thread : threads) {
    thread = {.name = thread_name,
              .host_thread_id = 0xffffffff,
              .rip = 0x900000000ull,
              .rsp = 0x7ec1c4000ull,
              .read = true};
  }
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.registers = registers;
  report.stack = stack;
  report.indirect = indirect;
  report.handled = &ring;
  report.modules = &modules;
  report.now_ns = ~0ull;
  report.threads = threads;
  report.threads_total = threads.size();
  static std::array<Common::GuestStore, Common::GuestStoreRing::Capacity>
      stores{};
  for (size_t i = 0; i < stores.size(); i++) {
    stores[i] = {.sequence = ~0ull - stores.size() + i,
                 .tsc = 0,
                 .address = 0x7ff700000000ull,
                 .size = ~0ull >> 1u,
                 .first = ~0ull,
                 .source = ~0ull,
                 .thread = ~0u,
                 .path = Common::GuestStorePath::MetadataClearPublish};
  }
  static std::array<Loader::GuestFaultInterest, Loader::GuestFaultInterestMax>
      interest{};
  const auto interest_set = Loader::CollectGuestFaultInterest(
      ~0ull, registers, stack, indirect, &ring, interest);
  report.stores = stores;
  report.stores_total = ~0ull;
  report.store_tsc_now = ~0ull;
  report.interest = {interest.data(), interest_set.count};
  report.interest_dropped = interest_set.dropped;
  static Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  Check(!out.Truncated(), "a report with everything readable fits");
  Check(out.View().size() < Loader::GuestFaultMemoryText::Capacity * 9 / 10,
        "with room to spare");
}

void TestCopyLocation() {
  // A fault inside a red-zone function copy names the guest code it runs.
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x9100000040ull;
  info.rsp = 0x7ec1c3f80ull;
  Loader::GuestFaultReport report;
  report.thread = "Foreground Worker #1";
  report.info = &info;
  report.copy = Loader::GuestCopyLocation{.module = 0x900000000ull,
                                          .function = 0x9001feffeull,
                                          .copy = 0x9100000000ull,
                                          .instruction = 0x9001ff0f3ull,
                                          .lowered = true};
  Loader::GuestFaultText out;
  Loader::FormatGuestFaultReport(report, &out);
  const std::string text(out.View());
  Check(Contains(text,
                 "pc is in the red-zone copy of guest function 0x9001feffe "
                 "(module+0x1feffe, copy at 0x9100000000), at its "
                 "instruction 0x9001ff0f3 (module+0x1ff0f3)\n"),
        "copy location");
  Check(Contains(text, "guest rsp=0x00000007ec1c4000\n"),
        "guest rsp of a lowered copy");

  report.copy->lowered = false;
  report.copy->instruction = 0;
  Loader::GuestFaultText plain;
  Loader::FormatGuestFaultReport(report, &plain);
  const std::string plain_text(plain.View());
  Check(Contains(plain_text, "(module+0x1feffe, copy at 0x9100000000)\n"),
        "copy location without an instruction");
  Check(!Contains(plain_text, "guest rsp="),
        "no rsp line where RSP is the guest's");
}

void TestGuestStoreRing() {
  static Common::GuestStoreRing ring;
  using Path = Common::GuestStorePath;
  std::array<Common::GuestStore, 8> few{};
  Check(ring.Count() == 0 && ring.Snapshot(few.data(), few.size()) == 0,
        "empty store ring");
  const uint64_t word = 0x1122334455667788ull;
  ring.Record(Path::ImageWriteBack, 0x30c0000000ull, 0x1000, &word, 7, 100);
  const uint16_t half = 0xabcd;
  ring.Record(Path::Label, 0x30c0001000ull, sizeof(half), &half, 8, 200);
  ring.Record(Path::BufferReadback, 0x30c0002000ull, 0x10, nullptr, 9, 300,
              0x30bf000000ull);
  Check(ring.Snapshot(few.data(), few.size()) == 3, "three stores kept");
  Check(few[0].source == 0 && few[2].source == 0x30bf000000ull,
        "a store keeps the buffer or image it came from");
  Check(few[0].sequence == 1 && few[0].path == Path::ImageWriteBack &&
            few[0].address == 0x30c0000000ull && few[0].size == 0x1000 &&
            few[0].first == word && few[0].thread == 7 && few[0].tsc == 100,
        "a store keeps its path, range, first bytes, thread and time");
  Check(few[1].first == 0xabcd, "a short store keeps only its bytes");
  Check(few[2].sequence == 3 && few[2].first == 0, "no data, no bytes");
  Check(ring.Snapshot(few.data(), 2) == 2 && few[1].sequence == 2,
        "a short buffer takes the oldest stores");

  // Past the capacity the ring keeps the newest stores, oldest first.
  for (uint64_t i = 0; i < Common::GuestStoreRing::Capacity + 10; i++) {
    ring.Record(Path::Backing, i, 8, &i, 1, i);
  }
  static std::array<Common::GuestStore, Common::GuestStoreRing::Capacity> all{};
  const auto kept = ring.Snapshot(all.data(), all.size());
  const auto total = ring.Count();
  Check(total == Common::GuestStoreRing::Capacity + 13, "every store counted");
  Check(kept == Common::GuestStoreRing::Capacity &&
            all[0].sequence == total - kept + 1 &&
            all[kept - 1].sequence == total,
        "the ring keeps the last Capacity stores");
  for (size_t i = 0; i < kept; i++) {
    Check(all[i].first == all[i].address, "a kept store is the one recorded");
  }

  // Concurrent writers, and a reader meanwhile: every slot read is one whole
  // store.
  constexpr uint64_t writers = 4;
  constexpr uint64_t each = 20000;
  std::vector<std::thread> threads;
  for (uint64_t t = 0; t < writers; t++) {
    threads.emplace_back([t] {
      for (uint64_t i = 0; i < each; i++) {
        const uint64_t address = (t << 32u) | i;
        ring.Record(Common::GuestStorePath::AsyncReadback, address, 8, &address,
                    static_cast<uint32_t>(t), i);
      }
    });
  }
  for (int round = 0; round < 20; round++) {
    const auto seen = ring.Snapshot(all.data(), all.size());
    for (size_t i = 0; i < seen; i++) {
      Check(all[i].first == all[i].address, "a torn store was read");
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
  Check(ring.Count() == total + writers * each, "concurrent stores counted");
  const auto after = ring.Snapshot(all.data(), all.size());
  Check(after == Common::GuestStoreRing::Capacity, "full ring after writers");
  for (size_t i = 0; i < after; i++) {
    Check(all[i].first == all[i].address &&
              all[i].thread == (all[i].address >> 32u) &&
              all[i].tsc == (all[i].address & 0xffffffffu),
          "a concurrent store is whole");
    Check(i == 0 || all[i].sequence == all[i - 1].sequence + 1,
          "kept stores are consecutive");
  }

  Common::GuestStore point{.address = 0x2000, .size = 0};
  Check(Common::GuestStoreOverlaps(point, 0x2000, 0x2001) &&
            !Common::GuestStoreOverlaps(point, 0x2001, 0x3000),
        "an empty store overlaps its own byte only");
  Common::GuestStore range{.address = 0x1000, .size = 0x1000};
  Check(!Common::GuestStoreOverlaps(range, 0x2000, 0x3000) &&
            Common::GuestStoreOverlaps(range, 0x1fff, 0x2000) &&
            Common::GuestStoreOverlaps(range, 0, 0x1001),
        "store ranges overlap by byte");
}

void TestStoreReport() {
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x900ff8048ull;
  info.access_violation_vaddr = 0x3204420158ull;
  std::array<uint8_t, PointerBytes> bytes{};
  const std::array<Loader::GuestFaultMemory, 1> registers{
      MakeWindow("rax", 0x3300000020ull, 16, bytes, PointerRows)};
  Loader::HandledFaultRing ring;
  ring.Record({.address = 0x3204420158ull,
               .pc = 0x900ff8030ull,
               .rsp = 0,
               .time_ns = 10,
               .kind = Loader::HandledFaultKind::Read,
               .stores = 42});
  std::array<Loader::GuestFaultInterest, 8> interest{};
  const auto interest_set = Loader::CollectGuestFaultInterest(
      info.access_violation_vaddr, registers, {}, {}, &ring, interest);
  const auto interest_count = interest_set.count;
  Check(interest_count == 2 && interest_set.dropped == 0 &&
            interest[0].begin == 0x3204420000ull &&
            interest[0].end == 0x3204430000ull &&
            interest[0].reasons == (Loader::GuestFaultNearFault |
                                    Loader::GuestFaultNearResolvedFault) &&
            interest[1].begin == 0x3300000000ull &&
            interest[1].end == 0x3300010000ull &&
            interest[1].reasons ==
                (Loader::GuestFaultNearRegister | Loader::GuestFaultNearDump),
        "interest: the fault's granule, the register's granule with its dump, "
        "sorted and merged");
  using Path = Common::GuestStorePath;
  const std::array<Common::GuestStore, 4> stores{
      // Over the second chunk of the faulting array: the store to find.
      Common::GuestStore{.sequence = 41,
                         .tsc = 1000,
                         .address = 0x3204420000ull,
                         .size = 0x1000,
                         .first = ~0ull,
                         .thread = 5,
                         .path = Path::MetadataClearPublish},
      // Elsewhere.
      Common::GuestStore{.sequence = 42,
                         .tsc = 2000,
                         .address = 0x5000000000ull,
                         .size = 8,
                         .first = 1,
                         .thread = 6,
                         .path = Path::Label},
      // In the granule of rax.
      Common::GuestStore{.sequence = 43,
                         .tsc = 3000,
                         .address = 0x330000f000ull,
                         .size = 0x2000,
                         .first = 2,
                         .source = 0x3300000000ull,
                         .thread = 7,
                         .path = Path::BufferReadback},
      Common::GuestStore{.sequence = 44,
                         .tsc = 4000,
                         .address = 0x5000001000ull,
                         .size = 8,
                         .first = 3,
                         .thread = 6,
                         .path = Path::Label},
  };
  const std::array<Loader::GuestThreadState, 1> threads{
      Loader::GuestThreadState{.name = "Background Worker #6",
                               .host_thread_id = 99,
                               .rip = 0x7ff600002000ull,
                               .rsp = 0x7ec000000ull,
                               .read = true,
                               .host_module = "kyty_emulator.exe",
                               .host_offset = 0x2000,
                               .guest_caller = 0x900e7cab9ull}};
  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x4000000ull, "eboot.bin");
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.modules = &modules;
  report.handled = &ring;
  report.threads = threads;
  report.threads_total = 1;
  report.stores = stores;
  report.stores_total = 44;
  report.store_tsc_now = 5000;
  report.store_ticks_per_us = 10;
  report.interest = {interest.data(), interest_count};
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  const std::string text(out.View());
  Check(Contains(text, " after store #42\n"),
        "a resolved fault names the stores made before it");
  Check(Contains(text, "rsp=0x00000007ec000000 guest caller 0x0000000900e7cab9 "
                       "(eboot.bin+0xe7cab9)\n"),
        "a thread in host code names its guest caller");
  Check(Contains(text,
                 "--- The emulator's own stores into guest memory near this "
                 "fault (2 of the last 4; 44 recorded; 2 ranges) ---\n"
                 "  #41 400 us before: metadata clear publish "
                 "0x0000003204420000 size 0x1000 first ffffffffffffffff "
                 "thread 5 (near: fault address, resolved fault)\n"
                 "  #43 200 us before: buffer readback 0x000000330000f000 "
                 "size 0x2000 first 0000000000000002 thread 7 from buffer "
                 "0x0000003300000000 (near: register, dump)\n"
                 "--- The emulator's stores since the oldest of those "
                 "resolved faults (2 made, 2 kept) ---\n"
                 "  -- resolved read fault 0x0000003204420158 "
                 "pc=0x0000000900ff8030 (eboot.bin+0xff8030) --\n"
                 "  #43 200 us before: buffer readback 0x000000330000f000 "
                 "size 0x2000 first 0000000000000002 thread 7 from buffer "
                 "0x0000003300000000 (near: register, dump)\n"
                 "  #44 100 us before: label 0x0000005000001000 size 0x8 "
                 "first 0000000000000003 thread 6\n"
                 "--- The emulator's last 4 stores into guest memory ---\n"
                 "  #41 "),
        "stores near the fault, since the resolved fault, then the last "
        "ones");
  Check(text.ends_with("  #44 100 us before: label 0x0000005000001000 size 0x8 "
                       "first 0000000000000003 thread 6\n"),
        "the newest store ends the report");
}

// The ff8048 shape (issue #17): a heightfield at r8 holds its spans pointer at
// +0x48, past the first 0x40 bytes of the dump; the thread had just taken
// write faults on the array's second 128 KiB (a memset) and faults reading
// it. Stores there are found through the deep pointer scan and the resolved
// faults' granules, and every store since the oldest resolved fault is listed
// with the faults at their place.
void TestStoreInterestDeep() {
  constexpr uint64_t hf = 0x11c21c6700ull, spans = 0x332b550000ull;
  Common::HostException::ExceptionInfo info;
  info.exception_address = 0x900ff8048ull;
  info.access_violation_vaddr = ~0ull;
  std::array<uint8_t, PointerBytes> bytes{};
  auto r8 = MakeWindow("r8", hf, 16, bytes, PointerRows);
  std::memset(bytes.data(), 0, bytes.size());
  std::memcpy(bytes.data() + 16 + 0x48, &spans, sizeof(spans));
  const std::array<Loader::GuestFaultMemory, 1> registers{r8};
  Loader::HandledFaultRing ring;
  for (uint64_t page = 0; page < 3; page++) {
    ring.Record({.address = spans + 0x24000 + page * 0x1000,
                 .pc = 0x8000105aeull,
                 .rsp = 0,
                 .time_ns = 0,
                 .kind = Loader::HandledFaultKind::Write,
                 .stores = 100});
  }
  ring.Record({.address = 0x332b57a828ull,
               .pc = 0x900ff81e6ull,
               .rsp = 0,
               .time_ns = 0,
               .kind = Loader::HandledFaultKind::Write,
               .stores = 101});
  std::array<Loader::GuestFaultInterest, 16> interest{};
  const auto set = Loader::CollectGuestFaultInterest(
      info.access_violation_vaddr, registers, {}, {}, &ring, interest);
  const std::span<const Loader::GuestFaultInterest> ranges{interest.data(),
                                                           set.count};
  using Path = Common::GuestStorePath;
  const Common::GuestStore in_spans{.sequence = 102,
                                    .tsc = 102,
                                    .address = spans + 0x8000,
                                    .size = 0x1000,
                                    .path = Path::AsyncReadback};
  const Common::GuestStore second_chunk{.sequence = 103,
                                        .tsc = 103,
                                        .address = spans + 0x203e8,
                                        .size = 8,
                                        .path = Path::BufferReadback};
  const Common::GuestStore elsewhere{.sequence = 104,
                                     .tsc = 104,
                                     .address = 0x1ec29e0000ull,
                                     .size = 0x4000,
                                     .path = Path::AsyncReadback};
  Check(Loader::GuestStoreInterest(ranges, in_spans) ==
            Loader::GuestFaultNearDumpPointer,
        "a pointer past the first 0x40 bytes of a dump is an interest range");
  Check(Loader::GuestStoreInterest(ranges, second_chunk) ==
            Loader::GuestFaultNearResolvedFault,
        "the granule of a resolved fault is an interest range");
  Check(Loader::GuestStoreInterest(ranges, elsewhere) == 0,
        "other memory is not");
  // A pointer marks the memory after its granule to the end of its mapping (the
  // issue #17 second sighting: the bad entry at spans + 0x20030, 128 KiB past
  // the pointer, with no resolved fault there), at most 1 MiB.
  std::array<Loader::GuestFaultInterest, 16> reach{};
  const auto reach_set = Loader::CollectGuestFaultInterest(
      info.access_violation_vaddr, registers, {}, {}, nullptr, reach,
      [](uint64_t address) {
        return address >= spans && address < spans + 0x30000 ? spans + 0x30000
                                                             : 0;
      });
  const std::span<const Loader::GuestFaultInterest> reach_ranges{
      reach.data(), reach_set.count};
  Check(Loader::GuestStoreInterest(reach_ranges, second_chunk) ==
            Loader::GuestFaultNearDumpPointer,
        "a pointer marks its mapping past its granule");
  Check(Loader::GuestStoreInterest(
            reach_ranges, {.address = spans + 0x30000, .size = 8}) == 0,
        "and no further than the mapping's end");
  const auto capped_set = Loader::CollectGuestFaultInterest(
      info.access_violation_vaddr, registers, {}, {}, nullptr, reach,
      [](uint64_t) { return ~0ull; });
  const std::span<const Loader::GuestFaultInterest> capped{reach.data(),
                                                           capped_set.count};
  Check(Loader::GuestStoreInterest(
            capped, {.address = spans + 0xff000, .size = 8}) != 0 &&
            Loader::GuestStoreInterest(
                capped, {.address = spans + 0x100000, .size = 8}) == 0,
        "a pointer's reach stops at 1 MiB");
  Check(Loader::GuestStoreInterest(capped,
                                   {.address = hf + 0xf0000, .size = 8}) != 0,
        "a register marks its mapping too");

  // Pointers in the stack dumps and in the indirect dumps count too, and the
  // stack dumps are interest ranges themselves.
  std::array<uint8_t, 0x120> frame{};
  auto rbp = MakeWindow("rbp", 0x7ec1c4100ull, 0x100, frame, 0x3ffffull);
  std::memset(frame.data(), 0, frame.size());
  const uint64_t local = 0x6100000000ull;
  std::memcpy(frame.data() + 0x40, &local, sizeof(local));
  std::array<uint8_t, 0x110> target{};
  auto deeper = MakeWindow("[r8+0x0]", 0x6200000000ull, 0, target, 0x1ffff);
  std::memset(target.data(), 0, target.size());
  const uint64_t field = 0x6300000000ull;
  std::memcpy(target.data() + 0x100, &field, sizeof(field));
  const std::array<Loader::GuestFaultMemory, 1> frames{rbp};
  const std::array<Loader::GuestFaultMemory, 1> pointed{deeper};
  std::array<Loader::GuestFaultInterest, 16> more{};
  const auto more_set = Loader::CollectGuestFaultInterest(
      ~0ull, {}, frames, pointed, nullptr, more);
  const std::span<const Loader::GuestFaultInterest> more_ranges{more.data(),
                                                                more_set.count};
  Check(Loader::GuestStoreInterest(more_ranges,
                                   {.address = 0x6100008000ull, .size = 8}) ==
            Loader::GuestFaultNearDumpPointer,
        "a pointer in the stack dumps is an interest range");
  Check(Loader::GuestStoreInterest(more_ranges,
                                   {.address = 0x6300000100ull, .size = 8}) ==
            Loader::GuestFaultNearDumpPointer,
        "a pointer deep in an indirect dump is an interest range");
  Check((Loader::GuestStoreInterest(more_ranges,
                                    {.address = 0x7ec1c4080ull, .size = 8}) &
         Loader::GuestFaultNearDump) != 0,
        "a stack dump is an interest range");
  for (size_t i = 1; i < set.count; i++) {
    Check(interest[i - 1].end < interest[i].begin, "ranges sorted, disjoint");
  }

  // The stores since the oldest resolved fault (#100): #101 .. #104, with the
  // faults at their place, and #99 before them left out.
  const std::array<Common::GuestStore, 5> stores{
      Common::GuestStore{.sequence = 99, .address = 0x5000000000ull, .size = 8},
      Common::GuestStore{.sequence = 101,
                         .tsc = 101,
                         .address = 0x5000000100ull,
                         .size = 8,
                         .path = Path::Label},
      in_spans, second_chunk, elsewhere};
  Loader::GuestModuleMap modules;
  modules.Add(0x900000000ull, 0x4000000ull, "eboot.bin");
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.modules = &modules;
  report.handled = &ring;
  report.stores = stores;
  report.stores_total = 104;
  report.store_tsc_now = 105;
  report.store_ticks_per_us = 1;
  report.interest = ranges;
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  const std::string text(out.View());
  Check(Contains(text, "near this fault (2 of the last 5; 104 recorded; "),
        "both stores over the array are near the fault");
  Check(
      Contains(
          text,
          "--- The emulator's stores since the oldest of those resolved "
          "faults (4 made, 4 kept) ---\n"
          "  -- 3 resolved write faults 0x000000332b574000 .. "
          "0x000000332b576000 pc=0x00000008000105ae --\n"
          "  #101 4 us before: label 0x0000005000000100 size 0x8 first "
          "0000000000000000 thread 0\n"
          "  -- resolved write fault 0x000000332b57a828 pc=0x0000000900ff81e6 "
          "(eboot.bin+0xff81e6) --\n"
          "  #102 3 us before: async readback 0x000000332b558000 size 0x1000 "
          "first 0000000000000000 thread 0 (near: pointer in a dump)\n"
          "  #103 2 us before: buffer readback 0x000000332b5703e8 size 0x8 "
          "first 0000000000000000 thread 0 (near: resolved fault)\n"
          "  #104 1 us before: async readback"),
      "the stores since the oldest resolved fault, faults in between");

  // More than the window holds: its first and last half, and how many were
  // left out; stores the ring no longer holds are counted.
  static std::array<Common::GuestStore, 600> many{};
  for (size_t i = 0; i < many.size(); i++) {
    many[i] = {.sequence = 1000 + i, .address = 0x5000000000ull, .size = 8};
  }
  Loader::HandledFaultRing old_ring;
  old_ring.Record({.address = 0x5000000000ull,
                   .kind = Loader::HandledFaultKind::Write,
                   .stores = 900});
  report.handled = &old_ring;
  report.stores = many;
  report.stores_total = 1599;
  report.store_tsc_now = 0;
  report.interest = {};
  static Loader::GuestFaultMemoryText long_out;
  Loader::FormatGuestFaultMemory(report, &long_out);
  const std::string long_text(long_out.View());
  Check(Contains(long_text, "faults (699 made, 600 kept) ---\n"
                            "  (99 of them are no longer in the ring)\n"
                            "  -- resolved write fault"),
        "a window larger than the ring says so");
  Check(Contains(long_text, "  #1127 0 us before") &&
            Contains(long_text, "  ... 344 stores left out ...\n  #1472 ") &&
            !Contains(long_text, "  #1128 ") && !Contains(long_text, "#1471 "),
        "a long window keeps its first and its last half");
}

// A table too small for every range: it merges first, then leaves the lowest
// priority (pointers in dumps) out and counts them; the report says so.
void TestStoreInterestOverflow() {
  std::array<uint8_t, PointerBytes> bytes{};
  auto rax = MakeWindow("rax", 0x3300000020ull, 16, bytes, PointerRows);
  for (size_t i = 0; i < bytes.size() / 8; i++) {
    const uint64_t pointer = 0x4000000000ull + i * 0x100000;
    std::memcpy(bytes.data() + i * 8, &pointer, sizeof(pointer));
  }
  const std::array<Loader::GuestFaultMemory, 1> registers{rax};
  std::array<Loader::GuestFaultInterest, 4> interest{};
  const auto set = Loader::CollectGuestFaultInterest(0x3204420158ull, registers,
                                                     {}, {}, nullptr, interest);
  Check(set.count == 4 && set.dropped == bytes.size() / 8 - 2,
        "a full table keeps what fits and counts the rest");
  Check(interest[0].begin == 0x3204420000ull &&
            interest[0].reasons == Loader::GuestFaultNearFault &&
            interest[1].begin == 0x3300000000ull &&
            interest[1].reasons ==
                (Loader::GuestFaultNearRegister | Loader::GuestFaultNearDump),
        "the fault and the register come first");
  Common::HostException::ExceptionInfo info;
  const std::array<Common::GuestStore, 1> stores{
      Common::GuestStore{.sequence = 1, .address = 0x3204420000ull, .size = 8}};
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.stores = stores;
  report.stores_total = 1;
  report.interest = {interest.data(), set.count};
  report.interest_dropped = set.dropped;
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  Check(Contains(std::string(out.View()),
                 "(1 of the last 1; 1 recorded; 4 ranges, 34 more left out) "
                 "---\n"),
        "the report says ranges were left out");
}

// The cache directory: slots set, cleared and flagged by the owner, read from
// another thread meanwhile without a torn range.
void TestRangeDirectory() {
  static Common::RangeDirectory directory;
  directory.Set(3, 0x332b540000ull, 0x20000);
  directory.Set(7, 0x332b570000ull, 0x10000, 1);
  directory.Set(9, 0x5000000000ull, 0x1000);
  directory.Set(Common::RangeDirectory::Capacity, 0x332b570000ull, 0x10000);
  std::vector<std::pair<uint32_t, uint64_t>> seen;
  auto collect = [&](uint64_t begin, uint64_t end) {
    seen.clear();
    return directory.ForEachOverlap(
        begin, end,
        [&](uint32_t index, const Common::RangeDirectory::Range &r) {
          seen.emplace_back(index, r.base | r.flags);
        });
  };
  Check(collect(0x332b550000ull, 0x332b580000ull) == 0 && seen.size() == 2 &&
            seen[0].first == 3 && seen[1].first == 7 &&
            seen[1].second == (0x332b570000ull | 1),
        "the ranges over an address range, with their flags");
  directory.SetFlags(7, 1, false);
  directory.Clear(3);
  Check(collect(0x332b550000ull, 0x332b580000ull) == 0 && seen.size() == 1 &&
            seen[0].second == 0x332b570000ull,
        "a cleared slot and a cleared flag are gone");
  Check(collect(0x332b580000ull, 0x332b590000ull) == 0 && seen.empty(),
        "ranges that end before the query are left out");

  std::atomic<bool> stop{false};
  std::thread writer([&] {
    for (uint64_t i = 0; !stop.load(); i++) {
      directory.Set(11, 0x7000000000ull + (i % 16) * 0x10000,
                    0x10000 + (i % 16));
    }
  });
  for (int round = 0; round < 200000; round++) {
    directory.ForEachOverlap(
        0x7000000000ull, 0x7000100000ull,
        [&](uint32_t, const Common::RangeDirectory::Range &r) {
          Check(r.size - 0x10000 == ((r.base - 0x7000000000ull) >> 16u),
                "a range read while it is written is whole");
        });
  }
  stop = true;
  writer.join();
}

// Lines the GPU caches give, most telling ranges first, within the caps.
void TestCacheLines() {
  const std::array<Loader::GuestFaultInterest, 4> ranges{
      Loader::GuestFaultInterest{0x1000000ull, 0x1010000ull,
                                 Loader::GuestFaultNearDumpPointer},
      Loader::GuestFaultInterest{0x2000000ull, 0x2010000ull,
                                 Loader::GuestFaultNearDump},
      Loader::GuestFaultInterest{0x3000000ull, 0x3010000ull,
                                 Loader::GuestFaultNearRegister},
      Loader::GuestFaultInterest{0x4000000ull, 0x4010000ull,
                                 Loader::GuestFaultNearResolvedFault}};
  std::vector<uint64_t> asked;
  auto describe = [&](uint64_t begin, uint64_t,
                      Common::FaultReportLines *lines) {
    asked.push_back(begin);
    if (begin != 0x3000000ull) {
      (void)lines->Format("    buffer 0x{:x}", begin);
    }
  };
  std::array<Loader::GuestFaultLine, 16> storage{};
  Loader::GuestFaultLineCollector lines(storage);
  const auto ranges_described =
      Loader::CollectGuestFaultCacheLines(ranges, 8, describe, &lines);
  Check(ranges_described == 3 && asked.size() == 3 &&
            asked[0] == 0x4000000ull && asked[1] == 0x3000000ull &&
            asked[2] == 0x1000000ull,
        "resolved faults first, then registers, then pointers; no stack dumps");
  Check(lines.Count() == 6 &&
            lines.Lines()[0].View() ==
                "  0x0000000004000000-0x0000000004010000:" &&
            lines.Lines()[1].View() == "    buffer 0x4000000" &&
            lines.Lines()[3].View() == "    nothing cached",
        "a line naming each range, then what is cached there");

  std::array<Loader::GuestFaultLine, 3> small{};
  Loader::GuestFaultLineCollector few(small);
  Check(Loader::CollectGuestFaultCacheLines(ranges, 8, describe, &few) == 2 &&
            few.Count() == 3 && few.Dropped() == 1,
        "lines past the storage are counted");
  asked.clear();
  Loader::GuestFaultLineCollector one(storage);
  Check(Loader::CollectGuestFaultCacheLines(ranges, 1, describe, &one) == 1 &&
            asked.size() == 1,
        "at most max_ranges ranges");

  Common::HostException::ExceptionInfo info;
  const std::array<Common::GuestStore, 1> stores{
      Common::GuestStore{.sequence = 1,
                         .address = 0x4000000ull,
                         .size = 8,
                         .path = Common::GuestStorePath::FileRead}};
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.stores = stores;
  report.stores_total = 1;
  report.interest = ranges;
  report.cache_lines = few.Lines();
  report.cache_ranges = 2;
  report.cache_lines_dropped = few.Dropped();
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  Check(Contains(std::string(out.View()),
                 "--- GPU caches over 2 of those ranges (buffers, tracked "
                 "pages, images; read without waiting for a lock) ---\n"
                 "  0x0000000004000000-0x0000000004010000:\n"
                 "    buffer 0x4000000\n"
                 "  0x0000000003000000-0x0000000003010000:\n"
                 "  (1 more lines left out)\n"),
        "the cache lines in the report");
  Check(Contains(std::string(out.View()), ": file read 0x0000000004000000"),
        "an HLE store names its path");
}

void TestTwoLevelIndirect() {
  // rax -> A; A -> B, A -> A and A -> rax's window; B -> C. Two levels dump A
  // and B; pointers into a dump are not dumped again (cycles end), and C,
  // a third level, is not dumped.
  constexpr uint64_t a = 0x31000000ull, b = 0x32000000ull, c = 0x33000000ull;
  constexpr uint64_t rax_value = 0x30000000ull;
  std::array<uint8_t, PointerBytes> rax_bytes{};
  auto rax = MakeWindow("rax", rax_value, 16, rax_bytes, PointerRows);
  std::memset(rax_bytes.data(), 0, rax_bytes.size());
  std::memcpy(rax_bytes.data() + 16, &a, 8);
  static std::array<std::array<uint8_t, 0x110>, 8> memory{};
  const auto contents = [](uint64_t address) -> std::array<uint8_t, 0x110> & {
    return memory[(address >> 24u) & 7u];
  };
  std::memcpy(contents(a).data(), &b, 8);
  std::memcpy(contents(a).data() + 8, &a, 8);
  std::memcpy(contents(a).data() + 16, &rax_value, 8);
  std::memcpy(contents(b).data(), &c, 8);
  const std::array<Loader::GuestFaultMemory, 1> registers{rax};
  std::array<Loader::GuestFaultMemory, Loader::GuestFaultIndirectMax> out{};
  std::array<std::array<char, Loader::GuestFaultIndirectName>,
             Loader::GuestFaultIndirectMax>
      names{};
  std::vector<uint64_t> dumped;
  const auto count = Loader::CollectGuestFaultIndirect(
      registers, {}, out, names,
      [](uint64_t target) { return target >= 0x10000; },
      [&](std::string_view name, uint64_t target, size_t) {
        dumped.push_back(target);
        Loader::GuestFaultMemory window;
        window.name = name;
        window.value = target;
        window.start = target;
        window.bytes = contents(target);
        window.readable_rows = 0x1ffff;
        return window;
      });
  Check(count == 2 && out[0].value == a && out[0].name == "[rax+0x0]" &&
            out[1].value == b && out[1].name == "[[rax+0x0]+0x0]",
        "two levels of pointers, named by their path");
  Check(dumped.size() == 2 &&
            std::find(dumped.begin(), dumped.end(), c) == dumped.end(),
        "pointers into a dump end a cycle; no third level");
}

} // namespace

int main() {
  TestFullReport();
  TestMissingParts();
  TestTruncation();
  TestMemoryWindows();
  TestIndirectAndThreads();
  TestModuleMap();
  TestThreadRegistry();
  TestFaultReportGate();
  TestHandledFaultRing();
  TestHandledFaultReport();
  TestMemoryReportFits();
  TestCopyLocation();
  TestGuestStoreRing();
  TestStoreReport();
  TestTwoLevelIndirect();
  TestStoreInterestDeep();
  TestStoreInterestOverflow();
  TestRangeDirectory();
  TestCacheLines();
  std::printf("GuestFaultReportTests: all cases passed\n");
  return 0;
}
