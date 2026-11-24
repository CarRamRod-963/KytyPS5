#include "loader/guestFaultReport.h"

#include <algorithm>
#include <array>
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
  for (uint64_t i = 4; i <= 20; i++) {
    ring.Record({.address = i, .pc = 0x100 + i, .rsp = 0, .time_ns = i * 1000});
  }
  Check(ring.Count() == 20, "every fault is counted");
  Check(ring.Size() == Loader::HandledFaultRing::Capacity, "the ring keeps 8");
  for (size_t i = 0; i < ring.Size(); i++) {
    Check(ring.At(i).address == 13 + i, "the last 8 faults, oldest first");
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
  for (uint64_t i = 0; i < 10; i++) {
    ring.Record({.address = i == 8 ? 0 : 0x30cbf00000ull + i * 0x1000,
                 .pc = i == 9 ? 0x904000123ull : 0x900e7c000ull + i,
                 .rsp = i == 9 ? 0x7ec1c3fc0ull : 0x7ec1c4040ull,
                 .time_ns = 1'000'000 + i * 10'000,
                 .kind = i == 0   ? Kind::Read
                         : i == 8 ? Kind::EmulatedInstruction
                                  : Kind::Write});
  }
  Loader::GuestFaultMemoryReport report;
  report.info = &info;
  report.handled = &ring;
  report.modules = &modules;
  report.now_ns = 1'500'000;
  Loader::GuestFaultMemoryText out;
  Loader::FormatGuestFaultMemory(report, &out);
  const std::string text(out.View());
  Check(Contains(text, "--- Faults this thread had resolved before (newest "
                       "last, 10 in total) ---\n"
                       "  480 us before: write 0x00000030cbf02000 "
                       "pc=0x0000000900e7c002 (eboot.bin+0xe7c002) "
                       "rsp=0x00000007ec1c4040 (fault rsp+0x40)\n"),
        "the oldest kept fault first, with its age and module offset");
  Check(Contains(text, "  420 us before: emulated instruction "
                       "pc=0x0000000900e7c008 (eboot.bin+0xe7c008) "),
        "an emulated instruction has no data address");
  Check(text.ends_with("  410 us before: write 0x00000030cbf09000 "
                       "pc=0x0000000904000123 (eboot.bin patch area+0x123) "
                       "rsp=0x00000007ec1c3fc0 (fault rsp-0x40)\n"),
        "the newest fault last; a pc in the patch area is named");
  Check(!Contains(text, "read 0x"), "faults older than the ring are gone");
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
  ring.Record(Path::BufferReadback, 0x30c0002000ull, 0x10, nullptr, 9, 300);
  Check(ring.Snapshot(few.data(), few.size()) == 3, "three stores kept");
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
  std::array<Loader::GuestFaultInterest, 8> interest{};
  const auto interest_count = Loader::CollectGuestFaultInterest(
      info.access_violation_vaddr, registers, {}, interest);
  Check(interest_count == 3 && interest[0].begin == 0x3204420000ull &&
            interest[0].end == 0x3204430000ull &&
            interest[1].begin == 0x3300000000ull &&
            interest[2].begin == 0x3300000010ull &&
            interest[2].end == 0x3300000010ull + PointerBytes,
        "interest: the fault's granule, the register's granule and dump");
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
  Loader::HandledFaultRing ring;
  ring.Record({.address = 0x3204420158ull,
               .pc = 0x900ff8030ull,
               .rsp = 0,
               .time_ns = 10,
               .kind = Loader::HandledFaultKind::Read,
               .stores = 42});
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
                 "fault (2 of the last 4; 44 recorded) ---\n"
                 "  #41 400 us before: metadata clear publish "
                 "0x0000003204420000 size 0x1000 first ffffffffffffffff "
                 "thread 5\n"
                 "  #43 200 us before: buffer readback 0x000000330000f000 "
                 "size 0x2000 first 0000000000000002 thread 7\n"
                 "--- The emulator's last 4 stores into guest memory ---\n"
                 "  #41 "),
        "stores near the fault, then the last ones");
  Check(text.ends_with("  #44 100 us before: label 0x0000005000001000 size 0x8 "
                       "first 0000000000000003 thread 6\n"),
        "the newest store ends the report");
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
  TestHandledFaultRing();
  TestHandledFaultReport();
  TestMemoryReportFits();
  TestCopyLocation();
  TestGuestStoreRing();
  TestStoreReport();
  TestTwoLevelIndirect();
  std::printf("GuestFaultReportTests: all cases passed\n");
  return 0;
}
