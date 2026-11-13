#include "loader/guestFaultReport.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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

} // namespace

int main() {
  TestFullReport();
  TestMissingParts();
  TestTruncation();
  std::printf("GuestFaultReportTests: all cases passed\n");
  return 0;
}
