#include "graphics/host_gpu/commandLog.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::CommandLog::Checkpoint;
using Libs::Graphics::CommandLog::Command;
using Libs::Graphics::CommandLog::CompletedTick;
using Libs::Graphics::CommandLog::Describe;
using Libs::Graphics::CommandLog::Entry;
using Libs::Graphics::CommandLog::FlagAsyncCompute;
using Libs::Graphics::CommandLog::FlagBindless;
using Libs::Graphics::CommandLog::Format;
using Libs::Graphics::CommandLog::FormatCounterNote;
using Libs::Graphics::CommandLog::Kind;
using Libs::Graphics::CommandLog::MakeMarker;
using Libs::Graphics::CommandLog::Ring;
using Libs::Graphics::CommandLog::ShownCommands;
using Libs::Graphics::CommandLog::ShownShaders;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "CommandLogTests: failed: %s\n", text);
    std::abort();
  }
}

void CheckText(const std::string &actual, const std::string &expected,
               const char *text) {
  if (actual != expected) {
    std::fprintf(stderr, "CommandLogTests: got:\n%s\nexpected:\n%s\n",
                 actual.c_str(), expected.c_str());
  }
  Check(actual == expected, text);
}

size_t Count(const std::string &text, std::string_view part) {
  size_t count = 0;
  for (size_t at = text.find(part); at != std::string::npos;
       at = text.find(part, at + 1)) {
    count++;
  }
  return count;
}

Command Dispatch(uint64_t shader, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
  return {.kind = Kind::Dispatch, .shader0 = shader, .x = x, .y = y, .z = z};
}

void TestReportOfLostSubmission() {
  Ring ring(3, 64);
  ring.Note(5, Dispatch(0x11, 1));
  ring.Note(6, Dispatch(0xf716f0efbbc7e677, 1440));
  ring.Note(6, {.kind = Kind::DrawIndexed,
                .shader0 = 0xaa,
                .shader1 = 0xbb,
                .x = 36,
                .y = 2});
  ring.Note(6, {.kind = Kind::Internal, .name = "tiling pass"});
  ring.Note(6, {.kind = Kind::DispatchIndirect,
                .flags = FlagAsyncCompute,
                .shader0 = 0xf716f0efbbc7e677,
                .x = 0x40010,
                .y = 0x30,
                .address = 0x4300a0100});
  ring.Note(7, Dispatch(0x22, 3));
  ring.Note(8, Dispatch(0x22, 4));
  CheckText(
      Format(ring, "main", 5, 8),
      "Lost submission (main): tick 6, GPU at 5, submitted up to 8, 4 "
      "command(s) noted\n"
      "Lost submission (main) command 0: dispatch CS f716f0efbbc7e677 groups "
      "1440x1x1\n"
      "Lost submission (main) command 1: draw indexed VS 00000000000000aa PS "
      "00000000000000bb count 36 instances 2\n"
      "Lost submission (main) command 2: internal tiling pass\n"
      "Lost submission (main) command 3: dispatch indirect CS "
      "f716f0efbbc7e677 args 0x4300a0100 (guest 0x3000040010) (compute "
      "queue)\n"
      "Lost submission (main) shader: CS f716f0efbbc7e677 x 2\n"
      "Lost submission (main) shader: VS 00000000000000aa PS "
      "00000000000000bb x 1\n"
      "Lost submission (main) later submissions: 2 command(s) noted\n",
      "lost submission report");
  // A submission without noted commands, and one not submitted yet.
  CheckText(Format(ring, "present", 6, 6),
            "Lost submission (present): none, GPU at tick 6\n",
            "nothing submitted past the GPU");
  CheckText(Format(ring, "main", 8, 9),
            "Lost submission (main): tick 9, GPU at 8, submitted up to 9, 0 "
            "command(s) noted\n",
            "submission without noted commands");
}

// The facts that tell the causes of a boot-time loss apart: what an upload
// scatter read, whether a command binds the bindless table and how many slot
// writes it saw, where an indirect dispatch reads its arguments.
void TestLossFacts() {
  Ring ring(3, 64);
  ring.Note(11, {.kind = Kind::Scatter,
                 .x = 0x48d0000,
                 .y = 0x48f2000,
                 .z = 34,
                 .address = 0x500000000,
                 .table = 0x500a00100});
  ring.Note(11, {.kind = Kind::UploadCopies,
                 .x = 0x100,
                 .y = 0x200,
                 .address = 0x500000000});
  ring.Note(11, {.kind = Kind::Dispatch,
                 .flags = FlagBindless,
                 .shader0 = 0xb5710780d3d8eb7b,
                 .x = 32,
                 .y = 32,
                 .z = 16,
                 .bindless_writes = 412});
  ring.Note(11, {.kind = Kind::DrawIndexed,
                 .flags = FlagBindless | FlagAsyncCompute,
                 .shader0 = 0xaa,
                 .shader1 = 0xbb,
                 .x = 3,
                 .y = 1,
                 .bindless_writes = 413});
  ring.Note(11, {.kind = Kind::Internal, .name = "tiling pass"});
  CheckText(
      Format(ring, "main", 10, 11),
      "Lost submission (main): tick 11, GPU at 10, submitted up to 11, 5 "
      "command(s) noted\n"
      "Lost submission (main) command 0: upload scatter source 0x500000000 "
      "offsets 0x48d0000-0x48f2000 pieces 34 table 0x500a00100\n"
      "Lost submission (main) command 1: upload copies source 0x500000000 "
      "offsets 0x100-0x200\n"
      "Lost submission (main) command 2: dispatch CS b5710780d3d8eb7b groups "
      "32x32x16 bindless (slot writes 412)\n"
      "Lost submission (main) command 3: draw indexed VS 00000000000000aa PS "
      "00000000000000bb count 3 instances 1 bindless (slot writes 413) "
      "(compute queue)\n"
      "Lost submission (main) command 4: internal tiling pass\n"
      "Lost submission (main) shader: CS b5710780d3d8eb7b x 1\n"
      "Lost submission (main) shader: VS 00000000000000aa PS "
      "00000000000000bb x 1\n",
      "loss facts");
  // Each kind keeps its own field in the shared slot word.
  Entry entry;
  Check(ring.Read(1, entry) && entry.command.table == 0x500a00100 &&
            entry.command.bindless_writes == 0 &&
            entry.command.name == nullptr &&
            entry.command.address == 0x500000000,
        "scatter fields read back");
  Check(ring.Read(3, entry) && entry.command.bindless_writes == 412 &&
            entry.command.table == 0 && entry.command.name == nullptr,
        "bindless count read back");
  Check(ring.Read(5, entry) && entry.command.bindless_writes == 0 &&
            std::string_view(entry.command.name) == "tiling pass",
        "pass name read back");
  // Without the flag a command shows no bindless count.
  Check(Describe({.kind = Kind::Dispatch,
                  .shader0 = 1,
                  .x = 1,
                  .y = 1,
                  .z = 1,
                  .bindless_writes = 7})
                .find("bindless") == std::string::npos,
        "bindless count without the flag");
}

// After a lost device the timeline counter may read UINT64_MAX (or fail):
// such a value is no completed tick, and the report must still name the
// commands of a submission the GPU may not have finished.
void TestUnknownCompletedTick() {
  // Only a submitted tick is taken from the counter.
  Check(CompletedTick(true, 41, 30, 45) == 41, "counter taken");
  Check(CompletedTick(true, 45, 30, 45) == 45, "counter at the last submitted");
  Check(CompletedTick(true, UINT64_MAX, 30, 45) == 30, "UINT64_MAX ignored");
  Check(CompletedTick(true, 46, 30, 45) == 30,
        "counter past submitted ignored");
  Check(CompletedTick(false, 41, 30, 45) == 30, "failed read ignored");
  Check(CompletedTick(false, 0, 50, 45) == 45, "known value bounded");
  CheckText(FormatCounterNote("main", true, 41, 30, 45), "",
            "no note for a used counter");
  CheckText(FormatCounterNote("main", true, UINT64_MAX, 30, 45),
            "Lost submission (main): GPU counter after the loss "
            "18446744073709551615 is not a submitted tick; using the last one "
            "read before, 30\n",
            "note for UINT64_MAX");
  CheckText(FormatCounterNote("present", false, 0, 7, 9),
            "Lost submission (present): GPU counter unreadable after the "
            "loss; using the last one read before, 7\n",
            "note for a failed read");

  // The known tick is current: its successor's commands are listed.
  Ring ring(1, 8);
  ring.Note(30, Dispatch(0x30, 1));
  ring.Note(31, Dispatch(0x31, 1));
  ring.Note(31, Dispatch(0x32, 2));
  ring.Note(32, Dispatch(0x33, 3));
  const auto text =
      Format(ring, "main", CompletedTick(true, UINT64_MAX, 30, 32), 32);
  Check(text.find("Lost submission (main): tick 31, GPU at 30, submitted up "
                  "to 32, 2 command(s) noted\n") != std::string::npos &&
            text.find("command 1: dispatch CS 0000000000000032 groups 2x1x1") !=
                std::string::npos,
        "known tick used after an unreadable counter");

  // UINT64_MAX given directly: the oldest held submission is listed, not
  // "tick 0 ... 0 commands".
  const auto unknown = Format(ring, "main", UINT64_MAX, 32);
  Check(unknown.find("Lost submission (main): tick 30 (the oldest held: the "
                     "first unfinished one is not held), GPU at unknown, "
                     "submitted up to 32, 1 command(s) noted\n") !=
                std::string::npos &&
            unknown.find("command 0: dispatch CS 0000000000000030") !=
                std::string::npos &&
            unknown.find("later submissions: 3 command(s) noted\n") !=
                std::string::npos,
        "oldest held submission for an unknown tick");
  Check(unknown.find("tick 0,") == std::string::npos, "no tick 0 report");

  // A stale known tick whose successor was overwritten: the oldest held
  // submission is listed (bounded like any other).
  Ring small(1, 4);
  for (uint32_t i = 0; i < 9; i++) {
    small.Note(100 + i / 3, Dispatch(0x40 + i, i));
  }
  const auto stale = Format(small, "main", 50, 102);
  Check(
      stale.find("Lost submission (main): tick 101 (the oldest held: the "
                 "first unfinished one is not held), GPU at 50, submitted up "
                 "to 102, 1 command(s) noted (earlier ones overwritten)\n") !=
              std::string::npos &&
          stale.find("command 0: dispatch CS 0000000000000045 groups 5x1x1") !=
              std::string::npos,
      "oldest held submission for an overwritten tick");
}

void TestReadAndMarkers() {
  Ring ring(7, 4);
  Check(ring.Capacity() == 4, "capacity");
  Check(ring.LogId() == 7, "log id");
  Entry entry;
  Check(!ring.Read(0, entry), "sequence 0 read");
  Check(!ring.Read(1, entry), "read before any note");
  for (uint32_t i = 1; i <= 6; i++) {
    Check(ring.Note(i, Dispatch(i, i)) == i, "sequence numbers");
  }
  Check(ring.Count() == 6, "count");
  Check(!ring.Read(2, entry), "overwritten command read");
  Check(!ring.Read(7, entry), "future command read");
  Check(ring.Read(3, entry) && entry.tick == 3 && entry.command.x == 3 &&
            entry.command.shader0 == 3 && entry.sequence == 3,
        "held command read");
  Check(MakeMarker(7, 5) == ((uint64_t{7} << 56u) | 5u), "marker layout");
  Check(Ring(1, 5).Capacity() == 8, "capacity rounds up");
}

void TestOverwrittenStart() {
  Ring ring(1, 8);
  for (uint32_t i = 0; i < 20; i++) {
    ring.Note(4, Dispatch(0x33, i));
  }
  const auto text = Format(ring, "main", 3, 4);
  Check(text.find("tick 4, GPU at 3, submitted up to 4, 8 command(s) noted "
                  "(earlier ones overwritten)\n") != std::string::npos,
        "overwritten start reported");
  Check(text.find("command 0: dispatch CS 0000000000000033 groups 12x1x1") !=
            std::string::npos,
        "oldest held command first");
  Check(Count(text, " command ") == 8, "every held command shown");
}

void TestLongSubmission() {
  Ring ring(1, 1024);
  const uint64_t total = 3 * ShownCommands;
  for (uint64_t i = 0; i < total; i++) {
    ring.Note(10, Dispatch(0x44, static_cast<uint32_t>(i)));
  }
  const auto text = Format(ring, "main", 9, 10);
  Check(Count(text, ") command ") == 2 * ShownCommands, "first and last shown");
  Check(text.find("commands not shown: 64\n") != std::string::npos,
        "hidden commands counted");
  Check(text.find("command 63: dispatch CS 0000000000000044 groups 63x") !=
            std::string::npos,
        "last of the first part");
  Check(text.find("command 128: dispatch CS 0000000000000044 groups 128x") !=
            std::string::npos,
        "first of the last part");
  Check(text.find("command 64:") == std::string::npos, "middle hidden");
  Check(text.find("shader: CS 0000000000000044 x 192\n") != std::string::npos,
        "summary counts hidden commands");
}

void TestShaderSummaryLimit() {
  Ring ring(1, 256);
  for (uint64_t i = 0; i < ShownShaders + 8; i++) {
    ring.Note(2, Dispatch(0x1000 + i, 1));
    ring.Note(2, Dispatch(0x1000 + i, 2));
  }
  const auto text = Format(ring, "main", 1, 2);
  Check(Count(text, "shader: CS ") == ShownShaders, "distinct shaders capped");
  Check(text.find("shader: others x 16\n") != std::string::npos,
        "others counted");
}

void TestCheckpoints() {
  Ring ring(2, 16);
  ring.Note(1, Dispatch(0x55, 1));
  ring.Note(2, Dispatch(0x66, 2));
  const uint64_t third = ring.Note(2, Dispatch(0x77, 3));
  const Checkpoint checkpoints[] = {
      {"TopOfPipe", MakeMarker(2, third)},
      {"BottomOfPipe", MakeMarker(2, 1)},
      {"TopOfPipe", MakeMarker(3, third)}, // another log's marker
      {"BottomOfPipe", MakeMarker(2, 100)},
  };
  const auto text = Format(ring, "main", 1, 2, checkpoints);
  Check(text.find("Lost submission (main) checkpoint TopOfPipe: command 1 "
                  "of tick 2, dispatch CS 0000000000000077 groups 3x1x1\n") !=
            std::string::npos,
        "checkpoint in the lost submission");
  Check(text.find("Lost submission (main) checkpoint BottomOfPipe: tick 1, "
                  "dispatch CS 0000000000000055 groups 1x1x1\n") !=
            std::string::npos,
        "checkpoint of a finished submission");
  Check(text.find("checkpoint BottomOfPipe: command #100 (overwritten)\n") !=
            std::string::npos,
        "checkpoint without a held command");
  Check(Count(text, "checkpoint ") == 3, "other logs' checkpoints left out");
}

// A writer notes while a reader reads: a read either fails or returns the
// fields of one command, never a mix.
void TestConcurrentNotes() {
  Ring ring(1, 64);
  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (uint64_t i = 1; i <= 2000000; i++) {
      ring.Note(i, {.kind = Kind::Dispatch,
                    .shader0 = i * 3,
                    .shader1 = i * 5,
                    .x = static_cast<uint32_t>(i),
                    .y = static_cast<uint32_t>(i >> 32u),
                    .z = static_cast<uint32_t>(i * 7),
                    .address = i * 11,
                    .bindless_writes = i * 13});
    }
    done.store(true, std::memory_order_release);
  });
  uint64_t reads = 0;
  while (!done.load(std::memory_order_acquire)) {
    const uint64_t newest = ring.Count();
    for (uint64_t s = newest > 64 ? newest - 63 : 1; s <= newest; s++) {
      Entry entry;
      if (!ring.Read(s, entry)) {
        continue;
      }
      reads++;
      Check(entry.sequence == s && entry.tick == s &&
                entry.command.shader0 == s * 3 &&
                entry.command.shader1 == s * 5 &&
                entry.command.x == static_cast<uint32_t>(s) &&
                entry.command.z == static_cast<uint32_t>(s * 7) &&
                entry.command.address == s * 11 &&
                entry.command.bindless_writes == s * 13 &&
                entry.command.kind == Kind::Dispatch,
            "torn read");
    }
  }
  writer.join();
  Check(reads > 0, "no reads while writing");
}

} // namespace

int main() {
  TestReportOfLostSubmission();
  TestLossFacts();
  TestUnknownCompletedTick();
  TestReadAndMarkers();
  TestOverwrittenStart();
  TestLongSubmission();
  TestShaderSummaryLimit();
  TestCheckpoints();
  TestConcurrentNotes();
  std::printf("CommandLogTests: all passed\n");
  return 0;
}
