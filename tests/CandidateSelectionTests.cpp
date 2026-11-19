#include "graphics/host_gpu/renderer/pipeline/candidateSelection.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::CandidateSelection;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "CandidateSelectionTests: failed: %s\n", text);
    std::abort();
  }
}

constexpr uint32_t Quiet = CandidateSelection::QuietFrames;

// One frame of a table binding two candidates; `select` picks which ones the
// shader selected (bit 0: a, bit 1: b, bit 2: the root's own ordinal), then the
// frame is read.
struct Fixture {
  CandidateSelection selection;
  uint64_t frame = 1;

  void Frame(CandidateSelection::Table &table, CandidateSelection::Candidate &a,
             CandidateSelection::Candidate &b, uint32_t select) {
    if (select & 1u)
      CandidateSelection::Selected(table, &a, frame);
    if (select & 2u)
      CandidateSelection::Selected(table, &b, frame);
    if (select & 4u)
      CandidateSelection::Selected(table, nullptr, frame);
    CandidateSelection::Observed(table, nullptr, frame);
    CandidateSelection::Observed(table, &a, frame);
    CandidateSelection::Observed(table, &b, frame);
    ++frame;
  }
};

void TestQuietCandidate() {
  Fixture f;
  auto &table = f.selection.FindTable(1, f.frame);
  auto &a = CandidateSelection::FindCandidate(table, 10, f.frame);
  auto &b = CandidateSelection::FindCandidate(table, 11, f.frame);
  for (uint32_t i = 0; i + 1 < Quiet; i++)
    f.Frame(table, a, b, 1u);
  Check(!CandidateSelection::Demotes(table, b, false),
        "a candidate is demoted before QuietFrames observed frames");
  f.Frame(table, a, b, 1u);
  Check(CandidateSelection::Demotes(table, b, false),
        "a candidate unselected for QuietFrames frames stays bound");
  Check(!CandidateSelection::Demotes(table, a, false),
        "the selected candidate is demoted");
  Check(!CandidateSelection::Demotes(table, b, true),
        "a candidate of a written table is demoted");
  // Selected again while bound: quiet again from scratch, no miss.
  f.Frame(table, a, b, 2u);
  Check(!CandidateSelection::Demotes(table, b, false),
        "a reselected candidate stays demoted");
  Check(f.selection.Misses() == 0,
        "a selection of a bound candidate counts as a miss");
}

void TestUnobservedFramesDoNotCount() {
  Fixture f;
  auto &table = f.selection.FindTable(1, f.frame);
  auto &a = CandidateSelection::FindCandidate(table, 10, f.frame);
  auto &b = CandidateSelection::FindCandidate(table, 11, f.frame);
  f.Frame(table, a, b, 1u);
  // Frames pass without the program running: nothing is observed.
  f.frame += 10 * Quiet;
  Check(!CandidateSelection::Demotes(table, b, false),
        "frames in which the candidate was not bound made it quiet");
  // Observing a candidate twice in one frame counts once.
  CandidateSelection::Observed(table, &b, f.frame);
  CandidateSelection::Observed(table, &b, f.frame);
  Check(b.quiet == 2u, "a frame with two bindings counted twice");
}

void TestIdleTable() {
  Fixture f;
  auto &table = f.selection.FindTable(2, f.frame);
  auto &a = CandidateSelection::FindCandidate(table, 10, f.frame);
  auto &b = CandidateSelection::FindCandidate(table, 11, f.frame);
  for (uint32_t i = 0; i < Quiet; i++)
    f.Frame(table, a, b, 0u);
  // A table that selected nothing for QuietFrames frames demotes a new
  // candidate at once.
  auto &fresh = CandidateSelection::FindCandidate(table, 12, f.frame);
  Check(fresh.quiet == 0 && CandidateSelection::Demotes(table, fresh, false),
        "a new candidate of an idle table is bound");
  // The root's own ordinal (a key outside the table) keeps the table busy.
  auto &busy = f.selection.FindTable(3, f.frame);
  auto &c = CandidateSelection::FindCandidate(busy, 10, f.frame);
  auto &d = CandidateSelection::FindCandidate(busy, 11, f.frame);
  for (uint32_t i = 0; i < Quiet; i++)
    f.Frame(busy, c, d, 4u);
  auto &other = CandidateSelection::FindCandidate(busy, 12, f.frame);
  Check(!CandidateSelection::Demotes(busy, other, false),
        "a table selecting its root ordinal counts as idle");
}

void TestMissMakesTableEager() {
  Fixture f;
  auto &table = f.selection.FindTable(4, f.frame);
  auto &a = CandidateSelection::FindCandidate(table, 10, f.frame);
  auto &b = CandidateSelection::FindCandidate(table, 11, f.frame);
  for (uint32_t i = 0; i < Quiet; i++)
    f.Frame(table, a, b, 1u);
  Check(CandidateSelection::Demotes(table, b, false), "setup: b not demoted");
  Check(f.selection.Missed(table),
        "the first miss did not turn the table eager");
  Check(!f.selection.Missed(table), "a second miss reported a new eager table");
  Check(f.selection.EagerTables() == 1 && f.selection.Misses() == 2,
        "miss counters");
  for (uint32_t i = 0; i < 4 * Quiet; i++)
    f.Frame(table, a, b, 0u);
  Check(!CandidateSelection::Demotes(table, b, false) &&
            !CandidateSelection::Demotes(table, a, false),
        "an eager table demotes again");
}

void TestForget() {
  Fixture f;
  auto &table = f.selection.FindTable(5, 1);
  CandidateSelection::FindCandidate(table, 10, 1);
  CandidateSelection::FindCandidate(table, 11, 900);
  f.selection.Forget(1 + CandidateSelection::ForgetFrames);
  Check(table.candidates.size() == 2,
        "a candidate bound ForgetFrames ago was forgotten early");
  f.selection.Forget(2 + CandidateSelection::ForgetFrames);
  Check(table.candidates.size() == 1 && table.candidates.contains(11),
        "an old candidate was kept or a recent one forgotten");
}

} // namespace

int main() {
  TestQuietCandidate();
  TestUnobservedFramesDoNotCount();
  TestIdleTable();
  TestMissMakesTableEager();
  TestForget();
  std::printf("CandidateSelectionTests: ok\n");
  return 0;
}
