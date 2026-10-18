#include "graphics/host_gpu/renderer/pipeline/pendingDrawPolicy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::PendingDrawBlocker;
using Libs::Graphics::PendingDrawDescription;
using Libs::Graphics::PendingDrawMaySkip;
using Libs::Graphics::PendingDrawSkipBlocker;
using Libs::Graphics::PendingDrawSkipWindow;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "PendingDrawPolicyTests: failed: %s\n", text);
    std::abort();
  }
}

PendingDrawDescription PictureOnly() { return {.skip_enabled = true}; }

void TestBlockers() {
  Check(PendingDrawSkipBlocker(PictureOnly()) == PendingDrawBlocker::None,
        "a draw that only changes the picture may be skipped");
  Check(PendingDrawSkipBlocker({}) == PendingDrawBlocker::Disabled,
        "nothing is skipped with the option off");

  auto draw = PictureOnly();
  draw.compute = true;
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::Compute,
        "dispatches never skip");

  draw = PictureOnly();
  draw.writes_memory = true;
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::WritesMemory,
        "a draw whose shaders write memory waits");

  draw = PictureOnly();
  draw.query_active = true;
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::QueryActive,
        "a draw counted by a query waits");

  draw = PictureOnly();
  draw.linear_color_target = true;
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::LinearTarget,
        "a draw to a linear color target waits");

  draw = PictureOnly();
  draw.linear_depth_write = true;
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::LinearTarget,
        "a draw writing a linear depth target waits");

  draw = PictureOnly();
  draw.depth_stencil_clear = true;
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::DepthStencilClear,
        "a depth/stencil clear draw waits");

  // The option gates everything, even a draw that would be safe to skip.
  draw = {.compute = true, .writes_memory = true};
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::Disabled,
        "the option off reports Disabled before any other reason");
  draw = {.skip_enabled = true, .compute = true, .writes_memory = true};
  Check(PendingDrawSkipBlocker(draw) == PendingDrawBlocker::Compute,
        "a dispatch reports Compute whatever else it does");
}

void TestWindow() {
  using std::chrono::milliseconds;
  const auto window =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          PendingDrawSkipWindow);
  Check(PendingDrawMaySkip(PendingDrawBlocker::None, {}, window),
        "a freshly queued pipeline is skipped");
  Check(PendingDrawMaySkip(PendingDrawBlocker::None, window - milliseconds(1),
                           window),
        "skipping lasts the whole window");
  Check(!PendingDrawMaySkip(PendingDrawBlocker::None, window, window),
        "after the window, draws wait for the pipeline");
  Check(!PendingDrawMaySkip(PendingDrawBlocker::WritesMemory, {}, window),
        "a blocked draw waits even inside the window");
  Check(!PendingDrawMaySkip(PendingDrawBlocker::Disabled, {}, window),
        "nothing is skipped with the option off");
  Check(!PendingDrawMaySkip(PendingDrawBlocker::None, {}, {}),
        "an empty window never skips");
}

} // namespace

int main() {
  TestBlockers();
  TestWindow();
  std::printf("PendingDrawPolicyTests: all tests passed\n");
  return 0;
}
