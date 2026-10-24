#include "graphics/host_gpu/renderer/pendingLabels.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::PendingLabels;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "PendingLabelsTests: failed: %s\n", message);
    std::abort();
  }
}

auto Equals(uint64_t expected) {
  return [expected](uint64_t value) { return value == expected; };
}

void TestElideAndStore() {
  PendingLabels labels;
  Check(!labels.HasElided() && labels.ElidedTick(0, UINT64_MAX) == 0,
        "empty list elided");
  const auto id = labels.Add(0x1000, 8, 5);
  labels.SetTick(id, 7);
  Check(!labels.Elide(0x1000, 8, Equals(4)),
        "elided for a value the label does not store");
  Check(!labels.HasElided(), "a refused wait counted as elided");
  Check(!labels.Elide(0x2000, 8, Equals(5)), "elided a wait on other memory");
  Check(labels.Elide(0x1000, 8, Equals(5)), "a satisfied wait was not elided");
  Check(labels.HasElided(), "the elided label is not reported");
  Check(labels.ElidedTick(0x1004, 1) == 7,
        "an access inside the label missed its tick");
  Check(labels.ElidedTick(0x1008, 8) == 0,
        "an access after the label found it");
  Check(labels.ElidedTick(0x0ff8, 8) == 0,
        "an access before the label found it");
  Check(labels.Elide(0x1000, 8, Equals(5)),
        "a second wait on the label was not elided");
  labels.Stored(id);
  Check(!labels.HasElided() && labels.ElidedTick(0x1000, 8) == 0 &&
            labels.Size() == 0,
        "a stored label stays elided");
}

void TestPartialAndNewest() {
  PendingLabels labels;
  const auto low = labels.Add(0x1000, 8, 0x0000000200000001ull);
  labels.SetTick(low, 3);
  // The upper half of a 64-bit label, read as a 32-bit wait.
  Check(labels.Elide(0x1004, 4, Equals(2)),
        "a wait on the upper half was not elided");
  Check(!labels.Elide(0x1004, 8, Equals(2)),
        "a wait crossing the label's end was elided");
  // A later label of the same address wins: it is stored last.
  const auto high = labels.Add(0x1000, 4, 9);
  labels.SetTick(high, 4);
  Check(!labels.Elide(0x1000, 4, Equals(1)),
        "an overwritten value satisfied the wait");
  Check(labels.Elide(0x1000, 4, Equals(9)),
        "the newest value did not satisfy the wait");
  Check(labels.ElidedTick(0x1000, 4) == 4,
        "the elided tick is not the newest label's");
  labels.Stored(low);
  labels.Stored(high);
  Check(!labels.HasElided() && labels.Size() == 0,
        "labels left after both were stored");
}

void TestStoreCompleted() {
  PendingLabels labels;
  const auto first = labels.Add(0x1000, 4, 1);
  const auto second = labels.Add(0x1000, 4, 2);
  const auto later = labels.Add(0x1000, 4, 3);
  labels.SetTick(first, 5);
  labels.SetTick(second, 6);
  labels.SetTick(later, 9);
  Check(labels.Elide(0x1000, 4, Equals(3)),
        "the newest label did not satisfy the wait");
  std::vector<uint64_t> stored;
  labels.StoreCompleted(0x1000, 4, 6, [&](const PendingLabels::Label &label) {
    stored.push_back(label.value);
  });
  Check(stored == std::vector<uint64_t>{1, 2},
        "completed labels not stored in order");
  Check(!labels.ShouldStore(first) && !labels.ShouldStore(second),
        "the regular store would write a stored label again");
  Check(labels.ShouldStore(later) && labels.HasElided(),
        "a label of an incomplete submission was stored");
  labels.StoreCompleted(0x1000, 4, 9, [&](const PendingLabels::Label &label) {
    stored.push_back(label.value);
  });
  Check(stored == std::vector<uint64_t>{1, 2, 3} && !labels.HasElided(),
        "the elided label was not stored once completed");
  labels.Stored(first);
  labels.Stored(second);
  labels.Stored(later);
  Check(labels.Size() == 0 && labels.ShouldStore(first),
        "stored labels left in the list");
}

void TestPatchElided() {
  PendingLabels labels;
  const auto first = labels.Add(0x1004, 4, 0x11111111u);
  const auto second = labels.Add(0x1008, 8, 0x3333333322222222ull);
  const auto other = labels.Add(0x1010, 4, 0x44444444u);
  labels.SetTick(first, 1);
  labels.SetTick(second, 2);
  labels.SetTick(other, 3);
  uint32_t words[5] = {0, 0, 0, 0, 0};
  labels.PatchElided(0x1000, sizeof(words), words);
  Check(words[1] == 0 && words[2] == 0,
        "a label whose wait was not elided was overlaid");
  Check(labels.Elide(0x1004, 4, Equals(0x11111111u)) &&
            labels.Elide(0x100c, 4, Equals(0x33333333u)),
        "satisfied waits were not elided");
  labels.PatchElided(0x1000, sizeof(words), words);
  Check(words[0] == 0 && words[1] == 0x11111111u && words[2] == 0x22222222u &&
            words[3] == 0x33333333u && words[4] == 0,
        "elided labels were not overlaid exactly onto their bytes");
  uint8_t byte = 0;
  labels.PatchElided(0x100b, 1, &byte);
  Check(byte == 0x22, "a read inside a label got the wrong byte");
}

void TestUnknownValue() {
  PendingLabels labels;
  // A timestamp: its value is known only when its submission ends.
  const auto id = labels.Add(0x1000, 8, 5, false);
  labels.SetTick(id, 2);
  Check(!labels.Elide(0x1000, 8, Equals(5)) && !labels.HasElided(),
        "a wait was elided on a label whose value is not known yet");
  labels.SetValue(id, 9);
  Check(labels.Elide(0x1000, 8, Equals(9)),
        "a wait on a label with its final value was not elided");
  uint64_t stored = 0;
  labels.StoreCompleted(
      0x1000, 8, 2,
      [&stored](const PendingLabels::Label &label) { stored = label.value; });
  Check(stored == 9, "a settling access stored the provisional value");
}

} // namespace

int main() {
  TestElideAndStore();
  TestPartialAndNewest();
  TestStoreCompleted();
  TestPatchElided();
  TestUnknownValue();
  std::printf("PendingLabelsTests: ok\n");
  return 0;
}
