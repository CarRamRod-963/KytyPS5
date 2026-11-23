#include "graphics/host_gpu/renderer/cache/inPlaceReadbacks.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Config::InPlaceReadbacks;
using Libs::Graphics::InPlaceReadbacks::Decide;
using Libs::Graphics::InPlaceReadbacks::Sample;

constexpr uint64_t KiB = 1024;
constexpr uint64_t Limit = 512 * KiB;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "InPlaceReadbacksTests: failed: %s\n", text);
    std::abort();
  }
}

// Reads at `mbps` MB/s against GPU copies of `copy_us` us plus `copy_mbps`.
std::vector<Sample> Host(double mbps, double copy_us, double copy_mbps) {
  std::vector<Sample> samples;
  for (const uint64_t bytes : {4 * KiB, 16 * KiB, 64 * KiB, 256 * KiB, Limit}) {
    samples.push_back(
        {bytes, static_cast<int64_t>(static_cast<double>(bytes) * 1e3 / mbps),
         static_cast<int64_t>(copy_us * 1e3 +
                              static_cast<double>(bytes) * 1e3 / copy_mbps)});
  }
  return samples;
}

void TestModes() {
  const auto fast = Host(1500, 60, 10000);
  Check(Decide(InPlaceReadbacks::Off, fast, Limit).max_bytes == 0,
        "off never reads in place");
  const auto slow = Host(19, 60, 10000);
  Check(Decide(InPlaceReadbacks::On, slow, Limit).max_bytes == Limit,
        "on reads in place however slow");
  Check(Decide(InPlaceReadbacks::Auto, {}, Limit).max_bytes == 0,
        "auto without a measurement reads nothing in place");
}

void TestAuto() {
  // Faster than a copy at every size: the limit.
  Check(
      Decide(InPlaceReadbacks::Auto, Host(20000, 60, 10000), Limit).max_bytes ==
          Limit,
      "fast reads stay in place up to the limit");
  // 19 MB/s: 4 KiB alone takes 216 us, more than a 60 us copy: off.
  const auto slow = Decide(InPlaceReadbacks::Auto, Host(19, 60, 10000), Limit);
  Check(slow.max_bytes == 0, "19 MB/s reads are never in place");
  // 1.5 GB/s against 60 us + 10 GB/s copies: the lines cross at
  // 60 us / (1/1500 - 1/10000) us per byte... ~105.9 KB, between 64 and 256
  // KiB.
  const auto mid = Decide(InPlaceReadbacks::Auto, Host(1500, 60, 10000), Limit);
  const double exact = 60e-6 / (1.0 / 1500e6 - 1.0 / 10000e6);
  Check(mid.max_bytes > 64 * KiB && mid.max_bytes < 256 * KiB,
        "the crossing lies between the samples around it");
  Check(static_cast<double>(mid.max_bytes) > exact * 0.97 &&
            static_cast<double>(mid.max_bytes) < exact * 1.03,
        "linear costs cross where the model says");
  // A limit below the crossing caps it.
  Check(Decide(InPlaceReadbacks::Auto, Host(1500, 60, 10000), 64 * KiB)
                .max_bytes == 64 * KiB,
        "the limit caps the crossing");
}

} // namespace

int main() {
  TestModes();
  TestAuto();
  std::printf("InPlaceReadbacksTests: all passed\n");
  return 0;
}
