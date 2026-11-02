#include "common/gpuStageCores.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Common::LogicalCpu;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "GpuStageCoresTests: failed: %s\n", message);
    std::abort();
  }
}

// `cores` physical cores, `smt` logical processors each (siblings adjacent), from logical `first` on.
void AddCores(std::vector<LogicalCpu> &cpus, int first, uint32_t first_core, uint32_t cores, int smt,
              uint8_t efficiency, uint32_t llc, uint64_t llc_bytes) {
  for (uint32_t c = 0; c < cores; c++) {
    for (int s = 0; s < smt; s++) {
      cpus.push_back({first + static_cast<int>(c) * smt + s, first_core + c, efficiency, llc, llc_bytes});
    }
  }
}

void TestRyzen9800X3D() {
  std::vector<LogicalCpu> cpus;
  AddCores(cpus, 0, 0, 8, 2, 0, 0, 96u << 20u);
  Check(Common::ChooseGpuStageCpus(cpus) == std::array<int, 4>{2, 6, 8, 4},
        "9800X3D: not the list measured before (physical cores 1, 3, 4, 2)");
}

void TestTwoCcdsPicksTheLargerCache() {
  std::vector<LogicalCpu> cpus;
  AddCores(cpus, 0, 0, 8, 2, 0, 0, 32u << 20u);   // CCD 0
  AddCores(cpus, 16, 8, 8, 2, 0, 1, 96u << 20u);  // CCD 1 with 3D V-Cache
  Check(Common::ChooseGpuStageCpus(cpus) == std::array<int, 4>{16, 20, 22, 18},
        "two CCDs: the stages are not on the V-Cache CCD");
  std::vector<LogicalCpu> same;
  AddCores(same, 0, 0, 8, 2, 0, 0, 32u << 20u);
  AddCores(same, 16, 8, 8, 2, 0, 1, 32u << 20u);
  Check(Common::ChooseGpuStageCpus(same) == std::array<int, 4>{2, 6, 8, 4},
        "two equal CCDs: not the first one (core 0 left out)");
}

void TestIntelHybridUsesPerformanceCores() {
  std::vector<LogicalCpu> cpus;
  AddCores(cpus, 0, 0, 8, 2, 1, 0, 30u << 20u);   // 8 performance cores with SMT
  AddCores(cpus, 16, 8, 8, 1, 0, 0, 30u << 20u);  // 8 efficient cores
  Check(Common::ChooseGpuStageCpus(cpus) == std::array<int, 4>{2, 6, 8, 4},
        "hybrid: the stages are not on performance cores");
  std::vector<LogicalCpu> few;
  AddCores(few, 0, 0, 2, 2, 1, 0, 12u << 20u);    // 2 performance cores
  AddCores(few, 4, 2, 8, 1, 0, 0, 12u << 20u);    // 8 efficient cores
  Check(Common::ChooseGpuStageCpus(few) == std::array<int, 4>{-1, -1, -1, -1},
        "hybrid with two performance cores: pinned to fewer than four of them");
}

void TestSmallProcessors() {
  std::vector<LogicalCpu> six;
  AddCores(six, 0, 0, 6, 1, 0, 0, 16u << 20u);
  Check(Common::ChooseGpuStageCpus(six) == std::array<int, 4>{1, 3, 4, 2}, "6 cores without SMT: core 0 not left out");
  std::vector<LogicalCpu> four;
  AddCores(four, 0, 0, 4, 2, 0, 0, 8u << 20u);
  Check(Common::ChooseGpuStageCpus(four) == std::array<int, 4>{0, 4, 6, 2},
        "4 cores: core 0 must be used when only four exist");
  std::vector<LogicalCpu> three;
  AddCores(three, 0, 0, 3, 2, 0, 0, 8u << 20u);
  Check(Common::ChooseGpuStageCpus(three) == std::array<int, 4>{-1, -1, -1, -1}, "3 cores: pinned");
  Check(Common::ChooseGpuStageCpus({}) == std::array<int, 4>{-1, -1, -1, -1}, "no information: pinned");
}

} // namespace

int main() {
  TestRyzen9800X3D();
  TestTwoCcdsPicksTheLargerCache();
  TestIntelHybridUsesPerformanceCores();
  TestSmallProcessors();
  std::printf("GpuStageCoresTests: ok\n");
  return 0;
}
