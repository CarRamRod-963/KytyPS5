#include "graphics/host_gpu/subgroupPlan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

using Libs::Graphics::PlanSubgroups;
using Libs::Graphics::RequiredComputeSubgroupSize;
using Libs::Graphics::SubgroupDevice;
using Libs::Graphics::SubgroupPlan;
using Libs::Graphics::UseMaximalReconvergence;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "SubgroupPlanTests: failed: %s\n", text);
    std::abort();
  }
}

// The size a compute pipeline of a wave_size shader requires on a device
// planned with emulate_wave64; the shader is translated for the plan's host
// subgroup.
uint32_t Required(const SubgroupDevice &device, bool emulate_wave64,
                  uint32_t wave_size) {
  const auto plan = PlanSubgroups(device, emulate_wave64);
  return RequiredComputeSubgroupSize(plan, device, wave_size,
                                     plan.compute_host_subgroup_size);
}

// AMD RDNA: wave64 by default, 32 or 64 lanes on request.
void TestNativeWave64Device() {
  const SubgroupDevice amd{.default_size = 64,
                           .min_size = 32,
                           .max_size = 64,
                           .compute_size_control = true};

  const auto native = PlanSubgroups(amd, false);
  Check(native.compute_size_control,
        "a 32-64 device does not control compute subgroups");
  Check(native.compute_host_subgroup_size == 64,
        "wave64 does not run natively by default");
  Check(!native.emulation_unavailable,
        "native wave64 reports a failed emulation");
  Check(Required(amd, false, 64) == 64,
        "a native wave64 shader does not require 64 lanes");
  Check(Required(amd, false, 32) == 32,
        "a wave32 shader does not require 32 lanes");

  const auto emulated = PlanSubgroups(amd, true);
  Check(emulated.compute_size_control,
        "emulation turns the subgroup size control off");
  Check(emulated.compute_host_subgroup_size == 32,
        "emulated wave64 is not translated for 32-lane subgroups");
  Check(!emulated.emulation_unavailable,
        "an available emulation is reported unavailable");
  // Two guest lanes per invocation need 32-lane subgroups: a 64-lane subgroup
  // would hold two guest waves.
  Check(Required(amd, true, 64) == 32,
        "an emulated wave64 shader does not require 32 lanes");
  Check(Required(amd, true, 32) == 32,
        "a wave32 shader does not require 32 lanes when emulating");
}

// NVIDIA: 32 lanes only. Nothing changes with the option: no pipeline requires
// a size.
void TestWave32Device() {
  const SubgroupDevice nvidia{.default_size = 32,
                              .min_size = 32,
                              .max_size = 32,
                              .compute_size_control = true};
  for (const bool emulate : {false, true}) {
    const auto plan = PlanSubgroups(nvidia, emulate);
    Check(!plan.compute_size_control,
          "a 32-lane device controls compute subgroups");
    Check(plan.compute_host_subgroup_size == 32,
          "a 32-lane device runs wave64 natively");
    Check(!plan.emulation_unavailable,
          "a 32-lane device reports a failed emulation");
    Check(Required(nvidia, emulate, 64) == 0,
          "a 32-lane device requires a subgroup size");
    Check(Required(nvidia, emulate, 32) == 0,
          "a 32-lane device requires a subgroup size");
  }
}

// A 64-lane device without size control cannot run 32-lane compute subgroups.
void TestUncontrolledWave64Device() {
  const SubgroupDevice device{.default_size = 64,
                              .min_size = 64,
                              .max_size = 64,
                              .compute_size_control = false};
  const auto plan = PlanSubgroups(device, true);
  Check(plan.compute_host_subgroup_size == 64,
        "emulation ignores a fixed 64-lane device");
  Check(plan.emulation_unavailable, "an impossible emulation is not reported");
  Check(Required(device, true, 64) == 0,
        "an uncontrolled device requires a subgroup size");

  // Size control limited to 64 lanes: wave64 stays native.
  const SubgroupDevice fixed64{.default_size = 64,
                               .min_size = 64,
                               .max_size = 64,
                               .compute_size_control = true};
  const auto fixed = PlanSubgroups(fixed64, true);
  Check(fixed.compute_host_subgroup_size == 64 && fixed.emulation_unavailable,
        "a 64-only device takes the emulation");
  Check(Required(fixed64, true, 64) == 64,
        "a 64-only device does not require 64 lanes");
}

// Compute size control without 64 lanes (8-32 lane devices) stays off, as
// before the option.
void TestNarrowDevice() {
  const SubgroupDevice intel{.default_size = 32,
                             .min_size = 8,
                             .max_size = 32,
                             .compute_size_control = true};
  const auto plan = PlanSubgroups(intel, true);
  Check(!plan.compute_size_control,
        "an 8-32 lane device controls compute subgroups");
  Check(plan.compute_host_subgroup_size == 32,
        "an 8-32 lane device runs wave64 natively");
  Check(!plan.emulation_unavailable,
        "an 8-32 lane device reports a failed emulation");
  Check(Required(intel, true, 64) == 0,
        "an 8-32 lane device requires a subgroup size");
}

// Maximal reconvergence needs both device features and is off when disabled.
void TestMaximalReconvergence() {
  Check(UseMaximalReconvergence(true, true, false),
        "a supporting device does not request maximal reconvergence");
  Check(!UseMaximalReconvergence(true, true, true),
        "disabled maximal reconvergence is still requested");
  Check(!UseMaximalReconvergence(false, true, false),
        "maximal reconvergence without uniform control flow is requested");
  Check(!UseMaximalReconvergence(true, false, false),
        "maximal reconvergence without device support is requested");
}

} // namespace

int main() {
  TestNativeWave64Device();
  TestWave32Device();
  TestUncontrolledWave64Device();
  TestNarrowDevice();
  TestMaximalReconvergence();
  std::printf("SubgroupPlanTests: ok\n");
  return 0;
}
