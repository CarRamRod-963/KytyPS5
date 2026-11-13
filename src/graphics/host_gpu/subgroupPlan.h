#ifndef KYTY_GRAPHICS_HOST_GPU_SUBGROUP_PLAN_H_
#define KYTY_GRAPHICS_HOST_GPU_SUBGROUP_PLAN_H_

#include <cstdint>

namespace Libs::Graphics {

// The subgroup properties of the device that decide how compute shaders run.
struct SubgroupDevice {
	uint32_t default_size = 0;
	uint32_t min_size     = 0;
	uint32_t max_size     = 0;
	// subgroupSizeControl, and compute in requiredSubgroupSizeStages.
	bool compute_size_control = false;
};

// How guest compute waves map onto host subgroups.
struct SubgroupPlan {
	// The pipelines of compute shaders require a subgroup size (VK_EXT_subgroup_size_control).
	bool compute_size_control = false;
	// The host subgroup a wave64 compute shader runs in: 64 = one guest lane per invocation, 32 =
	// two guest lanes per invocation, as on a 32-lane device.
	uint32_t compute_host_subgroup_size = 32;
	// Wave64 emulation was asked for but the device cannot require 32-lane compute subgroups.
	bool emulation_unavailable = false;
};

// emulate_wave64 runs wave64 compute shaders 32 lanes wide also where the device has native 64-lane
// subgroups.
constexpr SubgroupPlan PlanSubgroups(const SubgroupDevice& device, bool emulate_wave64) {
	SubgroupPlan plan;
	plan.compute_size_control =
	    device.compute_size_control && device.min_size <= 64 && device.max_size >= 64;
	const bool native = device.default_size == 64u || plan.compute_size_control;
	if (!native) {
		plan.compute_host_subgroup_size = 32;
		return plan;
	}
	const bool can_require_32 =
	    plan.compute_size_control && device.min_size <= 32 && device.max_size >= 32;
	if (emulate_wave64 && can_require_32) {
		plan.compute_host_subgroup_size = 32;
		return plan;
	}
	plan.compute_host_subgroup_size = 64;
	plan.emulation_unavailable      = emulate_wave64;
	return plan;
}

// The subgroup size a compute pipeline requires for a shader of wave_size lanes translated for
// host_subgroup_size, or 0 for none (the driver's choice).
constexpr uint32_t RequiredComputeSubgroupSize(const SubgroupPlan&   plan,
                                               const SubgroupDevice& device, uint32_t wave_size,
                                               uint32_t host_subgroup_size) {
	if (!plan.compute_size_control) {
		return 0;
	}
	const uint32_t size = wave_size < host_subgroup_size ? wave_size : host_subgroup_size;
	return size >= device.min_size && size <= device.max_size ? size : 0u;
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_SUBGROUP_PLAN_H_ */
