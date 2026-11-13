// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_
#define KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_

#include "common/common.h"

#include <span>
#include <utility>
#include <vector>

namespace Loader {

struct GuestInstructionHostFeatures {
	bool sse4a = false;
	bool rdpid = false;
	bool clwb  = false;
};

GuestInstructionHostFeatures GetGuestInstructionHostFeatures();

struct InstructionPatchCounts {
	uint64_t found   = 0;
	uint64_t native  = 0;
	uint64_t trapped = 0;

	uint64_t Skipped() const { return found - native - trapped; }

	InstructionPatchCounts& operator+=(const InstructionPatchCounts& other) {
		found += other.found;
		native += other.native;
		trapped += other.trapped;
		return *this;
	}
};

struct GuestInstructionPatchResult {
	uint64_t               function_count                           = 0;
	uint64_t               instruction_count                        = 0;
	uint64_t               red_zone_function_count                  = 0;
	uint64_t               memory_instruction_count                 = 0;
	uint64_t               short_memory_instruction_count           = 0;
	uint64_t               patched_memory_instruction_count         = 0;
	uint64_t               stack_dependent_memory_instruction_count = 0;
	uint64_t               control_flow_memory_instruction_count    = 0;
	uint64_t               unrelocatable_memory_instruction_count   = 0;
	uint64_t               indirect_red_zone_function_count         = 0;
	uint64_t               discovered_function_count                = 0;
	uint64_t               red_zone_copy_candidate_count            = 0;
	uint64_t               rewritten_function_count                 = 0;
	uint64_t               rewritten_function_bytes                 = 0;
	uint64_t               partially_rewritten_function_count       = 0;
	uint64_t               unprotected_instruction_count            = 0;
	uint64_t               copy_entry_conflict_count                = 0;
	uint64_t               copy_unsupported_count                   = 0;
	uint64_t               copy_no_space_count                      = 0;
	InstructionPatchCounts reciprocal_sqrt;
	InstructionPatchCounts extrq;
	InstructionPatchCounts insertq;
	InstructionPatchCounts rdpid;
	InstructionPatchCounts clwb;
	InstructionPatchCounts rdtsc;
};

void RegisterGuestInstructionPatchModule(void* module_ptr, uint64_t module_size,
                                         void* trampoline_area_ptr, uint64_t trampoline_area_size);
void UnregisterGuestInstructionPatchModule(void* module_ptr);

// Apply enabled instruction fixes using native trampolines or safe trap fallbacks. A nonzero
// tsc_multiplier rewrites RDTSC/RDTSCP to return host TSC * tsc_multiplier >> 32.
// copy_red_zone_functions runs every function that keeps live data in its SysV red zone while it
// accesses memory from a copy that keeps RSP below the red zone (Windows writes exception frames
// directly below RSP); function_starts are completed by a sweep of the segment. protect_memory
// additionally patches single instructions of the functions that could not be copied. data_ranges
// (address, size) are readable module ranges that may hold jump tables.
GuestInstructionPatchResult PatchGuestInstructions(
    uint64_t segment_addr, uint64_t segment_size, std::span<const uintptr_t> function_starts,
    bool protect_memory, bool emulate_amd,
    GuestInstructionHostFeatures host_features = GetGuestInstructionHostFeatures(),
    uint64_t tsc_multiplier = 0, bool copy_red_zone_functions = false,
    std::span<const std::pair<uint64_t, uint64_t>> data_ranges = {});

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_ */
