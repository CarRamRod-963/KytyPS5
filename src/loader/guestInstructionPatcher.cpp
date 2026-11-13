// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "loader/guestInstructionPatcher.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/virtualMemory.h"

#include <Zydis/Zydis.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <climits>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

using namespace Xbyak::util;

namespace Loader {

using u8  = uint8_t;
using s8  = int8_t;
using u16 = uint16_t;
using s16 = int16_t;
using u32 = uint32_t;
using s32 = int32_t;
using u64 = uint64_t;
using s64 = int64_t;

#define ASSERT(condition) EXIT_IF(!(condition))

constexpr size_t NearJumpSize = 5;

struct PatchModule {
	std::mutex           mutex {};
	u8*                  start = nullptr;
	u8*                  end   = nullptr;
	std::set<u8*>        patched;
	Xbyak::CodeGenerator patch_gen;
	Xbyak::CodeGenerator trampoline_gen;
	bool                 trampoline_exhaustion_reported = false;

	PatchModule(u8* module_ptr, u64 module_size, u8* trampoline_ptr, u64 trampoline_size)
	    : start(module_ptr), end(module_ptr + module_size), patch_gen(module_size, module_ptr),
	      trampoline_gen(trampoline_size, trampoline_ptr) {}
};

static std::map<u64, PatchModule> g_patch_modules;

static PatchModule* GetContainingModule(const void* ptr) {
	auto upper = g_patch_modules.upper_bound(reinterpret_cast<u64>(ptr));
	if (upper == g_patch_modules.begin()) {
		return nullptr;
	}
	auto* module  = &std::prev(upper)->second;
	auto* address = static_cast<const u8*>(ptr);
	return address >= module->start && address < module->end ? module : nullptr;
}

static bool HandleTrampolineError(PatchModule* module, int error) {
	if (error != Xbyak::ERR_CODE_IS_TOO_BIG) {
		return false;
	}
	if (!module->trampoline_exhaustion_reported) {
		LOGF("Guest instruction trampoline space exhausted for module %p\n",
		     static_cast<void*>(module->start));
		module->trampoline_exhaustion_reported = true;
	}
	return true;
}

static ZydisDecoder& GetDecoder() {
	static ZydisDecoder decoder = [] {
		ZydisDecoder value {};
		EXIT_IF(!ZYAN_SUCCESS(
		    ZydisDecoderInit(&value, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64)));
		return value;
	}();
	return decoder;
}

namespace {

constexpr size_t GuestRedZoneSize = 128;
constexpr size_t ShortJumpSize    = 2;
using RedZoneMask                 = std::bitset<GuestRedZoneSize>;

struct DecodedCodeInstruction {
	uintptr_t                                                address {};
	ZydisDecodedInstruction                                  instruction {};
	std::array<ZydisDecodedOperand, ZYDIS_MAX_OPERAND_COUNT> operands {};
	RedZoneMask                                              red_zone_use {};
	RedZoneMask                                              red_zone_def {};
	RedZoneMask                                              red_zone_live {};
	bool                                                     accesses_memory {};
	bool                                                     uses_stack_pointer {};
	bool                                                     has_red_zone_operand {};
	bool                                                     has_unmodeled_red_zone_operand {};
	bool                                                     changes_stack_pointer {};
	bool                                                     replaces_stack_pointer {};
	std::optional<s64>                                       stack_pointer_delta;
};

// A resolved `movsxd target, [table + index * 4]; add target, table; jmp target` dispatch.
struct JumpTable {
	uintptr_t              load {};    // the MOVSXD
	uintptr_t              address {}; // table base: entries are offsets from it
	std::vector<uintptr_t> entries;    // target of each index
};

// Readable module memory outside the code segment (jump tables live in read-only data).
using AddressRanges = std::span<const std::pair<uint64_t, uint64_t>>;

struct DecodedFunction {
	std::map<uintptr_t, DecodedCodeInstruction> instructions;
	std::map<uintptr_t, JumpTable>              jump_tables; // by branch address
	std::set<uintptr_t>                         branch_targets;
	bool                                        uses_red_zone {};
	bool                                        has_indirect_branch {};
	bool                                        requires_conservative_red_zone_tracking {};
	bool                                        red_zone_liveness_exact {};
};

enum class InstructionReplacement {
	None,
	ReciprocalSquareRoot,
	ExtractQ,
	InsertQ,
	ReadProcessorId,
	CacheLineWriteBack,
	ReadTimeStampCounter
};

struct InstructionRewrite {
	bool                   protect_red_zone {};
	bool                   protected_indirect_call {};
	InstructionReplacement replacement {};
};

InstructionPatchCounts& ReplacementCounts(GuestInstructionPatchResult& result,
                                          InstructionReplacement       replacement) {
	ASSERT(replacement != InstructionReplacement::None);
	switch (replacement) {
		case InstructionReplacement::ReciprocalSquareRoot: return result.reciprocal_sqrt;
		case InstructionReplacement::ExtractQ: return result.extrq;
		case InstructionReplacement::InsertQ: return result.insertq;
		case InstructionReplacement::ReadProcessorId: return result.rdpid;
		case InstructionReplacement::ReadTimeStampCounter: return result.rdtsc;
		default: return result.clwb;
	}
}

bool UsesInstructionTrap(InstructionReplacement replacement, bool trap_replacements) {
	// RDTSC does not fault, so a trap fallback would keep the host rate.
	return replacement != InstructionReplacement::None &&
	       replacement != InstructionReplacement::ReadTimeStampCounter &&
	       (trap_replacements || replacement == InstructionReplacement::InsertQ ||
	        replacement == InstructionReplacement::ReadProcessorId);
}

bool IsStackPointerRegister(ZydisRegister reg) {
	return reg != ZYDIS_REGISTER_NONE &&
	       ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg) == ZYDIS_REGISTER_RSP;
}

bool IsControlFlowTerminator(const ZydisDecodedInstruction& instruction) {
	return instruction.meta.category == ZYDIS_CATEGORY_UNCOND_BR ||
	       instruction.meta.category == ZYDIS_CATEGORY_RET ||
	       instruction.meta.category == ZYDIS_CATEGORY_INTERRUPT ||
	       instruction.meta.category == ZYDIS_CATEGORY_SYSRET ||
	       instruction.mnemonic == ZYDIS_MNEMONIC_UD2;
}

uintptr_t GetRelativeTarget(const DecodedCodeInstruction& decoded) {
	for (u8 index = 0; index < decoded.instruction.operand_count_visible; ++index) {
		const auto& operand = decoded.operands[index];
		if (operand.type != ZYDIS_OPERAND_TYPE_IMMEDIATE || !operand.imm.is_relative) {
			continue;
		}

		ZyanU64 target {};
		if (ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&decoded.instruction, &operand, decoded.address,
		                                          &target))) {
			return target;
		}
	}
	return 0;
}

DecodedCodeInstruction DecodeCodeInstruction(uintptr_t address, uintptr_t end) {
	DecodedCodeInstruction decoded {.address = address};
	const auto status = ZydisDecoderDecodeFull(&GetDecoder(), reinterpret_cast<void*>(address),
	                                           end - address, &decoded.instruction,
	                                           decoded.operands.data());
	if (!ZYAN_SUCCESS(status)) {
		decoded.instruction.length = 0;
		return decoded;
	}

	for (u8 index = 0; index < decoded.instruction.operand_count; ++index) {
		const auto& operand = decoded.operands[index];
		if (operand.type == ZYDIS_OPERAND_TYPE_REGISTER) {
			decoded.uses_stack_pointer |= IsStackPointerRegister(operand.reg.value);
			if (IsStackPointerRegister(operand.reg.value) &&
			    (operand.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0 &&
			    decoded.instruction.meta.category != ZYDIS_CATEGORY_CALL &&
			    decoded.instruction.meta.category != ZYDIS_CATEGORY_RET) {
				decoded.changes_stack_pointer = true;
			}
			continue;
		}
		if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY) {
			continue;
		}

		const bool stack_relative =
		    IsStackPointerRegister(operand.mem.base) || IsStackPointerRegister(operand.mem.index);
		decoded.uses_stack_pointer |= stack_relative;
		constexpr ZydisOperandActions MemoryAccessMask =
		    ZYDIS_OPERAND_ACTION_MASK_READ | ZYDIS_OPERAND_ACTION_MASK_WRITE;
		if (decoded.instruction.mnemonic != ZYDIS_MNEMONIC_LEA &&
		    decoded.instruction.mnemonic != ZYDIS_MNEMONIC_NOP && !stack_relative &&
		    (operand.actions & MemoryAccessMask) != 0) {
			decoded.accesses_memory = true;
		}
	}

	if (decoded.changes_stack_pointer) {
		const auto& operands = decoded.operands;
		if (operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
		    IsStackPointerRegister(operands[0].reg.value) &&
		    (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_RDPID ||
		     (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_MOV &&
		      operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY))) {
			decoded.replaces_stack_pointer = true;
		} else if ((decoded.instruction.mnemonic == ZYDIS_MNEMONIC_ADD ||
		            decoded.instruction.mnemonic == ZYDIS_MNEMONIC_SUB) &&
		           operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
		           IsStackPointerRegister(operands[0].reg.value) &&
		           operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
			const s64 immediate = operands[1].imm.is_signed
			                          ? operands[1].imm.value.s
			                          : static_cast<s64>(operands[1].imm.value.u);
			decoded.stack_pointer_delta =
			    decoded.instruction.mnemonic == ZYDIS_MNEMONIC_ADD ? immediate : -immediate;
		} else if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_LEA &&
		           operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
		           IsStackPointerRegister(operands[0].reg.value) &&
		           operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY &&
		           IsStackPointerRegister(operands[1].mem.base) &&
		           operands[1].mem.index == ZYDIS_REGISTER_NONE) {
			decoded.stack_pointer_delta = operands[1].mem.disp.value;
		} else if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_PUSH ||
		           decoded.instruction.mnemonic == ZYDIS_MNEMONIC_PUSHF ||
		           decoded.instruction.mnemonic == ZYDIS_MNEMONIC_PUSHFD ||
		           decoded.instruction.mnemonic == ZYDIS_MNEMONIC_PUSHFQ) {
			decoded.stack_pointer_delta = -static_cast<s64>(sizeof(u64));
		} else if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_POP ||
		           decoded.instruction.mnemonic == ZYDIS_MNEMONIC_POPF ||
		           decoded.instruction.mnemonic == ZYDIS_MNEMONIC_POPFD ||
		           decoded.instruction.mnemonic == ZYDIS_MNEMONIC_POPFQ) {
			decoded.stack_pointer_delta = sizeof(u64);
		}
	}

	if (decoded.instruction.meta.category == ZYDIS_CATEGORY_CALL) {
		// The return address and the callee's frame overwrite everything below RSP, so the ABI
		// keeps no red-zone data across a call.
		decoded.red_zone_def.set();
	}
	for (u8 index = 0; index < decoded.instruction.operand_count_visible; ++index) {
		const auto& operand = decoded.operands[index];
		if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY ||
		    !IsStackPointerRegister(operand.mem.base) || operand.mem.disp.size == 0 ||
		    operand.mem.disp.value >= 0) {
			continue;
		}

		decoded.has_red_zone_operand = true;
		if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_LEA) {
			if (decoded.stack_pointer_delta.has_value()) {
				decoded.has_red_zone_operand = false;
				continue;
			}
			decoded.has_unmodeled_red_zone_operand = true;
			continue;
		}

		const s64 access_start = operand.mem.disp.value;
		const s64 access_size  = std::max<s64>(operand.size / 8, 1);
		const s64 range_start  = std::max(access_start, -static_cast<s64>(GuestRedZoneSize));
		const s64 range_end    = std::min<s64>(access_start + access_size, 0);
		for (s64 offset = range_start; offset < range_end; ++offset) {
			const size_t bit = static_cast<size_t>(offset + static_cast<s64>(GuestRedZoneSize));
			if ((operand.actions & ZYDIS_OPERAND_ACTION_MASK_READ) != 0) {
				decoded.red_zone_use.set(bit);
			}
			if ((operand.actions & ZYDIS_OPERAND_ACTION_WRITE) != 0) {
				decoded.red_zone_def.set(bit);
			}
		}
	}
	return decoded;
}

bool IsSameRegister(ZydisRegister lhs, ZydisRegister rhs) {
	return lhs != ZYDIS_REGISTER_NONE && rhs != ZYDIS_REGISTER_NONE &&
	       ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, lhs) ==
	           ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, rhs);
}

bool WritesRegister(const DecodedCodeInstruction& decoded, ZydisRegister reg) {
	return std::ranges::any_of(
	    std::span {decoded.operands}.first(decoded.instruction.operand_count),
	    [reg](const ZydisDecodedOperand& operand) {
		    return operand.type == ZYDIS_OPERAND_TYPE_REGISTER &&
		           (operand.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0 &&
		           IsSameRegister(operand.reg.value, reg);
	    });
}

std::optional<std::vector<uintptr_t>>
ResolveBoundedJumpTable(const DecodedFunction& function, uintptr_t branch_address,
                        uintptr_t function_start, uintptr_t function_end, uintptr_t segment_start,
                        uintptr_t segment_end, AddressRanges data_ranges, JumpTable* table) {
	const auto branch = function.instructions.find(branch_address);
	if (branch == function.instructions.end() ||
	    branch->second.instruction.mnemonic != ZYDIS_MNEMONIC_JMP ||
	    branch->second.operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER) {
		return std::nullopt;
	}
	const ZydisRegister target_reg = branch->second.operands[0].reg.value;

	const auto previous_contiguous = [&function](auto instruction) {
		if (instruction == function.instructions.begin()) {
			return function.instructions.end();
		}
		const auto previous = std::prev(instruction);
		return previous->first + previous->second.instruction.length == instruction->first
		           ? previous
		           : function.instructions.end();
	};

	constexpr size_t MaxInterveningInstructions = 4;
	auto             add                        = function.instructions.end();
	auto             pattern_cursor             = branch;
	for (size_t count = 0; count <= MaxInterveningInstructions; ++count) {
		const auto candidate = previous_contiguous(pattern_cursor);
		if (candidate == function.instructions.end()) {
			break;
		}
		const auto& decoded = candidate->second;
		if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_ADD &&
		    decoded.operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
		    IsSameRegister(decoded.operands[0].reg.value, target_reg) &&
		    decoded.operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
			add = candidate;
			break;
		}
		if (WritesRegister(decoded, target_reg) || IsControlFlowTerminator(decoded.instruction)) {
			return std::nullopt;
		}
		pattern_cursor = candidate;
	}
	if (add == function.instructions.end()) {
		return std::nullopt;
	}
	const ZydisRegister table_reg = add->second.operands[1].reg.value;

	const auto load = previous_contiguous(add);
	if (load == function.instructions.end() ||
	    load->second.instruction.mnemonic != ZYDIS_MNEMONIC_MOVSXD ||
	    load->second.operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
	    !IsSameRegister(load->second.operands[0].reg.value, target_reg) ||
	    load->second.operands[1].type != ZYDIS_OPERAND_TYPE_MEMORY ||
	    !IsSameRegister(load->second.operands[1].mem.base, table_reg) ||
	    load->second.operands[1].mem.index == ZYDIS_REGISTER_NONE ||
	    load->second.operands[1].mem.scale != sizeof(s32)) {
		return std::nullopt;
	}
	const ZydisRegister index_reg = load->second.operands[1].mem.index;

	constexpr size_t         MaxPatternInstructions = 64;
	std::optional<size_t>    table_size;
	std::optional<uintptr_t> guarded_path_start;
	auto                     cursor = load;
	for (size_t count = 0; count < MaxPatternInstructions; ++count) {
		cursor = previous_contiguous(cursor);
		if (cursor == function.instructions.end()) {
			break;
		}
		const auto& decoded = cursor->second;
		if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_CMP &&
		    decoded.operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
		    IsSameRegister(decoded.operands[0].reg.value, index_reg) &&
		    decoded.operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
			const uintptr_t next_address  = decoded.address + decoded.instruction.length;
			const auto      bounds_branch = function.instructions.find(next_address);
			if (bounds_branch == function.instructions.end()) {
				return std::nullopt;
			}
			const s64 bound = decoded.operands[1].imm.is_signed
			                      ? decoded.operands[1].imm.value.s
			                      : static_cast<s64>(decoded.operands[1].imm.value.u);
			if (bound < 0) {
				return std::nullopt;
			}
			if (bounds_branch->second.instruction.mnemonic == ZYDIS_MNEMONIC_JNBE) {
				table_size = static_cast<size_t>(bound) + 1;
			} else if (bounds_branch->second.instruction.mnemonic == ZYDIS_MNEMONIC_JNB) {
				table_size = static_cast<size_t>(bound);
			} else {
				return std::nullopt;
			}
			guarded_path_start = next_address + bounds_branch->second.instruction.length;
			break;
		}
		if (WritesRegister(decoded, index_reg)) {
			return std::nullopt;
		}
	}
	if (!table_size) {
		return std::nullopt;
	}
	if (std::ranges::any_of(function.branch_targets, [&](uintptr_t target) {
		    return target >= *guarded_path_start && target <= branch_address;
	    })) {
		return std::nullopt;
	}

	const auto decode_table_address =
	    [table_reg](const DecodedCodeInstruction& decoded) -> std::optional<uintptr_t> {
		if (decoded.instruction.mnemonic != ZYDIS_MNEMONIC_LEA ||
		    decoded.operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
		    !IsSameRegister(decoded.operands[0].reg.value, table_reg) ||
		    decoded.operands[1].type != ZYDIS_OPERAND_TYPE_MEMORY) {
			return std::nullopt;
		}
		ZyanU64 absolute_address {};
		if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&decoded.instruction, &decoded.operands[1],
		                                           decoded.address, &absolute_address))) {
			return std::nullopt;
		}
		return absolute_address;
	};

	std::set<uintptr_t> table_candidates;
	cursor = load;
	for (size_t count = 0; count < MaxPatternInstructions; ++count) {
		cursor = previous_contiguous(cursor);
		if (cursor == function.instructions.end()) {
			break;
		}
		if (!WritesRegister(cursor->second, table_reg)) {
			continue;
		}
		if (const auto address = decode_table_address(cursor->second)) {
			table_candidates.insert(*address);
		}
		break;
	}
	if (table_candidates.empty()) {
		for (const auto& [address, decoded]: function.instructions) {
			if (address >= branch_address) {
				break;
			}
			if (const auto table_address = decode_table_address(decoded)) {
				table_candidates.insert(*table_address);
			}
		}
	}
	constexpr size_t MaxJumpTableEntries = 4096;
	if (*table_size == 0 || *table_size > MaxJumpTableEntries) {
		return std::nullopt;
	}

	const auto table_fits = [&](uintptr_t table_address, uintptr_t start, uintptr_t end) {
		return table_address >= start && table_address <= end &&
		       *table_size <= (end - table_address) / sizeof(s32);
	};
	std::optional<std::vector<uintptr_t>> resolved_targets;
	for (const uintptr_t table_address: table_candidates) {
		if (!table_fits(table_address, segment_start, segment_end) &&
		    std::ranges::none_of(data_ranges, [&](const auto& range) {
			    return table_fits(table_address, range.first, range.first + range.second);
		    })) {
			continue;
		}

		std::vector<uintptr_t> targets;
		targets.reserve(*table_size);
		bool valid = true;
		for (size_t index = 0; index < *table_size; ++index) {
			s32 offset;
			std::memcpy(&offset,
			            reinterpret_cast<const void*>(table_address + index * sizeof(offset)),
			            sizeof(offset));
			const s64 target = static_cast<s64>(table_address) + offset;
			if (target < static_cast<s64>(function_start) ||
			    target >= static_cast<s64>(function_end)) {
				valid = false;
				break;
			}
			targets.push_back(static_cast<uintptr_t>(target));
		}
		if (!valid) {
			continue;
		}
		if (resolved_targets) {
			return std::nullopt;
		}
		if (table != nullptr) {
			*table = {.load = load->first, .address = table_address, .entries = targets};
		}
		std::ranges::sort(targets);
		targets.erase(std::ranges::unique(targets).begin(), targets.end());
		resolved_targets = std::move(targets);
	}
	return resolved_targets;
}

bool IsFramePointerRegister(ZydisRegister reg) {
	return reg != ZYDIS_REGISTER_NONE &&
	       ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg) == ZYDIS_REGISTER_RBP;
}

bool WritesFramePointer(const DecodedCodeInstruction& decoded) {
	for (u8 index = 0; index < decoded.instruction.operand_count; ++index) {
		const auto& operand = decoded.operands[index];
		if (operand.type == ZYDIS_OPERAND_TYPE_REGISTER &&
		    IsFramePointerRegister(operand.reg.value) &&
		    (operand.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0) {
			return true;
		}
	}
	return false;
}

// Clang's frame-pointer prologue (push rbp; mov rbp, rsp; push ...) lets a leaf function address
// its red zone through RBP: [rbp - d] lies below RSP once d exceeds the bytes pushed after the
// MOV. Windows writes its exception frame there, so map such operands onto the RSP-relative red
// zone bits the liveness analysis uses. Offsets come from a forward pass over the known RSP
// deltas; anything the pass cannot prove makes the function conservative.
void ModelFramePointerRedZone(DecodedFunction& function, uintptr_t function_start) {
	auto&      instructions = function.instructions;
	const auto push_rbp     = instructions.find(function_start);
	if (push_rbp == instructions.end() ||
	    push_rbp->second.instruction.mnemonic != ZYDIS_MNEMONIC_PUSH ||
	    push_rbp->second.operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
	    push_rbp->second.operands[0].reg.value != ZYDIS_REGISTER_RBP) {
		return;
	}
	const auto mov_rbp = instructions.find(function_start + push_rbp->second.instruction.length);
	if (mov_rbp == instructions.end() ||
	    mov_rbp->second.instruction.mnemonic != ZYDIS_MNEMONIC_MOV ||
	    mov_rbp->second.operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
	    mov_rbp->second.operands[0].reg.value != ZYDIS_REGISTER_RBP ||
	    mov_rbp->second.operands[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
	    mov_rbp->second.operands[1].reg.value != ZYDIS_REGISTER_RSP) {
		return;
	}

	// State before each instruction: RSP relative to RBP (RBP = entry RSP - 8), or nothing
	// when unknown or RBP no longer holds the frame.
	struct State {
		bool known {};
		s64  rsp_minus_rbp {};
	};
	std::map<uintptr_t, State> state_in;
	std::vector<uintptr_t>     work;
	bool                       conflict = false;
	const auto                 merge    = [&](uintptr_t address, State state) {
		if (!instructions.contains(address)) {
			return;
		}
		auto [entry, inserted] = state_in.try_emplace(address, state);
		if (inserted) {
			work.push_back(address);
			return;
		}
		if (entry->second.known &&
		    (!state.known || state.rsp_minus_rbp != entry->second.rsp_minus_rbp)) {
			conflict |= state.known;
			entry->second.known = false;
			work.push_back(address);
		}
	};
	merge(mov_rbp->first + mov_rbp->second.instruction.length, {.known = true, .rsp_minus_rbp = 0});
	while (!work.empty()) {
		const uintptr_t address = work.back();
		work.pop_back();
		const auto& decoded = instructions.at(address);
		State       out     = state_in.at(address);
		if (out.known) {
			if (WritesFramePointer(decoded)) {
				out.known = false;
			} else if (decoded.changes_stack_pointer) {
				if (decoded.stack_pointer_delta.has_value() && !decoded.replaces_stack_pointer) {
					out.rsp_minus_rbp += *decoded.stack_pointer_delta;
				} else {
					out.known = false;
				}
			}
		}
		const uintptr_t next_address  = address + decoded.instruction.length;
		const uintptr_t branch_target = GetRelativeTarget(decoded);
		if (decoded.instruction.meta.category == ZYDIS_CATEGORY_COND_BR) {
			merge(next_address, out);
			merge(branch_target, out);
		} else if (decoded.instruction.meta.category == ZYDIS_CATEGORY_UNCOND_BR) {
			merge(branch_target, out);
			if (const auto table = function.jump_tables.find(address);
			    table != function.jump_tables.end()) {
				for (const uintptr_t target: table->second.entries) {
					merge(target, out);
				}
			}
		} else if (!IsControlFlowTerminator(decoded.instruction)) {
			merge(next_address, out);
		}
	}

	for (auto& [address, decoded]: instructions) {
		const auto state               = state_in.find(address);
		bool       only_frame_operands = true;
		bool       has_memory_operand  = false;
		for (u8 index = 0; index < decoded.instruction.operand_count_visible; ++index) {
			const auto& operand = decoded.operands[index];
			if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY) {
				continue;
			}
			has_memory_operand = true;
			if (!IsFramePointerRegister(operand.mem.base) || operand.mem.disp.value >= 0) {
				only_frame_operands = false;
				continue;
			}
			if (state == state_in.end() || !state->second.known || conflict ||
			    operand.mem.index != ZYDIS_REGISTER_NONE ||
			    decoded.instruction.mnemonic == ZYDIS_MNEMONIC_LEA) {
				// A frame access whose place relative to RSP is unknown, or a pointer into it.
				if (state != state_in.end() || conflict) {
					decoded.has_red_zone_operand                     = true;
					decoded.has_unmodeled_red_zone_operand           = true;
					function.uses_red_zone                           = true;
					function.requires_conservative_red_zone_tracking = true;
				}
				only_frame_operands = false;
				continue;
			}
			// Offset of the access from the current RSP.
			const s64 access_start = operand.mem.disp.value - state->second.rsp_minus_rbp;
			const s64 access_size  = std::max<s64>(operand.size / 8, 1);
			if (access_start >= 0) {
				continue;
			}
			decoded.has_red_zone_operand = true;
			function.uses_red_zone       = true;
			const s64 range_start = std::max(access_start, -static_cast<s64>(GuestRedZoneSize));
			const s64 range_end   = std::min<s64>(access_start + access_size, 0);
			for (s64 offset = range_start; offset < range_end; ++offset) {
				const size_t bit = static_cast<size_t>(offset + static_cast<s64>(GuestRedZoneSize));
				if ((operand.actions & ZYDIS_OPERAND_ACTION_MASK_READ) != 0) {
					decoded.red_zone_use.set(bit);
				}
				if ((operand.actions & ZYDIS_OPERAND_ACTION_WRITE) != 0) {
					decoded.red_zone_def.set(bit);
				}
			}
		}
		// Stack accesses through the frame never fault: keep them out of the rewrite set.
		if (has_memory_operand && only_frame_operands && state != state_in.end() &&
		    state->second.known && !conflict) {
			decoded.accesses_memory = false;
		}
	}
}

DecodedFunction DecodeFunction(uintptr_t function_start, uintptr_t function_end,
                               uintptr_t segment_start, uintptr_t segment_end,
                               AddressRanges data_ranges = {}) {
	DecodedFunction               function;
	std::vector<uintptr_t>        blocks {function_start};
	std::unordered_set<uintptr_t> visited;
	std::set<uintptr_t>           indirect_branches;
	std::set<uintptr_t>           resolved_indirect_branches;

	while (true) {
		while (!blocks.empty()) {
			uintptr_t address = blocks.back();
			blocks.pop_back();

			while (address >= function_start && address < function_end &&
			       !visited.contains(address)) {
				visited.insert(address);
				auto decoded = DecodeCodeInstruction(address, function_end);
				if (decoded.instruction.length == 0) {
					break;
				}

				function.uses_red_zone |= decoded.has_red_zone_operand;
				function.requires_conservative_red_zone_tracking |=
				    decoded.has_unmodeled_red_zone_operand ||
				    (decoded.changes_stack_pointer && !decoded.stack_pointer_delta.has_value());

				const uintptr_t next_address  = address + decoded.instruction.length;
				const uintptr_t branch_target = GetRelativeTarget(decoded);
				if (branch_target >= function_start && branch_target < function_end) {
					function.branch_targets.insert(branch_target);
				}
				function.instructions.emplace(address, decoded);

				if (decoded.instruction.meta.category == ZYDIS_CATEGORY_COND_BR) {
					if (branch_target >= function_start && branch_target < function_end) {
						blocks.push_back(branch_target);
					}
				} else if (IsControlFlowTerminator(decoded.instruction)) {
					if (decoded.instruction.meta.category == ZYDIS_CATEGORY_UNCOND_BR &&
					    branch_target >= function_start && branch_target < function_end) {
						blocks.push_back(branch_target);
					} else if (decoded.instruction.meta.category == ZYDIS_CATEGORY_UNCOND_BR &&
					           branch_target == 0) {
						indirect_branches.insert(address);
					}
					break;
				}
				address = next_address;
			}
		}

		bool discovered_block = false;
		for (const uintptr_t branch_address: indirect_branches) {
			if (resolved_indirect_branches.contains(branch_address)) {
				continue;
			}
			JumpTable  table;
			const auto targets =
			    ResolveBoundedJumpTable(function, branch_address, function_start, function_end,
			                            segment_start, segment_end, data_ranges, &table);
			if (!targets) {
				continue;
			}
			function.jump_tables.emplace(branch_address, std::move(table));
			resolved_indirect_branches.insert(branch_address);
			for (const uintptr_t target: *targets) {
				function.branch_targets.insert(target);
				if (!visited.contains(target)) {
					blocks.push_back(target);
					discovered_block = true;
				}
			}
		}
		if (!discovered_block) {
			break;
		}
	}
	function.has_indirect_branch = indirect_branches.size() != resolved_indirect_branches.size();
	ModelFramePointerRedZone(function, function_start);
	return function;
}

std::optional<RedZoneMask> TranslateRedZoneMask(const RedZoneMask& mask, s64 stack_pointer_delta) {
	if (stack_pointer_delta == 0) {
		return mask;
	}
	RedZoneMask translated;
	for (size_t bit = 0; bit < GuestRedZoneSize; ++bit) {
		if (!mask.test(bit)) {
			continue;
		}
		const s64 after_offset  = static_cast<s64>(bit) - static_cast<s64>(GuestRedZoneSize);
		const s64 before_offset = after_offset + stack_pointer_delta;
		if (before_offset < -static_cast<s64>(GuestRedZoneSize) || before_offset >= 0) {
			// A later inverse RSP adjustment can bring this byte back into the red zone.
			// Losing its stack-slot identity makes the bounded analysis inconclusive.
			return std::nullopt;
		}
		translated.set(static_cast<size_t>(before_offset + static_cast<s64>(GuestRedZoneSize)));
	}
	return translated;
}

void AnalyzeRedZoneLiveness(DecodedFunction& function) {
	if (!function.uses_red_zone) {
		return;
	}

	const auto protect_function = [&] {
		for (auto& [_, decoded]: function.instructions) {
			decoded.red_zone_live.set();
		}
	};
	if (function.has_indirect_branch || function.requires_conservative_red_zone_tracking ||
	    std::ranges::any_of(function.branch_targets, [&function](uintptr_t target) {
		    return !function.instructions.contains(target);
	    })) {
		protect_function();
		return;
	}

	std::vector<DecodedCodeInstruction*> instructions;
	instructions.reserve(function.instructions.size());
	std::map<uintptr_t, size_t> indices;
	for (auto& [address, decoded]: function.instructions) {
		indices.emplace(address, instructions.size());
		instructions.push_back(&decoded);
	}

	std::vector<RedZoneMask> live_in(instructions.size());
	bool                     changed;
	do {
		changed = false;
		for (size_t reverse_index = instructions.size(); reverse_index-- > 0;) {
			const auto& decoded = *instructions[reverse_index];
			RedZoneMask live_out;

			const auto add_successor = [&](uintptr_t address) {
				if (const auto successor = indices.find(address); successor != indices.end()) {
					live_out |= live_in[successor->second];
				}
			};

			const uintptr_t next_address  = decoded.address + decoded.instruction.length;
			const uintptr_t branch_target = GetRelativeTarget(decoded);
			if (decoded.instruction.meta.category == ZYDIS_CATEGORY_COND_BR) {
				add_successor(next_address);
				add_successor(branch_target);
			} else if (decoded.instruction.meta.category == ZYDIS_CATEGORY_UNCOND_BR) {
				add_successor(branch_target);
				if (const auto table = function.jump_tables.find(decoded.address);
				    table != function.jump_tables.end()) {
					for (const uintptr_t target: table->second.entries) {
						add_successor(target);
					}
				}
			} else if (!IsControlFlowTerminator(decoded.instruction)) {
				add_successor(next_address);
			}

			const auto translated_live_out =
			    TranslateRedZoneMask(live_out, decoded.stack_pointer_delta.value_or(0));
			if (!translated_live_out) {
				protect_function();
				return;
			}
			const RedZoneMask new_live_in =
			    decoded.red_zone_use | (*translated_live_out & ~decoded.red_zone_def);
			if (new_live_in != live_in[reverse_index]) {
				live_in[reverse_index] = new_live_in;
				changed                = true;
			}
		}
	} while (changed);

	for (size_t index = 0; index < instructions.size(); ++index) {
		instructions[index]->red_zone_live = live_in[index];
	}
	function.red_zone_liveness_exact = true;
}

bool EncodeRelocatedInstruction(const DecodedCodeInstruction& decoded,
                                Xbyak::CodeGenerator& generator, s64 stack_adjustment = 0,
                                ZydisMnemonic replacement       = ZYDIS_MNEMONIC_INVALID,
                                s64           memory_adjustment = 0) {
	ZydisEncoderRequest request;
	if (!ZYAN_SUCCESS(ZydisEncoderDecodedInstructionToEncoderRequest(
	        &decoded.instruction, decoded.operands.data(),
	        decoded.instruction.operand_count_visible, &request))) {
		return false;
	}
	if (replacement != ZYDIS_MNEMONIC_INVALID) {
		request.mnemonic = replacement;
	}

	for (u8 index = 0; index < decoded.instruction.operand_count_visible; ++index) {
		const auto& operand = decoded.operands[index];
		if (operand.type == ZYDIS_OPERAND_TYPE_IMMEDIATE && operand.imm.is_relative) {
			ZyanU64 absolute_address {};
			if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&decoded.instruction, &operand,
			                                           decoded.address, &absolute_address))) {
				return false;
			}
			request.operands[index].imm.u = absolute_address;
		} else if (operand.type == ZYDIS_OPERAND_TYPE_MEMORY &&
		           (operand.mem.base == ZYDIS_REGISTER_RIP ||
		            operand.mem.base == ZYDIS_REGISTER_EIP)) {
			ZyanU64 absolute_address {};
			if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&decoded.instruction, &operand,
			                                           decoded.address, &absolute_address))) {
				return false;
			}
			request.operands[index].mem.displacement = static_cast<ZyanI64>(absolute_address);
		} else if (operand.type == ZYDIS_OPERAND_TYPE_MEMORY &&
		           IsStackPointerRegister(operand.mem.base)) {
			request.operands[index].mem.displacement += stack_adjustment;
		} else if (operand.type == ZYDIS_OPERAND_TYPE_MEMORY) {
			request.operands[index].mem.displacement += memory_adjustment;
		}
	}

	std::array<u8, ZYDIS_MAX_INSTRUCTION_LENGTH> encoded {};
	ZyanUSize                                    encoded_size = encoded.size();
	if (!ZYAN_SUCCESS(ZydisEncoderEncodeInstructionAbsolute(
	        &request, encoded.data(), &encoded_size,
	        reinterpret_cast<ZyanU64>(generator.getCurr())))) {
		return false;
	}
	generator.db(encoded.data(), encoded_size);
	return true;
}

bool GenerateProtectedIndirectCall(const DecodedCodeInstruction& decoded,
                                   Xbyak::CodeGenerator&         generator) {
	ASSERT(decoded.instruction.meta.category == ZYDIS_CATEGORY_CALL);
	const auto& target = decoded.operands[0];
	ASSERT(target.type == ZYDIS_OPERAND_TYPE_MEMORY && target.size == sizeof(uintptr_t) * CHAR_BIT);

	ZydisEncoderRequest request {};
	request.machine_mode          = ZYDIS_MACHINE_MODE_LONG_64;
	request.mnemonic              = ZYDIS_MNEMONIC_MOV;
	request.operand_count         = 2;
	request.operands[0].type      = ZYDIS_OPERAND_TYPE_REGISTER;
	request.operands[0].reg.value = ZYDIS_REGISTER_R11;

	ZydisEncoderRequest original_request {};
	if (!ZYAN_SUCCESS(ZydisEncoderDecodedInstructionToEncoderRequest(
	        &decoded.instruction, decoded.operands.data(),
	        decoded.instruction.operand_count_visible, &original_request))) {
		return false;
	}
	request.prefixes             = original_request.prefixes & ZYDIS_ATTRIB_HAS_SEGMENT;
	request.address_size_hint    = original_request.address_size_hint;
	request.operands[1]          = original_request.operands[0];
	request.operands[1].mem.size = sizeof(uintptr_t);

	if (target.mem.base == ZYDIS_REGISTER_RIP || target.mem.base == ZYDIS_REGISTER_EIP) {
		ZyanU64 absolute_address {};
		if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&decoded.instruction, &target, decoded.address,
		                                           &absolute_address))) {
			return false;
		}
		request.operands[1].mem.displacement = static_cast<ZyanI64>(absolute_address);
	}

	generator.lea(rsp, ptr[rsp - GuestRedZoneSize]);
	generator.push(r11);

	std::array<u8, ZYDIS_MAX_INSTRUCTION_LENGTH> encoded {};
	ZyanUSize                                    encoded_size = encoded.size();
	if (!ZYAN_SUCCESS(ZydisEncoderEncodeInstructionAbsolute(
	        &request, encoded.data(), &encoded_size,
	        reinterpret_cast<ZyanU64>(generator.getCurr())))) {
		return false;
	}
	generator.db(encoded.data(), encoded_size);

	generator.xchg(r11, ptr[rsp]);
	generator.lea(rsp, ptr[rsp + GuestRedZoneSize + sizeof(uintptr_t)]);
	generator.call(ptr[rsp - GuestRedZoneSize - sizeof(uintptr_t)]);
	return true;
}

void CollectRedZoneMemoryInstructions(const DecodedFunction&                   function,
                                      std::map<uintptr_t, InstructionRewrite>& rewrite_sites,
                                      GuestInstructionPatchResult&             result) {
	if (!function.uses_red_zone) {
		return;
	}
	for (const auto& [address, decoded]: function.instructions) {
		if (!decoded.accesses_memory || !decoded.red_zone_live.any() ||
		    rewrite_sites.contains(address)) {
			continue;
		}
		++result.memory_instruction_count;
		if (decoded.instruction.length < NearJumpSize) {
			++result.short_memory_instruction_count;
		}
		if (decoded.instruction.meta.category == ZYDIS_CATEGORY_CALL &&
		    decoded.operands[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
		    decoded.operands[0].size == sizeof(uintptr_t) * CHAR_BIT &&
		    !IsStackPointerRegister(decoded.operands[0].mem.base) &&
		    !IsStackPointerRegister(decoded.operands[0].mem.index)) {
			rewrite_sites[address] = {
			    .protect_red_zone        = true,
			    .protected_indirect_call = true,
			};
			continue;
		}
		if (decoded.uses_stack_pointer && !decoded.replaces_stack_pointer) {
			++result.stack_dependent_memory_instruction_count;
			continue;
		}
		if (decoded.instruction.meta.category == ZYDIS_CATEGORY_CALL ||
		    IsControlFlowTerminator(decoded.instruction)) {
			++result.control_flow_memory_instruction_count;
			continue;
		}
		rewrite_sites[address].protect_red_zone = true;
	}
}

// Compute 1/sqrt in double precision, rounded to float, with denormals treated as
// signed zero. Preserve guest MXCSR, RFLAGS and the scratch register's full YMM.
void GenerateReciprocalSquareRoot(const DecodedCodeInstruction& decoded,
                                  Xbyak::CodeGenerator&         generator) {
	const int destination   = decoded.operands[0].reg.value - ZYDIS_REGISTER_XMM0;
	const int source        = decoded.operands[1].reg.value - ZYDIS_REGISTER_XMM0;
	int       scratch_index = 0;
	while (scratch_index == destination || scratch_index == source) {
		++scratch_index;
	}
	const Xbyak::Ymm scratch(scratch_index);
	Xbyak::Label     one;
	Xbyak::Label     done;
	constexpr size_t SpillSize = 48;

	// Guest code uses the SysV red zone on both hosts; put our spills below it.
	// LEA/MOV/SIMD leave RFLAGS intact, including when the guest red zone is live.
	generator.lea(rsp, ptr[rsp - GuestRedZoneSize - SpillSize]);
	generator.vmovdqu(ptr[rsp], scratch);
	generator.stmxcsr(ptr[rsp + 32]);
	generator.mov(dword[rsp + 36], 0x1fc0); // round to nearest, exceptions masked, DAZ
	generator.ldmxcsr(ptr[rsp + 36]);
	generator.vcvtps2pd(scratch, Xbyak::Xmm(source));
	generator.vsqrtpd(scratch, scratch);
	generator.vbroadcastsd(Xbyak::Ymm(destination), ptr[rip + one]);
	generator.vdivpd(scratch, Xbyak::Ymm(destination), scratch);
	generator.vcvtpd2ps(Xbyak::Xmm(destination), scratch); // Clears upper YMM lanes.
	generator.ldmxcsr(ptr[rsp + 32]);
	generator.vmovdqu(scratch, ptr[rsp]);
	generator.lea(rsp, ptr[rsp + GuestRedZoneSize + SpillSize]);
	generator.jmp(done);
	// A failed label reference can have an incomplete displacement. Do not bind it.
	if (Xbyak::GetError() != 0) {
		return;
	}
	generator.L(one);
	generator.dq(0x3ff0000000000000ULL);
	generator.L(done);
}

bool IsSupportedBitFieldInstruction(const DecodedCodeInstruction& decoded) {
	const bool insert = decoded.instruction.mnemonic == ZYDIS_MNEMONIC_INSERTQ;
	if (!insert && decoded.instruction.mnemonic != ZYDIS_MNEMONIC_EXTRQ) {
		return false;
	}
	const auto is_xmm = [](const ZydisDecodedOperand& operand) {
		return operand.type == ZYDIS_OPERAND_TYPE_REGISTER &&
		       operand.reg.value >= ZYDIS_REGISTER_XMM0 &&
		       operand.reg.value <= ZYDIS_REGISTER_XMM15;
	};
	const size_t immediate_index = insert ? 2 : 1;
	const bool   immediate =
	    decoded.instruction.operand_count_visible == immediate_index + 2 &&
	    decoded.operands[immediate_index].type == ZYDIS_OPERAND_TYPE_IMMEDIATE &&
	    decoded.operands[immediate_index + 1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE;
	if (!is_xmm(decoded.operands[0]) || (insert && !is_xmm(decoded.operands[1])) ||
	    (!immediate &&
	     !(decoded.instruction.operand_count_visible == 2 && is_xmm(decoded.operands[1])))) {
		return false;
	}
	// Match the trap handler's encoding support: 66/F2 [REX] 0F 78/79 ModRM [imm8 imm8].
	const auto*  code   = reinterpret_cast<const u8*>(decoded.address);
	const size_t opcode = (code[1] & 0xf0u) == 0x40u ? 2 : 1;
	return code[0] == (insert ? 0xf2 : 0x66) && code[opcode] == 0x0f &&
	       code[opcode + 1] == (immediate ? 0x78 : 0x79) &&
	       decoded.instruction.length == opcode + (immediate ? 5 : 3);
}

void GenerateExtractQ(const DecodedCodeInstruction& decoded, Xbyak::CodeGenerator& generator) {
	const Xbyak::Xmm destination(decoded.operands[0].reg.value - ZYDIS_REGISTER_XMM0);
	// Keep spills below the guest red zone on both hosts, and restore all modified flags/GPRs.
	generator.lea(rsp, ptr[rsp - GuestRedZoneSize]);
	generator.pushfq();
	generator.push(rax);
	generator.push(rcx);
	generator.push(rdx);
	if (decoded.operands[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
		const u32 length = decoded.operands[1].imm.value.u & 63u;
		const u32 index  = decoded.operands[2].imm.value.u & 63u;
		const u64 mask   = length == 0 ? UINT64_MAX : (u64 {1} << length) - 1;
		generator.movq(rax, destination);
		if (index != 0) {
			generator.shr(rax, index);
		}
		generator.mov(rdx, mask);
	} else {
		const Xbyak::Xmm source(decoded.operands[1].reg.value - ZYDIS_REGISTER_XMM0);
		generator.movq(rax, source);
		generator.mov(ecx, eax);
		generator.neg(ecx);
		generator.mov(rdx, UINT64_MAX);
		// CL shifts use six bits: -length gives 64-length, with zero retaining all 64 bits.
		generator.shr(rdx, cl);
		generator.shr(rax, 8);
		generator.mov(ecx, eax);
		generator.movq(rax, destination);
		generator.shr(rax, cl);
	}
	generator.and_(rax, rdx);
	// Match the emulator: clear the upper XMM half, preserving the upper YMM half.
	generator.movq(destination, rax);
	generator.pop(rdx);
	generator.pop(rcx);
	generator.pop(rax);
	generator.popfq();
	generator.lea(rsp, ptr[rsp + GuestRedZoneSize]);
}

void CollectAmdInstructions(const DecodedFunction&                   function,
                            std::map<uintptr_t, InstructionRewrite>& rewrite_sites,
                            GuestInstructionPatchResult&             result,
                            GuestInstructionHostFeatures             host_features) {
	for (const auto& [address, decoded]: function.instructions) {
		const auto&            instruction = decoded.instruction;
		InstructionReplacement replacement {};
		if (instruction.mnemonic == ZYDIS_MNEMONIC_VRSQRTPS &&
		    instruction.encoding == ZYDIS_INSTRUCTION_ENCODING_VEX &&
		    instruction.raw.vex.offset == 0 && decoded.operands[0].size == 128 &&
		    decoded.operands[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
			replacement = InstructionReplacement::ReciprocalSquareRoot;
		} else if (!host_features.sse4a && IsSupportedBitFieldInstruction(decoded)) {
			replacement = instruction.mnemonic == ZYDIS_MNEMONIC_EXTRQ
			                  ? InstructionReplacement::ExtractQ
			                  : InstructionReplacement::InsertQ;
		} else if (!host_features.rdpid && instruction.mnemonic == ZYDIS_MNEMONIC_RDPID) {
			replacement = InstructionReplacement::ReadProcessorId;
		} else if (!host_features.clwb && instruction.mnemonic == ZYDIS_MNEMONIC_CLWB) {
			replacement = InstructionReplacement::CacheLineWriteBack;
		} else {
			continue;
		}
		++ReplacementCounts(result, replacement).found;
		rewrite_sites[address].replacement = replacement;
	}
}

// RDTSC/RDTSCP at the guest rate: EDX:EAX = host TSC * multiplier >> 32 with the upper halves of
// RAX and RDX zeroed; RDTSCP still sets ECX. The BMI2 sequence keeps RFLAGS and the stack.
void GenerateReadTimeStampCounter(const DecodedCodeInstruction& decoded, u64 multiplier,
                                  Xbyak::CodeGenerator& generator) {
	if (decoded.instruction.mnemonic == ZYDIS_MNEMONIC_RDTSCP) {
		generator.rdtscp();
	} else {
		generator.rdtsc();
	}
	generator.rorx(rdx, rdx, 32);
	generator.lea(rax, ptr[rax + rdx]);
	generator.mov(rdx, multiplier);
	generator.mulx(rdx, rax, rax);
	generator.rorx(rax, rax, 32);
	generator.mov(eax, eax);
	generator.mov(edx, edx);
}

void CollectTscInstructions(const DecodedFunction&                   function,
                            std::map<uintptr_t, InstructionRewrite>& rewrite_sites,
                            GuestInstructionPatchResult&             result) {
	for (const auto& [address, decoded]: function.instructions) {
		const auto mnemonic = decoded.instruction.mnemonic;
		if (mnemonic == ZYDIS_MNEMONIC_RDTSC || mnemonic == ZYDIS_MNEMONIC_RDTSCP) {
			++result.rdtsc.found;
			rewrite_sites[address].replacement = InstructionReplacement::ReadTimeStampCounter;
		}
	}
}

// Whether [start, end) holds an RDTSC (0F 31) or RDTSCP (0F 01 F9) encoding: only those functions
// are decoded when nothing else is patched.
bool MayReadTimeStampCounter(uintptr_t start, uintptr_t end) {
	constexpr std::array<u8, 2> Rdtsc {0x0f, 0x31};
	constexpr std::array<u8, 3> Rdtscp {0x0f, 0x01, 0xf9};
	const std::span             code(reinterpret_cast<const u8*>(start), end - start);
	return !std::ranges::search(code, Rdtsc).empty() || !std::ranges::search(code, Rdtscp).empty();
}

void MarkInstructionTrap(u8* code, const ZydisDecodedInstruction& instruction,
                         InstructionReplacement replacement) {
	if (replacement == InstructionReplacement::ReciprocalSquareRoot) {
		// Clear a reserved VEX.vvvv bit, retaining both register operands for the handler.
		code[instruction.raw.vex.size - 1] &= ~0x08u;
	}
	// Other replacements already raise #UD on hosts without their CPU feature.
}

bool GenerateInstructionTrap(const DecodedCodeInstruction& decoded,
                             InstructionReplacement replacement, Xbyak::CodeGenerator& generator,
                             s64 stack_adjustment) {
	if (replacement == InstructionReplacement::CacheLineWriteBack) {
		return EncodeRelocatedInstruction(decoded, generator, stack_adjustment);
	}
	std::array<u8, ZYDIS_MAX_INSTRUCTION_LENGTH> code {};
	std::memcpy(code.data(), reinterpret_cast<const void*>(decoded.address),
	            decoded.instruction.length);
	MarkInstructionTrap(code.data(), decoded.instruction, replacement);
	generator.db(code.data(), decoded.instruction.length);
	return true;
}

bool NeedsTrapRedZoneProtection(const DecodedCodeInstruction& decoded) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return decoded.red_zone_live.any();
#else
	return false;
#endif
}

void TrapUnrelocatedInstructions(const PatchModule& module, const DecodedFunction& function,
                                 const std::map<uintptr_t, InstructionRewrite>& rewrite_sites,
                                 GuestInstructionPatchResult&                   result) {
	for (const auto& [address, rewrite]: rewrite_sites) {
		if (rewrite.replacement == InstructionReplacement::None ||
		    rewrite.replacement == InstructionReplacement::ReadTimeStampCounter ||
		    module.patched.contains(reinterpret_cast<u8*>(address))) {
			continue;
		}
		const auto& decoded = function.instructions.at(address);
		if (NeedsTrapRedZoneProtection(decoded)) {
			if (rewrite.replacement != InstructionReplacement::ReciprocalSquareRoot) {
				// Leaving an unsupported instruction unchanged still traps with a live red zone.
				EXIT("AMD CPU compatibility: cannot safely trap %s at %p (guest red zone is "
				     "live)\n",
				     ZydisMnemonicGetString(decoded.instruction.mnemonic),
				     reinterpret_cast<void*>(address));
			}
			continue;
		}
		MarkInstructionTrap(reinterpret_cast<u8*>(address), decoded.instruction,
		                    rewrite.replacement);
		++ReplacementCounts(result, rewrite.replacement).trapped;
	}
}

void RelocateGuestInstructions(PatchModule* module, const DecodedFunction& function,
                               const std::map<uintptr_t, InstructionRewrite>& rewrite_sites,
                               GuestInstructionPatchResult& result, u64 tsc_multiplier) {
	struct RelocationSpan {
		std::vector<const DecodedCodeInstruction*> instructions;
		uintptr_t                                  patch_start {};
		uintptr_t                                  continuation {};
		size_t                                     patch_size {};
		bool                                       trap_replacements {};
	};

	const auto emit_span = [&](RelocationSpan& span) -> std::optional<size_t> {
		auto&        generator         = module->trampoline_gen;
		const size_t trampoline_offset = generator.getSize();
		const bool   has_replacements =
		    std::ranges::any_of(span.instructions, [&](const auto* decoded) {
			    const auto rewrite = rewrite_sites.find(decoded->address);
			    return rewrite != rewrite_sites.end() &&
			           rewrite->second.replacement != InstructionReplacement::None;
		    });
		for (bool trap_replacements: {false, true}) {
			span.trap_replacements = trap_replacements;
			Xbyak::ClearError();
			bool encoded = true;
			for (const auto* decoded: span.instructions) {
				const auto rewrite     = rewrite_sites.find(decoded->address);
				const auto replacement = rewrite != rewrite_sites.end()
				                             ? rewrite->second.replacement
				                             : InstructionReplacement::None;
				const bool uses_trap   = UsesInstructionTrap(replacement, trap_replacements);
				const bool protected_indirect_call =
				    rewrite != rewrite_sites.end() && rewrite->second.protected_indirect_call;
				const bool protect_red_zone =
				    (rewrite != rewrite_sites.end() && rewrite->second.protect_red_zone &&
				     !protected_indirect_call) ||
				    ((uses_trap || replacement == InstructionReplacement::CacheLineWriteBack) &&
				     NeedsTrapRedZoneProtection(*decoded));
				if (protect_red_zone) {
					generator.lea(rsp, ptr[rsp - GuestRedZoneSize]);
				}
				if (replacement != InstructionReplacement::None) {
					if (uses_trap) {
						encoded = GenerateInstructionTrap(*decoded, replacement, generator,
						                                  protect_red_zone ? GuestRedZoneSize : 0);
					} else if (replacement == InstructionReplacement::ReciprocalSquareRoot) {
						GenerateReciprocalSquareRoot(*decoded, generator);
					} else if (replacement == InstructionReplacement::ExtractQ) {
						GenerateExtractQ(*decoded, generator);
					} else if (replacement == InstructionReplacement::ReadTimeStampCounter) {
						GenerateReadTimeStampCounter(*decoded, tsc_multiplier, generator);
					} else {
						// CLFLUSH writes back dirty data too; invalidation is a permitted stronger
						// action.
						encoded = EncodeRelocatedInstruction(
						    *decoded, generator, protect_red_zone ? GuestRedZoneSize : 0,
						    ZYDIS_MNEMONIC_CLFLUSH);
					}
				} else if (protected_indirect_call) {
					encoded = GenerateProtectedIndirectCall(*decoded, generator);
				} else {
					encoded = EncodeRelocatedInstruction(*decoded, generator);
				}
				if (protect_red_zone && !decoded->replaces_stack_pointer) {
					generator.lea(rsp, ptr[rsp + GuestRedZoneSize]);
				}
				if (!encoded || Xbyak::GetError() != 0) {
					break;
				}
			}
			if (encoded && Xbyak::GetError() == 0) {
				generator.jmp(reinterpret_cast<void*>(span.continuation));
				if (Xbyak::GetError() == 0) {
					return trampoline_offset;
				}
			}
			const int error = Xbyak::GetError();
			// Roll back before retrying this uncommitted span with smaller trap replacements.
			generator.reset();
			generator.setSize(trampoline_offset);
			if (error != 0 && !HandleTrampolineError(module, error)) {
				EXIT("Guest instruction trampoline generation failed: %s\n",
				     Xbyak::ConvertErrorToString(error));
			}
			if (error != Xbyak::ERR_CODE_IS_TOO_BIG || !has_replacements) {
				break;
			}
		}
		return std::nullopt;
	};

	const auto record_rewrites = [&](const RelocationSpan& span) {
		for (const auto* decoded: span.instructions) {
			module->patched.insert(reinterpret_cast<u8*>(decoded->address));
			const auto rewrite = rewrite_sites.find(decoded->address);
			if (rewrite == rewrite_sites.end()) {
				continue;
			}
			if (rewrite->second.protect_red_zone && decoded->accesses_memory) {
				++result.patched_memory_instruction_count;
			}
			if (rewrite->second.replacement != InstructionReplacement::None) {
				auto& counts = ReplacementCounts(result, rewrite->second.replacement);
				if (UsesInstructionTrap(rewrite->second.replacement, span.trap_replacements)) {
					++counts.trapped;
				} else {
					++counts.native;
				}
			}
		}
	};

	std::vector<std::pair<uintptr_t, uintptr_t>> patched_spans;
	std::vector<uintptr_t>                       relay_slots;
	std::vector<uintptr_t>                       short_relay_slots;
	std::vector<uintptr_t>                       unresolved_sites;
	uintptr_t                                    covered_until {};
	for (auto site_it = rewrite_sites.begin(); site_it != rewrite_sites.end(); ++site_it) {
		const uintptr_t site = site_it->first;
		if (site < covered_until) {
			continue;
		}

		const auto collect_forward_span = [&]() -> std::optional<RelocationSpan> {
			RelocationSpan span {.patch_start = site, .continuation = site};
			while (span.patch_size < NearJumpSize) {
				const auto decoded_it = function.instructions.find(span.continuation);
				if (decoded_it == function.instructions.end() ||
				    (span.continuation != site &&
				     function.branch_targets.contains(span.continuation))) {
					return std::nullopt;
				}

				const auto& decoded = decoded_it->second;
				span.instructions.push_back(&decoded);
				span.patch_size += decoded.instruction.length;
				span.continuation += decoded.instruction.length;
				if (span.patch_size < NearJumpSize &&
				    IsControlFlowTerminator(decoded.instruction)) {
					return std::nullopt;
				}
			}
			return span;
		};

		const auto collect_backward_span = [&]() -> std::optional<RelocationSpan> {
			const auto site_instruction = function.instructions.find(site);
			ASSERT(site_instruction != function.instructions.end());
			RelocationSpan span {
			    .instructions = {&site_instruction->second},
			    .patch_start  = site,
			    .continuation = site + site_instruction->second.instruction.length,
			    .patch_size   = site_instruction->second.instruction.length,
			};

			while (span.patch_size < NearJumpSize) {
				const auto previous_end = function.instructions.lower_bound(span.patch_start);
				if (previous_end == function.instructions.begin() ||
				    function.branch_targets.contains(span.patch_start)) {
					return std::nullopt;
				}

				const auto  previous = std::prev(previous_end);
				const auto& decoded  = previous->second;
				if (previous->first + decoded.instruction.length != span.patch_start ||
				    previous->first < covered_until ||
				    IsControlFlowTerminator(decoded.instruction)) {
					return std::nullopt;
				}

				span.instructions.insert(span.instructions.begin(), &decoded);
				span.patch_start = previous->first;
				span.patch_size += decoded.instruction.length;
			}
			return span;
		};

		std::optional<RelocationSpan> selected_span;
		std::optional<size_t>         trampoline_offset;
		const auto&                   site_instruction       = function.instructions.at(site);
		const bool                    can_relocate_neighbors = !function.has_indirect_branch;
		if (can_relocate_neighbors || site_instruction.instruction.length >= NearJumpSize) {
			auto forward_span = collect_forward_span();
			if (forward_span) {
				trampoline_offset = emit_span(*forward_span);
				if (trampoline_offset) {
					selected_span = std::move(forward_span);
				}
			}
		}
		if (!selected_span && can_relocate_neighbors) {
			if (auto backward_span = collect_backward_span()) {
				trampoline_offset = emit_span(*backward_span);
				if (trampoline_offset) {
					selected_span = std::move(backward_span);
				}
			}
		}
		if (!selected_span) {
			unresolved_sites.push_back(site);
			continue;
		}

		const auto& span       = *selected_span;
		const auto* trampoline = module->trampoline_gen.getCode() + *trampoline_offset;

		auto& patch_gen = module->patch_gen;
		patch_gen.reset();
		patch_gen.setSize(span.patch_start - reinterpret_cast<uintptr_t>(patch_gen.getCode()));
		patch_gen.jmp(trampoline, Xbyak::CodeGenerator::LabelType::T_NEAR);
		patch_gen.nop(span.patch_size - NearJumpSize);
		EXIT_IF(Xbyak::GetError() != 0);

		record_rewrites(span);
		patched_spans.emplace_back(span.patch_start, span.continuation);
		if (span.patch_size >= NearJumpSize * 2) {
			relay_slots.push_back(span.patch_start + NearJumpSize);
		}
		if (span.patch_size >= NearJumpSize + ShortJumpSize) {
			short_relay_slots.push_back(span.patch_start + NearJumpSize);
		}
		covered_until = span.continuation;
	}

	const auto record_unsupported = [&](uintptr_t site) {
		const auto& decoded = function.instructions.at(site);
		const auto& rewrite = rewrite_sites.at(site);
		if (rewrite.protect_red_zone && decoded.accesses_memory) {
			++result.unrelocatable_memory_instruction_count;
		}
	};
	const auto overlaps_patched_span = [&patched_spans](uintptr_t start, uintptr_t end) {
		return std::ranges::any_of(patched_spans, [start, end](const auto& patched) {
			return start < patched.second && patched.first < end;
		});
	};

	for (const uintptr_t site: unresolved_sites) {
		if (module->patched.contains(reinterpret_cast<u8*>(site))) {
			continue;
		}

		const auto site_instruction = function.instructions.find(site);
		ASSERT(site_instruction != function.instructions.end());
		if (function.has_indirect_branch ||
		    site_instruction->second.instruction.length < ShortJumpSize) {
			record_unsupported(site);
			continue;
		}

		RelocationSpan site_span {
		    .instructions = {&site_instruction->second},
		    .patch_start  = site,
		    .continuation = site + site_instruction->second.instruction.length,
		    .patch_size   = site_instruction->second.instruction.length,
		};
		const size_t trampoline_start       = module->trampoline_gen.getSize();
		const auto   site_trampoline_offset = emit_span(site_span);
		if (!site_trampoline_offset) {
			record_unsupported(site);
			continue;
		}

		constexpr s64 ShortJumpMin = std::numeric_limits<s8>::min();
		constexpr s64 ShortJumpMax = std::numeric_limits<s8>::max();
		const auto    relay_slot = std::ranges::find_if(relay_slots, [site](uintptr_t address) {
			const s64 displacement =
			    static_cast<s64>(address) - static_cast<s64>(site + ShortJumpSize);
			return displacement >= ShortJumpMin && displacement <= ShortJumpMax;
		});
		if (relay_slot != relay_slots.end()) {
			const auto* site_trampoline =
			    module->trampoline_gen.getCode() + *site_trampoline_offset;

			auto& patch_gen = module->patch_gen;
			patch_gen.reset();
			patch_gen.setSize(*relay_slot - reinterpret_cast<uintptr_t>(patch_gen.getCode()));
			patch_gen.jmp(site_trampoline, Xbyak::CodeGenerator::LabelType::T_NEAR);
			EXIT_IF(Xbyak::GetError() != 0);

			patch_gen.reset();
			patch_gen.setSize(site - reinterpret_cast<uintptr_t>(patch_gen.getCode()));
			patch_gen.jmp(reinterpret_cast<void*>(*relay_slot),
			              Xbyak::CodeGenerator::LabelType::T_SHORT);
			patch_gen.nop(site_span.patch_size - ShortJumpSize);
			EXIT_IF(Xbyak::GetError() != 0);

			record_rewrites(site_span);
			patched_spans.emplace_back(site_span.patch_start, site_span.continuation);
			std::erase(short_relay_slots, *relay_slot);
			relay_slots.erase(relay_slot);
			continue;
		}

		const auto find_host = [&](uintptr_t short_jump_address) {
			std::pair<std::optional<RelocationSpan>, std::optional<size_t>> result;
			const s64 minimum_host = static_cast<s64>(short_jump_address) +
			                         static_cast<s64>(ShortJumpSize) -
			                         static_cast<s64>(NearJumpSize) + ShortJumpMin;
			const s64 maximum_host = static_cast<s64>(short_jump_address) +
			                         static_cast<s64>(ShortJumpSize) -
			                         static_cast<s64>(NearJumpSize) + ShortJumpMax;

			auto candidate = function.instructions.lower_bound(
			    static_cast<uintptr_t>(std::max<s64>(minimum_host, 0)));
			for (; candidate != function.instructions.end() &&
			       static_cast<s64>(candidate->first) <= maximum_host;
			     ++candidate) {
				RelocationSpan span {.patch_start  = candidate->first,
				                     .continuation = candidate->first};
				while (span.patch_size < NearJumpSize * 2) {
					const auto instruction = function.instructions.find(span.continuation);
					if (instruction == function.instructions.end() ||
					    (span.continuation != span.patch_start &&
					     function.branch_targets.contains(span.continuation))) {
						span.instructions.clear();
						break;
					}
					span.instructions.push_back(&instruction->second);
					span.patch_size += instruction->second.instruction.length;
					span.continuation += instruction->second.instruction.length;
					if (span.patch_size < NearJumpSize * 2 &&
					    IsControlFlowTerminator(instruction->second.instruction)) {
						span.instructions.clear();
						break;
					}
				}
				if (span.instructions.empty() ||
				    !(span.continuation <= site ||
				      span.patch_start >= site_span.continuation) ||
				    overlaps_patched_span(span.patch_start, span.continuation)) {
					continue;
				}

				const uintptr_t relay_address = span.patch_start + NearJumpSize;
				const s64 displacement = static_cast<s64>(relay_address) -
				                         static_cast<s64>(short_jump_address + ShortJumpSize);
				if (displacement < ShortJumpMin || displacement > ShortJumpMax) {
					continue;
				}

				const auto offset = emit_span(span);
				if (!offset) {
					continue;
				}
				result.first  = std::move(span);
				result.second = offset;
				break;
			}
			return result;
		};

		auto [host_span, host_trampoline_offset] = find_host(site);
		std::optional<uintptr_t>       final_relay_slot;
		uintptr_t                      final_short_jump = site;
		std::map<uintptr_t, uintptr_t> relay_parent {{site, site}};
		std::vector<uintptr_t>         relay_queue {site};
		for (size_t queue_index = 0;
		     !host_span && !final_relay_slot && queue_index < relay_queue.size();
		     ++queue_index) {
			const uintptr_t current = relay_queue[queue_index];
			if (current != site) {
				if (std::ranges::find(relay_slots, current) != relay_slots.end()) {
					final_relay_slot = current;
					final_short_jump = relay_parent.at(current);
					break;
				}
				const auto near_slot =
				    std::ranges::find_if(relay_slots, [current](uintptr_t address) {
					    const s64 displacement = static_cast<s64>(address) -
					                             static_cast<s64>(current + ShortJumpSize);
					    return address != current && displacement >= ShortJumpMin &&
					           displacement <= ShortJumpMax;
				    });
				if (near_slot != relay_slots.end()) {
					final_relay_slot = *near_slot;
					final_short_jump = current;
					break;
				}

				auto bridge_host = find_host(current);
				if (bridge_host.first) {
					host_span              = std::move(bridge_host.first);
					host_trampoline_offset = bridge_host.second;
					final_short_jump       = current;
					break;
				}
			}

			for (const uintptr_t slot: short_relay_slots) {
				if (relay_parent.contains(slot)) {
					continue;
				}
				const s64 displacement =
				    static_cast<s64>(slot) - static_cast<s64>(current + ShortJumpSize);
				if (displacement < ShortJumpMin || displacement > ShortJumpMax) {
					continue;
				}
				relay_parent.emplace(slot, current);
				relay_queue.push_back(slot);
			}
		}

		if (!host_span && !final_relay_slot) {
			module->trampoline_gen.setSize(trampoline_start);
			record_unsupported(site);
			continue;
		}

		const auto* site_trampoline =
		    module->trampoline_gen.getCode() + *site_trampoline_offset;
		auto&     patch_gen = module->patch_gen;
		uintptr_t relay_address {};
		if (final_relay_slot) {
			relay_address = *final_relay_slot;
			patch_gen.reset();
			patch_gen.setSize(relay_address - reinterpret_cast<uintptr_t>(patch_gen.getCode()));
			patch_gen.jmp(site_trampoline, Xbyak::CodeGenerator::LabelType::T_NEAR);
			EXIT_IF(Xbyak::GetError() != 0);
			std::erase(relay_slots, relay_address);
			std::erase(short_relay_slots, relay_address);
		} else {
			ASSERT(host_span && host_trampoline_offset);
			const auto* host_trampoline =
			    module->trampoline_gen.getCode() + *host_trampoline_offset;
			relay_address = host_span->patch_start + NearJumpSize;

			patch_gen.reset();
			patch_gen.setSize(host_span->patch_start -
			                  reinterpret_cast<uintptr_t>(patch_gen.getCode()));
			patch_gen.jmp(host_trampoline, Xbyak::CodeGenerator::LabelType::T_NEAR);
			patch_gen.jmp(site_trampoline, Xbyak::CodeGenerator::LabelType::T_NEAR);
			patch_gen.nop(host_span->patch_size - NearJumpSize * 2);
			EXIT_IF(Xbyak::GetError() != 0);
		}

		uintptr_t jump_target = relay_address;
		while (final_short_jump != site) {
			patch_gen.reset();
			patch_gen.setSize(final_short_jump -
			                  reinterpret_cast<uintptr_t>(patch_gen.getCode()));
			patch_gen.jmp(reinterpret_cast<void*>(jump_target),
			              Xbyak::CodeGenerator::LabelType::T_SHORT);
			EXIT_IF(Xbyak::GetError() != 0);
			jump_target       = final_short_jump;
			const auto parent = relay_parent.find(final_short_jump);
			ASSERT(parent != relay_parent.end());
			final_short_jump = parent->second;
			std::erase(relay_slots, jump_target);
			std::erase(short_relay_slots, jump_target);
		}

		patch_gen.reset();
		patch_gen.setSize(site - reinterpret_cast<uintptr_t>(patch_gen.getCode()));
		patch_gen.jmp(reinterpret_cast<void*>(jump_target),
		              Xbyak::CodeGenerator::LabelType::T_SHORT);
		patch_gen.nop(site_span.patch_size - ShortJumpSize);
		EXIT_IF(Xbyak::GetError() != 0);

		if (host_span) {
			record_rewrites(*host_span);
			patched_spans.emplace_back(host_span->patch_start, host_span->continuation);
			if (host_span->patch_size >= NearJumpSize * 3) {
				relay_slots.push_back(host_span->patch_start + NearJumpSize * 2);
			}
			if (host_span->patch_size >= NearJumpSize * 2 + ShortJumpSize) {
				short_relay_slots.push_back(host_span->patch_start + NearJumpSize * 2);
			}
		}
		record_rewrites(site_span);
		patched_spans.emplace_back(site_span.patch_start, site_span.continuation);
	}
}

// Instruction starts and direct branch targets of a code segment, from a linear sweep. Clang keeps
// jump tables and other data out of the text, so the sweep stays in step with the code.
struct CodeSweep {
	uintptr_t              start {};
	uintptr_t              end {};
	std::vector<u64>       boundaries;
	std::vector<u64>       targets;
	std::vector<u64>       stack_below; // [rsp - d] or [rbp - d] operands: possible red-zone uses
	std::vector<uintptr_t> references;

	static void SetBit(std::vector<u64>& bits, size_t index) {
		std::atomic_ref(bits[index / 64])
		    .fetch_or(u64 {1} << (index % 64), std::memory_order_relaxed);
	}
	[[nodiscard]] bool TestBit(const std::vector<u64>& bits, uintptr_t address) const {
		if (address < start || address >= end) {
			return false;
		}
		const size_t index = address - start;
		return (bits[index / 64] & (u64 {1} << (index % 64))) != 0;
	}
	[[nodiscard]] bool IsBoundary(uintptr_t address) const { return TestBit(boundaries, address); }
	[[nodiscard]] bool IsBranchTarget(uintptr_t address) const { return TestBit(targets, address); }
	// Whether an instruction in [first, last) addresses memory below RSP or RBP.
	[[nodiscard]] bool MayUseRedZone(uintptr_t first, uintptr_t last) const {
		first = std::max(first, start);
		last  = std::min(last, end);
		for (uintptr_t address = first; address < last;) {
			const size_t index = address - start;
			const u64    word  = stack_below[index / 64] >> (index % 64);
			if (word != 0) {
				return address + std::countr_zero(word) < last;
			}
			address += 64 - index % 64;
		}
		return false;
	}
};

void SweepPiece(CodeSweep& sweep, uintptr_t piece_start, uintptr_t piece_end,
                std::vector<uintptr_t>& references) {
	ZydisDecodedInstruction instruction {};
	for (uintptr_t address = piece_start; address < piece_end;) {
		if (!ZYAN_SUCCESS(ZydisDecoderDecodeInstruction(&GetDecoder(), nullptr,
		                                                reinterpret_cast<const void*>(address),
		                                                sweep.end - address, &instruction))) {
			++address;
			continue;
		}
		CodeSweep::SetBit(sweep.boundaries, address - sweep.start);
		if ((instruction.attributes & ZYDIS_ATTRIB_HAS_MODRM) != 0 &&
		    instruction.raw.modrm.mod != 3 && instruction.raw.disp.value < 0) {
			const bool sib  = instruction.raw.modrm.rm == 4;
			const u8   base = sib ? instruction.raw.sib.base : instruction.raw.modrm.rm;
			// RSP (4) or RBP (5); base 5 with mod 0 is RIP or none. REX.B selects R12/R13 instead;
			// VEX/EVEX/XOP bases are not told apart here (R12/R13 bases only cost an analysis).
			const bool rex_base =
			    (instruction.attributes & ZYDIS_ATTRIB_HAS_REX) != 0 && instruction.raw.rex.B != 0;
			if (!rex_base && (base == 4 || (base == 5 && instruction.raw.modrm.mod != 0))) {
				CodeSweep::SetBit(sweep.stack_below, address - sweep.start);
			}
		}
		const uintptr_t next     = address + instruction.length;
		const auto      category = instruction.meta.category;
		if ((category == ZYDIS_CATEGORY_CALL || category == ZYDIS_CATEGORY_COND_BR ||
		     category == ZYDIS_CATEGORY_UNCOND_BR) &&
		    instruction.raw.imm[0].is_relative) {
			const uintptr_t target = next + static_cast<uintptr_t>(instruction.raw.imm[0].value.s);
			if (target >= sweep.start && target < sweep.end) {
				CodeSweep::SetBit(sweep.targets, target - sweep.start);
				if (category == ZYDIS_CATEGORY_CALL) {
					references.push_back(target);
				}
			}
		} else if (instruction.mnemonic == ZYDIS_MNEMONIC_LEA &&
		           (instruction.attributes & ZYDIS_ATTRIB_HAS_MODRM) != 0 &&
		           instruction.raw.modrm.mod == 0 && instruction.raw.modrm.rm == 5) {
			// RIP-relative LEA: a function pointer taken in code.
			const uintptr_t target = next + static_cast<uintptr_t>(instruction.raw.disp.value);
			if (target >= sweep.start && target < sweep.end) {
				references.push_back(target);
			}
		}
		address = next;
	}
}

// Sweeps [start, end) in pieces that begin at known instruction starts (sorted), on several
// threads.
CodeSweep SweepCode(uintptr_t start, uintptr_t end, std::span<const uintptr_t> known_starts,
                    size_t thread_count) {
	CodeSweep    sweep {.start = start, .end = end};
	const size_t words = (end - start + 63) / 64;
	sweep.boundaries.assign(words, 0);
	sweep.targets.assign(words, 0);
	sweep.stack_below.assign(words, 0);

	std::vector<uintptr_t> pieces {start};
	for (size_t piece = 1; piece < thread_count; ++piece) {
		const uintptr_t wanted = start + (end - start) / thread_count * piece;
		const auto      known  = std::ranges::lower_bound(known_starts, wanted);
		if (known != known_starts.end() && *known > pieces.back() && *known < end) {
			pieces.push_back(*known);
		}
	}
	pieces.push_back(end);
	std::vector<std::vector<uintptr_t>> references(pieces.size() - 1);
	const auto                          sweep_piece = [&](size_t piece) {
		SweepPiece(sweep, pieces[piece], pieces[piece + 1], references[piece]);
	};
	std::vector<std::thread> threads;
	for (size_t piece = 1; piece + 1 < pieces.size(); ++piece) {
		threads.emplace_back(sweep_piece, piece);
	}
	sweep_piece(0);
	for (auto& thread: threads) {
		thread.join();
	}
	for (const auto& piece_references: references) {
		sweep.references.insert(sweep.references.end(), piece_references.begin(),
		                        piece_references.end());
	}
	// Only references that land on an instruction start can be function entries.
	std::erase_if(sweep.references, [&](uintptr_t target) { return !sweep.IsBoundary(target); });
	return sweep;
}

// Whole-function red-zone protection. Windows writes its exception frame directly below the
// interrupted RSP, so a guest fault inside a function that keeps data in its SysV red zone
// destroys that data (only hosts with unused XSAVE components at the top of the frame, such as
// AVX-512 hosts with clean ZMM state, happen to leave it intact). A protected function runs from
// a copy in which RSP stays 128 bytes below the guest value: red-zone bytes then lie above RSP,
// out of reach of an exception frame.
// - Instructions that use RSP as a register or implicitly (PUSH, POP, RET, LEAVE, SUB RSP,
//   MOV RBP, RSP, ...) run unchanged with the guest RSP restored around them. They access only
//   the stack, which never takes a handled fault.
// - Explicit stack operands [rsp + d] are re-encoded as [rsp + d + 128].
// - Branch targets inside the copy are entered with the lowered RSP. Every way out of the copy
//   (RET, branches outside the function, indirect jumps, calls into guest code, falling off the
//   end) restores the guest RSP first and continues at the original address, so code outside the
//   copy, including the unmodified original body, sees exactly the original state.
// The original entry is replaced by a jump to the copy.
class RedZoneFunctionWriter {
public:
	enum class Outcome {
		Rewritten,
		EntryConflict,
		Unsupported,
		NoSpace,
	};

	RedZoneFunctionWriter(PatchModule* module, const DecodedFunction& function,
	                      uintptr_t function_start, uintptr_t function_end, const CodeSweep& sweep,
	                      const std::map<uintptr_t, InstructionRewrite>& rewrite_sites,
	                      u64                                            tsc_multiplier)
	    : m_module(module), m_function(function), m_start(function_start), m_end(function_end),
	      m_sweep(sweep), m_rewrite_sites(rewrite_sites), m_tsc_multiplier(tsc_multiplier),
	      m_gen(module->trampoline_gen) {}

	Outcome Write(GuestInstructionPatchResult& result) {
		const auto entry_end = EntryPatchEnd();
		if (!entry_end) {
			return Outcome::EntryConflict;
		}
		const size_t offset = m_gen.getSize();
		Xbyak::ClearError();
		// Jump tables of the copy come first: their loads in the body need their addresses.
		m_tables.clear();
		for (const auto& [branch, table]: m_function.jump_tables) {
			m_gen.align(sizeof(s32));
			m_tables.emplace(table.load, CopiedTable {&table, m_gen.getCurr()});
			for (size_t index = 0; index < table.entries.size(); ++index) {
				m_gen.dd(0);
			}
		}
		const auto* copy    = m_gen.getCurr();
		bool        written = false;
		int         error   = 0;
		{
			std::map<uintptr_t, Xbyak::Label> labels;
			for (const uintptr_t target: m_function.branch_targets) {
				if (m_function.instructions.contains(target)) {
					labels[target];
				}
			}
			written = EmitBody(labels) && Xbyak::GetError() == 0 && !m_gen.hasUndefinedLabel();
			error   = Xbyak::GetError();
			if (written) {
				// Entries relative to the original table base, which the code still adds.
				for (const auto& [load, copied]: m_tables) {
					auto* entry = const_cast<u8*>(copied.entries);
					for (const uintptr_t target: copied.table->entries) {
						const auto label = labels.find(target);
						if (label == labels.end()) {
							written = false;
							break;
						}
						const auto offset_to_target =
						    reinterpret_cast<intptr_t>(label->second.getAddress()) -
						    static_cast<intptr_t>(copied.table->address);
						const auto value = static_cast<s32>(offset_to_target);
						written &= value == offset_to_target;
						std::memcpy(entry, &value, sizeof(value));
						entry += sizeof(value);
					}
				}
			}
			if (!written) {
				// Detach the labels before the rollback below forgets them.
				m_gen.reset();
			}
		}
		if (!written) {
			m_gen.reset();
			m_gen.setSize(offset);
			Xbyak::ClearError();
			if (error != 0 && !HandleTrampolineError(m_module, error)) {
				EXIT("Guest red-zone function copy failed: %s\n",
				     Xbyak::ConvertErrorToString(error));
			}
			return error == Xbyak::ERR_CODE_IS_TOO_BIG ? Outcome::NoSpace : Outcome::Unsupported;
		}

		auto& patch_gen = m_module->patch_gen;
		patch_gen.reset();
		patch_gen.setSize(m_start - reinterpret_cast<uintptr_t>(patch_gen.getCode()));
		patch_gen.jmp(copy, Xbyak::CodeGenerator::LabelType::T_NEAR);
		for (uintptr_t address = m_start + NearJumpSize; address < *entry_end; ++address) {
			patch_gen.db(0xcc);
		}
		EXIT_IF(Xbyak::GetError() != 0);
		m_module->patched.insert(reinterpret_cast<u8*>(m_start));
		result.rewritten_function_bytes += m_gen.getSize() - offset;
		result.unprotected_instruction_count += m_unprotected;
		result.partially_rewritten_function_count += m_leaves_early ? 1 : 0;
		for (const auto& [address, rewrite]: m_rewrite_sites) {
			if (rewrite.replacement != InstructionReplacement::None &&
			    m_function.instructions.contains(address)) {
				++ReplacementCounts(result, rewrite.replacement).native;
			}
		}
		return Outcome::Rewritten;
	}

private:
	struct CopiedTable {
		const JumpTable* table {};
		const u8*        entries {};
	};

	struct Shape {
		bool stack_register {}; // RSP read or written as a register, or implicitly.
		bool stack_operand {};  // An explicit [rsp + ...] operand.
		bool other_memory {};   // A data access that is not on the stack.
		bool rip_relative {};
		bool relative {}; // A relative immediate (branches).
	};

	// End of the instructions the entry jump overwrites, if no other code can land inside them.
	[[nodiscard]] std::optional<uintptr_t> EntryPatchEnd() const {
		uintptr_t address = m_start;
		while (address < m_start + NearJumpSize) {
			const auto decoded = m_function.instructions.find(address);
			if (decoded == m_function.instructions.end() || address >= m_end) {
				return std::nullopt;
			}
			address += decoded->second.instruction.length;
		}
		for (uintptr_t inside = m_start + 1; inside < address; ++inside) {
			if (m_function.branch_targets.contains(inside) || m_sweep.IsBranchTarget(inside)) {
				return std::nullopt;
			}
		}
		return address;
	}

	void Lower() {
		if (!m_lowered) {
			m_gen.lea(rsp, ptr[rsp - GuestRedZoneSize]);
			m_lowered = true;
		}
	}

	void Raise() {
		if (m_lowered) {
			m_gen.lea(rsp, ptr[rsp + GuestRedZoneSize]);
			m_lowered = false;
		}
	}

	// Leaves the copy: continue at an original address with the guest RSP.
	bool Exit(uintptr_t target) {
		if (target > m_start && target < m_start + NearJumpSize) {
			return false;
		}
		Raise();
		m_gen.jmp(reinterpret_cast<const void*>(target), Xbyak::CodeGenerator::LabelType::T_NEAR);
		return true;
	}

	void EmitConditionalJump(u8 opcode, const Xbyak::Label& label) {
		constexpr auto Near = Xbyak::CodeGenerator::LabelType::T_NEAR;
		switch (opcode & 0x0f) {
			case 0x0: m_gen.jo(label, Near); break;
			case 0x1: m_gen.jno(label, Near); break;
			case 0x2: m_gen.jb(label, Near); break;
			case 0x3: m_gen.jnb(label, Near); break;
			case 0x4: m_gen.jz(label, Near); break;
			case 0x5: m_gen.jnz(label, Near); break;
			case 0x6: m_gen.jbe(label, Near); break;
			case 0x7: m_gen.jnbe(label, Near); break;
			case 0x8: m_gen.js(label, Near); break;
			case 0x9: m_gen.jns(label, Near); break;
			case 0xa: m_gen.jp(label, Near); break;
			case 0xb: m_gen.jnp(label, Near); break;
			case 0xc: m_gen.jl(label, Near); break;
			case 0xd: m_gen.jnl(label, Near); break;
			case 0xe: m_gen.jle(label, Near); break;
			default: m_gen.jnle(label, Near); break;
		}
	}

	static Shape Classify(const DecodedCodeInstruction& decoded) {
		Shape       shape {};
		const auto& instruction = decoded.instruction;
		shape.relative          = GetRelativeTarget(decoded) != 0;
		for (u8 index = 0; index < instruction.operand_count; ++index) {
			const auto& operand = decoded.operands[index];
			if (operand.type == ZYDIS_OPERAND_TYPE_REGISTER) {
				shape.stack_register |= IsStackPointerRegister(operand.reg.value);
				continue;
			}
			if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY) {
				continue;
			}
			if (IsStackPointerRegister(operand.mem.index) ||
			    (IsStackPointerRegister(operand.mem.base) &&
			     index >= instruction.operand_count_visible)) {
				shape.stack_register = true;
			} else if (IsStackPointerRegister(operand.mem.base)) {
				shape.stack_operand = true;
			} else {
				shape.rip_relative |= operand.mem.base == ZYDIS_REGISTER_RIP ||
				                      operand.mem.base == ZYDIS_REGISTER_EIP;
				shape.other_memory |= instruction.mnemonic != ZYDIS_MNEMONIC_LEA &&
				                      instruction.mnemonic != ZYDIS_MNEMONIC_NOP &&
				                      operand.mem.type != ZYDIS_MEMOP_TYPE_AGEN;
			}
		}
		const auto category = instruction.meta.category;
		shape.stack_register |= category == ZYDIS_CATEGORY_CALL || category == ZYDIS_CATEGORY_RET ||
		                        category == ZYDIS_CATEGORY_SYSCALL ||
		                        category == ZYDIS_CATEGORY_SYSTEM ||
		                        category == ZYDIS_CATEGORY_INTERRUPT;
		return shape;
	}

	// Copies an instruction, fixing RIP-relative operands and (with adjust) stack operands.
	bool EmitInstruction(const DecodedCodeInstruction& decoded, const Shape& shape, bool adjust) {
		if (shape.rip_relative || adjust) {
			return EncodeRelocatedInstruction(decoded, m_gen, adjust ? GuestRedZoneSize : 0);
		}
		m_gen.db(reinterpret_cast<const u8*>(decoded.address), decoded.instruction.length);
		return true;
	}

	bool EmitReplacement(const DecodedCodeInstruction& decoded, InstructionReplacement replacement,
	                     const Shape& shape) {
		switch (replacement) {
			case InstructionReplacement::ReciprocalSquareRoot:
				Lower();
				GenerateReciprocalSquareRoot(decoded, m_gen);
				return true;
			case InstructionReplacement::ExtractQ:
				Lower();
				GenerateExtractQ(decoded, m_gen);
				return true;
			case InstructionReplacement::ReadTimeStampCounter:
				GenerateReadTimeStampCounter(decoded, m_tsc_multiplier, m_gen);
				return true;
			case InstructionReplacement::CacheLineWriteBack:
				Lower();
				return EncodeRelocatedInstruction(decoded, m_gen,
				                                  shape.stack_operand ? GuestRedZoneSize : 0,
				                                  ZYDIS_MNEMONIC_CLFLUSH);
			default:
				// Replacements that rely on a trap keep the original function.
				return false;
		}
	}

	static bool IsCounterBranch(ZydisMnemonic mnemonic) {
		return mnemonic == ZYDIS_MNEMONIC_JRCXZ || mnemonic == ZYDIS_MNEMONIC_JECXZ ||
		       mnemonic == ZYDIS_MNEMONIC_JCXZ || mnemonic == ZYDIS_MNEMONIC_LOOP ||
		       mnemonic == ZYDIS_MNEMONIC_LOOPE || mnemonic == ZYDIS_MNEMONIC_LOOPNE;
	}

	bool EmitBody(std::map<uintptr_t, Xbyak::Label>& labels) {
		struct Stub {
			Xbyak::Label label;
			uintptr_t    target {};
		};
		std::vector<std::unique_ptr<Stub>> stubs;
		// Address the previous instruction continues at, while control can fall through.
		std::optional<uintptr_t> fallthrough = m_start;

		for (const auto& [address, decoded]: m_function.instructions) {
			if (fallthrough && *fallthrough != address) {
				// The previous instruction continues in bytes that were not decoded.
				if (!Exit(*fallthrough)) {
					return false;
				}
				fallthrough.reset();
			}
			if (const auto label = labels.find(address); label != labels.end()) {
				if (fallthrough) {
					Lower();
				}
				m_lowered = true;
				m_gen.L(label->second);
			} else if (!fallthrough) {
				// Reached only from outside the copy: never executed here.
				m_lowered = true;
			}
			fallthrough.reset();

			const auto&     instruction = decoded.instruction;
			const auto      category    = instruction.meta.category;
			const Shape     shape       = Classify(decoded);
			const uintptr_t next        = address + instruction.length;
			const uintptr_t target      = GetRelativeTarget(decoded);

			if (const auto rewrite = m_rewrite_sites.find(address);
			    rewrite != m_rewrite_sites.end() &&
			    rewrite->second.replacement != InstructionReplacement::None) {
				if (!EmitReplacement(decoded, rewrite->second.replacement, shape)) {
					return false;
				}
				fallthrough = next;
				continue;
			}

			if (category == ZYDIS_CATEGORY_COND_BR) {
				if (!shape.relative || IsCounterBranch(instruction.mnemonic) ||
				    (target > m_start && target < m_start + NearJumpSize)) {
					return false;
				}
				Lower();
				if (const auto label = labels.find(target); label != labels.end()) {
					EmitConditionalJump(instruction.opcode, label->second);
				} else {
					auto& stub   = stubs.emplace_back(std::make_unique<Stub>());
					stub->target = target;
					EmitConditionalJump(instruction.opcode, stub->label);
				}
				fallthrough = next;
				continue;
			}

			if (category == ZYDIS_CATEGORY_UNCOND_BR) {
				if (shape.relative) {
					if (const auto label = labels.find(target); label != labels.end()) {
						Lower();
						m_gen.jmp(label->second, Xbyak::CodeGenerator::LabelType::T_NEAR);
					} else if (!Exit(target)) {
						return false;
					}
				} else if (m_function.jump_tables.contains(address)) {
					// The copied table sends the jump to a label of the copy.
					Lower();
					if (!EmitInstruction(decoded, shape, false)) {
						return false;
					}
				} else {
					// Other indirect jumps (tail calls) continue in original code.
					Raise();
					m_leaves_early = true;
					m_unprotected += shape.other_memory ? 1 : 0;
					if (!EmitInstruction(decoded, shape, false)) {
						return false;
					}
				}
				continue;
			}

			if (category == ZYDIS_CATEGORY_CALL) {
				if (shape.relative && (target < reinterpret_cast<uintptr_t>(m_module->start) ||
				                       target >= reinterpret_cast<uintptr_t>(m_module->end))) {
					// Emulator stubs patched into guest code (thread-local storage access) take no
					// stack arguments: call them with the lowered RSP, so that their return address
					// and spills stay below the red zone.
					Lower();
					m_gen.call(reinterpret_cast<const void*>(target));
					fallthrough = next;
					continue;
				}
				// A call into guest code runs from the original instruction: the callee returns to
				// original code and unwinders see original return addresses.
				m_leaves_early = true;
				if (!Exit(address)) {
					return false;
				}
				continue;
			}

			if (category == ZYDIS_CATEGORY_RET) {
				Raise();
				m_gen.db(reinterpret_cast<const u8*>(address), instruction.length);
				continue;
			}

			if (shape.relative) {
				// Other relative forms (XBEGIN) would reach original code with the lowered RSP.
				return false;
			}

			if (const auto table = m_tables.find(address); table != m_tables.end()) {
				Lower();
				if (!EncodeRelocatedInstruction(
				        decoded, m_gen, 0, ZYDIS_MNEMONIC_INVALID,
				        reinterpret_cast<intptr_t>(table->second.entries) -
				            static_cast<intptr_t>(table->second.table->address))) {
					return false;
				}
				fallthrough = next;
				continue;
			}

			if (shape.stack_register) {
				Raise();
				m_unprotected += shape.other_memory ? 1 : 0;
				if (!EmitInstruction(decoded, shape, false)) {
					return false;
				}
			} else if (shape.stack_operand) {
				Lower();
				if (!EmitInstruction(decoded, shape, true)) {
					return false;
				}
			} else {
				if (shape.other_memory) {
					Lower();
				}
				if (!EmitInstruction(decoded, shape, false)) {
					return false;
				}
			}
			if (!IsControlFlowTerminator(instruction)) {
				fallthrough = next;
			}
		}
		if (fallthrough && !Exit(*fallthrough)) {
			return false;
		}
		for (auto& stub: stubs) {
			m_gen.L(stub->label);
			m_lowered = true;
			if (!Exit(stub->target)) {
				return false;
			}
		}
		return true;
	}

	PatchModule*                                   m_module;
	const DecodedFunction&                         m_function;
	uintptr_t                                      m_start;
	uintptr_t                                      m_end;
	const CodeSweep&                               m_sweep;
	const std::map<uintptr_t, InstructionRewrite>& m_rewrite_sites;
	u64                                            m_tsc_multiplier;
	Xbyak::CodeGenerator&                          m_gen;
	bool                                           m_lowered      = false;
	bool                                           m_leaves_early = false;
	u64                                            m_unprotected  = 0;
	std::map<uintptr_t, CopiedTable>               m_tables; // by load address
};

size_t AnalysisThreadCount() {
	return std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 16);
}

// Whether a fault-capable access may run while red-zone bytes are live. When the liveness of a
// function is unknown (stack realignment, dynamic allocation, unresolved jumps), calls into guest
// code decide: compilers keep data in the red zone only in functions without calls (clang and GCC
// use it in leaf functions only), so such functions are not copied.
bool NeedsRedZoneFunctionCopy(const DecodedFunction& function, const PatchModule& module) {
	if (!function.uses_red_zone ||
	    std::ranges::none_of(function.instructions, [](const auto& entry) {
		    return entry.second.accesses_memory && entry.second.red_zone_live.any();
	    })) {
		return false;
	}
	if (function.red_zone_liveness_exact) {
		return true;
	}
	return std::ranges::none_of(function.instructions, [&](const auto& entry) {
		const auto& decoded = entry.second;
		if (decoded.instruction.meta.category != ZYDIS_CATEGORY_CALL) {
			return false;
		}
		const auto* target = reinterpret_cast<const u8*>(GetRelativeTarget(decoded));
		return target == nullptr || (target >= module.start && target < module.end);
	});
}
} // namespace

GuestInstructionHostFeatures GetGuestInstructionHostFeatures() {
	static const auto features = [] {
		const Xbyak::util::Cpu cpu;
		uint32_t               leaf[4] {};
		Xbyak::util::Cpu::getCpuid(0, leaf);
		bool rdpid = false;
		if (leaf[0] >= 7) {
			Xbyak::util::Cpu::getCpuidEx(7, 0, leaf);
			rdpid = (leaf[2] & (1u << 22)) != 0;
		}
		return GuestInstructionHostFeatures {cpu.has(Xbyak::util::Cpu::tSSE4a), rdpid,
		                                     cpu.has(Xbyak::util::Cpu::tCLWB)};
	}();
	return features;
}

GuestInstructionPatchResult PatchGuestInstructions(u64 segment_addr, u64 segment_size,
                                                   std::span<const uintptr_t> function_starts,
                                                   bool protect_memory, bool emulate_amd,
                                                   GuestInstructionHostFeatures host_features,
                                                   u64 tsc_multiplier, bool copy_red_zone_functions,
                                                   AddressRanges data_ranges) {
	GuestInstructionPatchResult result {};
	const bool tsc_only = !protect_memory && !emulate_amd && !copy_red_zone_functions;
	if (tsc_only && tsc_multiplier == 0) {
		return result;
	}
	auto*              module = GetContainingModule(reinterpret_cast<void*>(segment_addr));
	if (module == nullptr || (function_starts.empty() && !copy_red_zone_functions)) {
		return result;
	}

	const uintptr_t        segment_end = segment_addr + segment_size;
	std::vector<uintptr_t> starts;
	starts.reserve(function_starts.size());
	for (const uintptr_t start: function_starts) {
		if (start >= segment_addr && start < segment_end) {
			starts.push_back(start);
		}
	}
	std::optional<CodeSweep> sweep;
	if (copy_red_zone_functions) {
		// Unwind tables list only some functions (a large title had 17.6k FDEs for ~400k
		// functions): add direct call targets and function pointers taken in code.
		std::ranges::sort(starts);
		starts.erase(std::ranges::unique(starts).begin(), starts.end());
		sweep              = SweepCode(segment_addr, segment_end, starts, AnalysisThreadCount());
		const size_t known = starts.size();
		starts.insert(starts.end(), sweep->references.begin(), sweep->references.end());
		std::ranges::sort(starts);
		starts.erase(std::ranges::unique(starts).begin(), starts.end());
		result.discovered_function_count = starts.size() - std::min(known, starts.size());
	}
	std::ranges::sort(starts);
	const auto unique_end = std::ranges::unique(starts).begin();
	starts.erase(unique_end, starts.end());

	std::unique_lock lock {module->mutex};
	const size_t     trampoline_begin = module->trampoline_gen.getSize();
	bool             analyze_red_zone = protect_memory || copy_red_zone_functions;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	analyze_red_zone |= emulate_amd;
#endif
	// Decode and analyze on several threads; patch in address order on this one.
	struct FunctionWork {
		size_t                                  index {};
		DecodedFunction                         function;
		std::map<uintptr_t, InstructionRewrite> replacements;
		bool                                    copy {};
	};
	// Only the red-zone copy is wanted: functions without operands below RSP/RBP need nothing.
	const bool copy_only =
	    copy_red_zone_functions && !protect_memory && !emulate_amd && tsc_multiplier == 0;
	const auto function_end_of = [&](size_t index) {
		return index + 1 < starts.size() ? starts[index + 1] : segment_end;
	};
	const size_t                             thread_count = AnalysisThreadCount();
	std::vector<std::vector<FunctionWork>>   thread_work(thread_count);
	std::vector<GuestInstructionPatchResult> thread_results(thread_count);
	std::atomic<size_t>                      next_index {0};
	constexpr size_t                         Batch   = 256;
	const auto                               analyze = [&](size_t thread_index) {
		auto& work   = thread_work[thread_index];
		auto& counts = thread_results[thread_index];
		for (size_t first = next_index.fetch_add(Batch); first < starts.size();
		     first        = next_index.fetch_add(Batch)) {
			for (size_t index = first; index < std::min(first + Batch, starts.size()); ++index) {
				const uintptr_t function_start = starts[index];
				const uintptr_t function_end   = function_end_of(index);
				if (function_end <= function_start ||
				    (tsc_only && !MayReadTimeStampCounter(function_start, function_end))) {
					continue;
				}
				++counts.function_count;
				if (copy_only && !sweep->MayUseRedZone(function_start, function_end)) {
					continue;
				}
				FunctionWork item {.index = index};
				item.function  = DecodeFunction(function_start, function_end, segment_addr,
				                                segment_end, data_ranges);
				auto& function = item.function;
				if (analyze_red_zone) {
					AnalyzeRedZoneLiveness(function);
				}
				counts.instruction_count += function.instructions.size();
				if (function.uses_red_zone) {
					++counts.red_zone_function_count;
					counts.indirect_red_zone_function_count += function.has_indirect_branch;
				}
				if (emulate_amd) {
					CollectAmdInstructions(function, item.replacements, counts, host_features);
				}
				if (tsc_multiplier != 0) {
					CollectTscInstructions(function, item.replacements, counts);
				}
				item.copy = copy_red_zone_functions && NeedsRedZoneFunctionCopy(function, *module);
				if (item.copy || !item.replacements.empty() ||
				    (protect_memory && function.uses_red_zone)) {
					work.push_back(std::move(item));
				}
			}
		}
	};
	{
		std::vector<std::thread> threads;
		for (size_t thread_index = 1; thread_index < thread_count; ++thread_index) {
			threads.emplace_back(analyze, thread_index);
		}
		analyze(0);
		for (auto& thread: threads) {
			thread.join();
		}
	}
	std::vector<FunctionWork*> ordered;
	for (size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
		const auto& counts = thread_results[thread_index];
		result.function_count += counts.function_count;
		result.instruction_count += counts.instruction_count;
		result.red_zone_function_count += counts.red_zone_function_count;
		result.indirect_red_zone_function_count += counts.indirect_red_zone_function_count;
		result.reciprocal_sqrt += counts.reciprocal_sqrt;
		result.extrq += counts.extrq;
		result.insertq += counts.insertq;
		result.rdpid += counts.rdpid;
		result.clwb += counts.clwb;
		result.rdtsc += counts.rdtsc;
		for (auto& item: thread_work[thread_index]) {
			ordered.push_back(&item);
		}
	}
	std::ranges::sort(ordered, {}, &FunctionWork::index);

	for (auto* item: ordered) {
		const uintptr_t function_start = starts[item->index];
		const uintptr_t function_end   = function_end_of(item->index);
		const auto&     function       = item->function;
		const auto&     replacements   = item->replacements;
		if (item->copy) {
			++result.red_zone_copy_candidate_count;
			RedZoneFunctionWriter writer(module, function, function_start, function_end, *sweep,
			                             replacements, tsc_multiplier);
			switch (writer.Write(result)) {
				case RedZoneFunctionWriter::Outcome::Rewritten:
					++result.rewritten_function_count;
					continue;
				case RedZoneFunctionWriter::Outcome::EntryConflict:
					++result.copy_entry_conflict_count;
					break;
				case RedZoneFunctionWriter::Outcome::Unsupported:
					++result.copy_unsupported_count;
					break;
				case RedZoneFunctionWriter::Outcome::NoSpace: ++result.copy_no_space_count; break;
			}
		}

		std::map<uintptr_t, InstructionRewrite> rewrite_sites;
		if (protect_memory) {
			CollectRedZoneMemoryInstructions(function, rewrite_sites, result);
		}
		for (const auto& [address, replacement]: replacements) {
			rewrite_sites[address].replacement = replacement.replacement;
		}
		if (!rewrite_sites.empty()) {
			RelocateGuestInstructions(module, function, rewrite_sites, result, tsc_multiplier);
			TrapUnrelocatedInstructions(*module, function, rewrite_sites, result);
		}
	}
	const auto trampoline_addr =
	    reinterpret_cast<u64>(module->trampoline_gen.getCode()) + trampoline_begin;
	const auto trampoline_size = module->trampoline_gen.getSize() - trampoline_begin;
	Common::VirtualMemory::FlushInstructionCache(segment_addr, segment_size);
	if (trampoline_size != 0) {
		Common::VirtualMemory::FlushInstructionCache(trampoline_addr, trampoline_size);
	}
	return result;
}

namespace {

constexpr uint8_t DW_EH_PE_FORMAT_MASK      = 0x0f;
constexpr uint8_t DW_EH_PE_APPLICATION_MASK = 0x70;
constexpr uint8_t DW_EH_PE_PCREL            = 0x10;
constexpr uint8_t DW_EH_PE_DATAREL          = 0x30;
constexpr uint8_t DW_EH_PE_INDIRECT         = 0x80;
constexpr uint8_t DW_EH_PE_OMIT             = 0xff;

template <typename T>
bool ReadEhValue(const uint8_t** cursor, const uint8_t* end, T* value) {
	if (cursor == nullptr || *cursor == nullptr || value == nullptr ||
	    static_cast<size_t>(end - *cursor) < sizeof(T)) {
		return false;
	}
	std::memcpy(value, *cursor, sizeof(T));
	*cursor += sizeof(T);
	return true;
}

bool ReadEncodedEhPointer(const uint8_t** cursor, const uint8_t* end, uint8_t encoding,
                          uintptr_t datarel_base, uintptr_t* value) {
	if (cursor == nullptr || *cursor == nullptr || value == nullptr || encoding == DW_EH_PE_OMIT) {
		return false;
	}

	const auto field_address = reinterpret_cast<uintptr_t>(*cursor);
	uint64_t   raw           = 0;
	bool       is_signed     = false;
	switch (encoding & DW_EH_PE_FORMAT_MASK) {
		case 0x00: {
			uintptr_t decoded = 0;
			if (!ReadEhValue(cursor, end, &decoded)) {
				return false;
			}
			raw = decoded;
			break;
		}
		case 0x02: {
			uint16_t decoded = 0;
			if (!ReadEhValue(cursor, end, &decoded)) {
				return false;
			}
			raw = decoded;
			break;
		}
		case 0x03: {
			uint32_t decoded = 0;
			if (!ReadEhValue(cursor, end, &decoded)) {
				return false;
			}
			raw = decoded;
			break;
		}
		case 0x04:
			if (!ReadEhValue(cursor, end, &raw)) {
				return false;
			}
			break;
		case 0x0a: {
			int16_t decoded = 0;
			if (!ReadEhValue(cursor, end, &decoded)) {
				return false;
			}
			raw       = static_cast<uint64_t>(static_cast<int64_t>(decoded));
			is_signed = true;
			break;
		}
		case 0x0b: {
			int32_t decoded = 0;
			if (!ReadEhValue(cursor, end, &decoded)) {
				return false;
			}
			raw       = static_cast<uint64_t>(static_cast<int64_t>(decoded));
			is_signed = true;
			break;
		}
		case 0x0c: {
			int64_t decoded = 0;
			if (!ReadEhValue(cursor, end, &decoded)) {
				return false;
			}
			raw       = static_cast<uint64_t>(decoded);
			is_signed = true;
			break;
		}
		default: return false;
	}

	uint64_t base = 0;
	switch (encoding & DW_EH_PE_APPLICATION_MASK) {
		case 0x00: break;
		case DW_EH_PE_PCREL: base = field_address; break;
		case DW_EH_PE_DATAREL: base = datarel_base; break;
		default: return false;
	}

	uint64_t decoded = 0;
	if (is_signed) {
		decoded = static_cast<uint64_t>(static_cast<int64_t>(base) + static_cast<int64_t>(raw));
	} else {
		if (raw > UINT64_MAX - base) {
			return false;
		}
		decoded = base + raw;
	}
	if ((encoding & DW_EH_PE_INDIRECT) != 0) {
		const auto* indirect = reinterpret_cast<const uint8_t*>(decoded);
		uintptr_t   target   = 0;
		std::memcpy(&target, indirect, sizeof(target));
		decoded = target;
	}
	*value = static_cast<uintptr_t>(decoded);
	return true;
}

} // namespace

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts) {
	if (function_starts == nullptr) {
		return false;
	}
	function_starts->clear();
	if (eh_frame_header_addr == 0 || eh_frame_header_size < 4 ||
	    eh_frame_header_size > UINTPTR_MAX - eh_frame_header_addr) {
		return false;
	}

	const auto* start             = reinterpret_cast<const uint8_t*>(eh_frame_header_addr);
	const auto* end               = start + eh_frame_header_size;
	const auto* cursor            = start;
	uint8_t     version           = 0;
	uint8_t     eh_frame_encoding = 0;
	uint8_t     count_encoding    = 0;
	uint8_t     table_encoding    = 0;
	if (!ReadEhValue(&cursor, end, &version) || !ReadEhValue(&cursor, end, &eh_frame_encoding) ||
	    !ReadEhValue(&cursor, end, &count_encoding) ||
	    !ReadEhValue(&cursor, end, &table_encoding) || version != 1) {
		return false;
	}

	uintptr_t ignored_eh_frame = 0;
	if (!ReadEncodedEhPointer(&cursor, end, eh_frame_encoding, eh_frame_header_addr,
	                          &ignored_eh_frame)) {
		return false;
	}
	if (count_encoding == DW_EH_PE_OMIT || table_encoding == DW_EH_PE_OMIT) {
		return true;
	}

	uintptr_t fde_count = 0;
	if (!ReadEncodedEhPointer(&cursor, end, count_encoding, eh_frame_header_addr, &fde_count)) {
		return false;
	}
	if (fde_count > eh_frame_header_size / 2u) {
		return false;
	}

	function_starts->reserve(static_cast<size_t>(fde_count));
	for (uintptr_t index = 0; index < fde_count; ++index) {
		uintptr_t function_start = 0;
		uintptr_t ignored_fde    = 0;
		if (!ReadEncodedEhPointer(&cursor, end, table_encoding, eh_frame_header_addr,
		                          &function_start) ||
		    !ReadEncodedEhPointer(&cursor, end, table_encoding, eh_frame_header_addr,
		                          &ignored_fde)) {
			function_starts->clear();
			return false;
		}
		function_starts->push_back(function_start);
	}
	return true;
}

void RegisterGuestInstructionPatchModule(void* module_ptr, uint64_t module_size,
                                         void* trampoline_area_ptr, uint64_t trampoline_area_size) {
	EXIT_IF(module_ptr == nullptr || module_size == 0 || trampoline_area_ptr == nullptr ||
	        trampoline_area_size == 0);
	const auto module_addr = reinterpret_cast<u64>(module_ptr);
	g_patch_modules.erase(module_addr);
	g_patch_modules.emplace(std::piecewise_construct, std::forward_as_tuple(module_addr),
	                        std::forward_as_tuple(static_cast<u8*>(module_ptr), module_size,
	                                              static_cast<u8*>(trampoline_area_ptr),
	                                              trampoline_area_size));
}

void UnregisterGuestInstructionPatchModule(void* module_ptr) {
	g_patch_modules.erase(reinterpret_cast<u64>(module_ptr));
}

#undef ASSERT

} // namespace Loader
