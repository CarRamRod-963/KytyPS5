#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <xbyak/xbyak.h>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};


} // namespace

// The general compiled walker. The simple walker above covers plans whose reads are plain
// user-data pointer chains with immediate offsets; most of this game's plans select pointers
// under control flow or compute offsets, so this one compiles the plan's value graph as a
// stack machine over the evaluation memo: every instruction value is computed into its memo
// slot (tagged with the generation) the first time a path needs it, which also lets the
// descriptor evaluation that follows hit the memo. Blocks are walked in index order with the
// interpreter's reachability rule: a block runs when a reachable predecessor selected it or had
// a condition the host could not decide, and a control phi takes the incoming of the reachable
// predecessors, which must agree. Plans whose edges go backwards stay with the interpreter.
// Guest reads go through the runtime readers, as in the interpreter. Anything the compiler
// does not model (floating point, varying loads, clean reads) leaves the plan to the
// interpreter, and a failing run falls back to it as well.
struct GeneralWalkerContext {
	const uint32_t*                         user_data       = nullptr;
	uint64_t                                user_data_count = 0;
	uint32_t*                               flat            = nullptr;
	ResourcePlan::EvaluationContext::Entry* values          = nullptr;
	uint64_t                                generation      = 0;
	uint64_t                                shader_base     = 0;
	uint8_t*                                visited         = nullptr;
	const SrtRuntime*                       runtime         = nullptr;
	// The clean evaluator's memo: values of conditions, select predicates and clean reads,
	// which read through the specialization reader.
	ResourcePlan::EvaluationContext::Entry* clean_values     = nullptr;
	uint64_t                                clean_generation = 0;
	uint32_t                                scratch          = 0;
	// Set when a reader refused an address: the interpreter would fail the same way, so the
	// run does not fall back to it.
	uint32_t read_failed = 0;
	// The stack pointer at entry: a failure with values still pushed unwinds to it.
	uint64_t frame = 0;
	// Stack pointers saved by nested logical operators that tolerate an unknown operand.
	uint64_t saved_sp[16] = {};
	// Memo arrays of the EXEC-factor assumptions: 2 * mask index + clean.
	ResourcePlan::EvaluationContext::Entry** assume_values = nullptr;
};
static_assert(offsetof(GeneralWalkerContext, user_data) == 0);
static_assert(offsetof(GeneralWalkerContext, user_data_count) == 8);
static_assert(offsetof(GeneralWalkerContext, flat) == 16);
static_assert(offsetof(GeneralWalkerContext, values) == 24);
static_assert(offsetof(GeneralWalkerContext, generation) == 32);
static_assert(offsetof(GeneralWalkerContext, shader_base) == 40);
static_assert(offsetof(GeneralWalkerContext, visited) == 48);
static_assert(offsetof(GeneralWalkerContext, runtime) == 56);
static_assert(offsetof(GeneralWalkerContext, clean_values) == 64);
static_assert(offsetof(GeneralWalkerContext, clean_generation) == 72);
static_assert(offsetof(GeneralWalkerContext, scratch) == 80);
static_assert(offsetof(GeneralWalkerContext, frame) == 88);
static_assert(offsetof(GeneralWalkerContext, saved_sp) == 96);
static_assert(offsetof(GeneralWalkerContext, assume_values) == 224);

static bool KYTY_SYSV_ABI GeneralWalkerRead(GeneralWalkerContext* context, uint64_t address,
                                            uint64_t clean) {
	const auto& runtime = *context->runtime;
	if (clean != 0u && runtime.read_specialization_memory == nullptr) {
		return false;
	}
	const auto reader = clean != 0u ? runtime.read_specialization_memory : runtime.read_memory;
	uint32_t   word   = 0;
	if (reader != nullptr) {
		if (!reader(runtime.userdata, address, {&word, 1})) {
			context->read_failed = 1;
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	context->scratch = word;
	return true;
}

// Bit field helpers return the value, or bit 32 set when the interpreter would fail.
static uint64_t KYTY_SYSV_ABI GeneralWalkerBitFieldUExtract(uint64_t a, uint64_t b, uint64_t c) {
	const auto offset = static_cast<uint32_t>(b);
	const auto width  = static_cast<uint32_t>(c);
	if (offset > 32u || width > 32u - offset) return uint64_t {1} << 32u;
	const auto mask = width == 32u ? UINT32_MAX : width == 0u ? 0u : (uint32_t {1} << width) - 1u;
	return width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
}

static uint64_t KYTY_SYSV_ABI GeneralWalkerBitFieldSExtract(uint64_t a, uint64_t b, uint64_t c) {
	const auto offset = static_cast<uint32_t>(b);
	const auto width  = static_cast<uint32_t>(c);
	if (offset > 32u || width > 32u - offset) return uint64_t {1} << 32u;
	if (width == 0u) return 0u;
	const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
	auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
	if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) bits |= ~mask;
	return bits;
}

static uint64_t KYTY_SYSV_ABI GeneralWalkerBitFieldInsert(uint64_t a, uint64_t b, uint64_t c,
                                                          uint64_t d) {
	const auto offset = static_cast<uint32_t>(c);
	const auto width  = static_cast<uint32_t>(d);
	if (offset > 32u || width > 32u - offset) return uint64_t {1} << 32u;
	if (width == 0u) return static_cast<uint32_t>(a);
	const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
	return (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
}

class GeneralWalkerCompiler {
public:
	explicit GeneralWalkerCompiler(ResourcePlan& program)
	    : m_program(program),
	      m_clean_evaluator(program.capture_specialization_reads || !program.source_reads.empty() ||
	                        program.requires_specialization_memory) {}

	bool Compile() {
		auto&      program = m_program;
		const auto blocks  = program.control_flow.size();
		for (uint32_t index = 0; index < blocks; ++index) {
			const auto& block = program.control_flow[index];
			if (block.condition.IsEmpty() ? block.successors.size() > 1u
			                              : block.successors.size() != 2u) {
				return false;
			}
			for (const auto successor: block.successors) {
				if (successor >= blocks) return false;
			}
		}
		// Blocks are emitted in a topological order so every predecessor of a block has run
		// (or been skipped) before it: the reachability rule and the phis depend on it.
		std::vector<uint32_t> order;
		{
			std::vector<uint32_t> pending(blocks, 0u);
			for (const auto& block: program.control_flow) {
				for (const auto successor: block.successors)
					++pending[successor];
			}
			std::vector<uint32_t> ready;
			for (uint32_t index = 0; index < blocks; ++index) {
				if (pending[index] == 0u) ready.push_back(index);
			}
			while (!ready.empty()) {
				const auto index = ready.back();
				ready.pop_back();
				order.push_back(index);
				for (const auto successor: program.control_flow[index].successors) {
					if (--pending[successor] == 0u) ready.push_back(successor);
				}
			}
			if (order.size() != blocks) return false;
		}
		for (const auto& [phi, predecessors]: program.control_phi_predecessors) {
			uint32_t owner = UINT32_MAX;
			for (uint32_t index = 0; index < blocks; ++index) {
				const bool all = std::ranges::all_of(predecessors, [&](uint32_t pred) {
					return pred < blocks &&
					       std::ranges::find(program.control_flow[pred].successors, index) !=
					           program.control_flow[pred].successors.end();
				});
				if (all) {
					if (owner != UINT32_MAX) return false;
					owner = index;
				}
			}
			if (owner == UINT32_MAX) return false;
			m_phi_blocks.emplace(phi, owner);
		}
		auto code_holder = std::make_unique<Xbyak::CodeGenerator>(16384, Xbyak::AutoGrow);
		m_code           = code_holder.get();
		auto& code       = *m_code;
		code.push(code.rbp);
		code.push(code.rbx);
		code.push(code.r12);
		code.push(code.r13);
		code.push(code.r14);
		code.push(code.r15);
		code.mov(code.rbx, code.rdi);
		code.mov(code.r15, code.qword[code.rbx + 0]);
		code.mov(code.r13, code.qword[code.rbx + 16]);
		code.mov(code.r12, code.qword[code.rbx + 24]);
		code.mov(code.r14, code.qword[code.rbx + 32]);
		code.mov(code.qword[code.rbx + 88], code.rsp);
		std::vector<Xbyak::Label> labels(blocks);
		if (blocks == 0) {
			for (const auto& read: program.srt_reads) {
				if (!EmitRead(read)) return false;
			}
		} else {
			for (size_t position = 0; position < order.size(); ++position) {
				const auto  index = order[position];
				const auto& block = program.control_flow[index];
				auto& next = position + 1 < order.size() ? labels[order[position + 1]] : m_finished;
				code.L(labels[index]);
				m_base.by_mode.clear();
				m_scopes.clear();
				code.mov(code.rax, code.qword[code.rbx + 48]);
				code.cmp(code.byte[code.rax + index], 0);
				code.je(next, Xbyak::CodeGenerator::T_NEAR);
				m_fail = &m_failed;
				for (const auto slot: block.srt_reads) {
					if (slot >= program.srt_reads.size() || !EmitRead(program.srt_reads[slot])) {
						return false;
					}
				}
				const auto reach = [&](uint32_t successor) {
					code.mov(code.rax, code.qword[code.rbx + 48]);
					code.mov(code.byte[code.rax + successor], 1);
				};
				if (block.condition.IsEmpty()) {
					for (const auto successor: block.successors)
						reach(successor);
					code.jmp(next, Xbyak::CodeGenerator::T_NEAR);
					continue;
				}
				// An undecidable condition reaches both successors, like the interpreter. The
				// condition is evaluated by the clean evaluator when there is one.
				Xbyak::Label otherwise, unknown;
				code.mov(code.rbp, code.rsp);
				m_fail = &unknown;
				EnterClean();
				const bool condition = Emit(block.condition);
				LeaveClean();
				if (!condition) return false;
				m_fail = &m_failed;
				code.test(code.eax, code.eax);
				code.jz(otherwise, Xbyak::CodeGenerator::T_NEAR);
				reach(block.successors[0]);
				code.jmp(next, Xbyak::CodeGenerator::T_NEAR);
				code.L(otherwise);
				reach(block.successors[1]);
				code.jmp(next, Xbyak::CodeGenerator::T_NEAR);
				code.L(unknown);
				code.mov(code.rsp, code.rbp);
				RestoreNormalMemo();
				reach(block.successors[0]);
				reach(block.successors[1]);
				code.jmp(next, Xbyak::CodeGenerator::T_NEAR);
			}
		}
		code.L(m_finished);
		code.mov(code.eax, 1);
		code.jmp(m_exit, Xbyak::CodeGenerator::T_NEAR);
		code.L(m_failed);
		code.mov(code.rsp, code.qword[code.rbx + 88]);
		code.xor_(code.eax, code.eax);
		code.L(m_exit);
		code.pop(code.r15);
		code.pop(code.r14);
		code.pop(code.r13);
		code.pop(code.r12);
		code.pop(code.rbx);
		code.pop(code.rbp);
		code.ret();
		code.readyRE();
		if (Xbyak::GetError() != 0) {
			Xbyak::ClearError();
			return false;
		}
		program.srt_general_code = std::move(code_holder);
		program.jit_assumptions  = static_cast<uint32_t>(m_masks.size());
		return true;
	}

private:
	using Code = Xbyak::CodeGenerator;

	bool EmitRead(const SrtRead& read) {
		if (read.flat_offset >= m_program.srt_reads.size() || !Emit(read.value)) return false;
		m_code->mov(m_code->dword[m_code->r13 + read.flat_offset * sizeof(uint32_t)], m_code->eax);
		return true;
	}

	// Leaves the value in rax: 32-bit results zero-extended, 64-bit results whole.
	bool Emit(Value value) {
		auto& code = *m_code;
		value      = value.Resolve();
		if (value.IsImmediate()) {
			uint64_t immediate = 0;
			switch (value.GetType()) {
				case Type::U1: immediate = value.U1(); break;
				case Type::U8: immediate = value.U8(); break;
				case Type::U16: immediate = value.U16(); break;
				case Type::U32: immediate = value.U32(); break;
				case Type::U64: immediate = value.U64(); break;
				case Type::F32: immediate = std::bit_cast<uint32_t>(value.F32Value()); break;
				default: return false;
			}
			code.mov(code.rax, immediate);
			return true;
		}
		if (!m_assume.empty() && value == m_masks[m_assume.back()]) {
			code.mov(code.eax, 1);
			return true;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		// A clean read belongs to the clean evaluator: its memo, its reader.
		if (m_clean_evaluator && m_clean_depth == 0 && IsRawRead(m_program, *inst) &&
		    inst->Flags<SrtReadFlags>().clean != 0u) {
			EnterClean();
			const bool done = Emit(value);
			LeaveClean();
			return done;
		}
		const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
		const auto slot =
		    static_cast<uint32_t>(index) * sizeof(ResourcePlan::EvaluationContext::Entry);
		if (IsEmitted(inst)) {
			code.mov(code.rax, code.qword[code.r12 + slot]);
			return true;
		}
		// The code may not grow without bound: a plan that re-emits shared values in many
		// nested regions is left to the interpreter.
		if (code.getSize() > MaxCodeSize) return false;
		Xbyak::Label compute, done;
		code.mov(code.rax, code.qword[code.r12 + slot + 8]);
		code.cmp(code.rax, code.r14);
		code.jne(compute, Code::T_NEAR);
		code.mov(code.rax, code.qword[code.r12 + slot]);
		code.jmp(done, Code::T_NEAR);
		code.L(compute);
		if (!EmitInst(*inst)) return false;
		code.mov(code.qword[code.r12 + slot], code.rax);
		code.mov(code.qword[code.r12 + slot + 8], code.r14);
		code.L(done);
		if (!IsUndef(*inst)) Emitted().insert(inst);
		return true;
	}

	static bool IsUndef(const Inst& inst) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU32: return true;
			default: return false;
		}
	}

	// Whether evaluating the value can end as "unknown" rather than fail the walk: undefined
	// values, and phis (their reachable incomings may disagree).
	bool MayBeUnknown(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) return false;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return true;
		if (const auto known = m_may_be_unknown.find(inst); known != m_may_be_unknown.end()) {
			return known->second;
		}
		m_may_be_unknown.emplace(inst, false);
		bool result = IsUndef(*inst) || inst->GetOpcode() == ValueOpcode::Phi;
		for (size_t index = 0; index < inst->NumArgs() && !result; ++index) {
			result = MayBeUnknown(inst->Arg(index));
		}
		m_may_be_unknown[inst] = result;
		return result;
	}

	// Evaluates a value whose unknown outcome is handled locally: on "unknown" control
	// continues at `unknown` with the stack pointer restored.
	bool EmitTolerant(Value value, Xbyak::Label& unknown) {
		auto& code = *m_code;
		if (m_tolerant_depth >= 16) return false;
		const auto slot = 96 + m_tolerant_depth++ * 8;
		code.mov(code.qword[code.rbx + slot], code.rsp);
		Xbyak::Label restore, done;
		auto* const  previous = m_fail;
		m_fail                = &restore;
		PushScope();
		const bool emitted = Emit(value);
		PopScope();
		m_fail = previous;
		--m_tolerant_depth;
		if (!emitted) return false;
		code.jmp(done, Code::T_NEAR);
		code.L(restore);
		code.mov(code.rsp, code.qword[code.rbx + slot]);
		LoadMemoBase(code.r12);
		if (m_clean_evaluator) {
			code.mov(code.r14, code.qword[code.rbx + (m_clean_depth > 0 ? 72 : 32)]);
		}
		code.jmp(unknown, Code::T_NEAR);
		code.L(done);
		return true;
	}

	// Values whose memo slot is known to hold the current generation: those emitted
	// unconditionally earlier in the block, and those emitted earlier in every enclosing
	// conditional region. A region's set is dropped when the region ends because a later
	// path may skip it. Each mode (plain, or under an EXEC-factor assumption) has its own
	// memo array and therefore its own sets.
	struct Sets {
		std::unordered_set<const Inst*> emitted;
		std::unordered_set<const Inst*> emitted_clean;
	};
	struct Scope {
		std::unordered_map<uint32_t, Sets> by_mode;
	};

	static constexpr uint32_t NoAssumption = ~0u;

	uint32_t Assumption() const { return m_assume.empty() ? NoAssumption : m_assume.back(); }

	std::unordered_set<const Inst*>& Emitted() {
		auto& sets = (m_scopes.empty() ? m_base : m_scopes.back()).by_mode[Assumption()];
		return m_clean_depth > 0 ? sets.emitted_clean : sets.emitted;
	}

	bool IsEmitted(const Inst* inst) const {
		const auto mode  = Assumption();
		const bool clean = m_clean_depth > 0;
		const auto has   = [&](const Scope& scope) {
			const auto found = scope.by_mode.find(mode);
			return found != scope.by_mode.end() &&
			       (clean ? found->second.emitted_clean : found->second.emitted).contains(inst);
		};
		if (has(m_base)) return true;
		for (const auto& scope: m_scopes) {
			if (has(scope)) return true;
		}
		return false;
	}

	void PushScope() { m_scopes.emplace_back(); }
	void PopScope() { m_scopes.pop_back(); }

	// A region that either completes or fails its enclosing context: what it emitted
	// dominates what follows it.
	void PopScopeMerging() {
		auto scope = std::move(m_scopes.back());
		m_scopes.pop_back();
		auto& parent = m_scopes.empty() ? m_base : m_scopes.back();
		for (auto& [mode, sets]: scope.by_mode) {
			auto& target = parent.by_mode[mode];
			target.emitted.merge(sets.emitted);
			target.emitted_clean.merge(sets.emitted_clean);
		}
	}

	uint32_t AssumptionIndex(Value mask) {
		for (uint32_t index = 0; index < m_masks.size(); ++index) {
			if (m_masks[index] == mask) return index;
		}
		m_masks.push_back(mask);
		return static_cast<uint32_t>(m_masks.size() - 1);
	}

	// Evaluates a value under the assumption that `mask` holds (the mask is true, selects
	// on it take their true arm), as the interpreter's active evaluator does. Unknown or
	// failure continues at `unknown` with the stack and the memo base of the enclosing
	// mode restored.
	bool EmitAssumed(Value value, Value mask, Xbyak::Label& unknown) {
		auto& code = *m_code;
		if (m_tolerant_depth >= 16) return false;
		const auto slot = 96 + m_tolerant_depth++ * 8;
		code.mov(code.qword[code.rbx + slot], code.rsp);
		m_assume.push_back(AssumptionIndex(mask.Resolve()));
		PushScope();
		code.push(code.r12);
		LoadMemoBase(code.r12);
		Xbyak::Label restore, done;
		auto* const  previous = m_fail;
		m_fail                = &restore;
		const bool emitted    = Emit(value);
		m_fail                = previous;
		PopScopeMerging();
		m_assume.pop_back();
		--m_tolerant_depth;
		if (!emitted) return false;
		code.pop(code.r12);
		code.jmp(done, Code::T_NEAR);
		code.L(restore);
		code.mov(code.rsp, code.qword[code.rbx + slot]);
		LoadMemoBase(code.r12);
		if (m_clean_evaluator) {
			code.mov(code.r14, code.qword[code.rbx + (m_clean_depth > 0 ? 72 : 32)]);
		}
		code.jmp(unknown, Code::T_NEAR);
		code.L(done);
		return true;
	}

	// A value computed from user data and constants alone: evaluating it where the
	// interpreter would not changes nothing observable.
	bool IsPure(const Inst& inst) {
		if (const auto found = m_pure.find(&inst); found != m_pure.end()) return found->second;
		bool pure = false;
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData:
			case ValueOpcode::GetShaderBase:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConditionRef:
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2:
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::BitFieldInsert:
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor:
			case ValueOpcode::LogicalNot: {
				pure = true;
				for (size_t index = 0; index < inst.NumArgs() && pure; ++index) {
					const auto* arg = inst.Arg(index).TryInstruction();
					pure            = arg == nullptr || IsPure(*arg);
				}
				break;
			}
			default: break;
		}
		m_pure.emplace(&inst, pure);
		return pure;
	}

	void CollectUses(Value value, std::unordered_map<const Inst*, Value>& uses, size_t& budget) {
		auto* const inst = value.TryInstruction();
		if (inst == nullptr || IsEmitted(inst) || uses.contains(inst) || budget == 0) return;
		--budget;
		uses.emplace(inst, value);
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			CollectUses(inst->Arg(index), uses, budget);
		}
	}

	// Values both arms of a select need are evaluated before the branch when that is
	// unobservable; otherwise each arm emits its own copy, which doubles the code at every
	// link of a select chain.
	bool HoistShared(const Inst& select) {
		std::unordered_map<const Inst*, Value> left;
		std::unordered_map<const Inst*, Value> right;
		size_t                                 budget = 4096;
		CollectUses(select.Arg(1), left, budget);
		CollectUses(select.Arg(2), right, budget);
		for (const auto& [inst, value]: left) {
			if (right.contains(inst) && IsPure(*inst) && !IsEmitted(inst) && !Emit(value)) {
				return false;
			}
		}
		return true;
	}

	// Loads the memo base of the current mode: the clean or the normal context.
	void LoadMemoBase(const Xbyak::Reg64& reg) {
		auto& code = *m_code;
		if (!m_assume.empty()) {
			code.mov(reg, code.qword[code.rbx + 224]);
			code.mov(reg, code.qword[reg + (2 * m_assume.back() + (m_clean_depth > 0 ? 1 : 0)) *
			                                   sizeof(void*)]);
		} else {
			code.mov(reg, code.qword[code.rbx + (m_clean_depth > 0 ? 64 : 24)]);
		}
	}

	// Clean mode swaps the memo base and generation for the clean evaluator's; without a
	// clean evaluator the mode changes nothing.
	void EnterClean() {
		if (!m_clean_evaluator) return;
		auto& code = *m_code;
		if (m_clean_depth++ == 0) {
			code.push(code.r12);
			code.push(code.r14);
			LoadMemoBase(code.r12);
			code.mov(code.r14, code.qword[code.rbx + 72]);
		}
	}

	void LeaveClean() {
		if (!m_clean_evaluator) return;
		auto& code = *m_code;
		if (--m_clean_depth == 0) {
			code.pop(code.r14);
			code.pop(code.r12);
		}
	}

	// After a jump out of clean mode the pushes are gone with the stack pointer.
	void RestoreNormalMemo() {
		auto& code = *m_code;
		LoadMemoBase(code.r12);
		if (m_clean_evaluator) code.mov(code.r14, code.qword[code.rbx + 32]);
	}

	// Evaluates two arguments: the first into rax, the second into rcx.
	bool EmitPair(const Inst& inst, size_t first, size_t second) {
		auto& code = *m_code;
		if (!Emit(inst.Arg(first))) return false;
		code.push(code.rax);
		if (!Emit(inst.Arg(second))) return false;
		code.mov(code.rcx, code.rax);
		code.pop(code.rax);
		return true;
	}

	void EmitCompare(void (Code::*set)(const Xbyak::Operand&)) {
		auto& code = *m_code;
		code.cmp(code.eax, code.ecx);
		(code.*set)(code.al);
		code.movzx(code.eax, code.al);
	}

	// Calls a SysV helper with up to four evaluated arguments; the result is in rax.
	bool EmitHelperCall(const Inst& inst, size_t args, const void* function) {
		auto& code = *m_code;
		for (size_t index = 0; index < args; ++index) {
			if (!Emit(inst.Arg(index))) return false;
			code.push(code.rax);
		}
		if (args > 3) code.pop(code.rcx);
		if (args > 2) code.pop(code.rdx);
		if (args > 1) code.pop(code.rsi);
		code.pop(code.rdi);
		code.push(code.rbp);
		code.mov(code.rbp, code.rsp);
		code.and_(code.rsp, -16);
		code.mov(code.rax, reinterpret_cast<uint64_t>(function));
		code.call(code.rax);
		code.mov(code.rsp, code.rbp);
		code.pop(code.rbp);
		return true;
	}

	bool EmitInst(const Inst& inst) {
		auto& code = *m_code;
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				if (inst.Arg(0).GetType() != Type::ScalarReg) return false;
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) return false;
				const auto slot = reg - m_program.user_data_base;
				code.cmp(code.qword[code.rbx + 8], slot);
				code.jbe(*m_fail, Code::T_NEAR);
				code.mov(code.eax, code.dword[code.r15 + slot * sizeof(uint32_t)]);
				return true;
			}
			case ValueOpcode::GetShaderBase:
				code.mov(code.rax, code.qword[code.rbx + 40]);
				return true;
			case ValueOpcode::Phi: return EmitPhi(inst);
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConditionRef: return Emit(inst.Arg(0));
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: {
				const auto index = inst.Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
					return false;
				}
				if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
					if (!Emit(inst.Arg(0))) return false;
					if (index.U32() == 1u) code.shr(code.rax, 32);
					code.mov(code.eax, code.eax);
					return true;
				}
				const auto* source = inst.Arg(0).ResolveInstruction();
				if (source == nullptr) return false;
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					return Emit(source->Arg(index.U32()));
				}
				return false;
			}
			case ValueOpcode::CompositeConstructU64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.mov(code.eax, code.eax);
				code.shl(code.rcx, 32);
				code.or_(code.rax, code.rcx);
				return true;
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: return EmitRawRead(inst);
			case ValueOpcode::IAdd32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.add(code.eax, code.ecx);
				return true;
			case ValueOpcode::IAdd64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.add(code.rax, code.rcx);
				return true;
			case ValueOpcode::ISub32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.sub(code.eax, code.ecx);
				return true;
			case ValueOpcode::ISub64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.sub(code.rax, code.rcx);
				return true;
			case ValueOpcode::IMul32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.imul(code.eax, code.ecx);
				return true;
			case ValueOpcode::UMulHi:
				if (!EmitPair(inst, 0, 1)) return false;
				code.mul(code.ecx);
				code.mov(code.eax, code.edx);
				return true;
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.cmp(code.eax, code.ecx);
				if (inst.GetOpcode() == ValueOpcode::UMin32) {
					code.cmova(code.eax, code.ecx);
				} else {
					code.cmovb(code.eax, code.ecx);
				}
				return true;
			case ValueOpcode::BitwiseAnd32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.and_(code.eax, code.ecx);
				return true;
			case ValueOpcode::BitwiseAnd64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.and_(code.rax, code.rcx);
				return true;
			case ValueOpcode::BitwiseOr32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.or_(code.eax, code.ecx);
				return true;
			case ValueOpcode::BitwiseXor32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.xor_(code.eax, code.ecx);
				return true;
			case ValueOpcode::BitwiseNot32:
				if (!Emit(inst.Arg(0))) return false;
				code.not_(code.eax);
				return true;
			case ValueOpcode::ShiftLeftLogical32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.shl(code.eax, code.cl);
				return true;
			case ValueOpcode::ShiftLeftLogical64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.shl(code.rax, code.cl);
				return true;
			case ValueOpcode::ShiftRightLogical32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.shr(code.eax, code.cl);
				return true;
			case ValueOpcode::ShiftRightLogical64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.shr(code.rax, code.cl);
				return true;
			case ValueOpcode::ShiftRightArithmetic32:
				if (!EmitPair(inst, 0, 1)) return false;
				code.sar(code.eax, code.cl);
				return true;
			case ValueOpcode::ShiftRightArithmetic64:
				if (!EmitPair(inst, 0, 1)) return false;
				code.sar(code.rax, code.cl);
				return true;
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::BitFieldInsert: {
				const bool  insert = inst.GetOpcode() == ValueOpcode::BitFieldInsert;
				const void* function =
				    insert ? reinterpret_cast<const void*>(&GeneralWalkerBitFieldInsert)
				    : inst.GetOpcode() == ValueOpcode::BitFieldUExtract
				        ? reinterpret_cast<const void*>(&GeneralWalkerBitFieldUExtract)
				        : reinterpret_cast<const void*>(&GeneralWalkerBitFieldSExtract);
				if (!EmitHelperCall(inst, insert ? 4u : 3u, function)) return false;
				code.bt(code.rax, 32);
				code.jc(*m_fail, Code::T_NEAR);
				return true;
			}
			case ValueOpcode::ReadFirstLane: {
				if (inst.NumArgs() != 2) return false;
				const auto mask = inst.Arg(1).Resolve();
				if (!m_assume.empty() && m_masks[m_assume.back()] == mask) {
					return Emit(inst.Arg(0));
				}
				return EmitAssumed(inst.Arg(0), mask, *m_fail);
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				if (!m_assume.empty() && inst.NumArgs() == 3 &&
				    inst.Arg(0).Resolve() == m_masks[m_assume.back()]) {
					return Emit(inst.Arg(1));
				}
				Xbyak::Label otherwise, done;
				if (!HoistShared(inst)) return false;
				EnterClean();
				const bool predicate = Emit(inst.Arg(0));
				LeaveClean();
				if (!predicate) return false;
				code.test(code.eax, code.eax);
				code.jz(otherwise, Code::T_NEAR);
				PushScope();
				const bool branches = Emit(inst.Arg(1));
				PopScope();
				code.jmp(done, Code::T_NEAR);
				code.L(otherwise);
				PushScope();
				const bool other = branches && Emit(inst.Arg(2));
				PopScope();
				code.L(done);
				return other;
			}
			case ValueOpcode::IEqual32:
				if (!EmitPair(inst, 0, 1)) return false;
				EmitCompare(&Code::sete);
				return true;
			case ValueOpcode::INotEqual32:
				if (!EmitPair(inst, 0, 1)) return false;
				EmitCompare(&Code::setne);
				return true;
			case ValueOpcode::ULessThan32:
				if (!EmitPair(inst, 0, 1)) return false;
				EmitCompare(&Code::setb);
				return true;
			case ValueOpcode::ULessThanEqual32: {
				Xbyak::Label one, done;
				if (!Emit(inst.Arg(0))) return false;
				code.test(code.eax, code.eax);
				code.jz(one, Code::T_NEAR);
				code.push(code.rax);
				PushScope();
				const bool right = Emit(inst.Arg(1));
				PopScope();
				if (!right) return false;
				code.mov(code.rcx, code.rax);
				code.pop(code.rax);
				EmitCompare(&Code::setbe);
				code.jmp(done, Code::T_NEAR);
				code.L(one);
				code.mov(code.eax, 1);
				code.L(done);
				return true;
			}
			case ValueOpcode::UGreaterThan32:
				if (!EmitPair(inst, 0, 1)) return false;
				EmitCompare(&Code::seta);
				return true;
			case ValueOpcode::SGreaterThanEqual32:
				if (!EmitPair(inst, 0, 1)) return false;
				EmitCompare(&Code::setge);
				return true;
			case ValueOpcode::LogicalAnd: {
				// A false left operand decides without the right one. An unknown left operand
				// still decides the conjunction when the right one is false, plainly or under
				// the assumption that the left EXEC factor holds; otherwise it is unknown.
				Xbyak::Label zero, done, left_unknown;
				const bool   tolerant = MayBeUnknown(inst.Arg(0));
				if (tolerant) {
					if (!EmitTolerant(inst.Arg(0), left_unknown)) return false;
				} else if (!Emit(inst.Arg(0))) {
					return false;
				}
				code.test(code.eax, code.eax);
				code.jz(zero, Code::T_NEAR);
				PushScope();
				const bool right = Emit(inst.Arg(1));
				PopScope();
				if (!right) return false;
				code.test(code.eax, code.eax);
				code.setne(code.al);
				code.movzx(code.eax, code.al);
				code.jmp(done, Code::T_NEAR);
				if (tolerant) {
					Xbyak::Label right_unknown;
					code.L(left_unknown);
					if (!EmitTolerant(inst.Arg(1), right_unknown)) return false;
					code.test(code.eax, code.eax);
					code.jz(zero, Code::T_NEAR);
					code.L(right_unknown);
					// The interpreter evaluates the assumed operand through the clean evaluator.
					EnterClean();
					const bool assumed = EmitAssumed(inst.Arg(1), inst.Arg(0), *m_fail);
					LeaveClean();
					if (!assumed) return false;
					code.test(code.eax, code.eax);
					code.jnz(*m_fail, Code::T_NEAR);
				}
				code.L(zero);
				code.xor_(code.eax, code.eax);
				code.L(done);
				return true;
			}
			case ValueOpcode::LogicalOr: {
				// A true left operand decides; a true right operand decides even when the left
				// is unknown; otherwise an unknown left operand is unknown.
				Xbyak::Label one, right, done, left_unknown;
				if (MayBeUnknown(inst.Arg(0))) {
					if (!EmitTolerant(inst.Arg(0), left_unknown)) return false;
					code.test(code.eax, code.eax);
					code.jnz(one, Code::T_NEAR);
					PushScope();
					const bool emitted = Emit(inst.Arg(1));
					PopScope();
					if (!emitted) return false;
					code.test(code.eax, code.eax);
					code.setne(code.al);
					code.movzx(code.eax, code.al);
					code.jmp(done, Code::T_NEAR);
					code.L(left_unknown);
					PushScope();
					const bool again = Emit(inst.Arg(1));
					PopScope();
					if (!again) return false;
					code.test(code.eax, code.eax);
					code.jz(*m_fail, Code::T_NEAR);
					code.jmp(one, Code::T_NEAR);
				} else {
					if (!Emit(inst.Arg(0))) return false;
					code.test(code.eax, code.eax);
					code.jnz(one, Code::T_NEAR);
					PushScope();
					const bool emitted = Emit(inst.Arg(1));
					PopScope();
					if (!emitted) return false;
					code.test(code.eax, code.eax);
					code.setne(code.al);
					code.movzx(code.eax, code.al);
					code.jmp(done, Code::T_NEAR);
				}
				code.L(one);
				code.mov(code.eax, 1);
				code.L(done);
				return true;
			}
			case ValueOpcode::LogicalXor:
				if (!EmitPair(inst, 0, 1)) return false;
				code.test(code.eax, code.eax);
				code.setne(code.al);
				code.test(code.ecx, code.ecx);
				code.setne(code.cl);
				code.xor_(code.al, code.cl);
				code.movzx(code.eax, code.al);
				return true;
			case ValueOpcode::LogicalNot:
				if (!Emit(inst.Arg(0))) return false;
				code.test(code.eax, code.eax);
				code.sete(code.al);
				code.movzx(code.eax, code.al);
				return true;
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU32: code.jmp(*m_fail, Code::T_NEAR); return true;
		}
	}

	bool EmitPhi(const Inst& inst) {
		auto&      code  = *m_code;
		const auto owner = m_phi_blocks.find(&inst);
		if (owner == m_phi_blocks.end()) {
			const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
			return !value.IsEmpty() && Emit(value);
		}
		const auto& predecessors = m_program.control_phi_predecessors.at(&inst);
		if (predecessors.size() != inst.NumArgs()) return false;
		// [rsp] = incoming value, [rsp + 8] = found; reachable predecessors must agree.
		code.push(0);
		code.push(0);
		for (size_t index = 0; index < predecessors.size(); ++index) {
			Xbyak::Label skip, first;
			PushScope();
			code.mov(code.rax, code.qword[code.rbx + 48]);
			code.cmp(code.byte[code.rax + predecessors[index]], 0);
			code.je(skip, Code::T_NEAR);
			if (!Emit(inst.Arg(index))) return false;
			code.cmp(code.qword[code.rsp + 8], 0);
			code.je(first, Code::T_NEAR);
			code.cmp(code.qword[code.rsp], code.rax);
			code.jne(*m_fail, Code::T_NEAR);
			code.jmp(skip, Code::T_NEAR);
			code.L(first);
			code.mov(code.qword[code.rsp], code.rax);
			code.mov(code.qword[code.rsp + 8], 1);
			code.L(skip);
			PopScope();
		}
		code.cmp(code.qword[code.rsp + 8], 0);
		code.je(*m_fail, Code::T_NEAR);
		code.pop(code.rax);
		code.pop(code.rcx);
		return true;
	}

	bool EmitRawRead(const Inst& inst) {
		auto&      code  = *m_code;
		const auto flags = inst.Flags<SrtReadFlags>();
		if (!IsRawRead(m_program, inst) || flags.index >= m_program.memory_info.size() ||
		    (flags.clean != 0u && !m_clean_evaluator)) {
			return false;
		}
		const auto& memory = m_program.memory_info[flags.index];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr) return false;
		const bool constant_buffer = inst.GetOpcode() == ValueOpcode::ReadConstBuffer;
		if (constant_buffer && handle->NumArgs() != 4u) return false;
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(memory.offset));
		if (constant_buffer && immediate < 0) return false;
		// base = ((high << 32) | low) & AddressMask; a V# read also keeps the raw high dword for
		// its stride.
		if (!EmitPair(*handle, 0, 1)) return false;
		code.mov(code.eax, code.eax);
		code.mov(code.rdx, code.rcx);
		code.shl(code.rdx, 32);
		code.or_(code.rax, code.rdx);
		code.mov(code.rdx, AddressMask);
		code.and_(code.rax, code.rdx);
		code.push(code.rax);
		if (constant_buffer) code.push(code.rcx);
		if (!Emit(inst.Arg(1))) return false;
		code.mov(code.eax, code.eax);
		code.and_(code.rax, -4);
		Xbyak::Label zero, read, done;
		if (constant_buffer) {
			// byte_offset = (immediate & ~3) + (offset & ~3); zero past the end of the buffer.
			code.add(code.rax, static_cast<int32_t>(immediate & ~int64_t {3}));
			code.push(code.rax);
			if (!Emit(handle->Arg(2))) return false;
			code.mov(code.eax, code.eax);
			code.mov(code.rdx, code.rax);
			code.mov(code.rcx, code.qword[code.rsp + 8]);
			code.shr(code.rcx, 16);
			code.and_(code.ecx, 0x3fff);
			Xbyak::Label sized;
			code.test(code.ecx, code.ecx);
			code.jz(sized, Code::T_NEAR);
			code.imul(code.rdx, code.rcx);
			code.L(sized);
			code.pop(code.rax);
			code.pop(code.rcx);
			code.pop(code.rcx);
			// rax = byte_offset, rcx = base, rdx = size
			code.cmp(code.rax, code.rdx);
			code.ja(zero, Code::T_NEAR);
			code.sub(code.rdx, code.rax);
			code.cmp(code.rdx, 4);
			code.jb(zero, Code::T_NEAR);
			code.and_(code.rcx, -4);
			code.add(code.rax, code.rcx);
			code.jmp(read, Code::T_NEAR);
			code.L(zero);
			code.xor_(code.eax, code.eax);
			code.jmp(done, Code::T_NEAR);
		} else {
			code.add(code.rax, static_cast<int32_t>(immediate & ~int64_t {3}));
			code.pop(code.rcx);
			code.and_(code.rcx, -4);
			code.add(code.rax, code.rcx);
			code.mov(code.rcx, AddressMask - 3);
			code.cmp(code.rax, code.rcx);
			code.ja(m_failed, Code::T_NEAR);
		}
		code.L(read);
		code.mov(code.rsi, code.rax);
		code.mov(code.rdi, code.rbx);
		code.mov(code.edx, m_clean_depth > 0 ? 1 : 0);
		code.push(code.rbp);
		code.mov(code.rbp, code.rsp);
		code.and_(code.rsp, -16);
		code.mov(code.rax, reinterpret_cast<uint64_t>(&GeneralWalkerRead));
		code.call(code.rax);
		code.mov(code.rsp, code.rbp);
		code.pop(code.rbp);
		code.test(code.al, code.al);
		code.jz(m_failed, Code::T_NEAR);
		code.mov(code.eax, code.dword[code.rbx + 80]);
		code.L(done);
		return true;
	}

	ResourcePlan&                             m_program;
	Xbyak::CodeGenerator*                     m_code = nullptr;
	Xbyak::Label                              m_finished;
	Xbyak::Label                              m_failed;
	Xbyak::Label                              m_exit;
	static constexpr size_t                   MaxCodeSize = 4u << 20u;

	Scope                                     m_base;
	std::vector<Scope>                        m_scopes;
	std::vector<Value>                        m_masks;
	std::vector<uint32_t>                     m_assume;
	std::unordered_map<const Inst*, bool>     m_pure;
	bool                                      m_clean_evaluator = false;
	int                                       m_clean_depth     = 0;
	int                                       m_tolerant_depth  = 0;
	std::unordered_map<const Inst*, bool>     m_may_be_unknown;
	Xbyak::Label*                             m_fail = &m_failed;
	std::unordered_map<const Inst*, uint32_t> m_phi_blocks;
};

void CompileGeneralSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.srt_reads.empty()) {
		return;
	}
	GeneralWalkerCompiler compiler(program);
	if (!compiler.Compile()) {
		program.srt_general_code.reset();
	}
}

void CompileSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.capture_specialization_reads || !program.control_flow.empty() ||
	    program.srt_reads.empty()) return;

	using Reads = std::map<int32_t, const SrtRead*>;
	std::map<uint32_t, Reads> roots;
	std::unordered_map<const Inst*, Reads> pointer_uses;
	std::unordered_map<const Inst*, int32_t> offsets;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    inst->NumArgs() < 2 || !IsRawRead(program, *inst) ||
		    read.flat_offset >= program.srt_reads.size()) return;
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return;
		const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
		const auto relative =
		    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
		    static_cast<int64_t>(offset.U32() & ~3u);
		if (relative < INT32_MIN || relative > INT32_MAX ||
		    !offsets.emplace(inst, static_cast<int32_t>(relative)).second) return;
	}
	uint32_t user_data_count = 0;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) return;
		const auto* low = handle->Arg(0).Resolve().TryInstruction();
		const auto* high = handle->Arg(1).Resolve().TryInstruction();
		if (low == nullptr || high == nullptr) return;
		Reads* uses = nullptr;
		if (low->GetOpcode() == ValueOpcode::GetUserData &&
		    high->GetOpcode() == ValueOpcode::GetUserData && low->NumArgs() == 1 &&
		    high->NumArgs() == 1 && low->Arg(0).GetType() == Type::ScalarReg &&
		    high->Arg(0).GetType() == Type::ScalarReg) {
			const auto reg = RegIndex(low->Arg(0).ScalarRegister());
			if (reg < program.user_data_base ||
			    RegIndex(high->Arg(0).ScalarRegister()) != reg + 1u) return;
			const auto base = reg - program.user_data_base;
			user_data_count = std::max(user_data_count, base + 2u);
			uses = &roots[base];
		} else {
			const auto lo = offsets.find(low);
			const auto hi = offsets.find(high);
			if (lo == offsets.end() || hi == offsets.end() ||
			    static_cast<int64_t>(lo->second) + 4 != hi->second ||
			    !EquivalentValue(program, low->Arg(0), high->Arg(0))) return;
			uses = &pointer_uses[low];
		}
		if (!uses->emplace(offsets.at(inst), &read).second) return;
	}
	std::unordered_set<const Inst*> visited;
	const auto reachable = [&](auto&& self, const Reads& reads) -> bool {
		for (const auto& [offset, read]: reads) {
			const auto* inst = read->value.ResolveInstruction();
			const auto child = pointer_uses.find(inst);
			if (child != pointer_uses.end() &&
			    (!visited.insert(inst).second || !self(self, child->second))) return false;
		}
		return true;
	};
	for (const auto& [reg, reads]: roots) {
		if (!reachable(reachable, reads)) return;
	}
	if (visited.size() != pointer_uses.size()) return;

	program.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code = *program.srt_code;
	Xbyak::Label failed, finished;
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.cmp(code.r8, user_data_count);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	code.mov(code.r9, AddressMask & ~uint64_t {3});
	// Follow shadPS4's VisitPointer: copy one table level in offset order, then
	// descend through its child pointers. Seed the descriptor evaluator's memo.
	const auto visit = [&](auto&& self, int32_t pointer_offset, const Reads& reads) -> void {
		code.push(code.rdi);
		code.mov(code.rdi, code.qword[code.rdi + pointer_offset]);
		code.and_(code.rdi, code.r9);
		const auto first = reads.begin()->first;
		const auto last = reads.rbegin()->first;
		if (first < 0) {
			code.mov(code.r10, -static_cast<int64_t>(first));
			code.cmp(code.rdi, code.r10);
			code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		if (last > 0) {
			code.mov(code.r10, AddressMask - last);
			code.cmp(code.rdi, code.r10);
			code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		for (const auto& [offset, read]: reads) {
			const auto index = read->value.ResolveInstruction()->EvaluationIndex(program.evaluation_value_count);
			using Entry = ResourcePlan::EvaluationContext::Entry;
			const auto slot = code.rdx + index * sizeof(Entry);
			code.mov(code.r10d, code.dword[code.rdi + offset]);
			code.mov(code.dword[code.rsi + read->flat_offset * sizeof(uint32_t)], code.r10d);
			code.mov(code.qword[slot + offsetof(Entry, value)], code.r10);
			code.mov(code.qword[slot + offsetof(Entry, generation)], code.rcx);
		}
		for (const auto& [offset, read]: reads) {
			const auto child = pointer_uses.find(read->value.ResolveInstruction());
			if (child == pointer_uses.end()) continue;
			self(self, offset, child->second);
		}
		code.pop(code.rdi);
	};
	for (const auto& [reg, reads]: roots) {
		visit(visit, reg * sizeof(uint32_t), reads);
	}
	code.mov(code.eax, 1);
	code.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
	code.L(failed);
	code.xor_(code.eax, code.eax);
	code.L(finished);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

void CompileSrtWalkers(ResourcePlan& program) {
	CompileSrtWalker(program);
	CompileGeneralSrtWalker(program);
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     SrtWalker* clean_evaluator)
    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_session(this),
      m_read_policy(clean_evaluator != nullptr ? 2u : 0u), m_context(AcquireContext()) {}

SrtWalker::SrtWalker(SrtWalker& parent, Value active_mask, bool clean,
                     SrtWalker* clean_evaluator)
    : m_program(parent.m_program), m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
                    (clean_evaluator != nullptr ? 2u : 0u)), m_context(AcquireContext()) {}

SrtWalker::~SrtWalker() {
	if (m_session != this) return;
	for (auto& context: m_program.evaluation_contexts)
		if (context.session == this) context.session = nullptr;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext() {
	ResourcePlan::EvaluationContext* unused = nullptr;
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session == m_session && context.active_mask == m_active_mask &&
		    context.read_policy == m_read_policy) return context;
		if (context.session == nullptr && unused == nullptr) unused = &context;
	}
	auto& context = unused != nullptr ? *unused : m_program.evaluation_contexts.emplace_back();
	context.session = m_session;
	context.active_mask = m_active_mask;
	context.read_policy = m_read_policy;
	context.generation += 4;
	return context;
}

void SrtWalker::ResetUnknowns() {
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session != m_session) continue;
		for (auto& entry: context.values)
			if (entry.generation == (context.generation | 2u)) entry.generation = 0;
	}
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation in this
	// context. Only completed unknowns are retried when reachability becomes tighter.
	if (m_context.values[index].generation == (m_context.generation | 1u) ||
	    m_context.values[index].generation == (m_context.generation | 2u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = m_context.generation | 2u;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	if (const auto control = m_program.control_phi_predecessors.find(&inst);
	    control != m_program.control_phi_predecessors.end()) {
		bool found = false;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (m_reachable_blocks != nullptr &&
			    !m_reachable_blocks->at(control->second[index])) continue;
			uint64_t incoming = 0;
			if (!Arg(inst, index, incoming) || (found && incoming != result)) return false;
			result = incoming;
			found = true;
		}
		return found;
	}
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector) return false;
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			SrtWalker clean_active(*this, inst.Arg(1), true);
			SrtWalker active(*this, inst.Arg(1), false, &clean_active);
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (m_clean_evaluator != nullptr &&
				    (inst.Flags<SrtReadFlags>().clean != 0u ||
				     (inst.GetOpcode() == ValueOpcode::LoadBufferU32 &&
				      m_clean_evaluator->m_active_mask == m_active_mask))) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
			if (binary()) {
				result = inst.GetOpcode() == ValueOpcode::UMin32
				    ? std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
				    : std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				// The conversion saturates and maps NaN to zero.
				const auto value = Float32(a);
				result           = std::isnan(value) || value <= 0.0f ? 0u
				                   : value >= 4294967296.0f ? UINT32_MAX
				                                            : static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::IEqual64:
		case ValueOpcode::INotEqual64:
			if (!binary()) return false;
			result = inst.GetOpcode() == ValueOpcode::IEqual64 ? a == b : a != b;
			return true;
		case ValueOpcode::ULessThan32:
			// No U32 is below zero, including an unknown loop-carried value.
			if (!Arg(inst, 1, b)) return false;
			if (static_cast<uint32_t>(b) == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 0, a)) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			if (!Arg(inst, 0, a)) return false;
			if (a == 0u) {
				result = 1u;
				return true;
			}
			if (Arg(inst, 1, b)) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 1, b) && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction
				// true. Use a separate memo so this assumption cannot escape its RHS.
				SrtWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
				                 inst.Arg(0));
				if (active.EvaluateWide(inst.Arg(1), b) && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU32: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	if (m_program.srt_code && m_runtime.read_memory == nullptr) {
		flat.resize(m_program.srt_reads.size());
		m_program.active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		return m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		    m_context.values.data(), m_context.generation, m_runtime.user_data.size());
	}
	if (m_program.srt_general_code && m_active_mask.IsEmpty() &&
	    (m_clean_evaluator == nullptr) ==
	        !(m_program.capture_specialization_reads || !m_program.source_reads.empty() ||
	          m_program.requires_specialization_memory)) {
		const auto result = RunGeneralWalker(flat);
		if (result != 0) {
			return result > 0;
		}
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const auto* inst = read.value.ResolveInstruction();
		const bool clean = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		return read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	auto& visited = m_program.visited_blocks;
	auto& reachable = m_program.reachable_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 1u);
	ResetUnknowns();
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->ResetUnknowns();
		m_clean_evaluator->m_reachable_blocks = &visited;
	}
	for (;;) {
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) continue;
			reachable[index] = 1u;
			const auto& block = m_program.control_flow[index];
			uint32_t condition = 0;
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads) &&
			    predicate.Evaluate(block.condition, condition)) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		if (m_program.control_phi_predecessors.empty()) break;
		// Only unknown values need another evaluation: successful Phi facts remain
		// true as their possible predecessors shrink. Keep uniform-read memos stable.
		ResetUnknowns();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
	}
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow)
		for (const auto source: block.sources) active.at(source) = 0u;
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (!visited[index]) continue;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto slot: block.srt_reads)
			if (!refresh(slot)) return false;
	}
	return true;
}

// 1: refreshed; 0: run the interpreter; -1: a read failed.
int SrtWalker::RunGeneralWalker(std::vector<uint32_t>& flat) {
	const auto blocks = m_program.control_flow.size();
	m_program.jit_visited.assign(blocks, 0u);
	if (blocks != 0) m_program.jit_visited[0] = 1u;
	m_context.values.resize(m_program.evaluation_value_count);
	auto& clean_context = m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
	clean_context.values.resize(m_program.evaluation_value_count);
	flat.assign(m_program.srt_reads.size(), 0u);
	const auto assumptions = m_program.jit_assumptions * 2u;
	m_program.jit_assume_values.resize(assumptions);
	m_program.jit_assume_bases.resize(assumptions);
	for (uint32_t index = 0; index < assumptions; ++index) {
		auto& values = m_program.jit_assume_values[index];
		if (values.size() != m_program.evaluation_value_count) {
			values.assign(m_program.evaluation_value_count, {});
		}
		m_program.jit_assume_bases[index] = values.data();
	}
	GeneralWalkerContext context {
	    .user_data        = m_runtime.user_data.data(),
	    .user_data_count  = m_runtime.user_data.size(),
	    .flat             = flat.data(),
	    .values           = m_context.values.data(),
	    .generation       = m_context.generation,
	    .shader_base      = m_runtime.shader_base,
	    .visited          = m_program.jit_visited.data(),
	    .runtime          = &m_runtime,
	    .clean_values     = clean_context.values.data(),
	    .clean_generation = clean_context.generation,
	    .assume_values    = m_program.jit_assume_bases.data(),
	};
	using Walker = bool KYTY_SYSV_ABI (*)(GeneralWalkerContext*);
	if (!m_program.srt_general_code->getCode<Walker>()(&context)) {
		if (context.read_failed != 0u) {
			return -1;
		}
		// A failed run leaves memo entries of this generation behind; the interpreter
		// recomputes them under a fresh one.
		m_context.generation += 4;
		if (&clean_context != &m_context) clean_context.generation += 4;
		return 0;
	}
	auto& active = m_program.active_sources;
	if (blocks == 0) {
		active.clear();
		return 1;
	}
	// Descriptor phis evaluated after the walk see the same reachable blocks.
	m_program.visited_blocks = m_program.jit_visited;
	m_reachable_blocks       = &m_program.visited_blocks;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->m_reachable_blocks = &m_program.visited_blocks;
	}
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow)
		for (const auto source: block.sources)
			active.at(source) = 0u;
	for (uint32_t index = 0; index < blocks; ++index) {
		if (!m_program.jit_visited[index]) continue;
		for (const auto source: m_program.control_flow[index].sources)
			active[source] = 1u;
	}
	return 1;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
