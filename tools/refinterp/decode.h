// RDNA2 instruction decoder of the reference interpreter. The encodings, bit layouts and opcodes come
// from AMD's machine-readable ISA (generated/isa_tables.inc); nothing here is shared with the emulator's
// own decoder, so a decoder bug in the emulator shows up as a difference instead of being copied.
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace refinterp {

#include "generated/isa_tables.inc"

enum class Enc : uint8_t {
	SOP1,
	SOPC,
	SOPP,
	SOPK,
	SOP2,
	SMEM,
	VOP1,
	VOPC,
	VOP2,
	VINTRP,
	VOP3,
	VOP3P,
	DS,
	MUBUF,
	MTBUF,
	MIMG,
	EXP,
	FLAT,
	Unknown
};

inline const char* EncName(Enc e) {
	static const char* const names[] = {"SOP1", "SOPC", "SOPP", "SOPK",  "SOP2",  "SMEM", "VOP1", "VOPC",  "VOP2", "VINTRP",
	                                    "VOP3", "VOP3P", "DS",  "MUBUF", "MTBUF", "MIMG", "EXP",  "FLAT", "?"};
	return names[static_cast<int>(e)];
}

// Operand source codes (9-bit SRC fields, AMD RDNA2 ISA "OPR_SRC").
namespace Src {
constexpr uint32_t VccLo       = 106;
constexpr uint32_t VccHi       = 107;
constexpr uint32_t M0          = 124;
constexpr uint32_t Null        = 125;
constexpr uint32_t ExecLo      = 126;
constexpr uint32_t ExecHi      = 127;
constexpr uint32_t Zero        = 128;
constexpr uint32_t Dpp8        = 233;
constexpr uint32_t Dpp8Fi      = 234;
constexpr uint32_t SharedBase  = 235;
constexpr uint32_t SharedLimit = 236;
constexpr uint32_t PrivBase    = 237;
constexpr uint32_t PrivLimit   = 238;
constexpr uint32_t PopsWaveId  = 239;
constexpr uint32_t Sdwa        = 249;
constexpr uint32_t Dpp16       = 250;
constexpr uint32_t Vccz        = 251;
constexpr uint32_t Execz       = 252;
constexpr uint32_t Scc         = 253;
constexpr uint32_t LdsDirect   = 254;
constexpr uint32_t Literal     = 255;
constexpr uint32_t Vgpr0       = 256;
} // namespace Src

struct Inst {
	uint32_t    pc        = 0; // byte offset in the program
	uint32_t    words     = 1;
	uint32_t    raw[5]    = {};
	Enc         enc       = Enc::Unknown;
	const char* enc_name  = "";
	uint32_t    opcode    = 0;
	Op          op        = Op::INVALID;
	const char* operands  = ""; // XML operand list of this instruction encoding
	bool        sdwa      = false;
	bool        dpp16     = false;
	bool        dpp8      = false;
	bool        has_literal = false;
	uint32_t    literal   = 0;
	std::map<std::string, uint32_t, std::less<>> f; // field values of the encoding layout (SDWA/DPP dword included)

	[[nodiscard]] uint32_t F(const char* name, uint32_t fallback = 0) const {
		const auto it = f.find(name);
		return it == f.end() ? fallback : it->second;
	}
	[[nodiscard]] bool Has(const char* name) const { return f.contains(name); }
};

inline const GenEncoding* FindEncoding(const char* name) {
	for (const auto& e: kEncodings) {
		if (std::strcmp(e.name, name) == 0) return &e;
	}
	return nullptr;
}

inline uint64_t ExtractBits(const uint32_t* words, uint32_t word_count, uint32_t offset, uint32_t count) {
	uint64_t value = 0;
	for (uint32_t b = 0; b < count; b++) {
		const uint32_t bit = offset + b;
		const uint32_t w   = bit / 32u;
		if (w >= word_count) break;
		value |= static_cast<uint64_t>((words[w] >> (bit % 32u)) & 1u) << b;
	}
	return value;
}

inline void ReadFields(Inst& inst, const GenEncoding& enc) {
	for (uint32_t i = 0; i < enc.field_count; i++) {
		const auto& field = enc.fields[i];
		uint64_t    value = 0;
		uint32_t    shift = 0;
		for (uint32_t r = 0; r < field.ranges; r++) {
			value |= ExtractBits(inst.raw, inst.words, field.offset[r], field.count[r]) << shift;
			shift += field.count[r];
		}
		inst.f[field.name] = static_cast<uint32_t>(value);
	}
}

struct Decoder {
	// (encoding name, opcode) -> instruction table entry
	std::map<std::pair<std::string, uint32_t>, const GenInstruction*> table;

	Decoder() {
		for (const auto& gi: kInstructions) table[{gi.encoding, gi.opcode}] = &gi;
	}

	[[nodiscard]] const GenInstruction* Lookup(const char* enc, uint32_t opcode) const {
		const auto it = table.find({enc, opcode});
		return it == table.end() ? nullptr : it->second;
	}

	// Decodes the instruction at dword index `index` of `code`; returns false for an unknown encoding.
	bool Decode(const std::vector<uint32_t>& code, uint32_t index, Inst& inst, std::string& error) const {
		inst      = Inst {};
		inst.pc   = index * 4u;
		const uint32_t w0 = code[index];
		static const char* const base_names[] = {"ENC_SOP1", "ENC_SOPC", "ENC_SOPP", "ENC_SOPK", "ENC_SOP2", "ENC_SMEM",
		                                         "ENC_VOP1", "ENC_VOPC", "ENC_VOP2", "ENC_VINTRP", "ENC_VOP3",
		                                         "ENC_VOP3P", "ENC_DS", "ENC_MUBUF", "ENC_MTBUF", "ENC_MIMG", "ENC_EXP",
		                                         "ENC_FLAT"};
		// Priority by the XML Order attribute (SOP1 first: its identifier overlaps SOP2/SOPK).
		const GenEncoding* found = nullptr;
		int                found_index = -1;
		int                best_order  = 1 << 30;
		for (int i = 0; i < 18; i++) {
			const auto* enc = FindEncoding(base_names[i]);
			const auto  v   = (w0 >> enc->enc_offset) & ((1u << enc->enc_count) - 1u);
			if (v == enc->enc_value && enc->order < best_order) {
				best_order  = enc->order;
				found       = enc;
				found_index = i;
			}
		}
		if (found == nullptr) {
			error = "unknown encoding";
			return false;
		}
		inst.enc   = static_cast<Enc>(found_index);
		inst.words = found->bits / 32u;
		for (uint32_t w = 0; w < inst.words && index + w < code.size(); w++) inst.raw[w] = code[index + w];
		ReadFields(inst, *found);
		inst.opcode = inst.F("OP");
		const char* lookup_enc = found->name;
		// VOP3 opcodes with a scalar destination use the VOP3_SDST_ENC layout.
		if (inst.enc == Enc::VOP3 && Lookup("VOP3_SDST_ENC", inst.opcode) != nullptr) {
			const auto* sdst = FindEncoding("VOP3_SDST_ENC");
			inst.f.clear();
			ReadFields(inst, *sdst);
			lookup_enc = "VOP3_SDST_ENC";
		}
		// FLAT/GLOBAL/SCRATCH share an encoding; SEG selects the opcode table.
		if (inst.enc == Enc::FLAT) {
			const uint32_t seg = inst.F("SEG");
			lookup_enc         = seg == 2 ? "ENC_FLAT_GLBL" : seg == 1 ? "ENC_FLAT_SCRATCH" : "ENC_FLAT";
		}
		const auto* gi = Lookup(lookup_enc, inst.opcode);
		if (gi == nullptr) {
			error = std::string("unknown opcode ") + std::to_string(inst.opcode) + " in " + lookup_enc;
			return false;
		}
		inst.op       = gi->op;
		inst.operands = gi->operands;
		inst.enc_name = lookup_enc;
		// Extra dwords: SDWA / DPP (VOP1/VOP2/VOPC with SRC0 = 249 / 250 / 233 / 234), literal (a source
		// field = 255; also S_SETREG_IMM32_B32), MIMG NSA address dwords.
		auto extra = [&](const char* layout_name) {
			const auto* layout = FindEncoding(layout_name);
			inst.words         = 2;
			inst.raw[1]        = index + 1 < code.size() ? code[index + 1] : 0;
			std::map<std::string, uint32_t, std::less<>> keep = inst.f;
			ReadFields(inst, *layout);
			for (const auto& [k, v]: keep) {
				if (!inst.f.contains(k)) inst.f[k] = v;
			}
		};
		if (inst.enc == Enc::VOP1 || inst.enc == Enc::VOP2 || inst.enc == Enc::VOPC) {
			const uint32_t src0 = inst.F("SRC0");
			const char*    kind = inst.enc == Enc::VOP1 ? "VOP1" : inst.enc == Enc::VOP2 ? "VOP2" : "VOPC";
			if (src0 == Src::Sdwa) {
				inst.sdwa = true;
				if (inst.enc == Enc::VOPC) {
					extra("VOPC_VOP_SDWA_SDST_ENC");
				} else if (inst.enc == Enc::VOP2 && Lookup("VOP2_VOP_SDWA_SDST_ENC", inst.opcode) != nullptr) {
					extra("VOP2_VOP_SDWA_SDST_ENC");
				} else {
					extra(inst.enc == Enc::VOP1 ? "VOP1_VOP_SDWA" : "VOP2_VOP_SDWA");
				}
			} else if (src0 == Src::Dpp16) {
				inst.dpp16 = true;
				if (inst.enc == Enc::VOPC) {
					error = "VOPC DPP not supported";
					return false;
				}
				extra(inst.enc == Enc::VOP1 ? "VOP1_VOP_DPP16" : "VOP2_VOP_DPP16");
			} else if (src0 == Src::Dpp8 || src0 == Src::Dpp8Fi) {
				inst.dpp8 = true;
				if (inst.enc == Enc::VOPC) {
					error = "VOPC DPP8 not supported";
					return false;
				}
				extra(inst.enc == Enc::VOP1 ? "VOP1_VOP_DPP8" : "VOP2_VOP_DPP8");
			}
			(void)kind;
		}
		auto is_lit = [&](const char* name) { return inst.Has(name) && inst.F(name) == Src::Literal; };
		bool literal = false;
		switch (inst.enc) {
			case Enc::SOP1:
			case Enc::SOPC:
			case Enc::SOP2: literal = is_lit("SSRC0") || is_lit("SSRC1"); break;
			case Enc::SOPK: literal = inst.op == Op::S_SETREG_IMM32_B32; break;
			case Enc::VOP1:
			case Enc::VOP2:
			case Enc::VOPC:
				literal = !inst.sdwa && !inst.dpp16 && !inst.dpp8 && is_lit("SRC0");
				// FMAMK/FMAAK/MADMK/MADAK always carry a literal (the K constant).
				if (inst.op == Op::V_FMAMK_F32 || inst.op == Op::V_FMAAK_F32 || inst.op == Op::V_MADMK_F32 ||
				    inst.op == Op::V_MADAK_F32 || inst.op == Op::V_FMAMK_F16 || inst.op == Op::V_FMAAK_F16) {
					literal = true;
				}
				break;
			case Enc::VOP3:
			case Enc::VOP3P: literal = is_lit("SRC0") || is_lit("SRC1") || is_lit("SRC2"); break;
			default: break;
		}
		if (literal) {
			inst.has_literal = true;
			inst.literal     = index + inst.words < code.size() ? code[index + inst.words] : 0;
			inst.raw[inst.words] = inst.literal;
			inst.words++;
		}
		if (inst.enc == Enc::MIMG) {
			const uint32_t nsa = inst.F("NSA");
			for (uint32_t k = 0; k < nsa; k++) {
				inst.raw[inst.words] = index + inst.words < code.size() ? code[index + inst.words] : 0;
				inst.words++;
			}
		}
		return true;
	}
};

} // namespace refinterp
