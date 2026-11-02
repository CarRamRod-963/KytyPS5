// Wave execution of the reference interpreter (RDNA1 semantics + BVH; PS5 compute shaders).
//
// Conventions follow AMD's RDNA ISA: EXEC masks lanes, VCC/EXEC are 64-bit in wave64 and 32-bit in
// wave32, FP32 denormals per FLOAT_MODE (0xc0 = FP32 flushed, FP16/FP64 kept), legacy V_MAD/V_MAC are
// not fused, V_FMA* are fused. Hardware approximations (RCP/RSQ/SQRT/EXP/LOG/SIN/COS) are computed in
// double and rounded, so they can differ by ~1 ulp from the GPU; the report counts such ops.
#pragma once

#include "capture.h"
#include "decode.h"
#include "fp.h"
#include "image.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace refinterp {

struct Options {
	std::vector<std::array<uint32_t, 3>> groups;
	bool     all_groups  = false;
	uint32_t sample      = 16;
	int      trace_group = -1; // index into the run list
	int      trace_lane  = -1;
	bool     prec8       = false;
	uint32_t max_report  = 20;
	uint64_t max_steps   = 200000000ull;
	float    force_compare = -1.0f; // debug: compare-sample result override (<0 = off)
	bool     compare_flip  = false;
	uint64_t watch_addr  = 0; // debug: report every buffer/global store covering this guest address
	// debug (--zero-op NAME[,NAME] / --zero-pc HEX[,HEX]): the listed load instructions return zeros (vector
	// destinations of the active lanes), to find which input a wrong (all-zero) GPU result could come from.
	std::vector<std::string> zero_ops;
	// debug (--probe PC:vN|sN[:COUNT]): same as the emulator's KYTY_DIAG_PROBE: COUNT registers from N are copied
	// into v252.. after the instruction at PC (byte offset), every IMAGE_STORE writes v252.. instead of its data,
	// lanes that never execute PC store -100.0.
	struct ProbePoint {
		uint32_t pc = 0;
		bool     scalar = false, to_float = false;
		uint32_t reg = 0, count = 1, slot = 0;
	};
	std::vector<ProbePoint> probes; // PC:vN[:COUNT][+PC:vN...]: points fill v252..v255 in order
	// debug (--probe-nth N, emulator KYTY_DIAG_PROBE_NTH): copy only at the N-th execution of the probe pc per lane
	// (1 = first; 0 = after every execution, the last one wins). Counters in v248..v251 (one per distinct pc).
	uint32_t probe_nth = 0;
	std::vector<uint32_t>    zero_pcs;
	bool     pc_profile = false; // --pc-profile FILE: wave executions and active lanes per instruction (pc_hits)
	int      watch_image = -1, watch_x = 0, watch_y = 0, watch_z = 0; // debug: report the lane storing this texel // debug: texel OP ref instead of ref OP texel
};

struct Wave {
	uint32_t lanes = 64;
	uint32_t s[128] {};
	uint64_t exec = 0, vcc = 0;
	bool     scc  = false;
	uint32_t m0   = 0;
	std::vector<std::array<uint32_t, 64>> v = std::vector<std::array<uint32_t, 64>>(256);
	uint32_t pc      = 0; // dword index
	bool     done    = false;
	bool     barrier = false;
	uint32_t id      = 0;
	uint64_t steps   = 0;

	[[nodiscard]] uint64_t Mask() const { return lanes == 64 ? ~0ull : 0xffffffffull; }
	[[nodiscard]] bool Active(uint32_t lane) const { return ((exec >> lane) & 1u) != 0; }
};

struct Group {
	uint32_t             id[3] {};
	std::vector<uint8_t> lds;
	std::vector<uint8_t> lds_written; // 1 per LDS byte stored by this group (reads of others are reported)
	std::vector<Wave>    waves;
};

class Interpreter {
public:
	Interpreter(Capture& capture, Options options): m_cap(capture), m_opt(std::move(options)) {
		ftz32 = ((m_cap.float_mode >> 4u) & 3u) == 0u;
	}

	std::map<std::string, uint64_t> unsupported;
	std::map<std::string, uint64_t> approx_ops;
	std::vector<std::string>        errors;
	uint64_t                        total_steps = 0;
	std::map<uint32_t, std::array<uint64_t, 2>> pc_hits; // byte pc -> {wave executions, active lanes}
	bool                            tracing     = false;
	uint32_t                        m_cur_group[3] {0, 0, 0}; // group being run (WATCH reports)
	int                             trace_lane  = -1;

	bool RunGroup(uint32_t gx, uint32_t gy, uint32_t gz) {
		Group group;
		group.id[0] = gx;
		group.id[1] = gy;
		group.id[2] = gz;
		m_cur_group[0] = gx, m_cur_group[1] = gy, m_cur_group[2] = gz;
		group.lds.assign(static_cast<size_t>(std::max(m_cap.lds_dwords, 1u)) * 4u + 65536u, 0);
		group.lds_written.assign(group.lds.size(), 0);
		const uint32_t threads = m_cap.threads[0] * m_cap.threads[1] * m_cap.threads[2];
		const uint32_t lanes   = m_cap.wave_size;
		const uint32_t waves   = (threads + lanes - 1u) / lanes;
		group.waves.resize(waves);
		for (uint32_t w = 0; w < waves; w++) {
			auto& wave = group.waves[w];
			wave.lanes = lanes;
			wave.id    = w;
			uint32_t sg = 0;
			for (uint32_t i = 0; i < m_cap.user_sgpr && i < m_cap.user_data.size(); i++) wave.s[sg++] = m_cap.user_data[i];
			sg = m_cap.user_sgpr;
			if (m_cap.tgid_en[0]) wave.s[sg++] = gx;
			if (m_cap.tgid_en[1]) wave.s[sg++] = gy;
			if (m_cap.tgid_en[2]) wave.s[sg++] = gz;
			// TG_SIZE: [5:0] waves in the group, [11:6] wave index (AMD GFX10 compute layout).
			if (m_cap.tg_size_en) wave.s[sg++] = (waves & 0x3fu) | ((w & 0x3fu) << 6u);
			for (uint32_t l = 0; l < lanes; l++) {
				const uint32_t t = w * lanes + l;
				if (t >= threads) continue;
				wave.exec |= 1ull << l;
				const uint32_t x = t % m_cap.threads[0];
				const uint32_t y = (t / m_cap.threads[0]) % m_cap.threads[1];
				const uint32_t z = t / (m_cap.threads[0] * m_cap.threads[1]);
				wave.v[0][l]     = x;
				if (m_cap.tidig_comp_cnt >= 1) wave.v[1][l] = y;
				if (m_cap.tidig_comp_cnt >= 2) wave.v[2][l] = z;
			}
		}
		// Cooperative scheduling: run every wave until it ends or waits at a barrier.
		while (true) {
			bool progress = false;
			bool all_done = true;
			for (auto& wave: group.waves) {
				while (!wave.done && !wave.barrier) {
					if (!Step(wave, group)) return false;
					progress = true;
				}
				all_done = all_done && wave.done;
			}
			if (all_done) break;
			bool all_waiting = true;
			for (auto& wave: group.waves) all_waiting = all_waiting && (wave.done || wave.barrier);
			if (all_waiting) {
				for (auto& wave: group.waves) wave.barrier = false;
				progress = true;
			}
			if (!progress) {
				errors.push_back("deadlock in group");
				return false;
			}
		}
		return true;
	}

private:
	Capture& m_cap;
	Options  m_opt;
	Decoder  m_dec;
	std::map<uint32_t, Inst> m_insts;
	bool     ftz32 = true;

	const Inst* Fetch(uint32_t index) {
		auto it = m_insts.find(index);
		if (it != m_insts.end()) return &it->second;
		if (index >= m_cap.code.size()) return nullptr;
		Inst        inst;
		std::string error;
		if (!m_dec.Decode(m_cap.code, index, inst, error)) {
			errors.push_back("decode failed at pc 0x" + Hex(index * 4u) + ": " + error);
			return nullptr;
		}
		return &m_insts.emplace(index, std::move(inst)).first->second;
	}

	static std::string Hex(uint64_t v) {
		char text[32];
		std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(v));
		return text;
	}

	void Unsupported(const Inst& inst, const char* what = "") {
		unsupported[std::string(kOpNames[static_cast<int>(inst.op)]) + what]++;
	}

	// ---------------------------------------------------------------- scalar operands
	uint32_t ReadS(const Wave& w, uint32_t code, const Inst& inst) const {
		if (code <= 105) return w.s[code];
		switch (code) {
			case Src::VccLo: return static_cast<uint32_t>(w.vcc);
			case Src::VccHi: return static_cast<uint32_t>(w.vcc >> 32u);
			case Src::M0: return w.m0;
			case Src::Null: return 0;
			case Src::ExecLo: return static_cast<uint32_t>(w.exec);
			case Src::ExecHi: return static_cast<uint32_t>(w.exec >> 32u);
			case Src::Vccz: return w.vcc == 0 ? 1u : 0u;
			case Src::Execz: return w.exec == 0 ? 1u : 0u;
			case Src::Scc: return w.scc ? 1u : 0u;
			case Src::Literal: return inst.literal;
			default: break;
		}
		if (code >= 128 && code <= 192) return code - 128u;
		if (code >= 193 && code <= 208) return static_cast<uint32_t>(-static_cast<int32_t>(code - 192u));
		if (code >= 240 && code <= 248) return InlineF32(code);
		if (code >= 108 && code <= 123) return 0; // TTMP
		return 0;
	}
	uint64_t ReadS64(const Wave& w, uint32_t code, const Inst& inst) const {
		if (code <= 104) return w.s[code] | (static_cast<uint64_t>(w.s[code + 1u]) << 32u);
		if (code == Src::VccLo) return w.vcc;
		if (code == Src::ExecLo) return w.exec;
		if (code >= 128 && code <= 192) return code - 128u;
		if (code >= 193 && code <= 208) return static_cast<uint64_t>(-static_cast<int64_t>(code - 192u));
		if (code >= 240 && code <= 248) return InlineF64Bits(code);
		if (code == Src::Literal) return inst.literal;
		if (code == Src::Scc) return w.scc ? 1u : 0u;
		if (code == Src::Vccz) return w.vcc == 0 ? 1u : 0u;
		if (code == Src::Execz) return w.exec == 0 ? 1u : 0u;
		return ReadS(w, code, inst);
	}
	void WriteS(Wave& w, uint32_t code, uint32_t value) const {
		if (code <= 105) {
			w.s[code] = value;
			return;
		}
		switch (code) {
			case Src::VccLo: w.vcc = (w.vcc & 0xffffffff00000000ull) | value; break;
			case Src::VccHi: w.vcc = (w.vcc & 0xffffffffull) | (static_cast<uint64_t>(value) << 32u); break;
			case Src::M0: w.m0 = value; break;
			case Src::ExecLo: w.exec = ((w.exec & 0xffffffff00000000ull) | value) & w.Mask(); break;
			case Src::ExecHi: w.exec = ((w.exec & 0xffffffffull) | (static_cast<uint64_t>(value) << 32u)) & w.Mask(); break;
			default: break;
		}
	}
	void WriteS64(Wave& w, uint32_t code, uint64_t value) const {
		if (code <= 104) {
			w.s[code]      = static_cast<uint32_t>(value);
			w.s[code + 1u] = static_cast<uint32_t>(value >> 32u);
		} else if (code == Src::VccLo) {
			w.vcc = value;
		} else if (code == Src::ExecLo) {
			w.exec = value & w.Mask();
		} else if (code != Src::Null) {
			WriteS(w, code, static_cast<uint32_t>(value));
		}
	}
	// Lane masks (EXEC/VCC-sized) as scalar operands: 64-bit in wave64, 32-bit in wave32.
	uint64_t ReadMask(const Wave& w, uint32_t code, const Inst& inst) const {
		return w.lanes == 64 ? ReadS64(w, code, inst) : ReadS(w, code, inst);
	}
	void WriteMask(Wave& w, uint32_t code, uint64_t value) const {
		if (w.lanes == 64) WriteS64(w, code, value);
		else WriteS(w, code, static_cast<uint32_t>(value));
	}

	static uint32_t InlineF32(uint32_t code) {
		static const uint32_t table[] = {0x3f000000u, 0xbf000000u, 0x3f800000u, 0xbf800000u, 0x40000000u,
		                                 0xc0000000u, 0x40800000u, 0xc0800000u, 0x3e22f983u};
		return table[code - 240u];
	}
	static uint32_t InlineF16(uint32_t code) {
		static const uint32_t table[] = {0x3800u, 0xb800u, 0x3c00u, 0xbc00u, 0x4000u, 0xc000u, 0x4400u, 0xc400u, 0x3118u};
		return table[code - 240u];
	}
	static uint64_t InlineF64Bits(uint32_t code) {
		static const double table[] = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.15915494309189532};
		return B64(table[code - 240u]);
	}

	// ---------------------------------------------------------------- vector operand types
	enum class Ty { B32, F32, F16, I16, U16, PkF16, PkI16, B64, F64 };

	static Ty TypeOf(const std::string& fmt) {
		if (fmt == "FMT_NUM_F32") return Ty::F32;
		if (fmt == "FMT_NUM_F16") return Ty::F16;
		if (fmt == "FMT_NUM_I16") return Ty::I16;
		if (fmt == "FMT_NUM_U16" || fmt == "FMT_NUM_B16") return Ty::U16;
		if (fmt == "FMT_NUM_PK2_F16") return Ty::PkF16;
		if (fmt == "FMT_NUM_PK2_I16" || fmt == "FMT_NUM_PK2_U16" || fmt == "FMT_NUM_PK2_B16") return Ty::PkI16;
		if (fmt == "FMT_NUM_F64") return Ty::F64;
		if (fmt == "FMT_NUM_B64" || fmt == "FMT_NUM_U64" || fmt == "FMT_NUM_I64") return Ty::B64;
		return Ty::B32;
	}

	struct OpInfo {
		Ty       src[3] {Ty::B32, Ty::B32, Ty::B32};
		Ty       dst = Ty::B32;
		uint32_t nsrc = 0;
	};
	std::map<const char*, OpInfo> m_opinfo;

	const OpInfo& Info(const Inst& inst) {
		auto it = m_opinfo.find(inst.operands);
		if (it != m_opinfo.end()) return it->second;
		OpInfo      info;
		std::string list(inst.operands);
		size_t      start = 0;
		while (start < list.size()) {
			auto end = list.find(' ', start);
			if (end == std::string::npos) end = list.size();
			const auto op = list.substr(start, end - start);
			start         = end + 1;
			// FIELD:TYPE:SIZE:FMT:io
			std::string parts[5];
			size_t      p = 0, k = 0;
			while (k < 5) {
				const auto q = op.find(':', p);
				parts[k++]   = op.substr(p, q == std::string::npos ? std::string::npos : q - p);
				if (q == std::string::npos) break;
				p = q + 1;
			}
			const Ty ty = TypeOf(parts[3]);
			if (parts[4] == "o") {
				if (parts[0] == "VDST" || parts[0] == "SDST" || parts[0] == "-") info.dst = ty;
			} else if (parts[0] == "SRC0" || parts[0] == "LITERAL" || parts[0] == "SSRC0") {
				info.src[0] = ty;
				info.nsrc   = std::max(info.nsrc, 1u);
			} else if (parts[0] == "SRC1" || parts[0] == "VSRC1" || parts[0] == "SSRC1") {
				info.src[1] = ty;
				info.nsrc   = std::max(info.nsrc, 2u);
			} else if (parts[0] == "SRC2") {
				info.src[2] = ty;
				info.nsrc   = std::max(info.nsrc, 3u);
			}
		}
		return m_opinfo.emplace(inst.operands, info).first->second;
	}

	// Raw 32-bit value of a 9-bit source for one lane (inline constants encoded for the operand type).
	uint32_t ReadV(const Wave& w, uint32_t code, uint32_t lane, Ty ty, const Inst& inst) const {
		if (code >= Src::Vgpr0) return w.v[code - Src::Vgpr0][lane];
		if (code >= 240 && code <= 248) {
			switch (ty) {
				case Ty::F16: return InlineF16(code);
				case Ty::PkF16: return InlineF16(code) | (InlineF16(code) << 16u);
				case Ty::F64: return static_cast<uint32_t>(InlineF64Bits(code) >> 32u);
				default: return InlineF32(code);
			}
		}
		if (code == Src::Literal && ty == Ty::F64) return inst.literal; // high dword of a f64 literal
		return ReadS(w, code, inst);
	}
	uint64_t ReadV64(const Wave& w, uint32_t code, uint32_t lane, Ty ty, const Inst& inst) const {
		if (code >= Src::Vgpr0) return w.v[code - Src::Vgpr0][lane] | (static_cast<uint64_t>(w.v[code - Src::Vgpr0 + 1u][lane]) << 32u);
		if (code >= 240 && code <= 248) return ty == Ty::F64 ? InlineF64Bits(code) : static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(InlineF32(code))));
		if (code == Src::Literal) return ty == Ty::F64 ? (static_cast<uint64_t>(inst.literal) << 32u) : inst.literal;
		return ReadS64(w, code, inst);
	}

	// ---------------------------------------------------------------- float helpers
	float In32(uint32_t bits) const {
		float f = F32(bits);
		return ftz32 ? FlushF32(f) : f;
	}
	uint32_t Out32(float f) const { return B32(ftz32 ? FlushF32(f) : f); }

	static uint32_t ApplyMod32(uint32_t bits, bool abs, bool neg) {
		if (abs) bits &= 0x7fffffffu;
		if (neg) bits ^= 0x80000000u;
		return bits;
	}
	static uint32_t ApplyMod16(uint32_t bits, bool abs, bool neg) {
		if (abs) bits &= 0x7fffu;
		if (neg) bits ^= 0x8000u;
		return bits & 0xffffu;
	}

	uint32_t Omod32(uint32_t bits, uint32_t omod, bool clamp) const {
		float f = F32(bits);
		if (omod == 1) f *= 2.0f;
		else if (omod == 2) f *= 4.0f;
		else if (omod == 3) f *= 0.5f;
		if (clamp) f = ClampF(f, m_cap.dx10_clamp);
		return Out32(f);
	}
	uint32_t Omod16(uint32_t bits, uint32_t omod, bool clamp) const {
		float f = F16ToF32(static_cast<uint16_t>(bits));
		if (omod == 0 && !clamp) return bits & 0xffffu;
		if (omod == 1) f *= 2.0f;
		else if (omod == 2) f *= 4.0f;
		else if (omod == 3) f *= 0.5f;
		if (clamp) f = ClampF(f, m_cap.dx10_clamp);
		return F32ToF16(f);
	}

	// ---------------------------------------------------------------- the ALU of one lane
	// a/b/c: modified sources (float sign modifiers already applied). Returns the 32-bit result.
	// carry_in: VCC/SGPR bit for *_CI ops; carry_out set for *_CO ops. ok=false when unsupported.
	uint32_t Alu(const Inst& inst, Op op, uint32_t a, uint32_t b, uint32_t c, uint32_t d_old, uint32_t lane, bool carry_in,
	             bool& carry_out, bool& ok) {
		auto f  = [&](uint32_t x) { return In32(x); };
		auto o  = [&](float x) { return Out32(x); };
		auto h  = [](uint32_t x) { return static_cast<double>(F16ToF32(static_cast<uint16_t>(x & 0xffffu))); };
		auto oh = [](double x) { return static_cast<uint32_t>(F64ToF16(x)); };
		const uint32_t K = inst.literal;
		switch (op) {
			// ---- moves / selects
			case Op::V_MOV_B32: return a;
			case Op::V_NOT_B32: return ~a;
			case Op::V_BFREV_B32: {
				uint32_t r = 0;
				for (int i = 0; i < 32; i++) r |= ((a >> i) & 1u) << (31 - i);
				return r;
			}
			// ---- integer
			case Op::V_ADD_NC_U32: return a + b;
			case Op::V_SUB_NC_U32: return a - b;
			case Op::V_SUBREV_NC_U32: return b - a;
			case Op::V_ADD_CO_U32: {
				const uint64_t r = static_cast<uint64_t>(a) + b;
				carry_out        = (r >> 32u) != 0;
				return static_cast<uint32_t>(r);
			}
			case Op::V_SUB_CO_U32: carry_out = b > a; return a - b;
			case Op::V_SUBREV_CO_U32: carry_out = a > b; return b - a;
			case Op::V_ADD_CO_CI_U32: {
				const uint64_t r = static_cast<uint64_t>(a) + b + (carry_in ? 1u : 0u);
				carry_out        = (r >> 32u) != 0;
				return static_cast<uint32_t>(r);
			}
			case Op::V_SUB_CO_CI_U32: {
				const uint64_t sub = static_cast<uint64_t>(b) + (carry_in ? 1u : 0u);
				carry_out          = sub > a;
				return static_cast<uint32_t>(static_cast<uint64_t>(a) - sub);
			}
			case Op::V_SUBREV_CO_CI_U32: {
				const uint64_t sub = static_cast<uint64_t>(a) + (carry_in ? 1u : 0u);
				carry_out          = sub > b;
				return static_cast<uint32_t>(static_cast<uint64_t>(b) - sub);
			}
			case Op::V_ADD_NC_I32: return static_cast<uint32_t>(static_cast<int32_t>(a) + static_cast<int32_t>(b));
			case Op::V_SUB_NC_I32: return static_cast<uint32_t>(static_cast<int32_t>(a) - static_cast<int32_t>(b));
			case Op::V_ADD3_U32: return a + b + c;
			case Op::V_LSHL_ADD_U32: return (a << (b & 31u)) + c;
			case Op::V_ADD_LSHL_U32: return (a + b) << (c & 31u);
			case Op::V_LSHL_OR_B32: return (a << (b & 31u)) | c;
			case Op::V_AND_OR_B32: return (a & b) | c;
			case Op::V_OR3_B32: return a | b | c;
			case Op::V_XOR3_B32: return a ^ b ^ c;
			case Op::V_XAD_U32: return (a ^ b) + c;
			case Op::V_AND_B32: return a & b;
			case Op::V_OR_B32: return a | b;
			case Op::V_XOR_B32: return a ^ b;
			case Op::V_XNOR_B32: return ~(a ^ b);
			case Op::V_LSHLREV_B32: return b << (a & 31u);
			case Op::V_LSHRREV_B32: return b >> (a & 31u);
			case Op::V_ASHRREV_I32: return static_cast<uint32_t>(static_cast<int32_t>(b) >> (a & 31u));
			case Op::V_MUL_LO_U32: return a * b;
			case Op::V_MUL_HI_U32: return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32u);
			case Op::V_MUL_HI_I32: return static_cast<uint32_t>((static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b)) >> 32u);
			case Op::V_MUL_U32_U24: return (a & 0xffffffu) * (b & 0xffffffu);
			case Op::V_MUL_HI_U32_U24: return static_cast<uint32_t>((static_cast<uint64_t>(a & 0xffffffu) * (b & 0xffffffu)) >> 32u);
			case Op::V_MUL_I32_I24: {
				const int32_t x = static_cast<int32_t>(a << 8u) >> 8u, y = static_cast<int32_t>(b << 8u) >> 8u;
				return static_cast<uint32_t>(x * y);
			}
			case Op::V_MUL_HI_I32_I24: {
				const int64_t x = static_cast<int32_t>(a << 8u) >> 8u, y = static_cast<int32_t>(b << 8u) >> 8u;
				return static_cast<uint32_t>((x * y) >> 32u);
			}
			case Op::V_MAD_U32_U24: return (a & 0xffffffu) * (b & 0xffffffu) + c;
			case Op::V_MAD_I32_I24: {
				const int32_t x = static_cast<int32_t>(a << 8u) >> 8u, y = static_cast<int32_t>(b << 8u) >> 8u;
				return static_cast<uint32_t>(x * y + static_cast<int32_t>(c));
			}
			case Op::V_BFE_U32: {
				const uint32_t off = b & 31u, width = c & 31u;
				return width == 0 ? 0u : (a >> off) & ((1u << width) - 1u);
			}
			case Op::V_BFE_I32: {
				const uint32_t off = b & 31u, width = c & 31u;
				if (width == 0) return 0;
				const uint32_t v = (a >> off) & ((1u << width) - 1u);
				return static_cast<uint32_t>(static_cast<int32_t>(v << (32u - width)) >> (32u - width));
			}
			case Op::V_BFI_B32: return (a & b) | (~a & c);
			case Op::V_BFM_B32: return ((1u << (a & 31u)) - 1u) << (b & 31u);
			case Op::V_ALIGNBIT_B32: return static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32u) | b) >> (c & 31u));
			case Op::V_ALIGNBYTE_B32: return static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32u) | b) >> (8u * (c & 3u)));
			case Op::V_PERM_B32: {
				const uint64_t bytes = (static_cast<uint64_t>(a) << 32u) | b;
				uint32_t       r     = 0;
				for (int i = 0; i < 4; i++) {
					const uint32_t sel = (c >> (8 * i)) & 0xffu;
					uint32_t       v   = 0;
					if (sel <= 7) v = static_cast<uint32_t>((bytes >> (8u * sel)) & 0xffu);
					else if (sel == 8) v = ((bytes >> 15u) & 1u) ? 0xffu : 0u;
					else if (sel == 9) v = ((bytes >> 31u) & 1u) ? 0xffu : 0u;
					else if (sel == 10) v = ((bytes >> 47u) & 1u) ? 0xffu : 0u;
					else if (sel == 11) v = ((bytes >> 63u) & 1u) ? 0xffu : 0u;
					else if (sel == 12) v = 0;
					else v = 0xffu;
					r |= v << (8 * i);
				}
				return r;
			}
			case Op::V_FFBL_B32: return a == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countr_zero(a));
			case Op::V_FFBH_U32: return a == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countl_zero(a));
			case Op::V_FFBH_I32: {
				const auto s = static_cast<int32_t>(a);
				if (s == 0 || s == -1) return 0xffffffffu;
				return static_cast<uint32_t>(std::countl_zero(static_cast<uint32_t>(s < 0 ? ~a : a))) - 1u;
			}
			case Op::V_BCNT_U32_B32: return static_cast<uint32_t>(std::popcount(a)) + b;
			case Op::V_MBCNT_LO_U32_B32: {
				const uint32_t mask = lane >= 32 ? 0xffffffffu : ((1u << lane) - 1u);
				return static_cast<uint32_t>(std::popcount(a & mask)) + b;
			}
			case Op::V_MBCNT_HI_U32_B32: {
				const uint32_t mask = lane < 32 ? 0u : (lane >= 64 ? 0xffffffffu : ((1u << (lane - 32u)) - 1u));
				return static_cast<uint32_t>(std::popcount(a & mask)) + b;
			}
			case Op::V_MIN_I32: return static_cast<uint32_t>(std::min(static_cast<int32_t>(a), static_cast<int32_t>(b)));
			case Op::V_MAX_I32: return static_cast<uint32_t>(std::max(static_cast<int32_t>(a), static_cast<int32_t>(b)));
			case Op::V_MIN_U32: return std::min(a, b);
			case Op::V_MAX_U32: return std::max(a, b);
			case Op::V_MIN3_I32: return static_cast<uint32_t>(std::min({static_cast<int32_t>(a), static_cast<int32_t>(b), static_cast<int32_t>(c)}));
			case Op::V_MAX3_I32: return static_cast<uint32_t>(std::max({static_cast<int32_t>(a), static_cast<int32_t>(b), static_cast<int32_t>(c)}));
			case Op::V_MIN3_U32: return std::min({a, b, c});
			case Op::V_MAX3_U32: return std::max({a, b, c});
			case Op::V_MED3_I32: {
				const int32_t x = static_cast<int32_t>(a), y = static_cast<int32_t>(b), z = static_cast<int32_t>(c);
				return static_cast<uint32_t>(std::max(std::min(x, y), std::min(std::max(x, y), z)));
			}
			case Op::V_MED3_U32: return std::max(std::min(a, b), std::min(std::max(a, b), c));
			case Op::V_SAD_U32: return (a > b ? a - b : b - a) + c;
			case Op::V_CVT_F32_U32: return o(static_cast<float>(a));
			case Op::V_CVT_F32_I32: return o(static_cast<float>(static_cast<int32_t>(a)));
			case Op::V_CVT_OFF_F32_I4: return o(static_cast<float>(static_cast<int32_t>(a << 28u) >> 28) / 16.0f);
			case Op::V_CVT_U32_F32: return CvtU32F32(f(a));
			case Op::V_CVT_I32_F32: return static_cast<uint32_t>(CvtI32F32(f(a)));
			case Op::V_CVT_RPI_I32_F32: return static_cast<uint32_t>(CvtI32F32(std::floor(f(a) + 0.5f)));
			case Op::V_CVT_FLR_I32_F32: return static_cast<uint32_t>(CvtI32F32(std::floor(f(a))));
			case Op::V_CVT_F32_UBYTE0: return o(static_cast<float>(a & 0xffu));
			case Op::V_CVT_F32_UBYTE1: return o(static_cast<float>((a >> 8u) & 0xffu));
			case Op::V_CVT_F32_UBYTE2: return o(static_cast<float>((a >> 16u) & 0xffu));
			case Op::V_CVT_F32_UBYTE3: return o(static_cast<float>(a >> 24u));
			case Op::V_CVT_PK_U16_U32: return (std::min(a, 0xffffu)) | (std::min(b, 0xffffu) << 16u);
			case Op::V_CVT_PK_I16_I32: {
				auto s = [](uint32_t x) { return static_cast<uint32_t>(std::clamp(static_cast<int32_t>(x), -32768, 32767)) & 0xffffu; };
				return s(a) | (s(b) << 16u);
			}
			case Op::V_CVT_PKNORM_I16_F32: {
				auto q = [&](uint32_t x) { float v = f(x); v = std::isnan(v) ? 0.0f : std::clamp(v, -1.0f, 1.0f); return static_cast<uint32_t>(static_cast<int32_t>(std::nearbyint(v * 32767.0f))) & 0xffffu; };
				return q(a) | (q(b) << 16u);
			}
			case Op::V_CVT_PKNORM_U16_F32: {
				auto q = [&](uint32_t x) { float v = f(x); v = std::isnan(v) ? 0.0f : std::clamp(v, 0.0f, 1.0f); return static_cast<uint32_t>(std::nearbyint(v * 65535.0f)) & 0xffffu; };
				return q(a) | (q(b) << 16u);
			}
			case Op::V_CVT_PK_U8_F32: {
				const uint32_t byte = b & 3u;
				const float    v    = f(a);
				const uint32_t q    = CvtU32F32(std::nearbyint(std::isnan(v) ? 0.0f : std::clamp(v, 0.0f, 255.0f)));
				return (c & ~(0xffu << (8u * byte))) | ((q & 0xffu) << (8u * byte));
			}
			// ---- FP32
			case Op::V_ADD_F32: return o(f(a) + f(b));
			case Op::V_SUB_F32: return o(f(a) - f(b));
			case Op::V_SUBREV_F32: return o(f(b) - f(a));
			case Op::V_MUL_F32: return o(f(a) * f(b));
			case Op::V_MUL_LEGACY_F32: {
				const float x = f(a), y = f(b);
				return (x == 0.0f || y == 0.0f) ? 0u : o(x * y);
			}
			case Op::V_MIN_F32: return o(MinF(f(a), f(b)));
			case Op::V_MAX_F32: return o(MaxF(f(a), f(b)));
			case Op::V_MIN3_F32: return o(MinF(MinF(f(a), f(b)), f(c)));
			case Op::V_MAX3_F32: return o(MaxF(MaxF(f(a), f(b)), f(c)));
			case Op::V_MED3_F32: return o(Med3F(f(a), f(b), f(c)));
			case Op::V_FMA_F32: return o(std::fma(f(a), f(b), f(c)));
			case Op::V_FMAC_F32: return o(std::fma(f(a), f(b), f(d_old)));
			case Op::V_FMAAK_F32: return o(std::fma(f(a), f(b), f(K)));
			case Op::V_FMAMK_F32: return o(std::fma(f(a), f(K), f(b)));
			// Legacy MAD/MAC: not fused, denormals flushed (product and sum each rounded).
			case Op::V_MAD_F32: return B32(FlushF32(FlushF32(FlushF32(F32(a)) * FlushF32(F32(b))) + FlushF32(F32(c))));
			case Op::V_MAC_F32: return B32(FlushF32(FlushF32(FlushF32(F32(a)) * FlushF32(F32(b))) + FlushF32(F32(d_old))));
			case Op::V_MADAK_F32: return B32(FlushF32(FlushF32(FlushF32(F32(a)) * FlushF32(F32(b))) + FlushF32(F32(K))));
			case Op::V_MADMK_F32: return B32(FlushF32(FlushF32(FlushF32(F32(a)) * FlushF32(F32(K))) + FlushF32(F32(b))));
			case Op::V_MAD_LEGACY_F32:
			case Op::V_MAC_LEGACY_F32: {
				const float x = FlushF32(F32(a)), y = FlushF32(F32(b));
				const float acc = FlushF32(F32(op == Op::V_MAC_LEGACY_F32 ? d_old : c));
				const float p   = (x == 0.0f || y == 0.0f) ? 0.0f : FlushF32(x * y);
				return B32(FlushF32(p + acc));
			}
			case Op::V_RCP_F32: approx_ops["V_RCP_F32"]++; return o(static_cast<float>(1.0 / static_cast<double>(f(a))));
			case Op::V_RCP_IFLAG_F32: approx_ops["V_RCP_IFLAG_F32"]++; return o(static_cast<float>(1.0 / static_cast<double>(f(a))));
			case Op::V_RSQ_F32: approx_ops["V_RSQ_F32"]++; return o(static_cast<float>(1.0 / std::sqrt(static_cast<double>(f(a)))));
			case Op::V_SQRT_F32: approx_ops["V_SQRT_F32"]++; return o(static_cast<float>(std::sqrt(static_cast<double>(f(a)))));
			case Op::V_EXP_F32: approx_ops["V_EXP_F32"]++; return o(static_cast<float>(std::exp2(static_cast<double>(f(a)))));
			case Op::V_LOG_F32: approx_ops["V_LOG_F32"]++; return o(static_cast<float>(std::log2(static_cast<double>(f(a)))));
			case Op::V_SIN_F32: approx_ops["V_SIN_F32"]++; return o(static_cast<float>(std::sin(static_cast<double>(f(a)) * 6.283185307179586)));
			case Op::V_COS_F32: approx_ops["V_COS_F32"]++; return o(static_cast<float>(std::cos(static_cast<double>(f(a)) * 6.283185307179586)));
			case Op::V_FLOOR_F32: return o(std::floor(f(a)));
			case Op::V_CEIL_F32: return o(std::ceil(f(a)));
			case Op::V_TRUNC_F32: return o(std::trunc(f(a)));
			case Op::V_RNDNE_F32: return o(RndneF(f(a)));
			case Op::V_FRACT_F32: return o(FractF(f(a)));
			case Op::V_LDEXP_F32: return o(std::ldexp(f(a), static_cast<int32_t>(b)));
			case Op::V_FREXP_MANT_F32: {
				const float x = f(a);
				if (std::isnan(x) || std::isinf(x)) return B32(x);
				int e = 0;
				return o(std::frexp(x, &e));
			}
			case Op::V_FREXP_EXP_I32_F32: {
				const float x = f(a);
				if (std::isnan(x) || std::isinf(x) || x == 0.0f) return 0;
				int e = 0;
				std::frexp(x, &e);
				return static_cast<uint32_t>(e);
			}
			case Op::V_CVT_F16_F32: return (d_old & 0xffff0000u) | F32ToF16(f(a));
			case Op::V_CVT_F32_F16: return o(F16ToF32(static_cast<uint16_t>(a)));
			case Op::V_CVT_PKRTZ_F16_F32: return F32ToF16(f(a), true) | (static_cast<uint32_t>(F32ToF16(f(b), true)) << 16u);
			case Op::V_CVT_PKNORM_I16_F16:
			case Op::V_CVT_PKNORM_U16_F16: {
				const bool s = op == Op::V_CVT_PKNORM_I16_F16;
				auto q = [&](uint32_t x) {
					double v = h(x);
					if (std::isnan(v)) v = 0;
					v = std::clamp(v, s ? -1.0 : 0.0, 1.0);
					return static_cast<uint32_t>(static_cast<int32_t>(std::nearbyint(v * (s ? 32767.0 : 65535.0)))) & 0xffffu;
				};
				return q(a) | (q(b) << 16u);
			}
			case Op::V_PACK_B32_F16: return (a & 0xffffu) | ((b & 0xffffu) << 16u);
			// ---- FP16 (scalar halves; op_sel picked the halves before)
			case Op::V_ADD_F16: return oh(h(a) + h(b));
			case Op::V_SUB_F16: return oh(h(a) - h(b));
			case Op::V_SUBREV_F16: return oh(h(b) - h(a));
			case Op::V_MUL_F16: return oh(h(a) * h(b));
			case Op::V_FMA_F16: return oh(std::fma(h(a), h(b), h(c)));
			case Op::V_FMAC_F16: return oh(std::fma(h(a), h(b), h(d_old)));
			case Op::V_FMAAK_F16: return oh(std::fma(h(a), h(b), h(K)));
			case Op::V_FMAMK_F16: return oh(std::fma(h(a), h(K), h(b)));
			case Op::V_MIN_F16: return oh(MinD(h(a), h(b)));
			case Op::V_MAX_F16: return oh(MaxD(h(a), h(b)));
			case Op::V_MIN3_F16: return oh(MinD(MinD(h(a), h(b)), h(c)));
			case Op::V_MAX3_F16: return oh(MaxD(MaxD(h(a), h(b)), h(c)));
			case Op::V_MED3_F16: return oh(MaxD(MinD(h(a), h(b)), MinD(MaxD(h(a), h(b)), h(c))));
			case Op::V_FLOOR_F16: return oh(std::floor(h(a)));
			case Op::V_CEIL_F16: return oh(std::ceil(h(a)));
			case Op::V_TRUNC_F16: return oh(std::trunc(h(a)));
			case Op::V_RNDNE_F16: return oh(std::nearbyint(h(a)));
			case Op::V_FRACT_F16: {
				const double x = h(a);
				const double r = x - std::floor(x);
				return r >= 1.0 ? 0x3bffu : oh(r);
			}
			case Op::V_RCP_F16: approx_ops["V_RCP_F16"]++; return oh(1.0 / h(a));
			case Op::V_RSQ_F16: approx_ops["V_RSQ_F16"]++; return oh(1.0 / std::sqrt(h(a)));
			case Op::V_SQRT_F16: approx_ops["V_SQRT_F16"]++; return oh(std::sqrt(h(a)));
			case Op::V_EXP_F16: approx_ops["V_EXP_F16"]++; return oh(std::exp2(h(a)));
			case Op::V_LOG_F16: approx_ops["V_LOG_F16"]++; return oh(std::log2(h(a)));
			case Op::V_SIN_F16: approx_ops["V_SIN_F16"]++; return oh(std::sin(h(a) * 6.283185307179586));
			case Op::V_COS_F16: approx_ops["V_COS_F16"]++; return oh(std::cos(h(a) * 6.283185307179586));
			case Op::V_LDEXP_F16: return oh(std::ldexp(h(a), static_cast<int16_t>(b & 0xffffu)));
			case Op::V_CVT_F16_U16: return oh(static_cast<double>(a & 0xffffu));
			case Op::V_CVT_F16_I16: return oh(static_cast<double>(static_cast<int16_t>(a & 0xffffu)));
			case Op::V_CVT_U16_F16: {
				const double x = h(a);
				return std::isnan(x) ? 0u : static_cast<uint32_t>(std::clamp(std::trunc(x), 0.0, 65535.0));
			}
			case Op::V_CVT_I16_F16: {
				const double x = h(a);
				return std::isnan(x) ? 0u : static_cast<uint32_t>(static_cast<int32_t>(std::clamp(std::trunc(x), -32768.0, 32767.0))) & 0xffffu;
			}
			// ---- 16-bit integer
			case Op::V_ADD_NC_U16: return (a + b) & 0xffffu;
			case Op::V_SUB_NC_U16: return (a - b) & 0xffffu;
			case Op::V_ADD_NC_I16: return (a + b) & 0xffffu;
			case Op::V_SUB_NC_I16: return (a - b) & 0xffffu;
			case Op::V_MUL_LO_U16: return (a * b) & 0xffffu;
			case Op::V_LSHLREV_B16: return ((b & 0xffffu) << (a & 15u)) & 0xffffu;
			case Op::V_LSHRREV_B16: return (b & 0xffffu) >> (a & 15u);
			case Op::V_ASHRREV_I16: return static_cast<uint32_t>(static_cast<int16_t>(b & 0xffffu) >> (a & 15u)) & 0xffffu;
			case Op::V_MAX_U16: return std::max(a & 0xffffu, b & 0xffffu);
			case Op::V_MIN_U16: return std::min(a & 0xffffu, b & 0xffffu);
			case Op::V_MAX_I16: return static_cast<uint32_t>(std::max(static_cast<int16_t>(a), static_cast<int16_t>(b))) & 0xffffu;
			case Op::V_MIN_I16: return static_cast<uint32_t>(std::min(static_cast<int16_t>(a), static_cast<int16_t>(b))) & 0xffffu;
			case Op::V_MAD_U16: return (a * b + c) & 0xffffu;
			case Op::V_MAD_U32_U16: return (a & 0xffffu) * (b & 0xffffu) + c;
			case Op::V_MAD_I32_I16: return static_cast<uint32_t>(static_cast<int16_t>(a) * static_cast<int16_t>(b) + static_cast<int32_t>(c));
			default: break;
		}
		ok = false;
		return 0;
	}

	// ---------------------------------------------------------------- VALU dispatch
	bool IsCompare(Op op) const {
		const char* name = kOpNames[static_cast<int>(op)];
		return std::strncmp(name, "V_CMP", 5) == 0;
	}
	// Compare op parsed once per opcode: condition and operand type from the AMD name.
	enum class Cond { F, LT, EQ, LE, GT, LG, NE, GE, O, U, NGE, NLG, NGT, NLE, NEQ, NLT, TRU, T, CLASS, Bad };
	enum class CTy { F16, F32, F64, I16, I32, I64, U16, U32, U64, Bad };
	struct CmpSpec {
		Cond cond = Cond::Bad;
		CTy  type = CTy::Bad;
		bool x    = false;
	};
	std::map<Op, CmpSpec> m_cmp;
	const CmpSpec& Spec(Op op) {
		auto it = m_cmp.find(op);
		if (it != m_cmp.end()) return it->second;
		CmpSpec           spec;
		const std::string name = kOpNames[static_cast<int>(op)];
		spec.x                 = name.rfind("V_CMPX_", 0) == 0;
		const std::string rest = name.substr(spec.x ? 7 : 6);
		const auto        us   = rest.rfind('_');
		const std::string cond = rest.substr(0, us), type = rest.substr(us + 1);
		static const std::pair<const char*, Cond> conds[] = {{"F", Cond::F}, {"LT", Cond::LT}, {"EQ", Cond::EQ}, {"LE", Cond::LE},
		    {"GT", Cond::GT}, {"LG", Cond::LG}, {"NE", Cond::NE}, {"GE", Cond::GE}, {"O", Cond::O}, {"U", Cond::U}, {"NGE", Cond::NGE},
		    {"NLG", Cond::NLG}, {"NGT", Cond::NGT}, {"NLE", Cond::NLE}, {"NEQ", Cond::NEQ}, {"NLT", Cond::NLT}, {"TRU", Cond::TRU},
		    {"T", Cond::T}, {"CLASS", Cond::CLASS}};
		for (const auto& [n, c]: conds) {
			if (cond == n) spec.cond = c;
		}
		static const std::pair<const char*, CTy> types[] = {{"F16", CTy::F16}, {"F32", CTy::F32}, {"F64", CTy::F64}, {"I16", CTy::I16},
		    {"I32", CTy::I32}, {"I64", CTy::I64}, {"U16", CTy::U16}, {"U32", CTy::U32}, {"U64", CTy::U64}};
		for (const auto& [n, t]: types) {
			if (type == n) spec.type = t;
		}
		return m_cmp.emplace(op, spec).first->second;
	}
	bool CompareLane(const CmpSpec& spec, uint64_t a, uint64_t b, bool& ok) const {
		const Cond cond = spec.cond;
		const CTy  type = spec.type;
		if (cond == Cond::Bad || type == CTy::Bad) {
			ok = false;
			return false;
		}
		if (cond == Cond::CLASS) {
			double v;
			if (type == CTy::F32) v = static_cast<double>(F32(static_cast<uint32_t>(a)));
			else if (type == CTy::F16) v = static_cast<double>(F16ToF32(static_cast<uint16_t>(a)));
			else v = F64(a);
			const uint32_t mask = static_cast<uint32_t>(b);
			bool           r    = false;
			const bool     sign = std::signbit(v);
			if (std::isnan(v)) {
				const bool snan = type == CTy::F32 ? IsSnanF32(static_cast<uint32_t>(a)) : false;
				r               = snan ? (mask & 1u) : (mask & 2u);
			} else if (std::isinf(v)) {
				r = sign ? (mask & 4u) : (mask & 0x200u);
			} else if (v == 0.0) {
				r = sign ? (mask & 0x20u) : (mask & 0x40u);
			} else {
				const bool denorm = type == CTy::F32   ? std::fabs(v) < 1.17549435e-38
				                    : type == CTy::F16 ? std::fabs(v) < 6.103515625e-05
				                                       : std::fabs(v) < 2.2250738585072014e-308;
				r = sign ? (denorm ? (mask & 0x10u) : (mask & 8u)) : (denorm ? (mask & 0x80u) : (mask & 0x100u));
			}
			return r;
		}
		if (type == CTy::F32 || type == CTy::F16 || type == CTy::F64) {
			double fa, fb;
			if (type == CTy::F32) {
				fa = static_cast<double>(In32(static_cast<uint32_t>(a)));
				fb = static_cast<double>(In32(static_cast<uint32_t>(b)));
			} else if (type == CTy::F16) {
				fa = static_cast<double>(F16ToF32(static_cast<uint16_t>(a)));
				fb = static_cast<double>(F16ToF32(static_cast<uint16_t>(b)));
			} else {
				fa = F64(a);
				fb = F64(b);
			}
			const bool un = std::isnan(fa) || std::isnan(fb);
			switch (cond) {
				case Cond::F: return false;
				case Cond::LT: return !un && fa < fb;
				case Cond::EQ: return !un && fa == fb;
				case Cond::LE: return !un && fa <= fb;
				case Cond::GT: return !un && fa > fb;
				case Cond::LG: return !un && fa != fb;
				case Cond::GE: return !un && fa >= fb;
				case Cond::O: return !un;
				case Cond::U: return un;
				case Cond::NGE: return !(fa >= fb);
				case Cond::NLG: return !(fa < fb || fa > fb);
				case Cond::NGT: return !(fa > fb);
				case Cond::NLE: return !(fa <= fb);
				case Cond::NEQ: return !(fa == fb);
				case Cond::NLT: return !(fa < fb);
				case Cond::TRU: return true;
				default: ok = false; return false;
			}
		}
		int64_t  sa = 0, sb = 0;
		uint64_t ua = a, ub = b;
		if (type == CTy::I32) {
			sa = static_cast<int32_t>(a);
			sb = static_cast<int32_t>(b);
		} else if (type == CTy::I16) {
			sa = static_cast<int16_t>(a);
			sb = static_cast<int16_t>(b);
		} else if (type == CTy::I64) {
			sa = static_cast<int64_t>(a);
			sb = static_cast<int64_t>(b);
		} else if (type == CTy::U32) {
			ua = static_cast<uint32_t>(a);
			ub = static_cast<uint32_t>(b);
		} else if (type == CTy::U16) {
			ua = a & 0xffffu;
			ub = b & 0xffffu;
		}
		const bool is_signed = type == CTy::I16 || type == CTy::I32 || type == CTy::I64;
		auto       cmp       = [&](auto x, auto y) -> bool {
            switch (cond) {
                case Cond::F: return false;
                case Cond::LT: return x < y;
                case Cond::EQ: return x == y;
                case Cond::LE: return x <= y;
                case Cond::GT: return x > y;
                case Cond::NE:
                case Cond::LG: return x != y;
                case Cond::GE: return x >= y;
                case Cond::T: return true;
                default: ok = false; return false;
            }
		};
		return is_signed ? cmp(sa, sb) : cmp(ua, ub);
	}

	static bool Is16(Ty t) { return t == Ty::F16 || t == Ty::I16 || t == Ty::U16; }

	// DPP16 source lane (or -1 for an invalid/out-of-row source).
	static int DppLane(uint32_t ctrl, int lane) {
		const int row = lane & ~15, in = lane & 15;
		if (ctrl <= 0xff) return (lane & ~3) | static_cast<int>((ctrl >> (2u * (lane & 3))) & 3u);
		if (ctrl >= 0x101 && ctrl <= 0x10f) { const int s = in + static_cast<int>(ctrl - 0x100); return s < 16 ? row + s : -1; }
		if (ctrl >= 0x111 && ctrl <= 0x11f) { const int s = in - static_cast<int>(ctrl - 0x110); return s >= 0 ? row + s : -1; }
		if (ctrl >= 0x121 && ctrl <= 0x12f) return row + ((in - static_cast<int>(ctrl - 0x120)) & 15);
		if (ctrl == 0x140) return row + (15 - in);
		if (ctrl == 0x141) return (lane & ~7) + (7 - (lane & 7));
		if (ctrl >= 0x150 && ctrl <= 0x15f) return row + static_cast<int>(ctrl - 0x150);
		if (ctrl >= 0x160 && ctrl <= 0x16f) return row + (in ^ static_cast<int>(ctrl - 0x160));
		return -1;
	}

	static uint32_t SdwaSel(uint32_t v, uint32_t sel, bool sext) {
		switch (sel) {
			case 0:
			case 1:
			case 2:
			case 3: {
				const uint32_t b = (v >> (8u * sel)) & 0xffu;
				return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(b))) : b;
			}
			case 4:
			case 5: {
				const uint32_t w = (v >> (16u * (sel - 4u))) & 0xffffu;
				return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(w))) : w;
			}
			default: return v;
		}
	}
	static uint32_t SdwaDst(uint32_t r, uint32_t old, uint32_t sel, uint32_t unused) {
		if (sel == 6) return r;
		uint32_t shift = 0, mask = 0;
		if (sel <= 3) { shift = 8u * sel; mask = 0xffu; }
		else { shift = 16u * (sel - 4u); mask = 0xffffu; }
		const uint32_t v = (r & mask) << shift;
		if (unused == 2) return (old & ~(mask << shift)) | v;
		if (unused == 1) {
			// sign extend upward from the selected field; lower bits zero
			const uint32_t top = shift + (mask == 0xffu ? 8u : 16u);
			const bool     neg = ((r & mask) >> ((mask == 0xffu ? 8u : 16u) - 1u)) & 1u;
			uint32_t       out = v;
			if (neg && top < 32) out |= ~0u << top;
			return out;
		}
		return v;
	}

	bool ExecValu(Wave& w, const Inst& inst) {
		const Op      op   = inst.op;
		const OpInfo& info = Info(inst);
		const bool    vop3 = inst.enc == Enc::VOP3;
		const bool    vop3p = inst.enc == Enc::VOP3P;
		// Lane-crossing / scalar-destination ops first.
		if (op == Op::V_READFIRSTLANE_B32) {
			int lane = 0;
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (w.Active(l)) { lane = static_cast<int>(l); break; }
			}
			const uint32_t src = inst.F("SRC0");
			WriteS(w, inst.F("VDST"), ReadV(w, src, static_cast<uint32_t>(lane), Ty::B32, inst));
			return true;
		}
		if (op == Op::V_READLANE_B32) {
			const uint32_t lane = ReadS(w, inst.F("SRC1"), inst) & (w.lanes - 1u);
			WriteS(w, inst.F("VDST"), ReadV(w, inst.F("SRC0"), lane, Ty::B32, inst));
			return true;
		}
		if (op == Op::V_WRITELANE_B32) {
			const uint32_t lane = ReadS(w, inst.F("SRC1"), inst) & (w.lanes - 1u);
			w.v[inst.F("VDST")][lane] = ReadS(w, inst.F("SRC0"), inst);
			return true;
		}
		if (op == Op::V_PERMLANE16_B32 || op == Op::V_PERMLANEX16_B32) {
			const uint32_t vsrc = inst.F("SRC0") - Src::Vgpr0;
			const uint64_t sel  = ReadS(w, inst.F("SRC1"), inst) | (static_cast<uint64_t>(ReadS(w, inst.F("SRC2"), inst)) << 32u);
			const bool     fi   = (inst.F("OP_SEL") & 1u) != 0;
			const bool     bc   = (inst.F("OP_SEL") & 2u) != 0;
			auto&          dst  = w.v[inst.F("VDST")];
			const auto     src  = w.v[vsrc];
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (!w.Active(l)) continue;
				const uint32_t row = l / 16u, in = l % 16u;
				const uint32_t s   = static_cast<uint32_t>((sel >> (4u * in)) & 15u);
				const uint32_t srow = op == Op::V_PERMLANEX16_B32 ? (row ^ 1u) : row;
				const uint32_t sl   = srow * 16u + s;
				if (sl >= w.lanes) continue;
				if (!w.Active(sl) && !fi) {
					if (bc) dst[l] = 0;
					continue;
				}
				dst[l] = src[sl];
			}
			return true;
		}
		const bool cmp = IsCompare(op);
		const CmpSpec cmp_spec = cmp ? Spec(op) : CmpSpec {};
		// Source codes.
		uint32_t code[3] {0, 0, 0};
		uint32_t nsrc = std::max(info.nsrc, 1u);
		if (vop3 || vop3p) {
			code[0] = inst.F("SRC0");
			code[1] = inst.F("SRC1");
			code[2] = inst.F("SRC2");
		} else {
			code[0] = inst.F("SRC0");
			code[1] = inst.Has("VSRC1") ? Src::Vgpr0 + inst.F("VSRC1") : 0;
			if (inst.sdwa || inst.dpp16 || inst.dpp8) code[0] = inst.F("VSRC0") + ((inst.sdwa && inst.F("S0")) ? 0u : Src::Vgpr0);
			if (inst.sdwa && inst.F("S1")) code[1] = inst.F("VSRC1");
		}
		const uint32_t abs_bits = vop3 ? inst.F("ABS") : 0;
		const uint32_t neg_bits = (vop3 || vop3p) ? inst.F("NEG") : 0;
		const uint32_t opsel    = vop3 ? inst.F("OP_SEL") : 0;
		const bool     clamp    = (vop3 || vop3p || inst.sdwa) ? inst.F("CLAMP") != 0 : false;
		const uint32_t omod     = (vop3 || inst.sdwa) ? inst.F("OMOD") : 0;
		// 64-bit ops
		const bool wide = info.dst == Ty::B64 || info.dst == Ty::F64 || info.src[0] == Ty::B64 || info.src[0] == Ty::F64 ||
		                  info.src[1] == Ty::B64 || info.src[1] == Ty::F64;
		uint64_t cmp_result = 0;
		uint64_t carry_mask = 0;
		const uint32_t vdst = inst.F("VDST");
		// carry-in mask
		uint64_t carry_in_mask = 0;
		if (op == Op::V_ADD_CO_CI_U32 || op == Op::V_SUB_CO_CI_U32 || op == Op::V_SUBREV_CO_CI_U32 || op == Op::V_CNDMASK_B32) {
			carry_in_mask = vop3 ? ReadMask(w, code[2], inst) : w.vcc;
			if (op == Op::V_CNDMASK_B32) nsrc = 2;
		}
		if (op == Op::V_CNDMASK_B32 && inst.sdwa) carry_in_mask = w.vcc;
		std::array<uint32_t, 64> result {};
		std::array<bool, 64>     written {};
		const uint32_t dpp_ctrl = inst.dpp16 ? inst.F("DPP_CTRL") : 0u, row_mask = inst.dpp16 ? inst.F("ROW_MASK") : 0xfu;
		const uint32_t bank_mask = inst.dpp16 ? inst.F("BANK_MASK") : 0xfu;
		const bool     dpp_fi = (inst.dpp16 && inst.F("FI")) || (inst.dpp8 && inst.F("SRC0") == Src::Dpp8Fi);
		const bool     dpp_bc = inst.dpp16 && inst.F("BOUND_CTRL");
		uint32_t       lane_sel[8] {};
		if (inst.dpp8) {
			static const char* const names[8] = {"LANE_SEL_0", "LANE_SEL_1", "LANE_SEL_2", "LANE_SEL_3",
			                                     "LANE_SEL_4", "LANE_SEL_5", "LANE_SEL_6", "LANE_SEL_7"};
			for (int k = 0; k < 8; k++) lane_sel[k] = inst.F(names[k]);
		}
		uint32_t sdwa_sel[2] {6, 6};
		bool     sdwa_sext[2] {false, false}, mod_abs[2] {false, false}, mod_neg[2] {false, false};
		if (inst.sdwa) {
			sdwa_sel[0]  = inst.F("SRC0_SEL", 6);
			sdwa_sel[1]  = inst.F("SRC1_SEL", 6);
			sdwa_sext[0] = inst.F("SRC0_SEXT") != 0;
			sdwa_sext[1] = inst.F("SRC1_SEXT") != 0;
		}
		if (inst.sdwa || inst.dpp16) {
			mod_abs[0] = inst.F("SRC0_ABS") != 0;
			mod_abs[1] = inst.F("SRC1_ABS") != 0;
			mod_neg[0] = inst.F("SRC0_NEG") != 0;
			mod_neg[1] = inst.F("SRC1_NEG") != 0;
		}
		const uint32_t dst_sel = inst.sdwa ? inst.F("DST_SEL", 6) : 6u, dst_unused = inst.sdwa ? inst.F("DST_UNUSED", 2) : 2u;
		// V_CNDMASK_B32 has B32 operands in the ISA tables, but the hardware applies abs/neg source modifiers to it (LLVM
		// selects f32 select(c, -x, |y|) as V_CNDMASK_B32_e64 with modifiers).
		// Input modifiers (neg/abs; SDWA, DPP, VOP3) also apply to V_MOV_B32: the hardware flips/clears the sign bit and
		// AMD's compiler folds fneg into v_mov_b32_sdwa (f82e549a 0x10cc: v_mov_b32 v9, -v2). Without this the reference
		// copied v2 unchanged (notes M123 section 35).
		const bool cnd_mods = op == Op::V_CNDMASK_B32 || op == Op::V_MOV_B32;
		for (uint32_t l = 0; l < w.lanes; l++) {
			if (!w.Active(l)) continue;
			// DPP source lane for src0
			uint32_t src_lane = l;
			bool     dpp_skip = false, dpp_zero = false;
			if (inst.dpp16) {
				const int sl = DppLane(dpp_ctrl, static_cast<int>(l));
				const bool row_ok  = (row_mask >> (l / 16u)) & 1u;
				const bool bank_ok = (bank_mask >> ((l % 16u) / 4u)) & 1u;
				if (!row_ok || !bank_ok) continue; // lane not written
				if (sl < 0 || static_cast<uint32_t>(sl) >= w.lanes || (!w.Active(static_cast<uint32_t>(sl)) && !dpp_fi)) {
					if (dpp_bc) dpp_zero = true;
					else dpp_skip = true;
				} else {
					src_lane = static_cast<uint32_t>(sl);
				}
				if (dpp_skip) continue;
			} else if (inst.dpp8) {
				src_lane = (l & ~7u) | lane_sel[l & 7u];
			}
			if (wide) {
				// 64-bit ALU subset
				uint64_t a = ReadV64(w, code[0], src_lane, info.src[0], inst);
				uint64_t b = nsrc > 1 ? ReadV64(w, code[1], l, info.src[1], inst) : 0;
				// 32-bit shift amounts / compares of 64-bit values
				if (op == Op::V_LSHLREV_B64 || op == Op::V_LSHRREV_B64 || op == Op::V_ASHRREV_I64) a = ReadV(w, code[0], src_lane, Ty::B32, inst);
				if (info.src[0] == Ty::F64) {
					if ((abs_bits & 1u)) a &= ~(1ull << 63u);
					if ((neg_bits & 1u)) a ^= 1ull << 63u;
				}
				if (info.src[1] == Ty::F64) {
					if ((abs_bits & 2u)) b &= ~(1ull << 63u);
					if ((neg_bits & 2u)) b ^= 1ull << 63u;
				}
				uint64_t r = 0;
				bool     ok = true;
				if (cmp) {
					if (CompareLane(cmp_spec, a, b, ok)) cmp_result |= 1ull << l;
					if (!ok) { Unsupported(inst); return true; }
					continue;
				}
				switch (op) {
					case Op::V_LSHLREV_B64: r = b << (a & 63u); break;
					case Op::V_LSHRREV_B64: r = b >> (a & 63u); break;
					case Op::V_ASHRREV_I64: r = static_cast<uint64_t>(static_cast<int64_t>(b) >> (a & 63u)); break;
					case Op::V_ADD_F64: r = B64(F64(a) + F64(b)); break;
					case Op::V_MUL_F64: r = B64(F64(a) * F64(b)); break;
					case Op::V_FMA_F64: {
						uint64_t c = ReadV64(w, code[2], l, Ty::F64, inst);
						if (abs_bits & 4u) c &= ~(1ull << 63u);
						if (neg_bits & 4u) c ^= 1ull << 63u;
						r = B64(std::fma(F64(a), F64(b), F64(c)));
						break;
					}
					case Op::V_MIN_F64: r = B64(MinD(F64(a), F64(b))); break;
					case Op::V_MAX_F64: r = B64(MaxD(F64(a), F64(b))); break;
					case Op::V_CVT_F64_F32: r = B64(static_cast<double>(In32(static_cast<uint32_t>(ReadV(w, code[0], src_lane, Ty::F32, inst))))); break;
					case Op::V_CVT_F32_F64: {
						result[l]  = Out32(static_cast<float>(F64(a)));
						written[l] = true;
						continue;
					}
					case Op::V_RCP_F64:
						// The hardware result is an approximation (shaders refine it with V_FMA_F64); the correctly
						// rounded reciprocal is within its error bound.
						r = B64(1.0 / F64(a));
						approx_ops["V_RCP_F64 (correctly rounded 1/x)"]++;
						break;
					case Op::V_CVT_F64_I32: r = B64(static_cast<double>(static_cast<int32_t>(ReadV(w, code[0], src_lane, Ty::B32, inst)))); break;
					case Op::V_CVT_F64_U32: r = B64(static_cast<double>(ReadV(w, code[0], src_lane, Ty::B32, inst))); break;
					case Op::V_CVT_I32_F64: {
						const double x = F64(a);
						result[l] = std::isnan(x) ? 0u : static_cast<uint32_t>(static_cast<int32_t>(std::clamp(std::trunc(x), -2147483648.0, 2147483647.0)));
						written[l] = true;
						continue;
					}
					case Op::V_CVT_U32_F64: {
						const double x = F64(a);
						result[l] = std::isnan(x) ? 0u : static_cast<uint32_t>(std::clamp(std::trunc(x), 0.0, 4294967295.0));
						written[l] = true;
						continue;
					}
					case Op::V_MAD_U64_U32: {
						const uint64_t c = ReadV64(w, code[2], l, Ty::B64, inst);
						r = static_cast<uint64_t>(static_cast<uint32_t>(ReadV(w, code[0], src_lane, Ty::B32, inst))) * ReadV(w, code[1], l, Ty::B32, inst) + c;
						break;
					}
					case Op::V_MAD_I64_I32: {
						const uint64_t c = ReadV64(w, code[2], l, Ty::B64, inst);
						r = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(ReadV(w, code[0], src_lane, Ty::B32, inst))) *
						                          static_cast<int32_t>(ReadV(w, code[1], l, Ty::B32, inst)) + static_cast<int64_t>(c));
						break;
					}
					default: Unsupported(inst); return true;
				}
				w.v[vdst][l]      = static_cast<uint32_t>(r);
				w.v[vdst + 1u][l] = static_cast<uint32_t>(r >> 32u);
				continue;
			}
			uint32_t s[3] {0, 0, 0};
			for (uint32_t i = 0; i < nsrc && i < 3; i++) {
				const uint32_t lane_i = (i == 0) ? src_lane : l;
				uint32_t       v      = ReadV(w, code[i], lane_i, info.src[i], inst);
				if (i == 0 && dpp_zero) v = 0;
				if (vop3p) {
					s[i] = v;
					continue;
				}
				if (inst.sdwa) {
					v = SdwaSel(v, sdwa_sel[i], sdwa_sext[i]);
					const bool a = mod_abs[i], n = mod_neg[i];
					if (info.src[i] == Ty::F32 || cnd_mods) v = ApplyMod32(v, a, n);
					else if (info.src[i] == Ty::F16) v = ApplyMod16(v, a, n);
				} else if (inst.dpp16 && i < 2) {
					const bool a = mod_abs[i], n = mod_neg[i];
					if (info.src[i] == Ty::F32 || cnd_mods) v = ApplyMod32(v, a, n);
					else if (info.src[i] == Ty::F16) v = ApplyMod16(v, a, n);
				} else if (vop3) {
					if (Is16(info.src[i]) && ((opsel >> i) & 1u)) v >>= 16u;
					const bool a = (abs_bits >> i) & 1u, n = (neg_bits >> i) & 1u;
					if (info.src[i] == Ty::F32 || cnd_mods) v = ApplyMod32(v, a, n);
					else if (info.src[i] == Ty::F16) v = ApplyMod16(v, a, n);
				}
				if (Is16(info.src[i])) v &= 0xffffu;
				s[i] = v;
			}
			if (cmp) {
				bool ok = true;
				if (CompareLane(cmp_spec, s[0], s[1], ok)) cmp_result |= 1ull << l;
				if (!ok) { Unsupported(inst); return true; }
				continue;
			}
			uint32_t r = 0;
			bool     ok = true;
			bool     carry = false;
			const uint32_t d_old = w.v[vdst][l];
			if (vop3p) {
				r = Packed(inst, op, s, d_old, ok);
			} else if (op == Op::V_CNDMASK_B32) {
				r = ((carry_in_mask >> l) & 1u) ? s[1] : s[0];
			} else if (op == Op::V_PK_FMAC_F16) {
				// RDNA1 VOP2 packed FMAC: d = a * b + d, per half
				auto hf = [](uint32_t x) { return static_cast<double>(F16ToF32(static_cast<uint16_t>(x & 0xffffu))); };
				const uint32_t lo = F64ToF16(std::fma(hf(s[0]), hf(s[1]), hf(d_old)));
				const uint32_t hi = F64ToF16(std::fma(hf(s[0] >> 16u), hf(s[1] >> 16u), hf(d_old >> 16u)));
				r = lo | (hi << 16u);
			} else {
				uint32_t old_in = d_old;
				if (Is16(info.dst) && vop3 && ((opsel >> 3u) & 1u)) old_in = d_old >> 16u;
				r = Alu(inst, op, s[0], s[1], s[2], old_in, l, (carry_in_mask >> l) & 1u, carry, ok);
			}
			if (!ok) {
				Unsupported(inst);
				return true;
			}
			if (carry) carry_mask |= 1ull << l;
			// output modifiers
			if (!vop3p && op != Op::V_PK_FMAC_F16) {
				if (info.dst == Ty::F32 && (clamp || omod)) r = Omod32(r, omod, clamp);
				else if (info.dst == Ty::F16 && (clamp || omod)) r = Omod16(r, omod, clamp);
				else if (clamp && (info.dst == Ty::B32) && (op == Op::V_ADD_NC_U32 || op == Op::V_SUB_NC_U32)) {
					// integer clamp (saturate) is rare; left unsupported for exactness
				}
			}
			// destination: 16-bit halves, SDWA selection
			if (inst.sdwa && !(inst.enc == Enc::VOPC)) {
				r = SdwaDst(r, d_old, dst_sel, dst_unused);
			} else if (Is16(info.dst) && !vop3p && op != Op::V_CVT_F16_F32) {
				if (vop3 && ((opsel >> 3u) & 1u)) r = (d_old & 0xffffu) | ((r & 0xffffu) << 16u);
				else r = (d_old & 0xffff0000u) | (r & 0xffffu); // RDNA: the untouched half is preserved
			}
			result[l]  = r;
			written[l] = true;
		}
		if (cmp) {
			const bool x = std::strncmp(kOpNames[static_cast<int>(op)], "V_CMPX", 6) == 0;
			if (x) {
				w.exec = cmp_result & w.Mask();
			} else if (vop3) {
				WriteMask(w, inst.F("VDST"), cmp_result);
			} else if (inst.sdwa && inst.F("SD")) {
				WriteMask(w, inst.F("SDST"), cmp_result);
			} else {
				w.vcc = cmp_result;
			}
			return true;
		}
		// 64-bit ops wrote their pairs directly; the 64 -> 32 bit conversions (V_CVT_F32_F64, V_CVT_I32_F64,
		// V_CVT_U32_F64) left 32-bit results in result[] like the 32-bit ops.
		for (uint32_t l = 0; l < w.lanes; l++) {
			if (written[l]) w.v[vdst][l] = result[l];
		}
		if (wide) return true;
		// carry-out destination
		if (op == Op::V_ADD_CO_U32 || op == Op::V_SUB_CO_U32 || op == Op::V_SUBREV_CO_U32 || op == Op::V_ADD_CO_CI_U32 ||
		    op == Op::V_SUB_CO_CI_U32 || op == Op::V_SUBREV_CO_CI_U32) {
			if (vop3) WriteMask(w, inst.F("SDST"), carry_mask);
			else w.vcc = carry_mask;
		}
		return true;
	}

	// VOP3P packed math: OP_SEL (lo halves) / OP_SEL_HI (hi halves) per source, NEG / NEG_HI.
	uint32_t Packed(const Inst& inst, Op op, const uint32_t s[3], uint32_t d_old, bool& ok) {
		const uint32_t opsel    = inst.F("OP_SEL");
		const uint32_t opsel_hi = inst.F("OP_SEL_HI") | (inst.F("OP_SEL_HI_2") << 2u);
		const uint32_t neg      = inst.F("NEG");
		const uint32_t neg_hi   = inst.F("NEG_HI");
		const bool     clamp    = inst.F("CLAMP") != 0;
		uint32_t       lo[3], hi[3];
		for (int i = 0; i < 3; i++) {
			lo[i] = ((opsel >> i) & 1u) ? (s[i] >> 16u) : (s[i] & 0xffffu);
			hi[i] = ((opsel_hi >> i) & 1u) ? (s[i] >> 16u) : (s[i] & 0xffffu);
		}
		const bool is_float = op == Op::V_PK_FMA_F16 || op == Op::V_PK_ADD_F16 || op == Op::V_PK_MUL_F16 ||
		                      op == Op::V_PK_MIN_F16 || op == Op::V_PK_MAX_F16 || op == Op::V_PK_FMAC_F16 ||
		                      op == Op::V_FMA_MIX_F32 || op == Op::V_FMA_MIXLO_F16 || op == Op::V_FMA_MIXHI_F16;
		if (is_float && op != Op::V_FMA_MIX_F32 && op != Op::V_FMA_MIXLO_F16 && op != Op::V_FMA_MIXHI_F16) {
			for (int i = 0; i < 3; i++) {
				if ((neg >> i) & 1u) lo[i] ^= 0x8000u;
				if ((neg_hi >> i) & 1u) hi[i] ^= 0x8000u;
			}
		}
		auto hf = [](uint32_t x) { return static_cast<double>(F16ToF32(static_cast<uint16_t>(x & 0xffffu))); };
		auto half = [&](const uint32_t* x, int which) -> uint32_t {
			double r = 0;
			switch (op) {
				case Op::V_PK_FMA_F16: r = std::fma(hf(x[0]), hf(x[1]), hf(x[2])); break;
				case Op::V_PK_ADD_F16: r = hf(x[0]) + hf(x[1]); break;
				case Op::V_PK_MUL_F16: r = hf(x[0]) * hf(x[1]); break;
				case Op::V_PK_MIN_F16: r = MinD(hf(x[0]), hf(x[1])); break;
				case Op::V_PK_MAX_F16: r = MaxD(hf(x[0]), hf(x[1])); break;
				case Op::V_PK_FMAC_F16: r = std::fma(hf(x[0]), hf(x[1]), hf(which ? (d_old >> 16u) : d_old)); break;
				case Op::V_PK_ADD_U16: return (x[0] + x[1]) & 0xffffu;
				case Op::V_PK_SUB_U16: return (x[0] - x[1]) & 0xffffu;
				case Op::V_PK_ADD_I16: return (x[0] + x[1]) & 0xffffu;
				case Op::V_PK_SUB_I16: return (x[0] - x[1]) & 0xffffu;
				case Op::V_PK_MUL_LO_U16: return (x[0] * x[1]) & 0xffffu;
				case Op::V_PK_MAD_U16: return (x[0] * x[1] + x[2]) & 0xffffu;
				case Op::V_PK_MAD_I16: return static_cast<uint32_t>(static_cast<int16_t>(x[0]) * static_cast<int16_t>(x[1]) + static_cast<int16_t>(x[2])) & 0xffffu;
				case Op::V_PK_LSHLREV_B16: return (x[1] << (x[0] & 15u)) & 0xffffu;
				case Op::V_PK_LSHRREV_B16: return (x[1] & 0xffffu) >> (x[0] & 15u);
				case Op::V_PK_ASHRREV_I16: return static_cast<uint32_t>(static_cast<int16_t>(x[1]) >> (x[0] & 15u)) & 0xffffu;
				case Op::V_PK_MAX_U16: return std::max(x[0] & 0xffffu, x[1] & 0xffffu);
				case Op::V_PK_MIN_U16: return std::min(x[0] & 0xffffu, x[1] & 0xffffu);
				case Op::V_PK_MAX_I16: return static_cast<uint32_t>(std::max(static_cast<int16_t>(x[0]), static_cast<int16_t>(x[1]))) & 0xffffu;
				case Op::V_PK_MIN_I16: return static_cast<uint32_t>(std::min(static_cast<int16_t>(x[0]), static_cast<int16_t>(x[1]))) & 0xffffu;
				default: ok = false; return 0;
			}
			if (clamp) r = std::isnan(r) ? (m_cap.dx10_clamp ? 0.0 : r) : std::clamp(r, 0.0, 1.0);
			return F64ToF16(r);
		};
		if (op == Op::V_FMA_MIX_F32 || op == Op::V_FMA_MIXLO_F16 || op == Op::V_FMA_MIXHI_F16) {
			// op_sel_hi[i] = 1: source i is f16 (half chosen by op_sel[i]); else f32 (whole dword).
			double x[3];
			for (int i = 0; i < 3; i++) {
				if ((opsel_hi >> i) & 1u) x[i] = hf(((opsel >> i) & 1u) ? (s[i] >> 16u) : s[i]);
				else x[i] = static_cast<double>(In32(s[i]));
				if ((neg_hi >> i) & 1u) x[i] = std::fabs(x[i]);
				if ((neg >> i) & 1u) x[i] = -x[i];
			}
			const double r = std::fma(x[0], x[1], x[2]);
			if (op == Op::V_FMA_MIX_F32) {
				float rf = static_cast<float>(r);
				if (clamp) rf = ClampF(rf, m_cap.dx10_clamp);
				return Out32(rf);
			}
			double rr = r;
			if (clamp) rr = std::isnan(rr) ? 0.0 : std::clamp(rr, 0.0, 1.0);
			const uint32_t hbits = F64ToF16(rr);
			return op == Op::V_FMA_MIXLO_F16 ? ((d_old & 0xffff0000u) | hbits) : ((d_old & 0xffffu) | (hbits << 16u));
		}
		const uint32_t rlo = half(lo, 0);
		const uint32_t rhi = half(hi, 1);
		return rlo | (rhi << 16u);
	}

	// ---------------------------------------------------------------- scalar ALU
	bool ExecSalu(Wave& w, const Inst& inst) {
		const Op op = inst.op;
		auto     s0 = [&]() { return ReadS(w, inst.F("SSRC0"), inst); };
		auto     s1 = [&]() { return ReadS(w, inst.F("SSRC1"), inst); };
		auto     s0_64 = [&]() { return ReadS64(w, inst.F("SSRC0"), inst); };
		auto     s1_64 = [&]() { return ReadS64(w, inst.F("SSRC1"), inst); };
		const uint32_t sdst = inst.F("SDST");
		auto d32 = [&](uint32_t v) { WriteS(w, sdst, v); };
		auto d64 = [&](uint64_t v) { WriteS64(w, sdst, v); };
		const uint32_t simm = inst.F("SIMM16");
		const int32_t  sext16 = static_cast<int16_t>(simm);
		switch (op) {
			case Op::S_MOV_B32: d32(s0()); return true;
			case Op::S_MOV_B64: d64(s0_64()); return true;
			case Op::S_CMOV_B32: if (w.scc) d32(s0()); return true;
			case Op::S_CMOV_B64: if (w.scc) d64(s0_64()); return true;
			case Op::S_NOT_B32: { const uint32_t r = ~s0(); d32(r); w.scc = r != 0; return true; }
			case Op::S_NOT_B64: { const uint64_t r = ~s0_64(); d64(r); w.scc = r != 0; return true; }
			case Op::S_WQM_B32:
			case Op::S_WQM_B64: {
				const uint64_t v = op == Op::S_WQM_B32 ? s0() : s0_64();
				uint64_t r = 0;
				for (int q = 0; q < 16; q++) if ((v >> (4 * q)) & 0xfu) r |= 0xfull << (4 * q);
				if (op == Op::S_WQM_B32) d32(static_cast<uint32_t>(r)); else d64(r);
				w.scc = r != 0;
				return true;
			}
			case Op::S_BREV_B32: { uint32_t v = s0(), r = 0; for (int i = 0; i < 32; i++) r |= ((v >> i) & 1u) << (31 - i); d32(r); return true; }
			case Op::S_BCNT0_I32_B32: { const uint32_t r = 32u - static_cast<uint32_t>(std::popcount(s0())); d32(r); w.scc = r != 0; return true; }
			case Op::S_BCNT1_I32_B32: { const uint32_t r = static_cast<uint32_t>(std::popcount(s0())); d32(r); w.scc = r != 0; return true; }
			case Op::S_BCNT1_I32_B64: { const uint32_t r = static_cast<uint32_t>(std::popcount(s0_64())); d32(r); w.scc = r != 0; return true; }
			case Op::S_BCNT0_I32_B64: { const uint32_t r = 64u - static_cast<uint32_t>(std::popcount(s0_64())); d32(r); w.scc = r != 0; return true; }
			case Op::S_FF1_I32_B32: { const uint32_t v = s0(); d32(v == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countr_zero(v))); return true; }
			case Op::S_FF1_I32_B64: { const uint64_t v = s0_64(); d32(v == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countr_zero(v))); return true; }
			case Op::S_FF0_I32_B32: { const uint32_t v = ~s0(); d32(v == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countr_zero(v))); return true; }
			case Op::S_FF0_I32_B64: { const uint64_t v = ~s0_64(); d32(v == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countr_zero(v))); return true; }
			case Op::S_FLBIT_I32_B32: { const uint32_t v = s0(); d32(v == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countl_zero(v))); return true; }
			case Op::S_FLBIT_I32_B64: { const uint64_t v = s0_64(); d32(v == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countl_zero(v))); return true; }
			case Op::S_SEXT_I32_I8: d32(static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(s0())))); return true;
			case Op::S_SEXT_I32_I16: d32(static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(s0())))); return true;
			case Op::S_BITSET0_B32: d32(ReadS(w, sdst, inst) & ~(1u << (s0() & 31u))); return true;
			case Op::S_BITSET1_B32: d32(ReadS(w, sdst, inst) | (1u << (s0() & 31u))); return true;
			case Op::S_BITSET0_B64: d64(ReadS64(w, sdst, inst) & ~(1ull << (s0() & 63u))); return true;
			case Op::S_BITSET1_B64: d64(ReadS64(w, sdst, inst) | (1ull << (s0() & 63u))); return true;
			case Op::S_ABS_I32: { const int32_t v = static_cast<int32_t>(s0()); const uint32_t r = static_cast<uint32_t>(v < 0 ? -v : v); d32(r); w.scc = r != 0; return true; }
			case Op::S_GETPC_B64: d64(m_cap.code_addr + inst.pc + inst.words * 4u); return true;
			case Op::S_SETPC_B64: w.pc = static_cast<uint32_t>((s0_64() - m_cap.code_addr) / 4u); return true;
			case Op::S_SWAPPC_B64: {
				const uint64_t target = s0_64();
				d64(m_cap.code_addr + inst.pc + inst.words * 4u);
				w.pc = static_cast<uint32_t>((target - m_cap.code_addr) / 4u);
				return true;
			}
			case Op::S_AND_SAVEEXEC_B64: case Op::S_OR_SAVEEXEC_B64: case Op::S_XOR_SAVEEXEC_B64: case Op::S_ANDN1_SAVEEXEC_B64:
			case Op::S_ANDN2_SAVEEXEC_B64: case Op::S_ORN1_SAVEEXEC_B64: case Op::S_ORN2_SAVEEXEC_B64: case Op::S_NAND_SAVEEXEC_B64:
			case Op::S_NOR_SAVEEXEC_B64: case Op::S_XNOR_SAVEEXEC_B64: case Op::S_AND_SAVEEXEC_B32: case Op::S_OR_SAVEEXEC_B32:
			case Op::S_XOR_SAVEEXEC_B32: case Op::S_ANDN1_SAVEEXEC_B32: case Op::S_ANDN2_SAVEEXEC_B32: case Op::S_ORN1_SAVEEXEC_B32:
			case Op::S_ORN2_SAVEEXEC_B32: case Op::S_NAND_SAVEEXEC_B32: case Op::S_NOR_SAVEEXEC_B32: case Op::S_XNOR_SAVEEXEC_B32:
			case Op::S_ANDN1_WREXEC_B64: case Op::S_ANDN2_WREXEC_B64: case Op::S_ANDN1_WREXEC_B32: case Op::S_ANDN2_WREXEC_B32: {
				const std::string name = kOpNames[static_cast<int>(op)];
				const bool b32  = name.find("_B32") != std::string::npos;
				const uint64_t mask = b32 ? 0xffffffffull : ~0ull;
				const uint64_t src  = (b32 ? static_cast<uint64_t>(s0()) : s0_64()) & mask;
				const uint64_t old  = w.exec & mask;
				uint64_t       nexec = 0;
				if (name.find("S_AND_") == 0) nexec = src & old;
				else if (name.find("S_OR_") == 0) nexec = src | old;
				else if (name.find("S_XOR_") == 0) nexec = src ^ old;
				else if (name.find("S_ANDN1_") == 0) nexec = ~src & old;
				else if (name.find("S_ANDN2_") == 0) nexec = src & ~old;
				else if (name.find("S_ORN1_") == 0) nexec = ~src | old;
				else if (name.find("S_ORN2_") == 0) nexec = src | ~old;
				else if (name.find("S_NAND_") == 0) nexec = ~(src & old);
				else if (name.find("S_NOR_") == 0) nexec = ~(src | old);
				else if (name.find("S_XNOR_") == 0) nexec = ~(src ^ old);
				nexec &= mask & w.Mask();
				const bool wrexec = name.find("WREXEC") != std::string::npos;
				if (b32) d32(static_cast<uint32_t>(wrexec ? nexec : old));
				else d64(wrexec ? nexec : old);
				w.exec = (w.exec & ~mask) | nexec;
				w.scc  = nexec != 0;
				return true;
			}
			case Op::S_MOVRELS_B32: d32(w.s[(inst.F("SSRC0") + w.m0) & 127u]); return true;
			case Op::S_MOVRELD_B32: WriteS(w, (sdst + w.m0) & 127u, s0()); return true;
			// SOP2
			case Op::S_ADD_U32: { const uint64_t r = static_cast<uint64_t>(s0()) + s1(); d32(static_cast<uint32_t>(r)); w.scc = (r >> 32u) != 0; return true; }
			case Op::S_SUB_U32: { const uint32_t a = s0(), b = s1(); d32(a - b); w.scc = b > a; return true; }
			case Op::S_ADDC_U32: { const uint64_t r = static_cast<uint64_t>(s0()) + s1() + (w.scc ? 1u : 0u); d32(static_cast<uint32_t>(r)); w.scc = (r >> 32u) != 0; return true; }
			case Op::S_SUBB_U32: { const uint64_t a = s0(), b = static_cast<uint64_t>(s1()) + (w.scc ? 1u : 0u); d32(static_cast<uint32_t>(a - b)); w.scc = b > a; return true; }
			case Op::S_ADD_I32: {
				const int32_t a = static_cast<int32_t>(s0()), b = static_cast<int32_t>(s1());
				const int32_t r = static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
				d32(static_cast<uint32_t>(r));
				w.scc = ((a >= 0) == (b >= 0)) && ((r >= 0) != (a >= 0));
				return true;
			}
			case Op::S_SUB_I32: {
				const int32_t a = static_cast<int32_t>(s0()), b = static_cast<int32_t>(s1());
				const int32_t r = static_cast<int32_t>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b));
				d32(static_cast<uint32_t>(r));
				w.scc = ((a >= 0) != (b >= 0)) && ((r >= 0) != (a >= 0));
				return true;
			}
			case Op::S_MIN_I32: { const int32_t a = static_cast<int32_t>(s0()), b = static_cast<int32_t>(s1()); w.scc = a < b; d32(static_cast<uint32_t>(a < b ? a : b)); return true; }
			case Op::S_MAX_I32: { const int32_t a = static_cast<int32_t>(s0()), b = static_cast<int32_t>(s1()); w.scc = a > b; d32(static_cast<uint32_t>(a > b ? a : b)); return true; }
			case Op::S_MIN_U32: { const uint32_t a = s0(), b = s1(); w.scc = a < b; d32(a < b ? a : b); return true; }
			case Op::S_MAX_U32: { const uint32_t a = s0(), b = s1(); w.scc = a > b; d32(a > b ? a : b); return true; }
			case Op::S_CSELECT_B32: d32(w.scc ? s0() : s1()); return true;
			case Op::S_CSELECT_B64: d64(w.scc ? s0_64() : s1_64()); return true;
			case Op::S_AND_B32: { const uint32_t r = s0() & s1(); d32(r); w.scc = r != 0; return true; }
			case Op::S_OR_B32: { const uint32_t r = s0() | s1(); d32(r); w.scc = r != 0; return true; }
			case Op::S_XOR_B32: { const uint32_t r = s0() ^ s1(); d32(r); w.scc = r != 0; return true; }
			case Op::S_ANDN2_B32: { const uint32_t r = s0() & ~s1(); d32(r); w.scc = r != 0; return true; }
			case Op::S_ORN2_B32: { const uint32_t r = s0() | ~s1(); d32(r); w.scc = r != 0; return true; }
			case Op::S_NAND_B32: { const uint32_t r = ~(s0() & s1()); d32(r); w.scc = r != 0; return true; }
			case Op::S_NOR_B32: { const uint32_t r = ~(s0() | s1()); d32(r); w.scc = r != 0; return true; }
			case Op::S_XNOR_B32: { const uint32_t r = ~(s0() ^ s1()); d32(r); w.scc = r != 0; return true; }
			case Op::S_AND_B64: { const uint64_t r = s0_64() & s1_64(); d64(r); w.scc = r != 0; return true; }
			case Op::S_OR_B64: { const uint64_t r = s0_64() | s1_64(); d64(r); w.scc = r != 0; return true; }
			case Op::S_XOR_B64: { const uint64_t r = s0_64() ^ s1_64(); d64(r); w.scc = r != 0; return true; }
			case Op::S_ANDN2_B64: { const uint64_t r = s0_64() & ~s1_64(); d64(r); w.scc = r != 0; return true; }
			case Op::S_ORN2_B64: { const uint64_t r = s0_64() | ~s1_64(); d64(r); w.scc = r != 0; return true; }
			case Op::S_NAND_B64: { const uint64_t r = ~(s0_64() & s1_64()); d64(r); w.scc = r != 0; return true; }
			case Op::S_NOR_B64: { const uint64_t r = ~(s0_64() | s1_64()); d64(r); w.scc = r != 0; return true; }
			case Op::S_XNOR_B64: { const uint64_t r = ~(s0_64() ^ s1_64()); d64(r); w.scc = r != 0; return true; }
			case Op::S_LSHL_B32: { const uint32_t r = s0() << (s1() & 31u); d32(r); w.scc = r != 0; return true; }
			case Op::S_LSHR_B32: { const uint32_t r = s0() >> (s1() & 31u); d32(r); w.scc = r != 0; return true; }
			case Op::S_ASHR_I32: { const uint32_t r = static_cast<uint32_t>(static_cast<int32_t>(s0()) >> (s1() & 31u)); d32(r); w.scc = r != 0; return true; }
			case Op::S_LSHL_B64: { const uint64_t r = s0_64() << (s1() & 63u); d64(r); w.scc = r != 0; return true; }
			case Op::S_LSHR_B64: { const uint64_t r = s0_64() >> (s1() & 63u); d64(r); w.scc = r != 0; return true; }
			case Op::S_ASHR_I64: { const uint64_t r = static_cast<uint64_t>(static_cast<int64_t>(s0_64()) >> (s1() & 63u)); d64(r); w.scc = r != 0; return true; }
			case Op::S_BFM_B32: d32(((1u << (s0() & 31u)) - 1u) << (s1() & 31u)); return true;
			case Op::S_BFM_B64: { const uint32_t n = s0() & 63u; d64((n == 0 ? 0ull : ((~0ull) >> (64u - n))) << (s1() & 63u)); return true; }
			case Op::S_MUL_I32: d32(s0() * s1()); return true;
			case Op::S_MUL_HI_U32: d32(static_cast<uint32_t>((static_cast<uint64_t>(s0()) * s1()) >> 32u)); return true;
			case Op::S_MUL_HI_I32: d32(static_cast<uint32_t>((static_cast<int64_t>(static_cast<int32_t>(s0())) * static_cast<int32_t>(s1())) >> 32u)); return true;
			case Op::S_BFE_U32: {
				const uint32_t off = s1() & 31u, width = (s1() >> 16u) & 0x7fu;
				const uint32_t r = width == 0 ? 0u : (width >= 32 ? (s0() >> off) : ((s0() >> off) & ((1u << width) - 1u)));
				d32(r); w.scc = r != 0; return true;
			}
			case Op::S_BFE_I32: {
				const uint32_t off = s1() & 31u, width = (s1() >> 16u) & 0x7fu;
				uint32_t r = 0;
				if (width != 0) {
					const uint32_t v = width >= 32 ? (s0() >> off) : ((s0() >> off) & ((1u << width) - 1u));
					r = width >= 32 ? v : static_cast<uint32_t>(static_cast<int32_t>(v << (32u - width)) >> (32u - width));
				}
				d32(r); w.scc = r != 0; return true;
			}
			case Op::S_BFE_U64: {
				const uint32_t off = s1() & 63u, width = (s1() >> 16u) & 0x7fu;
				const uint64_t r = width == 0 ? 0ull : (width >= 64 ? (s0_64() >> off) : ((s0_64() >> off) & ((1ull << width) - 1u)));
				d64(r); w.scc = r != 0; return true;
			}
			case Op::S_ABSDIFF_I32: {
				const int64_t a = static_cast<int32_t>(s0()), b = static_cast<int32_t>(s1());
				const uint32_t r = static_cast<uint32_t>(a > b ? a - b : b - a);
				d32(r); w.scc = r != 0; return true;
			}
			case Op::S_LSHL1_ADD_U32: case Op::S_LSHL2_ADD_U32: case Op::S_LSHL3_ADD_U32: case Op::S_LSHL4_ADD_U32: {
				const uint32_t n = op == Op::S_LSHL1_ADD_U32 ? 1u : op == Op::S_LSHL2_ADD_U32 ? 2u : op == Op::S_LSHL3_ADD_U32 ? 3u : 4u;
				const uint64_t r = (static_cast<uint64_t>(s0()) << n) + s1();
				d32(static_cast<uint32_t>(r)); w.scc = (r >> 32u) != 0; return true;
			}
			case Op::S_PACK_LL_B32_B16: d32((s0() & 0xffffu) | (s1() << 16u)); return true;
			case Op::S_PACK_LH_B32_B16: d32((s0() & 0xffffu) | (s1() & 0xffff0000u)); return true;
			case Op::S_PACK_HH_B32_B16: d32((s0() >> 16u) | (s1() & 0xffff0000u)); return true;
			// SOPC
			case Op::S_CMP_EQ_I32: case Op::S_CMP_EQ_U32: w.scc = s0() == s1(); return true;
			case Op::S_CMP_LG_I32: case Op::S_CMP_LG_U32: w.scc = s0() != s1(); return true;
			case Op::S_CMP_GT_I32: w.scc = static_cast<int32_t>(s0()) > static_cast<int32_t>(s1()); return true;
			case Op::S_CMP_GE_I32: w.scc = static_cast<int32_t>(s0()) >= static_cast<int32_t>(s1()); return true;
			case Op::S_CMP_LT_I32: w.scc = static_cast<int32_t>(s0()) < static_cast<int32_t>(s1()); return true;
			case Op::S_CMP_LE_I32: w.scc = static_cast<int32_t>(s0()) <= static_cast<int32_t>(s1()); return true;
			case Op::S_CMP_GT_U32: w.scc = s0() > s1(); return true;
			case Op::S_CMP_GE_U32: w.scc = s0() >= s1(); return true;
			case Op::S_CMP_LT_U32: w.scc = s0() < s1(); return true;
			case Op::S_CMP_LE_U32: w.scc = s0() <= s1(); return true;
			case Op::S_CMP_EQ_U64: w.scc = s0_64() == s1_64(); return true;
			case Op::S_CMP_LG_U64: w.scc = s0_64() != s1_64(); return true;
			case Op::S_BITCMP0_B32: w.scc = ((s0() >> (s1() & 31u)) & 1u) == 0; return true;
			case Op::S_BITCMP1_B32: w.scc = ((s0() >> (s1() & 31u)) & 1u) != 0; return true;
			case Op::S_BITCMP0_B64: w.scc = ((s0_64() >> (s1() & 63u)) & 1u) == 0; return true;
			case Op::S_BITCMP1_B64: w.scc = ((s0_64() >> (s1() & 63u)) & 1u) != 0; return true;
			// SOPK
			case Op::S_MOVK_I32: d32(static_cast<uint32_t>(sext16)); return true;
			case Op::S_CMOVK_I32: if (w.scc) d32(static_cast<uint32_t>(sext16)); return true;
			case Op::S_CMPK_EQ_I32: w.scc = static_cast<int32_t>(ReadS(w, sdst, inst)) == sext16; return true;
			case Op::S_CMPK_LG_I32: w.scc = static_cast<int32_t>(ReadS(w, sdst, inst)) != sext16; return true;
			case Op::S_CMPK_GT_I32: w.scc = static_cast<int32_t>(ReadS(w, sdst, inst)) > sext16; return true;
			case Op::S_CMPK_GE_I32: w.scc = static_cast<int32_t>(ReadS(w, sdst, inst)) >= sext16; return true;
			case Op::S_CMPK_LT_I32: w.scc = static_cast<int32_t>(ReadS(w, sdst, inst)) < sext16; return true;
			case Op::S_CMPK_LE_I32: w.scc = static_cast<int32_t>(ReadS(w, sdst, inst)) <= sext16; return true;
			case Op::S_CMPK_EQ_U32: w.scc = ReadS(w, sdst, inst) == simm; return true;
			case Op::S_CMPK_LG_U32: w.scc = ReadS(w, sdst, inst) != simm; return true;
			case Op::S_CMPK_GT_U32: w.scc = ReadS(w, sdst, inst) > simm; return true;
			case Op::S_CMPK_GE_U32: w.scc = ReadS(w, sdst, inst) >= simm; return true;
			case Op::S_CMPK_LT_U32: w.scc = ReadS(w, sdst, inst) < simm; return true;
			case Op::S_CMPK_LE_U32: w.scc = ReadS(w, sdst, inst) <= simm; return true;
			case Op::S_ADDK_I32: {
				const int32_t a = static_cast<int32_t>(ReadS(w, sdst, inst));
				const int32_t r = static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(sext16));
				d32(static_cast<uint32_t>(r));
				w.scc = ((a >= 0) == (sext16 >= 0)) && ((r >= 0) != (a >= 0));
				return true;
			}
			case Op::S_MULK_I32: d32(static_cast<uint32_t>(static_cast<int32_t>(ReadS(w, sdst, inst)) * sext16)); return true;
			case Op::S_GETREG_B32: d32(0); Unsupported(inst, "(hwreg read as 0)"); return true;
			case Op::S_SETREG_B32:
			case Op::S_SETREG_IMM32_B32: Unsupported(inst, "(ignored)"); return true;
			case Op::S_WAITCNT_VSCNT: case Op::S_WAITCNT_VMCNT: case Op::S_WAITCNT_EXPCNT: case Op::S_WAITCNT_LGKMCNT:
			case Op::S_SUBVECTOR_LOOP_BEGIN: case Op::S_SUBVECTOR_LOOP_END: case Op::S_VERSION:
				return true;
			default: break;
		}
		Unsupported(inst);
		return true;
	}

	// ---------------------------------------------------------------- scalar memory
	bool ExecSmem(Wave& w, const Inst& inst) {
		const Op       op    = inst.op;
		const uint32_t sbase = inst.F("SBASE") * 2u;
		const uint32_t sdata = inst.F("SDATA");
		int32_t        imm   = static_cast<int32_t>(inst.F("OFFSET") << 11u) >> 11u; // 21-bit signed
		const uint32_t soff_code = inst.F("SOFFSET");
		const uint32_t soff      = soff_code == Src::Null ? 0u : ReadS(w, soff_code, inst);
		uint32_t       count     = 0;
		bool           buffer    = false;
		switch (op) {
			case Op::S_LOAD_DWORD: count = 1; break;
			case Op::S_LOAD_DWORDX2: count = 2; break;
			case Op::S_LOAD_DWORDX4: count = 4; break;
			case Op::S_LOAD_DWORDX8: count = 8; break;
			case Op::S_LOAD_DWORDX16: count = 16; break;
			case Op::S_BUFFER_LOAD_DWORD: count = 1; buffer = true; break;
			case Op::S_BUFFER_LOAD_DWORDX2: count = 2; buffer = true; break;
			case Op::S_BUFFER_LOAD_DWORDX4: count = 4; buffer = true; break;
			case Op::S_BUFFER_LOAD_DWORDX8: count = 8; buffer = true; break;
			case Op::S_BUFFER_LOAD_DWORDX16: count = 16; buffer = true; break;
			case Op::S_DCACHE_INV: case Op::S_GL1_INV: return true;
			case Op::S_MEMTIME: case Op::S_MEMREALTIME: WriteS64(w, sdata, 0); return true;
			default: Unsupported(inst); return true;
		}
		// SBASE may name VCC (pair 53 = s106:s107), which is not stored in w.s: read through ReadS (the
		// 85312e99a3c69cef capture loads its light-scale constant through a pointer the shader put in VCC).
		const auto sreg = [&](uint32_t code) { return ReadS(w, code, inst); };
		if (!buffer) {
			const uint64_t base = (sreg(sbase) | (static_cast<uint64_t>(sreg(sbase + 1u) & 0xffffu) << 32u));
			const uint64_t addr = (base + static_cast<uint64_t>(static_cast<int64_t>(imm)) + soff) & ~3ull;
			for (uint32_t i = 0; i < count; i++) WriteS(w, sdata + i, m_cap.Read32(addr + 4ull * i));
		} else {
			const uint64_t base = (sreg(sbase) | (static_cast<uint64_t>(sreg(sbase + 1u) & 0xffffu) << 32u));
			const uint32_t num  = sreg(sbase + 2u);
			const uint32_t stride = (sreg(sbase + 1u) >> 16u) & 0x3fffu;
			const uint64_t limit = stride == 0 ? num : static_cast<uint64_t>(num) * stride;
			const uint64_t off   = (static_cast<uint64_t>(static_cast<int64_t>(imm)) + soff) & ~3ull;
			for (uint32_t i = 0; i < count; i++) {
				const uint64_t o = off + 4ull * i;
				WriteS(w, sdata + i, o + 4 <= limit ? m_cap.Read32(base + o) : 0u);
			}
		}
		return true;
	}

	// ---------------------------------------------------------------- buffer memory (MUBUF/MTBUF)
	struct VSharp {
		uint64_t base   = 0;
		uint32_t stride = 0, num = 0, oob = 0;
		bool     add_tid = false;
		explicit VSharp(const uint32_t* w) {
			base    = w[0] | (static_cast<uint64_t>(w[1] & 0xffffu) << 32u);
			stride  = (w[1] >> 16u) & 0x3fffu;
			num     = w[2];
			oob     = (w[3] >> 28u) & 3u;
			add_tid = ((w[3] >> 23u) & 1u) != 0;
		}
	};

	bool ExecMubuf(Wave& w, const Inst& inst) {
		const Op       op    = inst.op;
		const std::string name = kOpNames[static_cast<int>(op)];
		if (op == Op::BUFFER_GL0_INV || op == Op::BUFFER_GL1_INV) return true;
		const VSharp   v(&w.s[inst.F("SRSRC") * 4u]);
		const uint32_t vaddr = inst.F("VADDR"), vdata = inst.F("VDATA");
		const uint32_t soffset = ReadS(w, inst.F("SOFFSET"), inst);
		const uint32_t imm     = inst.F("OFFSET");
		const bool     offen = inst.F("OFFEN") != 0, idxen = inst.F("IDXEN") != 0;
		uint32_t bytes = 4, count = 1;
		bool     store = false, atomic = false, sext = false;
		if (name.find("ATOMIC") != std::string::npos) atomic = true;
		if (name.find("STORE") != std::string::npos) store = true;
		if (name.find("DWORDX2") != std::string::npos || name.find("_X2") != std::string::npos) count = 2;
		else if (name.find("DWORDX3") != std::string::npos) count = 3;
		else if (name.find("DWORDX4") != std::string::npos) count = 4;
		if (name.find("BYTE") != std::string::npos) bytes = 1;
		if (name.find("SHORT") != std::string::npos) bytes = 2;
		if (name.find("SBYTE") != std::string::npos || name.find("SSHORT") != std::string::npos) sext = true;
		// Plain D16 loads write 16 bits into the low (or _HI: high) half of VDATA and keep the other
		// half; D16_HI stores write the high half.
		const bool d16_plain = name.find("D16") != std::string::npos && name.find("FORMAT") == std::string::npos && !atomic;
		const bool d16_hi    = d16_plain && name.find("D16_HI") != std::string::npos;
		// BUFFER_LOAD_FORMAT_D16_*: the format-converted components are packed as 16-bit values, two per VGPR (see
		// PutResults); D16_HI_X writes the high half of VDATA. D16 format stores are not modelled.
		const bool d16_format_load = name.find("D16") != std::string::npos && name.find("FORMAT") != std::string::npos && !store && !atomic;
		if ((name.find("D16") != std::string::npos && !d16_plain && !d16_format_load) || (name.find("FORMAT") != std::string::npos && store)) {
			Unsupported(inst);
			return true;
		}
		if (name.find("FORMAT") != std::string::npos) {
			// GFX10 unified format (V# word3 [18:12]) and DST_SEL; plain formats only.
			const uint32_t* vs = &w.s[inst.F("SRSRC") * 4u];
			const uint32_t  fmt = (vs[3] >> 12u) & 0x7fu;
			struct BF { uint32_t comps, bits; int kind; bool packed = false; uint32_t pbits[4] {}, poff[4] {}; }; // kind: 0 unorm 1 snorm 2 uscaled 3 sscaled 4 uint 5 sint 6 float
			auto bf = [&](uint32_t f) -> BF {
				static const int k6[6] = {0, 1, 2, 3, 4, 5};
				if (f >= 1 && f <= 6) return {1, 8, k6[f - 1]};
				if (f >= 7 && f <= 13) return {1, 16, f == 13 ? 6 : k6[f - 7]};
				if (f >= 14 && f <= 19) return {2, 8, k6[f - 14]};
				if (f >= 20 && f <= 22) return {1, 32, f == 20 ? 4 : f == 21 ? 5 : 6};
				if (f >= 23 && f <= 29) return {2, 16, f == 29 ? 6 : k6[f - 23]};
				if (f >= 56 && f <= 61) return {4, 8, k6[f - 56]};
				if (f >= 62 && f <= 64) return {2, 32, f == 62 ? 4 : f == 63 ? 5 : 6};
				if (f >= 65 && f <= 71) return {4, 16, f == 71 ? 6 : k6[f - 65]};
				if (f >= 72 && f <= 74) return {3, 32, f == 72 ? 4 : f == 73 ? 5 : 6};
				if (f >= 75 && f <= 77) return {4, 32, f == 75 ? 4 : f == 76 ? 5 : 6};
				// Packed 2_10_10_10 (44-49: x 2 bits, then 10/10/10) and 10_10_10_2 (50-55: x/y/z 10 bits, w 2 bits) in one
				// dword, low bits first (BufferFormat.h).
				if (f >= 44 && f <= 55) {
					BF r {4, 32, k6[(f - 44) % 6], true};
					const bool x2 = f <= 49;
					for (uint32_t c = 0; c < 4; c++) {
						r.pbits[c] = x2 ? (c == 0 ? 2u : 10u) : (c == 3 ? 2u : 10u);
						r.poff[c]  = x2 ? (c == 0 ? 0u : 2u + 10u * (c - 1u)) : 10u * c;
					}
					return r;
				}
				return {0, 0, -1};
			};
			const BF b = bf(fmt);
			if (b.kind < 0) { Unsupported(inst, ("(buffer format " + std::to_string(fmt) + ")").c_str()); return true; }
			const uint32_t out_count = name.find("XYZW") != std::string::npos ? 4 : name.find("XYZ") != std::string::npos ? 3 : name.find("XY") != std::string::npos ? 2 : 1;
			const uint32_t dst_sel[4] = {vs[3] & 7u, (vs[3] >> 3u) & 7u, (vs[3] >> 6u) & 7u, (vs[3] >> 9u) & 7u};
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (!w.Active(l)) continue;
				uint32_t index = idxen ? w.v[vaddr][l] : 0u;
				uint32_t off   = offen ? w.v[vaddr + (idxen ? 1u : 0u)][l] : 0u;
				if (v.add_tid) index += l;
				const uint64_t inst_off = static_cast<uint64_t>(off) + imm;
				bool oob = false;
				if (v.stride == 0) oob = v.oob != 2 && inst_off + (b.comps * b.bits / 8u) > v.num;
				else oob = (v.oob != 2 && index >= v.num) || (v.oob == 0 && inst_off >= v.stride);
				const uint64_t addr = v.base + soffset + inst_off + static_cast<uint64_t>(index) * v.stride;
				uint32_t comp[4] = {0, 0, 0, B32(1.0f)};
				const bool integer = b.kind == 4 || b.kind == 5;
				if (integer) comp[3] = 1u;
				uint32_t packed_word = 0;
				if (b.packed && !oob) m_cap.Read(addr, &packed_word, 4);
				for (uint32_t c = 0; c < b.comps && !oob; c++) {
					uint32_t raw = 0;
					const uint32_t cbits = b.packed ? b.pbits[c] : b.bits;
					if (b.packed) raw = (packed_word >> b.poff[c]) & ((1u << cbits) - 1u);
					else m_cap.Read(addr + c * (b.bits / 8u), &raw, b.bits / 8u);
					const uint32_t mask = cbits == 32 ? 0xffffffffu : ((1u << cbits) - 1u);
					raw &= mask;
					const int32_t sraw = cbits == 32 ? static_cast<int32_t>(raw) : static_cast<int32_t>(raw << (32u - cbits)) >> (32u - cbits);
					switch (b.kind) {
						case 0: comp[c] = B32(static_cast<float>(static_cast<double>(raw) / static_cast<double>(mask))); break;
						case 1: comp[c] = B32(std::max(-1.0f, static_cast<float>(sraw) / static_cast<float>((1u << (cbits - 1u)) - 1u))); break;
						case 2: comp[c] = B32(static_cast<float>(raw)); break;
						case 3: comp[c] = B32(static_cast<float>(sraw)); break;
						case 4: comp[c] = raw; break;
						case 5: comp[c] = static_cast<uint32_t>(sraw); break;
						default: comp[c] = cbits == 16 ? B32(F16ToF32(static_cast<uint16_t>(raw))) : raw; break;
					}
				}
				uint32_t out[4] {0, 0, 0, 0};
				for (uint32_t c = 0; c < out_count; c++) {
					const uint32_t sel = dst_sel[c];
					out[c] = sel == 0 ? 0u : sel == 1 ? (integer ? 1u : B32(1.0f)) : sel >= 4 ? comp[sel - 4] : 0u;
				}
				if (d16_format_load && name.find("D16_HI") != std::string::npos) {
					// D16_HI_X: the single component goes to bits [31:16], the low half is kept.
					const uint32_t half = integer ? D16Integer(out[0], b.kind == 5) : static_cast<uint32_t>(F32ToF16(F32(out[0])));
					w.v[vdata][l]       = (w.v[vdata][l] & 0xffffu) | (half << 16u);
				} else {
					PutResults(w, vdata, l, out, out_count, d16_format_load, integer, b.kind == 5);
				}
			}
			return true;
		}
		const bool glc = inst.F("GLC") != 0;
		for (uint32_t l = 0; l < w.lanes; l++) {
			if (!w.Active(l)) continue;
			uint32_t index = idxen ? w.v[vaddr][l] : 0u;
			uint32_t off   = offen ? w.v[vaddr + (idxen ? 1u : 0u)][l] : 0u;
			if (v.add_tid) index += l;
			const uint64_t inst_off = static_cast<uint64_t>(off) + imm;
			bool oob = false;
			if (v.stride == 0 || !idxen) {
				const uint64_t byte = inst_off + static_cast<uint64_t>(index) * v.stride;
				if (v.oob != 2 && byte + bytes * count > static_cast<uint64_t>(v.num) * (v.stride == 0 ? 1u : 1u) && v.stride == 0) oob = true;
				if (v.stride != 0 && v.oob != 2 && index >= v.num) oob = true;
			} else {
				if (v.oob != 2 && index >= v.num) oob = true;
				if (v.oob == 0 && inst_off >= v.stride) oob = true;
			}
			const uint64_t addr = v.base + soffset + inst_off + static_cast<uint64_t>(index) * v.stride;
			if (atomic) {
				const bool x2 = count == 2;
				uint64_t   old = 0;
				m_cap.Read(addr, &old, x2 ? 8 : 4);
				uint64_t data = w.v[vdata][l] | (x2 ? (static_cast<uint64_t>(w.v[vdata + 1u][l]) << 32u) : 0ull);
				uint64_t cmp  = 0;
				if (name.find("CMPSWAP") != std::string::npos) {
					cmp = x2 ? (w.v[vdata + 2u][l] | (static_cast<uint64_t>(w.v[vdata + 3u][l]) << 32u)) : w.v[vdata + 1u][l];
				}
				uint64_t r = old;
				if (name.find("_ADD") != std::string::npos) r = old + data;
				else if (name.find("_SUB") != std::string::npos) r = old - data;
				else if (name.find("_SMIN") != std::string::npos) r = x2 ? static_cast<uint64_t>(std::min(static_cast<int64_t>(old), static_cast<int64_t>(data))) : static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(data)));
				else if (name.find("_UMIN") != std::string::npos) r = std::min(old, data);
				else if (name.find("_SMAX") != std::string::npos) r = x2 ? static_cast<uint64_t>(std::max(static_cast<int64_t>(old), static_cast<int64_t>(data))) : static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(data)));
				else if (name.find("_UMAX") != std::string::npos) r = std::max(old, data);
				else if (name.find("_AND") != std::string::npos) r = old & data;
				else if (name.find("_OR") != std::string::npos) r = old | data;
				else if (name.find("_XOR") != std::string::npos) r = old ^ data;
				else if (name.find("_SWAP") != std::string::npos && name.find("CMP") == std::string::npos) r = data;
				else if (name.find("CMPSWAP") != std::string::npos) r = old == cmp ? data : old;
				else if (!x2 && (name.find("_FMIN") != std::string::npos || name.find("_FMAX") != std::string::npos)) {
					// 32-bit float min/max: a NaN on either side, or two zeros, keep the old value; otherwise the IEEE
					// order of the two values (the emulator's EmitFloatAtomicReplacement).
					const auto o = static_cast<uint32_t>(old), d = static_cast<uint32_t>(data);
					const auto nan  = [](uint32_t b) { return (b & 0x7fffffffu) > 0x7f800000u; };
					const auto zero = [](uint32_t b) { return (b & 0x7fffffffu) == 0u; };
					const auto key  = [](uint32_t b) { return (b & 0x80000000u) != 0u ? ~b : (b ^ 0x80000000u); };
					const bool max  = name.find("_FMAX") != std::string::npos;
					const bool take = !(nan(o) || nan(d) || (zero(o) && zero(d))) && (max ? key(d) > key(o) : key(d) < key(o));
					r = take ? d : o;
				}
				else if (name.find("_INC") != std::string::npos) r = old >= data ? 0 : old + 1;
				else if (name.find("_DEC") != std::string::npos) r = (old == 0 || old > data) ? data : old - 1;
				else { Unsupported(inst); return true; }
				if (!x2) r &= 0xffffffffull;
				if (!oob) m_cap.Write(addr, &r, x2 ? 8 : 4);
				if (glc) {
					w.v[vdata][l] = oob ? 0u : static_cast<uint32_t>(old);
					if (x2) w.v[vdata + 1u][l] = oob ? 0u : static_cast<uint32_t>(old >> 32u);
				}
				continue;
			}
			for (uint32_t i = 0; i < count; i++) {
				if (store) {
					const uint32_t value = d16_hi ? (w.v[vdata + i][l] >> 16u) : w.v[vdata + i][l];
					WatchStore(inst, l, addr + 4ull * i, bytes, value, oob);
					if (!oob) m_cap.Write(addr + 4ull * i, &value, bytes);
				} else {
					uint32_t value = 0;
					if (!oob) m_cap.Read(addr + 4ull * i, &value, bytes);
					if (sext && bytes == 1) value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
					if (sext && bytes == 2) value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
					if (d16_plain) {
						value &= 0xffffu;
						const uint32_t old = w.v[vdata + i][l];
						value = d16_hi ? ((old & 0xffffu) | (value << 16u)) : ((old & 0xffff0000u) | value);
					}
					w.v[vdata + i][l] = value;
				}
			}
		}
		return true;
	}

	void WatchStore(const Inst& inst, uint32_t lane, uint64_t addr, uint32_t bytes, uint32_t value, bool oob) {
		if (m_opt.watch_addr == 0 || m_opt.watch_addr < addr || m_opt.watch_addr >= addr + bytes) return;
		float f = 0;
		std::memcpy(&f, &value, 4);
		std::printf("WATCH store pc=0x%x %s lane=%u addr=%llx value=0x%08x (%g)%s\n", inst.pc,
		            kOpNames[static_cast<int>(inst.op)], lane, static_cast<unsigned long long>(addr), value, f,
		            oob ? " OOB" : "");
	}

	// ---------------------------------------------------------------- GLOBAL (FLAT encoding, SEG=2)
	// Address = SGPR pair SADDR + zero-extended VGPR ADDR, or the VGPR pair ADDR when SADDR is
	// off (NULL/0x7f); plus the signed 12-bit OFFSET. Loads, stores and their D16 forms.
	bool ExecGlobal(Wave& w, const Inst& inst) {
		const std::string name = kOpNames[static_cast<int>(inst.op)];
		if (inst.F("SEG") != 2u || name.rfind("GLOBAL_", 0) != 0 || name.find("ATOMIC") != std::string::npos) {
			Unsupported(inst, "(flat)");
			return true;
		}
		const uint32_t vaddr = inst.F("ADDR"), vdata = inst.F("DATA"), vdst = inst.F("VDST"), saddr = inst.F("SADDR");
		int32_t        offset = static_cast<int32_t>(inst.F("OFFSET") & 0xfffu);
		if ((offset & 0x800) != 0) offset -= 0x1000;
		const bool store = name.find("STORE") != std::string::npos;
		uint32_t   bytes = 4, count = 1;
		bool       sext = false;
		if (name.find("DWORDX2") != std::string::npos) count = 2;
		else if (name.find("DWORDX3") != std::string::npos) count = 3;
		else if (name.find("DWORDX4") != std::string::npos) count = 4;
		if (name.find("BYTE") != std::string::npos) bytes = 1;
		if (name.find("SHORT") != std::string::npos) bytes = 2;
		if (name.find("SBYTE") != std::string::npos || name.find("SSHORT") != std::string::npos) sext = true;
		const bool d16    = name.find("D16") != std::string::npos;
		const bool d16_hi = name.find("D16_HI") != std::string::npos;
		const bool saddr_off = saddr == 0x7du || saddr == 0x7fu;
		const uint64_t sbase = saddr_off ? 0ull : (static_cast<uint64_t>(w.s[saddr]) | (static_cast<uint64_t>(w.s[saddr + 1u]) << 32u));
		for (uint32_t l = 0; l < w.lanes; l++) {
			if (!w.Active(l)) continue;
			const uint64_t base = saddr_off ? (static_cast<uint64_t>(w.v[vaddr][l]) | (static_cast<uint64_t>(w.v[vaddr + 1u][l]) << 32u))
			                                : sbase + static_cast<uint64_t>(w.v[vaddr][l]);
			const uint64_t addr = base + static_cast<int64_t>(offset);
			for (uint32_t i = 0; i < count; i++) {
				if (store) {
					const uint32_t value = d16_hi ? (w.v[vdata + i][l] >> 16u) : w.v[vdata + i][l];
					WatchStore(inst, l, addr + 4ull * i, bytes, value, false);
					m_cap.Write(addr + 4ull * i, &value, bytes);
				} else {
					uint32_t value = 0;
					m_cap.Read(addr + 4ull * i, &value, bytes);
					if (sext && bytes == 1) value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
					if (sext && bytes == 2) value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
					if (d16) {
						value &= 0xffffu;
						const uint32_t old = w.v[vdst + i][l];
						value = d16_hi ? ((old & 0xffffu) | (value << 16u)) : ((old & 0xffff0000u) | value);
					}
					w.v[vdst + i][l] = value;
				}
			}
		}
		return true;
	}

	// ---------------------------------------------------------------- LDS (DS)
	bool ExecDs(Wave& w, Group& g, const Inst& inst) {
		const Op           op   = inst.op;
		const std::string  name = kOpNames[static_cast<int>(op)];
		const uint32_t     addr_v = inst.F("ADDR"), d0 = inst.F("DATA0"), d1 = inst.F("DATA1"), vdst = inst.F("VDST");
		const uint32_t     off0 = inst.F("OFFSET0"), off1 = inst.F("OFFSET1");
		const uint32_t     offset = off0 | (off1 << 8u);
		if (inst.F("GDS")) { Unsupported(inst, "(gds)"); return true; }
		auto rd = [&](uint32_t a, void* out, uint32_t n) {
			if (static_cast<uint64_t>(a) + n <= g.lds.size()) {
				std::memcpy(out, g.lds.data() + a, n);
				// A read of LDS bytes this group never stored returns the interpreter's zero fill; on hardware
				// (and the host GPU) such bytes hold whatever an earlier workgroup left there.
				for (uint32_t i = 0; i < n; i++) {
					if (!g.lds_written[a + i]) {
						char key[96];
						std::snprintf(key, sizeof(key), "DS read of never-written LDS (zero fill) pc=0x%x", inst.pc);
						approx_ops[key]++;
						break;
					}
				}
			} else {
				std::memset(out, 0, n);
			}
		};
		auto wr = [&](uint32_t a, const void* in, uint32_t n) {
			if (static_cast<uint64_t>(a) + n <= g.lds.size()) {
				std::memcpy(g.lds.data() + a, in, n);
				std::memset(g.lds_written.data() + a, 1, n);
			}
		};
		if (op == Op::DS_READ_ADDTID_B32 || op == Op::DS_WRITE_ADDTID_B32) {
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (!w.Active(l)) continue;
				const uint32_t a = offset + (w.m0 & 0xffffu) + l * 4u;
				if (op == Op::DS_READ_ADDTID_B32) rd(a, &w.v[vdst][l], 4);
				else wr(a, &w.v[d0][l], 4);
			}
			approx_ops["DS_*_ADDTID (address = offset + M0[15:0] + lane*4)"]++;
			return true;
		}
		if (op == Op::DS_SWIZZLE_B32 || op == Op::DS_PERMUTE_B32 || op == Op::DS_BPERMUTE_B32) {
			const auto src = w.v[d0];
			std::array<uint32_t, 64> out = w.v[vdst];
			if (op == Op::DS_BPERMUTE_B32) {
				for (uint32_t l = 0; l < w.lanes; l++) {
					if (!w.Active(l)) continue;
					const uint32_t sl = ((w.v[addr_v][l] + offset) >> 2u) & (w.lanes - 1u);
					out[l] = w.Active(sl) ? src[sl] : 0u;
				}
			} else if (op == Op::DS_PERMUTE_B32) {
				for (uint32_t l = 0; l < w.lanes; l++) if (w.Active(l)) out[l] = 0;
				for (uint32_t l = 0; l < w.lanes; l++) {
					if (!w.Active(l)) continue;
					const uint32_t dl = ((w.v[addr_v][l] + offset) >> 2u) & (w.lanes - 1u);
					out[dl] = src[l];
				}
			} else {
				if (offset & 0x8000u) {
					for (uint32_t l = 0; l < w.lanes; l++) {
						if (!w.Active(l)) continue;
						const uint32_t base = l & ~3u;
						const uint32_t sl   = base | ((offset >> (2u * (l & 3u))) & 3u);
						out[l] = w.Active(sl) ? src[sl] : 0u;
					}
				} else {
					const uint32_t and_mask = offset & 0x1fu, or_mask = (offset >> 5u) & 0x1fu, xor_mask = (offset >> 10u) & 0x1fu;
					for (uint32_t l = 0; l < w.lanes; l++) {
						if (!w.Active(l)) continue;
						const uint32_t sl = (l & ~31u) | ((((l & 31u) & and_mask) | or_mask) ^ xor_mask);
						out[l] = w.Active(sl) ? src[sl] : 0u;
					}
				}
			}
			w.v[vdst] = out;
			return true;
		}
		for (uint32_t l = 0; l < w.lanes; l++) {
			if (!w.Active(l)) continue;
			const uint32_t a = w.v[addr_v][l];
			auto two = [&](uint32_t size, uint32_t scale) {
				return std::pair<uint32_t, uint32_t> {a + off0 * size * scale, a + off1 * size * scale};
			};
			if (name == "DS_READ_B32" || name == "DS_READ_B64" || name == "DS_READ_B96" || name == "DS_READ_B128") {
				const uint32_t n = name == "DS_READ_B32" ? 1 : name == "DS_READ_B64" ? 2 : name == "DS_READ_B96" ? 3 : 4;
				for (uint32_t i = 0; i < n; i++) rd(a + offset + 4 * i, &w.v[vdst + i][l], 4);
			} else if (name == "DS_WRITE_B32" || name == "DS_WRITE_B64" || name == "DS_WRITE_B96" || name == "DS_WRITE_B128") {
				const uint32_t n = name == "DS_WRITE_B32" ? 1 : name == "DS_WRITE_B64" ? 2 : name == "DS_WRITE_B96" ? 3 : 4;
				for (uint32_t i = 0; i < n; i++) wr(a + offset + 4 * i, &w.v[d0 + i][l], 4);
			} else if (name == "DS_READ2_B32" || name == "DS_READ2ST64_B32") {
				const auto [x, y] = two(4, name == "DS_READ2ST64_B32" ? 64u : 1u);
				rd(x, &w.v[vdst][l], 4);
				rd(y, &w.v[vdst + 1u][l], 4);
			} else if (name == "DS_READ2_B64" || name == "DS_READ2ST64_B64") {
				const auto [x, y] = two(8, name == "DS_READ2ST64_B64" ? 64u : 1u);
				rd(x, &w.v[vdst][l], 4); rd(x + 4, &w.v[vdst + 1u][l], 4);
				rd(y, &w.v[vdst + 2u][l], 4); rd(y + 4, &w.v[vdst + 3u][l], 4);
			} else if (name == "DS_WRITE2_B32" || name == "DS_WRITE2ST64_B32") {
				const auto [x, y] = two(4, name == "DS_WRITE2ST64_B32" ? 64u : 1u);
				wr(x, &w.v[d0][l], 4);
				wr(y, &w.v[d1][l], 4);
			} else if (name == "DS_WRITE2_B64" || name == "DS_WRITE2ST64_B64") {
				const auto [x, y] = two(8, name == "DS_WRITE2ST64_B64" ? 64u : 1u);
				wr(x, &w.v[d0][l], 4); wr(x + 4, &w.v[d0 + 1u][l], 4);
				wr(y, &w.v[d1][l], 4); wr(y + 4, &w.v[d1 + 1u][l], 4);
			} else if (name == "DS_READ_U8" || name == "DS_READ_I8" || name == "DS_READ_U16" || name == "DS_READ_I16") {
				uint32_t v = 0;
				const uint32_t n = name.find("8") != std::string::npos ? 1 : 2;
				rd(a + offset, &v, n);
				if (name == "DS_READ_I8") v = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(v)));
				if (name == "DS_READ_I16") v = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(v)));
				w.v[vdst][l] = v;
			} else if (name == "DS_WRITE_B8" || name == "DS_WRITE_B16") {
				wr(a + offset, &w.v[d0][l], name == "DS_WRITE_B8" ? 1 : 2);
			} else if (name.rfind("DS_", 0) == 0) {
				// atomics: <op>[_RTN]_<type>
				const bool rtn = name.find("_RTN") != std::string::npos;
				const bool b64 = name.find("64") != std::string::npos;
				const uint32_t n = b64 ? 8 : 4;
				uint64_t old = 0;
				rd(a + offset, &old, n);
				uint64_t data = w.v[d0][l] | (b64 ? (static_cast<uint64_t>(w.v[d0 + 1u][l]) << 32u) : 0ull);
				uint64_t data2 = w.v[d1][l] | (b64 ? (static_cast<uint64_t>(w.v[d1 + 1u][l]) << 32u) : 0ull);
				uint64_t r = old;
				std::string base = name.substr(3);
				if (rtn) base.erase(base.find("_RTN"), 4);
				const auto us = base.rfind('_');
				const auto kind = base.substr(0, us), type = base.substr(us + 1);
				if (kind == "ADD" && type == "F32") r = B32(F32(static_cast<uint32_t>(old)) + F32(static_cast<uint32_t>(data)));
				else if (kind == "ADD") r = old + data;
				else if (kind == "SUB") r = old - data;
				else if (kind == "RSUB") r = data - old;
				else if (kind == "INC") r = old >= data ? 0 : old + 1;
				else if (kind == "DEC") r = (old == 0 || old > data) ? data : old - 1;
				else if (kind == "AND") r = old & data;
				else if (kind == "OR") r = old | data;
				else if (kind == "XOR") r = old ^ data;
				else if (kind == "MIN" && type == "U32") r = std::min(static_cast<uint32_t>(old), static_cast<uint32_t>(data));
				else if (kind == "MAX" && type == "U32") r = std::max(static_cast<uint32_t>(old), static_cast<uint32_t>(data));
				else if (kind == "MIN" && type == "I32") r = static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(data)));
				else if (kind == "MAX" && type == "I32") r = static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(data)));
				else if (kind == "MIN" && type == "F32") r = B32(MinF(F32(static_cast<uint32_t>(old)), F32(static_cast<uint32_t>(data))));
				else if (kind == "MAX" && type == "F32") r = B32(MaxF(F32(static_cast<uint32_t>(old)), F32(static_cast<uint32_t>(data))));
				else if (kind == "MIN" && type == "U64") r = std::min(old, data);
				else if (kind == "MAX" && type == "U64") r = std::max(old, data);
				else if (kind == "WRXCHG") r = data;
				else if (kind == "CMPST") r = old == data ? data2 : old;
				else if (kind == "MSKOR") r = (old & ~data) | data2;
				else { Unsupported(inst); return true; }
				if (!b64) r &= 0xffffffffull;
				wr(a + offset, &r, n);
				if (rtn) {
					w.v[vdst][l] = static_cast<uint32_t>(old);
					if (b64) w.v[vdst + 1u][l] = static_cast<uint32_t>(old >> 32u);
				}
			} else {
				Unsupported(inst);
				return true;
			}
		}
		return true;
	}

	// D16 integer results saturate to the 16-bit range (upstream's BufferLoadFormatD16X* expectations; RADV treats AMD's
	// integer D16 returns as saturating).
	static uint32_t D16Integer(uint32_t value, bool is_signed) {
		if (!is_signed) return std::min(value, 0xffffu);
		return static_cast<uint32_t>(std::clamp(static_cast<int32_t>(value), -32768, 32767)) & 0xffffu;
	}

	// ---------------------------------------------------------------- images (MIMG)
	// Result registers of one lane: `n` values go to consecutive VDATA registers, one each, or with D16 two per register,
	// packed (the first in bits [15:0]). AMD RDNA ISA, MIMG D16: "convert 32 bit floating point texture return data to
	// 16 bit" - a float result (32-bit bits) becomes f16 (round to nearest even), an integer result (UINT/SINT formats)
	// saturates to 16 bits (see D16Integer); the unused half of the last register of an odd count keeps its old value
	// (as the D16 loads of buffer memory do).
	static void PutResults(Wave& w, uint32_t vdata, uint32_t lane, const uint32_t* values, uint32_t n, bool d16, bool integer,
	                       bool is_signed = false) {
		for (uint32_t k = 0; k < n; k++) {
			if (!d16) {
				w.v[vdata + k][lane] = values[k];
				continue;
			}
			const uint32_t half = integer ? D16Integer(values[k], is_signed) : static_cast<uint32_t>(F32ToF16(F32(values[k])));
			uint32_t&      reg  = w.v[vdata + k / 2u][lane];
			reg = (k & 1u) != 0 ? ((reg & 0xffffu) | (half << 16u)) : ((reg & 0xffff0000u) | half);
		}
	}

	bool ExecMimg(Wave& w, const Inst& inst) {
		const Op          op   = inst.op;
		const std::string name = kOpNames[static_cast<int>(op)];
		const uint32_t    srsrc = inst.F("SRSRC") * 4u, ssamp = inst.F("SSAMP") * 4u;
		const uint32_t    vdata = inst.F("VDATA"), dmask = inst.F("DMASK"), dim = inst.F("DIM");
		const bool        a16 = inst.F("A16") != 0, d16 = inst.F("D16") != 0, unorm = inst.F("UNORM") != 0;
		// D16 results (loads, samples, gathers) are packed by PutResults; D16 store data is unpacked where the store reads
		// VDATA. A16 address components are read by addr_raw / addr_f below.
		const TSharp      t(&w.s[srsrc]);
		CapImage*         image = m_cap.FindImage(t.base);
		if (image == nullptr) {
			// Under --probe the emulator's dead-code elimination can drop an image whose result only fed the
			// replaced store data: such a load returns zeros here instead of stopping the run.
			if (!m_opt.probes.empty() && op != Op::IMAGE_STORE) {
				Unsupported(inst, "(image not captured under --probe: zeros)");
				for (uint32_t c = 0; c < 4u; c++)
					for (uint32_t l = 0; l < w.lanes; l++)
						if ((w.exec >> l) & 1u) w.v[(vdata + c) & 255u][l] = 0;
				return true;
			}
			errors.push_back("image at 0x" + Hex(t.base) + " not captured (" + name + ")");
			return false;
		}
		ImageView view = MakeView(image, t);
		if (!view.format.ok) { Unsupported(inst, ("(format " + image->fmt + ")").c_str()); return true; }
		// address VGPRs (NSA or consecutive)
		const uint32_t nsa = inst.F("NSA");
		auto addr_reg = [&](uint32_t i) -> uint32_t {
			if (nsa == 0 || i == 0) return inst.F("VADDR") + (nsa == 0 ? i : 0u);
			const uint32_t k = i - 1u; // NSA dwords hold 4 bytes each
			return (inst.raw[2 + k / 4u] >> (8u * (k % 4u))) & 0xffu;
		};
		// Address component `j` of lane `l`, after `lead` leading 32-bit address dwords (the depth-compare reference of
		// IMAGE_SAMPLE_C* stays 32-bit). Without A16 it is the whole VGPR. With A16 (MIMG bit 62: "texture coordinates
		// size 1=16b") the 16-bit components - coordinates, array slice, mip/lod - are packed two per address dword in
		// order, the first in bits [15:0]: component j is half (j & 1) of address dword lead + j / 2 (e.g. 2D: x,y in
		// the first VGPR, then mip/lod or the next component in the low half of the second).
		auto addr_raw = [&](uint32_t lead, uint32_t j, uint32_t l) -> uint32_t {
			if (!a16) return w.v[addr_reg(lead + j)][l];
			return (w.v[addr_reg(lead + j / 2u)][l] >> (16u * (j & 1u))) & 0xffffu;
		};
		// ... as a float coordinate (sample / gather): an A16 component is an f16.
		auto addr_f = [&](uint32_t lead, uint32_t j, uint32_t l) -> float {
			const uint32_t raw = addr_raw(lead, j, l);
			return a16 ? F16ToF32(static_cast<uint16_t>(raw)) : F32(raw);
		};
		const bool three_d = dim == 2;
		const uint32_t ncoord = dim == 0 ? 1 : (dim == 1 ? 2 : (dim == 2 || dim == 3 || dim == 5) ? 3 : dim == 4 ? 2 : dim == 6 ? 3 : 4);
		const bool is_store  = name.rfind("IMAGE_STORE", 0) == 0;
		const bool is_load   = name.rfind("IMAGE_LOAD", 0) == 0;
		const bool is_sample = name.rfind("IMAGE_SAMPLE", 0) == 0;
		const bool is_gather = name.rfind("IMAGE_GATHER4", 0) == 0;
		const bool integer   = view.format.kind == FormatInfo::Uint || view.format.kind == FormatInfo::Sint;
		if (name == "IMAGE_GET_RESINFO") {
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (!w.Active(l)) continue;
				const uint32_t mip = addr_raw(0, 0, l) + t.base_level;
				const uint32_t vals[4] {std::max(1u, t.width >> mip), std::max(1u, t.height >> mip), three_d ? std::max(1u, t.depth >> mip) : t.depth, t.last_level - t.base_level + 1u};
				uint32_t k = 0;
				for (uint32_t c = 0; c < 4; c++) if ((dmask >> c) & 1u) w.v[vdata + k++][l] = vals[c];
			}
			return true;
		}
		if (is_store || is_load) {
			const bool mip_op = name.find("_MIP") != std::string::npos;
			if (name.find("PCK") != std::string::npos || name.find("BY") != std::string::npos) { Unsupported(inst); return true; }
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (!w.Active(l)) continue;
				int c[4] {0, 0, 0, 0};
				// integer coordinates: 32-bit (as signed), or with A16 unsigned 16-bit
				for (uint32_t i = 0; i < ncoord && i < 4; i++) c[i] = static_cast<int>(addr_raw(0, i, l));
				const uint32_t mip = t.base_level + (mip_op ? addr_raw(0, ncoord, l) : 0u);
				const int z = ncoord >= 3 ? c[2] : 0;
				if (is_store) {
					float    cur[4];
					uint32_t raw[4];
					uint32_t comp[4] {0, 0, 0, 0};
					// read-modify for components not in dmask
					if (view.Fetch(mip, c[0], ncoord >= 2 ? c[1] : 0, z, cur, raw)) {
						for (int k = 0; k < 4; k++) comp[k] = integer ? raw[k] : B32(cur[k]);
					}
					uint32_t k = 0;
					for (uint32_t ch = 0; ch < 4; ch++) {
						if (((dmask >> ch) & 1u) == 0) continue;
						if (!d16) {
							comp[ch] = w.v[vdata + k][l];
						} else {
							// D16 store: the data are 16-bit values, two per VGPR (the first in bits [15:0]). Float-type
							// formats (unorm/snorm/float) take them as f16 and widen to f32 (the format encoder then
							// packs/rounds as for a 32-bit store); UINT/SINT formats take 16-bit integers, zero/sign-extended.
							const uint32_t h = (w.v[vdata + k / 2u][l] >> (16u * (k & 1u))) & 0xffffu;
							comp[ch] = view.format.kind == FormatInfo::Sint   ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(h)))
							           : view.format.kind == FormatInfo::Uint ? h
							                                                  : B32(F16ToF32(static_cast<uint16_t>(h)));
						}
						k++;
					}
					if (mip != 0) { Unsupported(inst, "(store to mip>0 not compared)"); continue; }
					view.Store(mip, c[0], ncoord >= 2 ? c[1] : 0, z, comp);
					if (m_opt.watch_image >= 0 && image->index == static_cast<uint32_t>(m_opt.watch_image) && c[0] == m_opt.watch_x &&
					    (ncoord < 2 || c[1] == m_opt.watch_y) && z == m_opt.watch_z) {
						std::printf("WATCH store image %u (%d,%d,%d) by group %u,%u,%u wave %u lane %u pc=0x%x value=%08x %08x %08x %08x\n", image->index, c[0], c[1], z, m_cur_group[0], m_cur_group[1], m_cur_group[2], w.id, l, inst.pc, comp[0], comp[1], comp[2], comp[3]);
					}
				} else {
					float    val[4] {0, 0, 0, 0};
					uint32_t raw[4] {0, 0, 0, 0};
					view.Fetch(mip, c[0], ncoord >= 2 ? c[1] : 0, z, val, raw);
					uint32_t sw[4];
					Swizzle(t, val, raw, integer, sw);
					uint32_t out[4], n = 0;
					for (uint32_t ch = 0; ch < 4; ch++) if ((dmask >> ch) & 1u) out[n++] = sw[ch];
					PutResults(w, vdata, l, out, n, d16, integer, view.format.kind == FormatInfo::Sint);
				}
			}
			return true;
		}
		if (is_sample || is_gather) {
			const bool compare = name == "IMAGE_SAMPLE_C_LZ" || name == "IMAGE_SAMPLE_C_L";
			if (compare) {
				const SSharp s(&w.s[ssamp]);
				const bool   lz = name == "IMAGE_SAMPLE_C_LZ";
				for (uint32_t l = 0; l < w.lanes; l++) {
					if (!w.Active(l)) continue;
					// address: zcompare, coords..., [lod]
					// Fixed-point depth formats clamp the reference to [0, 1] before the comparison.
					float ref = F32(w.v[addr_reg(0)][l]);
					if (view.format.kind == FormatInfo::Unorm) ref = std::clamp(ref, 0.0f, 1.0f);
					float c[4] {0, 0, 0, 0};
					for (uint32_t i = 0; i < ncoord && i < 4; i++) c[i] = addr_f(1, i, l);
					const float lod = lz ? 0.0f : addr_f(1, ncoord, l);
					const bool  unn = unorm || s.force_unnorm;
					float       lv  = std::clamp(lod + s.lod_bias, std::max(s.min_lod, t.min_lod), s.max_lod);
					const bool  lin = ((lv <= 0.0f ? s.mag_filter : s.min_filter) & 1u) != 0;
					const uint32_t mip = t.base_level;
					const float layer  = (dim == 5) ? std::nearbyint(c[2]) : 0.0f;
					auto r = SampleCompareLevel(view, s, mip, c[0], c[1], layer, unn, lin, ref, m_opt.prec8, m_opt.compare_flip);
					if (m_opt.force_compare >= 0.0f) r.v[0] = r.v[1] = r.v[2] = r.v[3] = m_opt.force_compare;
					uint32_t out[4], n = 0;
					for (uint32_t ch = 0; ch < 4; ch++) if ((dmask >> ch) & 1u) out[n++] = B32(r.v[ch]);
					PutResults(w, vdata, l, out, n, d16, false);
				}
				approx_ops["image sample compare (filtering)"]++;
				return true;
			}
			if (name.find("_C") != std::string::npos || name.find("_D") != std::string::npos || name.find("_CD") != std::string::npos ||
			    name.find("_B") != std::string::npos || name.find("_CL") != std::string::npos || name.find("_O") != std::string::npos ||
			    (is_gather && name.find("4H") != std::string::npos)) {
				Unsupported(inst, "(sample variant)");
				return true;
			}
			const SSharp s(&w.s[ssamp]);
			const bool   explicit_lod = name.find("_L") != std::string::npos && name.find("_LZ") == std::string::npos;
			const bool   lz           = name.find("_LZ") != std::string::npos;
			if (!explicit_lod && !lz) { Unsupported(inst, "(implicit lod)"); return true; }
			for (uint32_t l = 0; l < w.lanes; l++) {
				if (!w.Active(l)) continue;
				float c[4] {0, 0, 0, 0};
				for (uint32_t i = 0; i < ncoord && i < 4; i++) c[i] = addr_f(0, i, l);
				const float lod = explicit_lod ? addr_f(0, ncoord, l) : 0.0f;
				const float zc  = (dim == 5 || dim == 4) ? std::nearbyint(c[ncoord - 1]) : c[2];
				if (is_gather) {
					// 4 texels of the bilinear footprint, component = lowest dmask bit; order (i0,j1),(i1,j1),(i1,j0),(i0,j0)
					const uint32_t comp = static_cast<uint32_t>(std::countr_zero(dmask | 0x10u));
					const uint32_t mip  = t.base_level;
					const int W = static_cast<int>(view.MipW(mip)), H = static_cast<int>(view.MipH(mip));
					const float x = (unorm || s.force_unnorm ? c[0] : c[0] * static_cast<float>(W)) - 0.5f;
					const float y = (unorm || s.force_unnorm ? c[1] : c[1] * static_cast<float>(H)) - 0.5f;
					const int   x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
					const int   px[4] {x0, x0 + 1, x0 + 1, x0}, py[4] {y0 + 1, y0 + 1, y0, y0};
					uint32_t    out[4] {0, 0, 0, 0};
					for (int k = 0; k < 4; k++) {
						const int ix = Address(px[k], W, s.clamp[0]), iy = Address(py[k], H, s.clamp[1]);
						float val[4] {0, 0, 0, 0};
						uint32_t raw[4] {0, 0, 0, 0};
						if (ix >= 0 && iy >= 0) view.Fetch(mip, ix, iy, (dim == 5) ? static_cast<int>(zc) : 0, val, raw);
						uint32_t sw[4];
						Swizzle(t, val, raw, integer, sw);
						out[k] = sw[comp & 3u];
					}
					PutResults(w, vdata, l, out, 4, d16, integer, view.format.kind == FormatInfo::Sint);
					continue;
				}
				auto r = Sample(view, t, s, c[0], ncoord >= 2 ? c[1] : 0.0f, zc, three_d, unorm, lod, m_opt.prec8);
				// A texel of a level the capture does not hold (mips > 0 of 3D/layered images) reads as 0: the reference is
				// not valid for that sample.
				if (!r.ok) Unsupported(inst, " (sampled mip/texel not captured)");
				if (tracing && static_cast<int>(l) == trace_lane) {
					std::printf("    SAMPLE pc=%04x img%u %s %ux%ux%u coord=(%.6g,%.6g,%.6g) lod=%g | S# clamp=%u/%u/%u mag=%u min=%u z=%u mip=%u mode=%u unnorm=%d | -> %.6g %.6g %.6g %.6g\n",
					            inst.pc, image->index, image->fmt.c_str(), image->w, image->h, image->d, c[0], c[1], zc, lod, s.clamp[0],
					            s.clamp[1], s.clamp[2], s.mag_filter, s.min_filter, s.z_filter, s.mip_filter, s.filter_mode,
					            (unorm || s.force_unnorm) ? 1 : 0, r.v[0], r.v[1], r.v[2], r.v[3]);
				}
				uint32_t raw[4] {B32(r.v[0]), B32(r.v[1]), B32(r.v[2]), B32(r.v[3])};
				uint32_t sw[4];
				Swizzle(t, r.v, raw, integer, sw);
				uint32_t out[4], n = 0;
				for (uint32_t ch = 0; ch < 4; ch++) if ((dmask >> ch) & 1u) out[n++] = sw[ch];
				PutResults(w, vdata, l, out, n, d16, integer, view.format.kind == FormatInfo::Sint);
			}
			approx_ops["image sample (filtering)"]++;
			return true;
		}
		Unsupported(inst);
		return true;
	}

	// ---------------------------------------------------------------- one instruction
	void TraceLine(const Wave& w, const Inst& inst) {
		if (!tracing) return;
		std::printf("  pc=%04x %-24s exec=%016llx scc=%d vcc=%016llx", inst.pc, kOpNames[static_cast<int>(inst.op)],
		            static_cast<unsigned long long>(w.exec), w.scc ? 1 : 0, static_cast<unsigned long long>(w.vcc));
		if (inst.Has("VDST") && inst.enc != Enc::SMEM) {
			const uint32_t d = inst.F("VDST");
			if (trace_lane >= 0 && d < 256) std::printf(" v%u[%d]=%08x", d, trace_lane, w.v[d][static_cast<uint32_t>(trace_lane)]);
		}
		if (inst.Has("SDST")) std::printf(" s%u=%08x", inst.F("SDST"), ReadS(w, inst.F("SDST"), inst));
		if (inst.enc == Enc::SMEM) std::printf(" s%u=%08x", inst.F("SDATA"), ReadS(w, inst.F("SDATA"), inst));
		std::printf("\n");
	}

	void ZeroLoad(Wave& w, const Inst& inst) {
		const std::string name = kOpNames[static_cast<int>(inst.op)];
		const bool by_name = std::find(m_opt.zero_ops.begin(), m_opt.zero_ops.end(), name) != m_opt.zero_ops.end();
		const bool by_pc   = std::find(m_opt.zero_pcs.begin(), m_opt.zero_pcs.end(), inst.pc) != m_opt.zero_pcs.end();
		if (!by_name && !by_pc) return;
		uint32_t reg = 0, count = 0;
		const auto ends = [&](const char* suffix) {
			const std::string t = suffix;
			return name.size() >= t.size() && name.compare(name.size() - t.size(), t.size(), t) == 0;
		};
		if (inst.enc == Enc::MUBUF || inst.enc == Enc::FLAT) {
			if (name.find("LOAD") == std::string::npos) return;
			reg   = inst.F(inst.enc == Enc::MUBUF ? "VDATA" : "VDST");
			count = ends("D16_XYZW") || ends("D16_XYZ") ? 2u : ends("XYZW") || ends("DWORDX4") ? 4u
			      : ends("XYZ") || ends("DWORDX3") ? 3u : (ends("XY") && name.find("D16") == std::string::npos) || ends("DWORDX2") ? 2u : 1u;
		} else if (inst.enc == Enc::MIMG) {
			if (name.find("STORE") != std::string::npos) return;
			reg   = inst.F("VDATA");
			count = static_cast<uint32_t>(std::popcount(inst.F("DMASK") & 0xfu));
		} else if (inst.enc == Enc::DS) {
			if (name.find("READ") == std::string::npos) return;
			reg   = inst.F("VDST");
			count = ends("B64") || name.find("READ2") != std::string::npos ? 2u : ends("B96") ? 3u : ends("B128") ? 4u : 1u;
		} else {
			return;
		}
		for (uint32_t i = 0; i < count && reg + i < w.v.size(); i++)
			for (uint32_t l = 0; l < w.lanes; l++)
				if ((w.exec >> l) & 1u) w.v[reg + i][l] = 0;
	}

	bool Step(Wave& w, Group& g) {
		const Inst* ip = Fetch(w.pc);
		if (ip == nullptr) {
			w.done = true;
			return false;
		}
		const Inst& inst = *ip;
		w.steps++;
		total_steps++;
		if (m_opt.pc_profile) {
			auto& h = pc_hits[inst.pc];
			h[0]++;
			h[1] += static_cast<uint64_t>(std::popcount(w.exec & w.Mask()));
		}
		if (w.steps > m_opt.max_steps) {
			errors.push_back("step limit reached (infinite loop?) at pc 0x" + Hex(inst.pc));
			return false;
		}
		uint32_t next = w.pc + inst.words;
		bool     ok   = true;
		const bool probing = !m_opt.probes.empty();
		if (probing && inst.pc == 0u) {
			for (uint32_t i = 0; i < 4u; i++)
				for (uint32_t l = 0; l < w.lanes; l++)
					if ((w.exec >> l) & 1u) {
						w.v[252u + i][l] = 0xc2c80000u;
						if (m_opt.probe_nth != 0u) w.v[248u + i][l] = 0u;
					}
		}
		if (probing && inst.enc == Enc::MIMG && inst.op == Op::IMAGE_STORE) {
			Inst patched = inst;
			patched.f["VDATA"] = 252u;
			ok = ExecMimg(w, patched);
			TraceLine(w, inst);
			w.pc = next;
			return ok;
		}
		// Shaders without image stores: dword buffer stores write the probe registers too (same as the emulator).
		if (probing && inst.enc == Enc::MUBUF &&
		    (inst.op == Op::BUFFER_STORE_DWORD || inst.op == Op::BUFFER_STORE_DWORDX2 ||
		     inst.op == Op::BUFFER_STORE_DWORDX3 || inst.op == Op::BUFFER_STORE_DWORDX4)) {
			Inst patched = inst;
			patched.f["VDATA"] = 252u;
			ok = ExecMubuf(w, patched);
			TraceLine(w, inst);
			w.pc = next;
			return ok;
		}
		switch (inst.enc) {
			case Enc::SOPP: {
				const int32_t simm = static_cast<int16_t>(inst.F("SIMM16"));
				const uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(w.pc + 1u) + simm);
				switch (inst.op) {
					case Op::S_ENDPGM: case Op::S_ENDPGM_SAVED: case Op::S_ENDPGM_ORDERED_PS_DONE: w.done = true; TraceLine(w, inst); return true;
					case Op::S_BRANCH: next = target; break;
					case Op::S_CBRANCH_SCC0: if (!w.scc) next = target; break;
					case Op::S_CBRANCH_SCC1: if (w.scc) next = target; break;
					case Op::S_CBRANCH_VCCZ: if ((w.vcc & w.Mask()) == 0) next = target; break;
					case Op::S_CBRANCH_VCCNZ: if ((w.vcc & w.Mask()) != 0) next = target; break;
					case Op::S_CBRANCH_EXECZ: if (w.exec == 0) next = target; break;
					case Op::S_CBRANCH_EXECNZ: if (w.exec != 0) next = target; break;
					case Op::S_BARRIER: w.barrier = true; break;
					case Op::S_NOP: case Op::S_WAITCNT: case Op::S_INST_PREFETCH: case Op::S_CLAUSE: case Op::S_SETPRIO:
					case Op::S_SLEEP: case Op::S_WAITCNT_DEPCTR: case Op::S_TTRACEDATA: case Op::S_TTRACEDATA_IMM:
					case Op::S_ICACHE_INV: case Op::S_DENORM_MODE: case Op::S_ROUND_MODE: case Op::S_CODE_END:
					case Op::S_WAKEUP: case Op::S_SETKILL: case Op::S_INCPERFLEVEL: case Op::S_DECPERFLEVEL: case Op::S_WAIT_IDLE:
						if (inst.op == Op::S_DENORM_MODE || inst.op == Op::S_ROUND_MODE) Unsupported(inst, "(mode change ignored)");
						break;
					case Op::S_SENDMSG: case Op::S_SENDMSGHALT: break;
					default: Unsupported(inst); break;
				}
				break;
			}
			case Enc::SOP1:
			case Enc::SOP2:
			case Enc::SOPC:
			case Enc::SOPK: ok = ExecSalu(w, inst); break;
			case Enc::SMEM: ok = ExecSmem(w, inst); break;
			case Enc::VOP1:
			case Enc::VOP2:
			case Enc::VOPC:
			case Enc::VOP3:
			case Enc::VOP3P:
				if (inst.op == Op::V_NOP || inst.op == Op::V_PIPEFLUSH || inst.op == Op::V_CLREXCP) break;
				ok = ExecValu(w, inst);
				break;
			case Enc::MUBUF: ok = ExecMubuf(w, inst); break;
			case Enc::DS: ok = ExecDs(w, g, inst); break;
			case Enc::MIMG: ok = ExecMimg(w, inst); break;
			case Enc::FLAT: ok = ExecGlobal(w, inst); break;
			default: Unsupported(inst, "(encoding)"); break;
		}
		if (!m_opt.zero_ops.empty() || !m_opt.zero_pcs.empty()) ZeroLoad(w, inst);
		if (probing) {
			for (size_t index = 0; index < m_opt.probes.size(); index++) {
				const auto& point = m_opt.probes[index];
				if (inst.pc != point.pc) continue;
				// --probe-nth: the first point with this pc owns the counter (v248 + its index), bumped once per execution.
				size_t owner = index;
				for (size_t k = 0; k < index; k++)
					if (m_opt.probes[k].pc == point.pc) { owner = k; break; }
				const uint32_t counter = 248u + static_cast<uint32_t>(std::min<size_t>(owner, 3u));
				if (m_opt.probe_nth != 0u && owner == index)
					for (uint32_t l = 0; l < w.lanes; l++)
						if ((w.exec >> l) & 1u) w.v[counter][l]++;
				for (uint32_t i = 0; i < point.count; i++)
					for (uint32_t l = 0; l < w.lanes; l++)
						if (((w.exec >> l) & 1u) && (m_opt.probe_nth == 0u || w.v[counter][l] == m_opt.probe_nth))
						{
							const uint32_t value = point.scalar ? ReadS(w, point.reg + i, inst) : w.v[point.reg + i][l];
							w.v[252u + point.slot + i][l] = point.to_float ? std::bit_cast<uint32_t>(static_cast<float>(value)) : value;
						}
			}
		}
		TraceLine(w, inst);
		// S_SETPC/S_SWAPPC changed pc themselves
		if (inst.op != Op::S_SETPC_B64 && inst.op != Op::S_SWAPPC_B64) w.pc = next;
		return ok;
	}
};

int RunCapture(int argc, char** argv);

} // namespace refinterp
