// Floating-point helpers of the reference interpreter: exact f16 conversions (RNE and RTZ), denormal
// flushing and AMD min/max/clamp rules.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace refinterp {

inline float F32(uint32_t bits) { return std::bit_cast<float>(bits); }
inline uint32_t B32(float value) { return std::bit_cast<uint32_t>(value); }
inline double F64(uint64_t bits) { return std::bit_cast<double>(bits); }
inline uint64_t B64(double value) { return std::bit_cast<uint64_t>(value); }

// f16 -> f32 (exact, denormals kept).
inline float F16ToF32(uint16_t h) {
	const uint32_t sign = (h >> 15u) & 1u;
	const uint32_t exp  = (h >> 10u) & 0x1fu;
	const uint32_t man  = h & 0x3ffu;
	float          value;
	if (exp == 0) {
		value = std::ldexp(static_cast<float>(man), -24);
	} else if (exp == 31) {
		value = man == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
		if (man != 0) {
			// keep the payload's quiet bit like the hardware does (qNaN out)
			return F32((sign << 31u) | 0x7fc00000u | (man << 13u));
		}
	} else {
		value = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
	}
	return sign != 0 ? -value : value;
}

// double -> f16 with round-to-nearest-even (rtz=false) or round-toward-zero (rtz=true). Using the double
// keeps a single rounding for f16 FMA results computed exactly in double.
// REFINTERP_F16_RTZ=1 (what-if): every f32 -> f16 conversion truncates (models a host GPU whose f16 conversion
// rounds toward zero) instead of rounding to nearest even.
inline bool ForceF16Rtz() {
	static const bool force = [] {
		const char* value = std::getenv("REFINTERP_F16_RTZ");
		return value != nullptr && value[0] == '1';
	}();
	return force;
}
inline uint16_t F64ToF16(double value, bool rtz = false) {
	rtz = rtz || ForceF16Rtz();
	const uint16_t sign = std::signbit(value) ? 0x8000u : 0u;
	if (std::isnan(value)) return sign | 0x7e00u;
	double a = std::fabs(value);
	if (std::isinf(a)) return sign | 0x7c00u;
	if (a == 0.0) return sign;
	int        e   = 0;
	const double m = std::frexp(a, &e); // a = m * 2^e, m in [0.5,1)
	int        exp = e - 1;             // a = (2m) * 2^exp, 2m in [1,2)
	if (exp < -14) {
		// denormal: value / 2^-24
		const double scaled = a * 16777216.0; // a / 2^-24
		double       q      = rtz ? std::floor(scaled) : std::nearbyint(scaled);
		if (!rtz) {
			// nearbyint uses the current rounding mode (round-to-nearest-even by default)
		}
		if (q >= 1024.0) return sign | 0x0400u; // rounds up to the smallest normal
		return sign | static_cast<uint16_t>(q);
	}
	const double frac = (2.0 * m - 1.0) * 1024.0; // 10-bit mantissa with fraction
	double       q    = rtz ? std::floor(frac) : std::nearbyint(frac);
	if (q >= 1024.0) {
		q = 0;
		exp++;
	}
	if (exp > 15) {
		return rtz ? static_cast<uint16_t>(sign | 0x7bffu) : static_cast<uint16_t>(sign | 0x7c00u);
	}
	return sign | static_cast<uint16_t>(((exp + 15) << 10) | static_cast<uint32_t>(q));
}

inline uint16_t F32ToF16(float value, bool rtz = false) { return F64ToF16(static_cast<double>(value), rtz); }

inline bool IsDenormF32(uint32_t bits) { return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0; }

inline float FlushF32(float value) {
	const uint32_t bits = B32(value);
	return IsDenormF32(bits) ? F32(bits & 0x80000000u) : value;
}

inline bool IsSnanF32(uint32_t bits) { return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu) != 0 && (bits & 0x400000u) == 0; }

// V_MIN/V_MAX: a NaN operand yields the other operand (IEEE minNum/maxNum); both NaN -> qNaN.
inline float MinF(float a, float b) {
	if (std::isnan(a)) return std::isnan(b) ? std::numeric_limits<float>::quiet_NaN() : b;
	if (std::isnan(b)) return a;
	if (a == 0.0f && b == 0.0f) return std::signbit(a) ? a : b;
	return a < b ? a : b;
}
inline float MaxF(float a, float b) {
	if (std::isnan(a)) return std::isnan(b) ? std::numeric_limits<float>::quiet_NaN() : b;
	if (std::isnan(b)) return a;
	if (a == 0.0f && b == 0.0f) return std::signbit(a) ? b : a;
	return a > b ? a : b;
}
inline double MinD(double a, double b) {
	if (std::isnan(a)) return std::isnan(b) ? std::numeric_limits<double>::quiet_NaN() : b;
	if (std::isnan(b)) return a;
	return a < b ? a : b;
}
inline double MaxD(double a, double b) {
	if (std::isnan(a)) return std::isnan(b) ? std::numeric_limits<double>::quiet_NaN() : b;
	if (std::isnan(b)) return a;
	return a > b ? a : b;
}
// V_MED3_F32: median; with a NaN input it returns min/max of the others as the ISA defines
// (MED3(a,b,c) = max(min(a,b), min(max(a,b), c))).
inline float Med3F(float a, float b, float c) { return MaxF(MinF(a, b), MinF(MaxF(a, b), c)); }

// Output clamp (VOP3 CLAMP bit) to [0, 1]; NaN -> 0 when DX10_CLAMP is set.
inline float ClampF(float value, bool dx10_clamp) {
	if (std::isnan(value)) return dx10_clamp ? 0.0f : value;
	return value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;
}

inline float RndneF(float value) { return std::nearbyint(value); }

inline float FractF(float value) {
	if (std::isinf(value)) return std::numeric_limits<float>::quiet_NaN();
	if (std::isnan(value)) return value;
	const float r = value - std::floor(value);
	return r >= 1.0f ? F32(0x3f7fffffu) : r;
}

inline uint32_t CvtU32F32(float value) {
	if (std::isnan(value)) return 0;
	if (value <= 0.0f) return 0;
	if (value >= 4294967295.0f) return 0xffffffffu;
	return static_cast<uint32_t>(value);
}
inline int32_t CvtI32F32(float value) {
	if (std::isnan(value)) return 0;
	if (value <= -2147483648.0f) return std::numeric_limits<int32_t>::min();
	if (value >= 2147483647.0f) return std::numeric_limits<int32_t>::max();
	return static_cast<int32_t>(value);
}

inline uint32_t Ulps(float a, float b) {
	if (std::isnan(a) || std::isnan(b)) return (std::isnan(a) && std::isnan(b)) ? 0u : 0xffffffffu;
	auto key = [](float v) {
		const int32_t i = std::bit_cast<int32_t>(v);
		return i < 0 ? static_cast<int64_t>(0x80000000LL - static_cast<int64_t>(i & 0x7fffffff)) : static_cast<int64_t>(i) + 0x80000000LL;
	};
	const int64_t d = key(a) - key(b);
	return static_cast<uint32_t>(d < 0 ? -d : d);
}

} // namespace refinterp
