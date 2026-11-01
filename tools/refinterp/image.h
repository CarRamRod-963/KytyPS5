// Image resources of the reference interpreter: T#/S# decoding (GFX10 layout), texel decode/encode of
// the dumped Vulkan formats, and sampling (point/bilinear/trilinear, clamp modes, min/max reduction).
#pragma once

#include "capture.h"
#include "fp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace refinterp {

struct TSharp {
	uint64_t base      = 0;
	uint32_t width     = 1, height = 1, depth = 1;
	uint32_t format    = 0;
	uint32_t dst_sel[4] {4, 5, 6, 7};
	uint32_t base_level = 0, last_level = 0;
	uint32_t type       = 9;
	float    min_lod    = 0.0f;

	explicit TSharp(const uint32_t* w) {
		base       = ((static_cast<uint64_t>(w[1] & 0xffu) << 32u) | w[0]) << 8u;
		min_lod    = static_cast<float>((w[1] >> 8u) & 0xfffu) / 256.0f;
		format     = (w[1] >> 20u) & 0x1ffu;
		width      = ((((w[2] & 0xfffu) << 2u) | (w[1] >> 30u)) + 1u);
		height     = ((w[2] >> 14u) & 0x3fffu) + 1u;
		dst_sel[0] = w[3] & 7u;
		dst_sel[1] = (w[3] >> 3u) & 7u;
		dst_sel[2] = (w[3] >> 6u) & 7u;
		dst_sel[3] = (w[3] >> 9u) & 7u;
		base_level = (w[3] >> 12u) & 0xfu;
		last_level = (w[3] >> 16u) & 0xfu;
		type       = (w[3] >> 28u) & 0xfu;
		depth      = (w[4] & 0x1fffu) + 1u;
	}
};

struct SSharp {
	uint32_t clamp[3] {0, 0, 0};
	bool     force_unnorm = false;
	uint32_t filter_mode  = 0; // 0 blend, 1 min, 2 max
	float    min_lod = 0.0f, max_lod = 15.0f, lod_bias = 0.0f;
	uint32_t mag_filter = 0, min_filter = 0, z_filter = 0, mip_filter = 0;
	uint32_t border_type = 0;
	uint32_t compare_func = 0; // DEPTH_COMPARE_FUNC: 0 never 1 less 2 equal 3 lequal 4 greater 5 notequal 6 gequal 7 always

	explicit SSharp(const uint32_t* w) {
		clamp[0]     = w[0] & 7u;
		clamp[1]     = (w[0] >> 3u) & 7u;
		clamp[2]     = (w[0] >> 6u) & 7u;
		compare_func = (w[0] >> 12u) & 7u;
		force_unnorm = ((w[0] >> 15u) & 1u) != 0;
		filter_mode  = (w[0] >> 29u) & 3u;
		min_lod      = static_cast<float>(w[1] & 0xfffu) / 256.0f;
		max_lod      = static_cast<float>((w[1] >> 12u) & 0xfffu) / 256.0f;
		int32_t bias = static_cast<int32_t>(w[2] & 0x3fffu);
		if (bias & 0x2000) bias -= 0x4000;
		lod_bias    = static_cast<float>(bias) / 256.0f;
		mag_filter  = (w[2] >> 20u) & 3u;
		min_filter  = (w[2] >> 22u) & 3u;
		z_filter    = (w[2] >> 24u) & 3u;
		mip_filter  = (w[2] >> 26u) & 3u;
		border_type = (w[3] >> 30u) & 3u;
	}
};

struct FormatInfo {
	uint32_t bytes = 0;
	uint32_t comps = 0;
	enum Kind { Unorm, Snorm, Float, Uint, Sint, Special } kind = Unorm;
	uint32_t bits = 0; // per component for plain formats
	bool     ok   = false;
	bool     srgb = false; // 8-bit unorm colour channels (not alpha) stored with the sRGB curve
};

inline float SrgbToLinear(float x) {
	return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}
inline float LinearToSrgb(float x) {
	return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

inline FormatInfo GetFormat(const std::string& fmt) {
	FormatInfo f;
	auto set = [&](uint32_t bytes, uint32_t comps, FormatInfo::Kind kind, uint32_t bits) {
		f.bytes = bytes;
		f.comps = comps;
		f.kind  = kind;
		f.bits  = bits;
		f.ok    = true;
	};
	if (fmt == "R8Unorm") set(1, 1, FormatInfo::Unorm, 8);
	else if (fmt == "R8G8Unorm") set(2, 2, FormatInfo::Unorm, 8);
	else if (fmt == "R8G8B8A8Unorm") set(4, 4, FormatInfo::Unorm, 8);
	else if (fmt == "R8G8B8A8Srgb") {
		set(4, 4, FormatInfo::Unorm, 8);
		f.srgb = true;
	}
	else if (fmt == "R8Uint") set(1, 1, FormatInfo::Uint, 8);
	else if (fmt == "R8G8Uint") set(2, 2, FormatInfo::Uint, 8);
	else if (fmt == "R8G8B8A8Uint") set(4, 4, FormatInfo::Uint, 8);
	else if (fmt == "R8Snorm") set(1, 1, FormatInfo::Snorm, 8);
	else if (fmt == "R8G8Snorm") set(2, 2, FormatInfo::Snorm, 8);
	else if (fmt == "R8G8B8A8Snorm") set(4, 4, FormatInfo::Snorm, 8);
	else if (fmt == "R16Unorm") set(2, 1, FormatInfo::Unorm, 16);
	else if (fmt == "R16G16Unorm") set(4, 2, FormatInfo::Unorm, 16);
	else if (fmt == "R16G16B16A16Unorm") set(8, 4, FormatInfo::Unorm, 16);
	else if (fmt == "R16Snorm") set(2, 1, FormatInfo::Snorm, 16);
	else if (fmt == "R16G16Snorm") set(4, 2, FormatInfo::Snorm, 16);
	else if (fmt == "R16G16B16A16Snorm") set(8, 4, FormatInfo::Snorm, 16);
	else if (fmt == "R16Sfloat") set(2, 1, FormatInfo::Float, 16);
	else if (fmt == "R16G16Sfloat") set(4, 2, FormatInfo::Float, 16);
	else if (fmt == "R16G16B16A16Sfloat") set(8, 4, FormatInfo::Float, 16);
	else if (fmt == "R16Uint") set(2, 1, FormatInfo::Uint, 16);
	else if (fmt == "R16G16Uint") set(4, 2, FormatInfo::Uint, 16);
	else if (fmt == "R16G16B16A16Uint") set(8, 4, FormatInfo::Uint, 16);
	// Depth aspect of a combined depth/stencil image: DebugQueueFullDump copies the depth aspect only, one 4-byte
	// float per texel (the stencil aspect is a separate file, see CapImage::stencil and MakeView).
	else if (fmt == "R32Sfloat" || fmt == "D32Sfloat" || fmt == "D32SfloatS8Uint") set(4, 1, FormatInfo::Float, 32);
	else if (fmt == "R32G32Sfloat") set(8, 2, FormatInfo::Float, 32);
	else if (fmt == "R32G32B32A32Sfloat") set(16, 4, FormatInfo::Float, 32);
	else if (fmt == "R32Uint") set(4, 1, FormatInfo::Uint, 32);
	else if (fmt == "R32G32Uint") set(8, 2, FormatInfo::Uint, 32);
	else if (fmt == "R32G32B32A32Uint") set(16, 4, FormatInfo::Uint, 32);
	else if (fmt == "R32Sint") set(4, 1, FormatInfo::Sint, 32);
	else if (fmt == "B10G11R11UfloatPack32" || fmt == "A2B10G10R10UnormPack32" || fmt == "A2B10G10R10UintPack32" ||
	         fmt == "E5B9G9R9UfloatPack32") {
		set(4, fmt == "B10G11R11UfloatPack32" || fmt == "E5B9G9R9UfloatPack32" ? 3 : 4, FormatInfo::Special, 0);
	} else if (fmt == "D16Unorm" || fmt == "D16UnormS8Uint") set(2, 1, FormatInfo::Unorm, 16);
	return f;
}

inline float UFloatToF32(uint32_t v, uint32_t mbits) {
	const uint32_t e = (v >> mbits) & 0x1fu;
	const uint32_t m = v & ((1u << mbits) - 1u);
	if (e == 0) return std::ldexp(static_cast<float>(m), -14 - static_cast<int>(mbits));
	if (e == 31) return m == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
	return std::ldexp(1.0f + static_cast<float>(m) / static_cast<float>(1u << mbits), static_cast<int>(e) - 15);
}
inline uint32_t F32ToUFloat(float value, uint32_t mbits) {
	if (std::isnan(value)) return (0x1fu << mbits) | 1u;
	if (value <= 0.0f) return 0;
	// via f16 (same exponent range); drop mantissa bits with round-to-nearest-even
	const uint16_t h   = F32ToF16(value);
	const uint32_t e   = (h >> 10u) & 0x1fu;
	uint32_t       m   = h & 0x3ffu;
	const uint32_t cut = 10u - mbits;
	uint32_t       r   = (e << mbits) | (m >> cut);
	const uint32_t rem = m & ((1u << cut) - 1u);
	const uint32_t half = 1u << (cut - 1u);
	if (rem > half || (rem == half && (r & 1u))) r++;
	return std::min(r, (0x1fu << mbits));
}

// Raw texel -> 4 floats (components missing in the format read as 0, alpha as 1) and raw integers.
inline void DecodeTexel(const std::string& fmt, const FormatInfo& f, const uint8_t* p, float out[4], uint32_t raw[4]) {
	for (int i = 0; i < 4; i++) {
		out[i] = i == 3 ? 1.0f : 0.0f;
		raw[i] = i == 3 ? 1u : 0u;
	}
	if (f.kind == FormatInfo::Special) {
		uint32_t v = 0;
		std::memcpy(&v, p, 4);
		if (fmt == "B10G11R11UfloatPack32") {
			out[0] = UFloatToF32(v & 0x7ffu, 6);
			out[1] = UFloatToF32((v >> 11u) & 0x7ffu, 6);
			out[2] = UFloatToF32((v >> 22u) & 0x3ffu, 5);
		} else if (fmt == "A2B10G10R10UnormPack32") {
			out[0] = static_cast<float>(v & 0x3ffu) / 1023.0f;
			out[1] = static_cast<float>((v >> 10u) & 0x3ffu) / 1023.0f;
			out[2] = static_cast<float>((v >> 20u) & 0x3ffu) / 1023.0f;
			out[3] = static_cast<float>(v >> 30u) / 3.0f;
		}
		for (int i = 0; i < 4; i++) raw[i] = B32(out[i]);
		return;
	}
	for (uint32_t c = 0; c < f.comps; c++) {
		uint32_t v = 0;
		std::memcpy(&v, p + c * (f.bits / 8u), f.bits / 8u);
		raw[c] = v;
		switch (f.kind) {
			case FormatInfo::Unorm:
				out[c] = static_cast<float>(v) / static_cast<float>((1ull << f.bits) - 1u);
				if (f.srgb && c < 3) out[c] = SrgbToLinear(out[c]);
				break;
			case FormatInfo::Snorm: {
				int32_t s = static_cast<int32_t>(v << (32u - f.bits)) >> (32u - f.bits);
				out[c]    = std::max(-1.0f, static_cast<float>(s) / static_cast<float>((1u << (f.bits - 1u)) - 1u));
				break;
			}
			case FormatInfo::Float: out[c] = f.bits == 16 ? F16ToF32(static_cast<uint16_t>(v)) : F32(v); break;
			case FormatInfo::Uint: out[c] = F32(v); break;
			case FormatInfo::Sint: out[c] = F32(v); break;
			default: break;
		}
	}
}

// 4 component values (float bits or integers) -> raw texel.
inline void EncodeTexel(const std::string& fmt, const FormatInfo& f, const uint32_t in[4], uint8_t* p) {
	if (f.kind == FormatInfo::Special) {
		uint32_t v = 0;
		if (fmt == "B10G11R11UfloatPack32") {
			v = F32ToUFloat(F32(in[0]), 6) | (F32ToUFloat(F32(in[1]), 6) << 11u) | (F32ToUFloat(F32(in[2]), 5) << 22u);
		} else if (fmt == "A2B10G10R10UnormPack32") {
			auto q = [](float x, float m) { return static_cast<uint32_t>(std::nearbyint(std::clamp(std::isnan(x) ? 0.0f : x, 0.0f, 1.0f) * m)); };
			v = q(F32(in[0]), 1023) | (q(F32(in[1]), 1023) << 10u) | (q(F32(in[2]), 1023) << 20u) | (q(F32(in[3]), 3) << 30u);
		}
		std::memcpy(p, &v, 4);
		return;
	}
	for (uint32_t c = 0; c < f.comps; c++) {
		uint32_t v = 0;
		float    x = F32(in[c]);
		if (f.srgb && c < 3 && !std::isnan(x)) x = LinearToSrgb(std::clamp(x, 0.0f, 1.0f));
		switch (f.kind) {
			case FormatInfo::Unorm: {
				const double m = static_cast<double>((1ull << f.bits) - 1u);
				v = static_cast<uint32_t>(std::nearbyint(std::clamp(std::isnan(x) ? 0.0 : static_cast<double>(x), 0.0, 1.0) * m));
				break;
			}
			case FormatInfo::Snorm: {
				const double m = static_cast<double>((1u << (f.bits - 1u)) - 1u);
				const auto   s = static_cast<int32_t>(std::nearbyint(std::clamp(std::isnan(x) ? 0.0 : static_cast<double>(x), -1.0, 1.0) * m));
				v = static_cast<uint32_t>(s) & static_cast<uint32_t>((1ull << f.bits) - 1u);
				break;
			}
			case FormatInfo::Float: v = f.bits == 16 ? F32ToF16(x) : in[c]; break;
			case FormatInfo::Uint:
			case FormatInfo::Sint: v = f.bits == 32 ? in[c] : in[c] & static_cast<uint32_t>((1ull << f.bits) - 1u); break;
			default: break;
		}
		std::memcpy(p + c * (f.bits / 8u), &v, f.bits / 8u);
	}
}

struct ImageView {
	CapImage*  image = nullptr;
	FormatInfo format;
	// Stencil aspect of a combined depth/stencil image: texels come from image->stencil (1 byte each, mip 0 only),
	// `format` is then the 8-bit format the guest T# names.
	bool stencil = false;

	[[nodiscard]] uint32_t MipW(uint32_t mip) const { return std::max(1u, image->w >> mip); }
	[[nodiscard]] uint32_t MipH(uint32_t mip) const { return std::max(1u, image->h >> mip); }
	[[nodiscard]] uint32_t MipD(uint32_t mip) const { return std::max(1u, image->d >> mip); }

	// Texel of a mip (layer folded into z for arrays); false when outside or the mip was not captured.
	bool Fetch(uint32_t mip, int x, int y, int z, float out[4], uint32_t raw[4]) const {
		if (stencil ? (mip != 0 || image->stencil.empty()) : (mip >= image->mips.size() || image->mips[mip].empty())) return false;
		const auto w = MipW(mip), h = MipH(mip);
		const auto d = std::max(MipD(mip), image->layers);
		if (x < 0 || y < 0 || z < 0 || static_cast<uint32_t>(x) >= w || static_cast<uint32_t>(y) >= h || static_cast<uint32_t>(z) >= d) return false;
		const uint64_t offset = ((static_cast<uint64_t>(z) * h + static_cast<uint64_t>(y)) * w + static_cast<uint64_t>(x)) * format.bytes;
		const auto&    data   = stencil ? image->stencil : (mip == 0) ? image->reference : image->mips[mip];
		if (offset + format.bytes > data.size()) return false;
		DecodeTexel(image->fmt, format, data.data() + offset, out, raw);
		return true;
	}
	bool Store(uint32_t mip, int x, int y, int z, const uint32_t in[4]) const {
		if (mip != 0 || stencil) return false;
		const auto w = MipW(0), h = MipH(0);
		const auto d = std::max(MipD(0), image->layers);
		if (x < 0 || y < 0 || z < 0 || static_cast<uint32_t>(x) >= w || static_cast<uint32_t>(y) >= h || static_cast<uint32_t>(z) >= d) return false;
		const uint64_t texel  = (static_cast<uint64_t>(z) * h + static_cast<uint64_t>(y)) * w + static_cast<uint64_t>(x);
		const uint64_t offset = texel * format.bytes;
		if (offset + format.bytes > image->reference.size()) return false;
		EncodeTexel(image->fmt, format, in, image->reference.data() + offset);
		image->ref_written[texel] = 1;
		return true;
	}
};

// The view a guest T# has of a captured image. A combined depth/stencil image (fmt ...S8Uint) holds two aspects
// and the guest picks one with the T# format: an 8-bit format (GFX10 IMG_FORMAT 1 = 8_UNORM, 5 = 8_UINT,
// 6 = 8_SINT) reads the stencil aspect (value 0..255, R8Uint), any wider format reads the depth aspect (32_FLOAT
// for D32, 16_UNORM for D16), which is what the emulator's dump of the depth aspect holds.
inline ImageView MakeView(CapImage* image, const TSharp& t) {
	ImageView view {image, GetFormat(image->fmt)};
	if (image->fmt.find("S8Uint") != std::string::npos && (t.format == 1 || t.format == 5 || t.format == 6)) {
		view.stencil = true;
		view.format  = GetFormat(t.format == 1 ? "R8Unorm" : "R8Uint");
	}
	// The T# format decides the decode, not the host image the emulator happened to create: 8_SRGB (128), 8_8_SRGB
	// (129) and 8_8_8_8_SRGB (130) colour channels are converted from sRGB on every read (and to sRGB on store).
	if ((t.format == 128u || t.format == 129u || t.format == 130u) && view.format.kind == FormatInfo::Unorm &&
	    view.format.bits == 8u && !view.stencil) {
		view.format.srgb = true;
	}
	return view;
}

// Texel index for a clamp mode (CLAMP_X/Y/Z of the S#): 0 wrap, 1 mirror, 2 clamp to last texel,
// 3 mirror once to last texel, 6 border; -1 = border texel.
inline int Address(int i, int size, uint32_t mode) {
	switch (mode) {
		case 0: return ((i % size) + size) % size;
		case 1: {
			const int period = 2 * size;
			int       m      = ((i % period) + period) % period;
			return m < size ? m : period - 1 - m;
		}
		case 2: return std::clamp(i, 0, size - 1);
		case 3: {
			const int m = i < 0 ? -1 - i : i;
			return std::clamp(m, 0, size - 1);
		}
		case 4:
		case 6: return (i < 0 || i >= size) ? -1 : i;
		case 5:
		case 7: {
			const int m = i < 0 ? -1 - i : i;
			return m >= size ? -1 : m;
		}
		default: return std::clamp(i, 0, size - 1);
	}
}

struct SampleResult {
	float v[4] {0, 0, 0, 0};
	bool  ok = true;
};

// One mip level, point or (bi/tri)linear in the dimensions given; reduction per FILTER_MODE.
inline SampleResult SampleLevel(const ImageView& view, const SSharp& s, uint32_t mip, float u, float v, float w, bool three_d,
                                bool unnorm, bool linear_xy, bool linear_z, bool prec8) {
	SampleResult r;
	const int    W = static_cast<int>(view.MipW(mip)), H = static_cast<int>(view.MipH(mip));
	const int    D = three_d ? static_cast<int>(view.MipD(mip)) : 1;
	const float  x = unnorm ? u : u * static_cast<float>(W);
	const float  y = unnorm ? v : v * static_cast<float>(H);
	const float  z = three_d ? (unnorm ? w : w * static_cast<float>(D)) : 0.0f;
	int          x0, y0, z0;
	float        fx = 0, fy = 0, fz = 0;
	auto split = [&](float t, bool lin, int& i0, float& f) {
		if (lin) {
			const float tt = t - 0.5f;
			i0             = static_cast<int>(std::floor(tt));
			f              = tt - std::floor(tt);
			if (prec8) f = std::floor(f * 256.0f) / 256.0f;
		} else {
			i0 = static_cast<int>(std::floor(t));
			f  = 0.0f;
		}
	};
	split(x, linear_xy, x0, fx);
	split(y, linear_xy, y0, fy);
	if (three_d) split(z, linear_z, z0, fz);
	else z0 = static_cast<int>(w); // array slice (already an index)
	float acc[4] {0, 0, 0, 0};
	float mn[4] {INFINITY, INFINITY, INFINITY, INFINITY}, mx[4] {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
	const int nx = linear_xy ? 2 : 1, ny = linear_xy ? 2 : 1, nz = (three_d && linear_z) ? 2 : 1;
	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				const float wgt = (nx == 2 ? (i ? fx : 1 - fx) : 1.0f) * (ny == 2 ? (j ? fy : 1 - fy) : 1.0f) *
				                  (nz == 2 ? (k ? fz : 1 - fz) : 1.0f);
				const int ix = Address(x0 + i, W, s.clamp[0]);
				const int iy = Address(y0 + j, H, s.clamp[1]);
				const int iz = three_d ? Address(z0 + k, D, s.clamp[2]) : z0;
				float     t[4] {0, 0, 0, 0};
				uint32_t  raw[4];
				if (ix >= 0 && iy >= 0 && iz >= 0) {
					if (!view.Fetch(mip, ix, iy, iz, t, raw)) r.ok = false;
				} else if (s.border_type == 1) {
					t[3] = 1.0f;
				} else if (s.border_type == 2) {
					t[0] = t[1] = t[2] = t[3] = 1.0f;
				}
				for (int c = 0; c < 4; c++) {
					acc[c] += wgt * t[c];
					if (wgt > 0.0f || (nx * ny * nz) == 1) {
						mn[c] = std::min(mn[c], t[c]);
						mx[c] = std::max(mx[c], t[c]);
					}
				}
			}
		}
	}
	for (int c = 0; c < 4; c++) {
		r.v[c] = s.filter_mode == 1 ? mn[c] : s.filter_mode == 2 ? mx[c] : acc[c];
		if (std::isinf(r.v[c]) && s.filter_mode != 0) r.v[c] = acc[c];
	}
	return r;
}

inline SampleResult Sample(const ImageView& view, const TSharp& t, const SSharp& s, float u, float v, float w, bool three_d,
                           bool unnorm, float lod, bool prec8) {
	unnorm = unnorm || s.force_unnorm;
	float l = lod + s.lod_bias;
	l       = std::clamp(l, std::max(s.min_lod, t.min_lod), s.max_lod);
	const bool mag      = l <= 0.0f;
	const bool linear   = (mag ? s.mag_filter : s.min_filter) & 1u;
	// Z_FILTER: 0 none (follow the XY filter), 1 point, 2 linear.
	const bool linear_z = s.z_filter == 2 || (s.z_filter == 0 && linear);
	const int  last     = static_cast<int>(std::min(t.last_level, view.image->levels - 1u));
	const int  base     = static_cast<int>(t.base_level);
	l                   = std::max(l, 0.0f);
	if (s.mip_filter == 2 && last > base) {
		const float lf  = std::min(l, static_cast<float>(last - base));
		const int   m0  = static_cast<int>(std::floor(lf));
		const float f   = lf - static_cast<float>(m0);
		const int   ma  = std::clamp(base + m0, base, last);
		const int   mb  = std::clamp(base + m0 + 1, base, last);
		auto        ra  = SampleLevel(view, s, static_cast<uint32_t>(ma), u, v, w, three_d, unnorm, linear, linear_z, prec8);
		if (f == 0.0f || ma == mb) return ra;
		auto rb = SampleLevel(view, s, static_cast<uint32_t>(mb), u, v, w, three_d, unnorm, linear, linear_z, prec8);
		for (int c = 0; c < 4; c++) {
			ra.v[c] = s.filter_mode == 1 ? std::min(ra.v[c], rb.v[c]) : s.filter_mode == 2 ? std::max(ra.v[c], rb.v[c]) : ra.v[c] * (1 - f) + rb.v[c] * f;
		}
		ra.ok = ra.ok && rb.ok;
		return ra;
	}
	int mip = base;
	if (s.mip_filter != 0) mip = std::clamp(base + static_cast<int>(std::floor(l + 0.5f)), base, last);
	return SampleLevel(view, s, static_cast<uint32_t>(mip), u, v, w, three_d, unnorm, linear, linear_z, prec8);
}


// Depth-compare sample (IMAGE_SAMPLE_C*): each footprint texel compares the reference against the
// texel (ref OP texel, the S# DEPTH_COMPARE_FUNC) and the 0/1 results are filtered like colours.
inline SampleResult SampleCompareLevel(const ImageView& view, const SSharp& s, uint32_t mip, float u, float v, float layer,
                                       bool unnorm, bool linear, float ref, bool prec8, bool flip = false) {
	SampleResult r;
	const int    W = static_cast<int>(view.MipW(mip)), H = static_cast<int>(view.MipH(mip));
	const float  x = unnorm ? u : u * static_cast<float>(W);
	const float  y = unnorm ? v : v * static_cast<float>(H);
	int          x0, y0;
	float        fx = 0, fy = 0;
	auto split = [&](float tt, int& i0, float& f) {
		if (linear) {
			const float q = tt - 0.5f;
			i0            = static_cast<int>(std::floor(q));
			f             = q - std::floor(q);
			if (prec8) f = std::floor(f * 256.0f) / 256.0f;
		} else {
			i0 = static_cast<int>(std::floor(tt));
			f  = 0.0f;
		}
	};
	split(x, x0, fx);
	split(y, y0, fy);
	auto pass = [&](float texel) {
		if (flip) std::swap(texel, ref);
		switch (s.compare_func) {
			case 0: return false;
			case 1: return ref < texel;
			case 2: return ref == texel;
			case 3: return ref <= texel;
			case 4: return ref > texel;
			case 5: return ref != texel;
			case 6: return ref >= texel;
			default: return true;
		}
	};
	float     acc = 0.0f;
	const int n   = linear ? 2 : 1;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			const float wgt = (n == 2 ? (i ? fx : 1 - fx) * (j ? fy : 1 - fy) : 1.0f);
			const int   ix  = Address(x0 + i, W, s.clamp[0]);
			const int   iy  = Address(y0 + j, H, s.clamp[1]);
			float       t4[4] {0, 0, 0, 0};
			uint32_t    raw[4];
			if (ix >= 0 && iy >= 0) {
				if (!view.Fetch(mip, ix, iy, static_cast<int>(layer), t4, raw)) r.ok = false;
			}
			acc += wgt * (pass(t4[0]) ? 1.0f : 0.0f);
		}
	}
	r.v[0] = r.v[1] = r.v[2] = r.v[3] = acc;
	return r;
}

// T# DST_SEL applied to decoded components: 0 -> 0, 1 -> 1, 4..7 -> X..W.
inline void Swizzle(const TSharp& t, const float in[4], const uint32_t raw_in[4], bool integer, uint32_t out[4]) {
	for (int c = 0; c < 4; c++) {
		const uint32_t sel = t.dst_sel[c];
		if (sel == 0) out[c] = 0;
		else if (sel == 1) out[c] = integer ? 1u : B32(1.0f);
		else if (sel >= 4) out[c] = integer ? raw_in[sel - 4] : B32(in[sel - 4]);
		else out[c] = 0;
	}
}

} // namespace refinterp
