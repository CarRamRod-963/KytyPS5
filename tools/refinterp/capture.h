// Loading of a KYTY_TRACE_REF_CAPTURE directory: dispatch registers, guest memory pages, bound buffers
// and images (as raw Vulkan texels dumped by the emulator) before and after the dispatch.
#pragma once

#include "fp.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace refinterp {

inline std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
	std::vector<uint8_t> bytes;
	FILE*                file = std::fopen(path.string().c_str(), "rb");
	if (file == nullptr) return bytes;
	std::fseek(file, 0, SEEK_END);
	const long size = std::ftell(file);
	std::fseek(file, 0, SEEK_SET);
	bytes.resize(static_cast<size_t>(size));
	if (!bytes.empty()) std::fread(bytes.data(), 1, bytes.size(), file);
	std::fclose(file);
	return bytes;
}

inline std::vector<uint32_t> ParseHexList(const std::string& text) {
	std::vector<uint32_t> out;
	std::stringstream     ss(text);
	std::string           item;
	while (std::getline(ss, item, ',')) {
		if (!item.empty()) out.push_back(static_cast<uint32_t>(std::stoul(item, nullptr, 16)));
	}
	return out;
}

struct CapImage {
	uint32_t                          index   = 0;
	bool                              written = false;
	std::vector<uint32_t>             desc;
	uint64_t                          addr = 0;
	uint32_t                          w = 1, h = 1, d = 1, layers = 1, levels = 1;
	std::string                       fmt;
	std::vector<std::vector<uint8_t>> mips;      // before
	// Combined depth/stencil images (fmt D32SfloatS8Uint ...) are dumped as two files: mips[0] holds the DEPTH
	// aspect (img<i>_before_<W>x<H>_<fmt>.raw: 4-byte float or 2-byte unorm per texel, tightly packed) and
	// `stencil` the STENCIL aspect (img<i>_before_<W>x<H>_S8Stencil.raw: 1 byte per texel).
	std::vector<uint8_t>              stencil;
	std::vector<uint8_t>              after;     // mip 0 after (written images)
	std::vector<uint8_t>              reference; // mip 0 as the interpreter leaves it
	std::vector<uint8_t>              ref_written; // per texel: 1 when the interpreter stored it
};

// The capture names an image by the emulator's cached backing format, which can differ from the guest T# format
// when the game reuses memory under another format of the same texel size (5ec7 stores f16 HDR alpha through a
// 16_FLOAT T# into memory that earlier held an R8G8 render target). The raw dumps are the memory bytes, so such an
// image is interpreted with the T# format (GFX10 IMG_FORMAT, dword1 bits 20-28) instead.
inline void OverrideWithGuestFormat(CapImage& image) {
	struct Entry {
		uint32_t    code;
		const char* name;
		uint32_t    bytes;
	};
	static const Entry kFormats[] = {
	    {1, "R8Unorm", 1},          {2, "R8Snorm", 1},
	    {5, "R8Uint", 1},           {7, "R16Unorm", 2},
	    {8, "R16Snorm", 2},         {11, "R16Uint", 2},
	    {13, "R16Sfloat", 2},       {14, "R8G8Unorm", 2},
	    {15, "R8G8Snorm", 2},       {18, "R8G8Uint", 2},
	    {20, "R32Uint", 4},         {21, "R32Sint", 4},
	    {22, "R32Sfloat", 4},       {23, "R16G16Unorm", 4},
	    {24, "R16G16Snorm", 4},     {27, "R16G16Uint", 4},
	    {29, "R16G16Sfloat", 4},    {36, "B10G11R11UfloatPack32", 4},
	    {50, "A2B10G10R10UnormPack32", 4}, {56, "R8G8B8A8Unorm", 4},
	    {57, "R8G8B8A8Snorm", 4},   {60, "R8G8B8A8Uint", 4},
	    {62, "R32G32Uint", 8},      {64, "R32G32Sfloat", 8},
	    {65, "R16G16B16A16Unorm", 8}, {66, "R16G16B16A16Snorm", 8},
	    {69, "R16G16B16A16Uint", 8}, {71, "R16G16B16A16Sfloat", 8},
	    {75, "R32G32B32A32Uint", 16}, {77, "R32G32B32A32Sfloat", 16},
	    {130, "R8G8B8A8Srgb", 4},
	};
	if (image.desc.size() < 2) return;
	const uint32_t code    = (image.desc[1] >> 20u) & 0x1ffu;
	const Entry*   guest   = nullptr;
	const Entry*   backing = nullptr;
	for (const auto& e: kFormats) {
		if (e.code == code) guest = &e;
		if (image.fmt == e.name) backing = &e;
	}
	if (guest == nullptr || backing == nullptr || guest == backing || guest->bytes != backing->bytes) return;
	std::printf("image %u: T# format %s overrides the cached backing format %s\n", image.index, guest->name,
	            backing->name);
	image.fmt = guest->name;
}

struct CapBuffer {
	uint32_t              index   = 0;
	bool                  written = false;
	std::vector<uint32_t> desc;
	uint64_t              addr = 0, size = 0;
	std::vector<uint8_t>  before, after;
};

struct Capture {
	std::filesystem::path dir;
	uint64_t              hash = 0, code_addr = 0;
	uint32_t              frame = 0;
	uint32_t              groups[3] {1, 1, 1}, threads[3] {1, 1, 1};
	// Indirect dispatch: meta.txt has groups=0,0,0 and indirect_args=<file>; the file (relative to the
	// capture dir) holds the vk::DispatchIndirectCommand (three u32 group counts) the GPU executed.
	std::string           indirect_args;
	uint32_t              wave_size = 64;
	bool                  tgid_en[3] {false, false, false};
	bool                  tg_size_en     = false;
	uint32_t              tidig_comp_cnt = 0;
	uint32_t              lds_dwords     = 0;
	uint32_t              float_mode     = 0xc0;
	bool                  ieee_mode      = false;
	bool                  dx10_clamp     = true;
	uint32_t              user_sgpr      = 0;
	std::vector<uint32_t> user_data;
	std::vector<uint32_t> code;
	std::vector<CapImage>  images;
	std::vector<CapBuffer> buffers;
	std::vector<std::vector<uint32_t>> samplers;
	std::map<uint64_t, std::vector<uint8_t>> pages; // 4 KiB guest pages (crawled + buffer dumps)

	bool use_guest_buffers = false; // --guest-buffers: bind the CPU view (buf<j>_guest.bin) instead of the GPU view

	bool Load(const std::filesystem::path& path, std::string& error) {
		dir = path;
		FILE* meta = std::fopen((dir / "meta.txt").string().c_str(), "r");
		if (meta == nullptr) {
			error = "no meta.txt";
			return false;
		}
		char line[4096];
		while (std::fgets(line, sizeof(line), meta) != nullptr) {
			std::string text(line);
			while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
			auto kv = [&](const std::string& key) -> std::optional<std::string> {
				if (text.rfind(key + "=", 0) == 0) return text.substr(key.size() + 1);
				return std::nullopt;
			};
			auto token = [&](const std::string& key) -> std::string {
				const auto pos = text.find(" " + key + "=");
				if (pos == std::string::npos) return {};
				const auto start = pos + key.size() + 2;
				const auto end   = text.find(' ', start);
				return text.substr(start, end == std::string::npos ? std::string::npos : end - start);
			};
			if (auto v = kv("hash")) hash = std::stoull(*v, nullptr, 16);
			else if (auto v = kv("frame")) frame = static_cast<uint32_t>(std::stoul(*v));
			else if (auto v = kv("code_addr")) code_addr = std::stoull(*v, nullptr, 16);
			else if (auto v = kv("groups")) std::sscanf(v->c_str(), "%u,%u,%u", &groups[0], &groups[1], &groups[2]);
			else if (auto v = kv("indirect_args")) indirect_args = *v;
			else if (auto v = kv("threads")) std::sscanf(v->c_str(), "%u,%u,%u", &threads[0], &threads[1], &threads[2]);
			else if (auto v = kv("wave_size")) wave_size = static_cast<uint32_t>(std::stoul(*v));
			else if (auto v = kv("tgid_en")) {
				int x = 0, y = 0, z = 0;
				std::sscanf(v->c_str(), "%d,%d,%d", &x, &y, &z);
				tgid_en[0] = x != 0;
				tgid_en[1] = y != 0;
				tgid_en[2] = z != 0;
			} else if (auto v = kv("tg_size_en")) tg_size_en = *v == "1";
			else if (auto v = kv("tidig_comp_cnt")) tidig_comp_cnt = static_cast<uint32_t>(std::stoul(*v));
			else if (auto v = kv("lds_dwords")) lds_dwords = static_cast<uint32_t>(std::stoul(*v));
			else if (auto v = kv("float_mode")) float_mode = static_cast<uint32_t>(std::stoul(*v));
			else if (auto v = kv("ieee_mode")) ieee_mode = *v == "1";
			else if (auto v = kv("dx10_clamp")) dx10_clamp = *v == "1";
			else if (auto v = kv("user_sgpr")) user_sgpr = static_cast<uint32_t>(std::stoul(*v));
			else if (auto v = kv("user_data")) user_data = ParseHexList(*v);
			else if (text.rfind("image ", 0) == 0) {
				CapImage image;
				image.index   = static_cast<uint32_t>(std::stoul(token("index")));
				image.written = token("written") == "1";
				image.desc    = ParseHexList(token("desc"));
				if (const auto a = token("addr"); !a.empty()) image.addr = std::stoull(a, nullptr, 16);
				if (const auto e = token("extent"); !e.empty()) std::sscanf(e.c_str(), "%ux%ux%u", &image.w, &image.h, &image.d);
				if (const auto l = token("layers"); !l.empty()) image.layers = static_cast<uint32_t>(std::stoul(l));
				if (const auto l = token("levels"); !l.empty()) image.levels = static_cast<uint32_t>(std::stoul(l));
				image.fmt = token("fmt");
				OverrideWithGuestFormat(image);
				images.push_back(std::move(image));
			} else if (text.rfind("buffer ", 0) == 0) {
				CapBuffer buffer;
				buffer.index   = static_cast<uint32_t>(std::stoul(token("index")));
				buffer.written = token("written") == "1";
				buffer.desc    = ParseHexList(token("desc"));
				if (const auto a = token("addr"); !a.empty()) buffer.addr = std::stoull(a, nullptr, 16);
				if (const auto s = token("size"); !s.empty()) buffer.size = std::stoull(s, nullptr, 16);
				buffers.push_back(std::move(buffer));
			} else if (text.rfind("sampler ", 0) == 0) {
				samplers.push_back(ParseHexList(token("desc")));
			}
		}
		std::fclose(meta);
		// Indirect dispatch: the group counts come from the dumped arguments, not from groups=.
		if (!indirect_args.empty()) {
			const auto args = ReadFile(dir / indirect_args);
			if (args.size() < 12u) {
				error = "indirect_args file '" + indirect_args + "' is missing or shorter than 12 bytes" +
				        " (the emulator writes it one present after the capture)";
				return false;
			}
			std::memcpy(groups, args.data(), 12u);
		}
		const auto code_bytes = ReadFile(dir / "code.bin");
		code.resize(code_bytes.size() / 4u);
		if (!code.empty()) std::memcpy(code.data(), code_bytes.data(), code.size() * 4u);
		// Guest pages.
		const auto mem = ReadFile(dir / "mem.bin");
		if (FILE* index = std::fopen((dir / "mem.txt").string().c_str(), "r")) {
			unsigned long long page = 0, size = 0, offset = 0;
			while (std::fscanf(index, "%llx %llx %llx", &page, &size, &offset) == 3) {
				if (offset + size <= mem.size()) pages[page] = std::vector<uint8_t>(mem.begin() + static_cast<long long>(offset), mem.begin() + static_cast<long long>(offset + size));
			}
			std::fclose(index);
		}
		// Buffers (GPU view before the dispatch) override crawled pages.
		for (auto& buffer: buffers) {
			buffer.before = ReadFile(dir / ("buf" + std::to_string(buffer.index) + (use_guest_buffers ? "_guest.bin" : "_before.bin")));
			if (buffer.written) buffer.after = ReadFile(dir / ("buf" + std::to_string(buffer.index) + "_after.bin"));
			for (uint64_t i = 0; i < buffer.before.size(); i++) WriteByte(buffer.addr + i, buffer.before[i]);
		}
		// Images: img<i>_before_<W>x<H>[x<D>][L<n>]_<fmt>.raw and img<i>_before_mip<N>_<W>x<H>_<fmt>.raw.
		for (auto& image: images) {
			image.mips.resize(image.levels);
			const std::string before = "img" + std::to_string(image.index) + "_before_";
			const std::string after  = "img" + std::to_string(image.index) + "_after_";
			for (const auto& entry: std::filesystem::directory_iterator(dir)) {
				const auto name = entry.path().filename().string();
				// The stencil aspect of a combined depth/stencil image goes to its own file.
				const bool is_stencil = name.size() > 14u && name.compare(name.size() - 14u, 14u, "_S8Stencil.raw") == 0;
				if (name.rfind(before, 0) == 0) {
					const auto rest = name.substr(before.size());
					if (is_stencil) {
						image.stencil = ReadFile(entry.path());
					} else if (rest.rfind("mip", 0) == 0) {
						const auto mip = static_cast<uint32_t>(std::stoul(rest.substr(3)));
						if (mip < image.levels) image.mips[mip] = ReadFile(entry.path());
					} else {
						image.mips[0] = ReadFile(entry.path());
					}
				} else if (name.rfind(after, 0) == 0 && name.find("_mip") == std::string::npos && !is_stencil) {
					image.after = ReadFile(entry.path());
				}
			}
			image.reference = image.mips.empty() ? std::vector<uint8_t> {} : image.mips[0];
			image.ref_written.assign(image.reference.size(), 0);
		}
		return true;
	}

	void WriteByte(uint64_t address, uint8_t value) {
		auto& page = pages[address & ~0xfffull];
		if (page.size() != 4096) page.resize(4096, 0);
		page[address & 0xfffu] = value;
	}

	std::set<uint64_t> missing_pages;

	bool Read(uint64_t address, void* out, uint64_t size) {
		auto* bytes = static_cast<uint8_t*>(out);
		bool  ok    = true;
		for (uint64_t i = 0; i < size; i++) {
			const auto a  = address + i;
			const auto it = pages.find(a & ~0xfffull);
			if (it == pages.end()) {
				bytes[i] = 0;
				missing_pages.insert(a & ~0xfffull);
				ok = false;
			} else {
				bytes[i] = it->second[a & 0xfffu];
			}
		}
		return ok;
	}
	uint32_t Read32(uint64_t address) {
		uint32_t value = 0;
		Read(address, &value, 4);
		return value;
	}
	// Stores by the reference run: applied to the memory view and recorded for the comparison.
	std::map<uint64_t, uint8_t> stores;
	void Write(uint64_t address, const void* data, uint64_t size) {
		const auto* bytes = static_cast<const uint8_t*>(data);
		for (uint64_t i = 0; i < size; i++) {
			WriteByte(address + i, bytes[i]);
			stores[address + i] = bytes[i];
		}
	}

	// The captured image a T# base address names: the Vulkan image's guest address first, then the base the
	// binding's own T# holds. The two differ for the stencil plane of a depth/stencil image: the guest keeps the
	// stencil plane at its own address (T# base), the emulator one combined image at the depth plane's address.
	// Several captured images can name one address (the same memory bound once as a read-only and once as a written
	// view, identical T#): the written one is taken, so the reference run's stores land in the image that is compared.
	CapImage* FindImage(uint64_t address) {
		CapImage* first = nullptr;
		for (auto& image: images) {
			if (image.addr != address) continue;
			if (image.written) return &image;
			if (first == nullptr) first = &image;
		}
		if (first != nullptr) return first;
		for (auto& image: images) {
			if (image.desc.size() >= 2u && image.addr != 0 &&
			    ((((static_cast<uint64_t>(image.desc[1]) & 0xffu) << 32u) | image.desc[0]) << 8u) == address) {
				return &image;
			}
		}
		return nullptr;
	}
};

} // namespace refinterp
