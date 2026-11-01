// refinterp: offline RDNA reference interpreter for PS5 compute shaders captured by the emulator
// (KYTY_TRACE_REF_CAPTURE). It executes the captured dispatch on the CPU from the AMD ISA definitions
// and compares the results with what the emulator's GPU translation wrote.
//
//   refinterp disasm <code.bin>                       list the decoded program
//   refinterp run <capture dir> [--groups N|all] [--group x,y,z]... [--trace-lane g:lane]
//
// Decode tables: generated/isa_tables.inc (AMD machine-readable ISA, RDNA1 + BVH). Semantics: interp.h.
#include "decode.h"
#include "interp.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace refinterp;

static std::vector<uint32_t> ReadWords(const std::string& path) {
	std::vector<uint32_t> words;
	FILE*                 file = std::fopen(path.c_str(), "rb");
	if (file == nullptr) return words;
	std::fseek(file, 0, SEEK_END);
	const long size = std::ftell(file);
	std::fseek(file, 0, SEEK_SET);
	words.resize(static_cast<size_t>(size) / 4u);
	if (!words.empty()) std::fread(words.data(), 4, words.size(), file);
	std::fclose(file);
	return words;
}

static int Disasm(const std::string& path) {
	const auto code = ReadWords(path);
	Decoder    decoder;
	uint32_t   index = 0;
	while (index < code.size()) {
		Inst        inst;
		std::string error;
		if (!decoder.Decode(code, index, inst, error)) {
			std::printf("0x%08x: <%s> word=%08x\n", index * 4u, error.c_str(), code[index]);
			return 1;
		}
		std::printf("0x%08x: %-28s %-6s", inst.pc, kOpNames[static_cast<int>(inst.op)], EncName(inst.enc));
		for (const auto& [k, v]: inst.f) {
			if (k == "ENCODING" || k == "OP") continue;
			std::printf(" %s=%u", k.c_str(), v);
		}
		if (inst.has_literal) std::printf(" lit=0x%08x", inst.literal);
		std::printf("\n");
		if (inst.op == Op::S_ENDPGM) break;
		index += inst.words;
	}
	return 0;
}

int main(int argc, char** argv) {
	if (argc >= 3 && std::string(argv[1]) == "disasm") return Disasm(argv[2]);
	if (argc >= 3 && std::string(argv[1]) == "run") return RunCapture(argc, argv);
	std::fprintf(stderr, "usage: refinterp disasm <code.bin> | refinterp run <capture dir> [options]\n");
	return 2;
}

// ------------------------------------------------------------------------------------------------
// run: execute selected workgroups of the captured dispatch and compare with the GPU result.
int refinterp::RunCapture(int argc, char** argv) {
	Capture     cap;
	std::string error;
	for (int i = 3; i < argc; i++) {
		if (std::string(argv[i]) == "--guest-buffers") cap.use_guest_buffers = true;
	}
	if (!cap.Load(argv[2], error)) {
		std::fprintf(stderr, "load failed: %s\n", error.c_str());
		return 1;
	}
	Options     opt;
	std::string pc_profile_path;
	std::string save_ref; // --save-ref <prefix>: <prefix>_img<i>.raw (reference texels) + _mask.raw (1 = stored)
	for (int i = 3; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--groups" && i + 1 < argc) {
			const std::string v = argv[++i];
			if (v == "all") opt.all_groups = true;
			else opt.sample = static_cast<uint32_t>(std::stoul(v));
		} else if (a == "--group" && i + 1 < argc) {
			std::array<uint32_t, 3> g {0, 0, 0};
			std::sscanf(argv[++i], "%u,%u,%u", &g[0], &g[1], &g[2]);
			opt.groups.push_back(g);
		} else if (a == "--trace" && i + 1 < argc) {
			std::sscanf(argv[++i], "%d:%d", &opt.trace_group, &opt.trace_lane);
		} else if (a == "--prec8") {
			opt.prec8 = true;
		} else if (a == "--report" && i + 1 < argc) {
			opt.max_report = static_cast<uint32_t>(std::stoul(argv[++i]));
		} else if (a == "--force-compare" && i + 1 < argc) {
			opt.force_compare = std::stof(argv[++i]);
		} else if (a == "--watch" && i + 1 < argc) {
			std::sscanf(argv[++i], "%d:%d,%d,%d", &opt.watch_image, &opt.watch_x, &opt.watch_y, &opt.watch_z);
		} else if (a == "--watch-addr" && i + 1 < argc) {
			opt.watch_addr = std::stoull(argv[++i], nullptr, 16);
		} else if (a == "--save-ref" && i + 1 < argc) {
			save_ref = argv[++i];
		} else if (a == "--probe-nth" && i + 1 < argc) {
			opt.probe_nth = static_cast<uint32_t>(std::stoul(argv[++i]));
		} else if (a == "--probe" && i + 1 < argc) {
			const std::string list = argv[++i];
			uint32_t          slot = 0;
			for (size_t start = 0; start < list.size() && slot < 4u;) {
				const size_t end = std::min(list.find('+', start), list.size());
				char     kind = 'v';
				unsigned pc = 0, reg = 0, count = 1;
				const int n = std::sscanf(list.substr(start, end - start).c_str(), "%x:%c%u:%u", &pc, &kind, &reg, &count);
				if (n >= 3) {
					Options::ProbePoint point;
					point.pc     = pc;
					point.scalar   = kind == 's';
					point.to_float = kind == 'u';
					point.reg    = reg;
					point.count  = std::clamp(count, 1u, 4u - slot);
					point.slot   = slot;
					slot += point.count;
					opt.probes.push_back(point);
				}
				start = end + 1;
			}
		} else if (a == "--zero-op" && i + 1 < argc) {
			std::string list = argv[++i];
			for (size_t start = 0; start < list.size();) {
				const size_t end = std::min(list.find(',', start), list.size());
				if (end > start) opt.zero_ops.push_back(list.substr(start, end - start));
				start = end + 1;
			}
		} else if (a == "--zero-pc" && i + 1 < argc) {
			std::string list = argv[++i];
			for (size_t start = 0; start < list.size();) {
				const size_t end = std::min(list.find(',', start), list.size());
				if (end > start) opt.zero_pcs.push_back(static_cast<uint32_t>(std::stoul(list.substr(start, end - start), nullptr, 16)));
				start = end + 1;
			}
		} else if (a == "--compare-flip") {
			opt.compare_flip = true;
		} else if (a == "--pc-profile" && i + 1 < argc) {
			opt.pc_profile = true;
			pc_profile_path = argv[++i];
		} else if (a == "--max-steps" && i + 1 < argc) {
			opt.max_steps = std::stoull(argv[++i]);
		}
	}
	const std::string indirect_note = cap.indirect_args.empty() ? std::string() : " (indirect, from " + cap.indirect_args + ")";
	std::printf("capture %s: hash=%016llx frame=%u groups=%ux%ux%u%s threads=%ux%ux%u wave%u lds=%u dwords user_sgpr=%u\n",
	            cap.dir.string().c_str(), static_cast<unsigned long long>(cap.hash), cap.frame, cap.groups[0], cap.groups[1],
	            cap.groups[2], indirect_note.c_str(), cap.threads[0], cap.threads[1], cap.threads[2], cap.wave_size,
	            cap.lds_dwords, cap.user_sgpr);
	std::printf("float_mode=0x%x (fp32 denormals %s) ieee=%d dx10_clamp=%d, %zu pages, %zu images, %zu buffers\n",
	            cap.float_mode, ((cap.float_mode >> 4u) & 3u) == 0 ? "flushed" : "kept", cap.ieee_mode ? 1 : 0,
	            cap.dx10_clamp ? 1 : 0, cap.pages.size(), cap.images.size(), cap.buffers.size());
	// Workgroup list.
	std::vector<std::array<uint32_t, 3>> run = opt.groups;
	if (run.empty()) {
		const uint64_t total = static_cast<uint64_t>(cap.groups[0]) * cap.groups[1] * cap.groups[2];
		if (opt.all_groups || total <= opt.sample) {
			for (uint32_t z = 0; z < cap.groups[2]; z++)
				for (uint32_t y = 0; y < cap.groups[1]; y++)
					for (uint32_t x = 0; x < cap.groups[0]; x++) run.push_back({x, y, z});
		} else {
			// corners, centre, then a fixed pseudo-random spread
			auto add = [&](uint32_t x, uint32_t y, uint32_t z) {
				std::array<uint32_t, 3> g {x, y, z};
				if (std::find(run.begin(), run.end(), g) == run.end()) run.push_back(g);
			};
			const uint32_t X = cap.groups[0] - 1, Y = cap.groups[1] - 1, Z = cap.groups[2] - 1;
			add(0, 0, 0);
			add(X, 0, 0);
			add(0, Y, 0);
			add(X, Y, Z);
			add(X / 2, Y / 2, Z / 2);
			uint64_t seed = 0x9e3779b97f4a7c15ull;
			while (run.size() < opt.sample) {
				seed = seed * 6364136223846793005ull + 1442695040888963407ull;
				const uint64_t idx = (seed >> 17u) % total;
				add(static_cast<uint32_t>(idx % cap.groups[0]), static_cast<uint32_t>((idx / cap.groups[0]) % cap.groups[1]),
				    static_cast<uint32_t>(idx / (static_cast<uint64_t>(cap.groups[0]) * cap.groups[1])));
			}
		}
	}
	Interpreter interp(cap, opt);
	for (size_t i = 0; i < run.size(); i++) {
		interp.tracing    = static_cast<int>(i) == opt.trace_group;
		interp.trace_lane = opt.trace_lane;
		if (interp.tracing) std::printf("trace group %u,%u,%u lane %d\n", run[i][0], run[i][1], run[i][2], opt.trace_lane);
		if (!interp.RunGroup(run[i][0], run[i][1], run[i][2])) {
			std::printf("group %u,%u,%u stopped: %s\n", run[i][0], run[i][1], run[i][2],
			            interp.errors.empty() ? "?" : interp.errors.back().c_str());
			break;
		}
	}
	std::printf("ran %zu workgroups, %llu wave instructions\n", run.size(), static_cast<unsigned long long>(interp.total_steps));
	if (!pc_profile_path.empty()) {
		if (FILE* f = std::fopen(pc_profile_path.c_str(), "w")) {
			for (const auto& [pc, h]: interp.pc_hits)
				std::fprintf(f, "%08x %llu %llu\n", pc, static_cast<unsigned long long>(h[0]), static_cast<unsigned long long>(h[1]));
			std::fclose(f);
		}
	}
	for (const auto& e: interp.errors) std::printf("ERROR %s\n", e.c_str());
	for (const auto& [k, v]: interp.unsupported) std::printf("UNSUPPORTED %s x%llu\n", k.c_str(), static_cast<unsigned long long>(v));
	for (const auto& [k, v]: interp.approx_ops) {
		std::printf("approximate op %s x%llu (may differ by ~1 ulp)\n", k.c_str(), static_cast<unsigned long long>(v));
	}
	if (!cap.missing_pages.empty()) {
		std::printf("MISSING guest pages read (returned 0): %zu, first:", cap.missing_pages.size());
		int n = 0;
		for (const auto p: cap.missing_pages) {
			if (n++ >= 8) break;
			std::printf(" %llx", static_cast<unsigned long long>(p));
		}
		std::printf("\n");
	}
	// ---- compare written images
	for (auto& image: cap.images) {
		if (!image.written) continue;
		if (!save_ref.empty()) {
			const std::string base = save_ref + "_img" + std::to_string(image.index);
			if (FILE* file = std::fopen((base + ".raw").c_str(), "wb")) {
				std::fwrite(image.reference.data(), 1, image.reference.size(), file);
				std::fclose(file);
			}
			if (FILE* file = std::fopen((base + "_mask.raw").c_str(), "wb")) {
				std::fwrite(image.ref_written.data(), 1, image.ref_written.size(), file);
				std::fclose(file);
			}
		}
		const auto f = GetFormat(image.fmt);
		if (!f.ok || image.after.empty()) {
			std::printf("image %u (%s): no comparable GPU result\n", image.index, image.fmt.c_str());
			continue;
		}
		uint64_t compared = 0, exact = 0, ulp1 = 0, ulp16 = 0, rel3 = 0, worse = 0, nz_ref = 0, nz_gpu = 0;
		double   sum_ref = 0, sum_gpu = 0, max_ref = 0;
		struct Bad {
			double   rel;
			uint64_t texel;
			uint32_t comp;
			float    ref, gpu;
		};
		std::vector<Bad> bads;
		const uint64_t   texels = image.reference.size() / f.bytes;
		for (uint64_t t = 0; t < texels; t++) {
			if (!image.ref_written[t]) continue;
			float    r[4], g[4];
			uint32_t rr[4], gr[4];
			DecodeTexel(image.fmt, f, image.reference.data() + t * f.bytes, r, rr);
			DecodeTexel(image.fmt, f, image.after.data() + t * f.bytes, g, gr);
			for (uint32_t c = 0; c < std::max(f.comps, 1u); c++) {
				compared++;
				if (r[c] != 0.0f) nz_ref++;
				if (g[c] != 0.0f) nz_gpu++;
				sum_ref += r[c];
				sum_gpu += g[c];
				max_ref = std::max(max_ref, static_cast<double>(std::fabs(r[c])));
				if (rr[c] == gr[c]) {
					exact++;
					continue;
				}
				const double d   = std::fabs(static_cast<double>(r[c]) - g[c]);
				const double m   = std::max(std::fabs(static_cast<double>(r[c])), std::fabs(static_cast<double>(g[c])));
				const double rel = m > 0 ? d / m : d;
				uint32_t     ulps;
				if (f.kind == FormatInfo::Float && f.bits == 16) {
					ulps = static_cast<uint32_t>(std::abs(static_cast<int32_t>(rr[c] & 0xffffu) - static_cast<int32_t>(gr[c] & 0xffffu)));
				} else if (f.kind == FormatInfo::Float) {
					ulps = Ulps(r[c], g[c]);
				} else {
					ulps = static_cast<uint32_t>(std::abs(static_cast<int64_t>(rr[c]) - static_cast<int64_t>(gr[c])));
				}
				if (ulps <= 1) ulp1++;
				else if (ulps <= 16) ulp16++;
				else if (rel <= 1e-3) rel3++;
				else worse++;
				bads.push_back({std::isnan(rel) ? 1e30 : rel, t, c, r[c], g[c]});
			}
		}
		std::sort(bads.begin(), bads.end(), [](const Bad& a, const Bad& b) { return a.rel > b.rel; });
		std::printf("\nimage %u %s %ux%ux%u: compared %llu values written by the reference run\n", image.index, image.fmt.c_str(),
		            image.w, image.h, image.d, static_cast<unsigned long long>(compared));
		std::printf("  exact %llu | 1 ulp %llu | <=16 ulp %llu | rel<=1e-3 %llu | WORSE %llu\n",
		            static_cast<unsigned long long>(exact), static_cast<unsigned long long>(ulp1),
		            static_cast<unsigned long long>(ulp16), static_cast<unsigned long long>(rel3),
		            static_cast<unsigned long long>(worse));
		std::printf("  non-zero values: reference %llu, gpu %llu; sum reference %.9g gpu %.9g; max |reference| %.9g\n",
		            static_cast<unsigned long long>(nz_ref), static_cast<unsigned long long>(nz_gpu), sum_ref, sum_gpu, max_ref);
		for (size_t k = 0; k < bads.size() && k < opt.max_report; k++) {
			const auto&    b = bads[k];
			const uint64_t x = b.texel % image.w, y = (b.texel / image.w) % image.h;
			const uint64_t z = b.texel / (static_cast<uint64_t>(image.w) * image.h);
			std::printf("  texel (%llu,%llu,%llu) c%u: reference %.9g  gpu %.9g  rel %.3g\n", static_cast<unsigned long long>(x),
			            static_cast<unsigned long long>(y), static_cast<unsigned long long>(z), b.comp, b.ref, b.gpu, b.rel);
		}
	}
	// ---- compare buffer stores
	for (auto& buffer: cap.buffers) {
		if (!buffer.written || buffer.after.empty()) continue;
		if (!save_ref.empty() && !buffer.before.empty()) {
			// --save-ref: <prefix>_buf<i>.bin = the buffer before the dispatch with the reference run's stores applied.
			auto reference = buffer.before;
			for (const auto& [addr, value]: cap.stores) {
				if (addr >= buffer.addr && addr < buffer.addr + reference.size()) reference[addr - buffer.addr] = value;
			}
			if (FILE* file = std::fopen((save_ref + "_buf" + std::to_string(buffer.index) + ".bin").c_str(), "wb")) {
				std::fwrite(reference.data(), 1, reference.size(), file);
				std::fclose(file);
			}
		}
		uint64_t compared = 0, diff = 0;
		for (const auto& [addr, value]: cap.stores) {
			if (addr < buffer.addr || addr >= buffer.addr + buffer.after.size()) continue;
			compared++;
			if (buffer.after[addr - buffer.addr] != value) diff++;
		}
		std::printf("\nbuffer %u @%llx: %llu bytes stored by the reference run, %llu differ from the GPU result\n", buffer.index,
		            static_cast<unsigned long long>(buffer.addr), static_cast<unsigned long long>(compared),
		            static_cast<unsigned long long>(diff));
		if (diff == 0) continue;
		// Per stored dword: distance in fp32 ulps between reference and GPU, plus the worst offsets.
		std::map<uint64_t, std::array<uint8_t, 4>> ref_dwords;
		for (const auto& [addr, value]: cap.stores) {
			if (addr < buffer.addr || addr >= buffer.addr + buffer.after.size()) continue;
			const uint64_t off  = addr - buffer.addr;
			const uint64_t base = off & ~3ull;
			auto [it, fresh]    = ref_dwords.try_emplace(base);
			if (fresh) std::memcpy(it->second.data(), buffer.after.data() + base, std::min<uint64_t>(4, buffer.after.size() - base));
			it->second[off & 3u] = value;
		}
		uint64_t d_exact = 0, d_ulp1 = 0, d_ulp16 = 0, d_rel3 = 0, d_worse = 0, reported = 0;
		for (const auto& [off, bytes]: ref_dwords) {
			if (off + 4 > buffer.after.size()) continue;
			uint32_t ru = 0, gu = 0;
			std::memcpy(&ru, bytes.data(), 4);
			std::memcpy(&gu, buffer.after.data() + off, 4);
			if (ru == gu) {
				d_exact++;
				continue;
			}
			float rf = 0, gf = 0;
			std::memcpy(&rf, &ru, 4);
			std::memcpy(&gf, &gu, 4);
			const int64_t ri   = (ru & 0x80000000u) ? -static_cast<int64_t>(ru & 0x7fffffffu) : static_cast<int64_t>(ru);
			const int64_t gi   = (gu & 0x80000000u) ? -static_cast<int64_t>(gu & 0x7fffffffu) : static_cast<int64_t>(gu);
			const int64_t ulps = ri > gi ? ri - gi : gi - ri;
			const double  mag  = std::max(std::fabs(static_cast<double>(rf)), std::fabs(static_cast<double>(gf)));
			const double  rel  = std::fabs(static_cast<double>(rf) - gf) / std::max(1e-30, mag);
			if (ulps <= 1) d_ulp1++;
			else if (ulps <= 16) d_ulp16++;
			else if (rel <= 1e-3) d_rel3++;
			else {
				d_worse++;
				if (reported++ < opt.max_report)
					std::printf("  +0x%llx: reference 0x%08x (%.9g)  gpu 0x%08x (%.9g)\n", static_cast<unsigned long long>(off), ru, rf, gu, gf);
			}
		}
		std::printf("  dwords: exact %llu | 1 ulp %llu | <=16 ulp %llu | rel<=1e-3 %llu | WORSE %llu\n",
		            static_cast<unsigned long long>(d_exact), static_cast<unsigned long long>(d_ulp1),
		            static_cast<unsigned long long>(d_ulp16), static_cast<unsigned long long>(d_rel3),
		            static_cast<unsigned long long>(d_worse));
	}
	return 0;
}
