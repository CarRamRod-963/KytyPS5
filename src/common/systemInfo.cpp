#include "common/systemInfo.h"

#include "common/assert.h"
#include "cpuinfo.h"

#include <fmt/format.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace Common {

static CpuFeatures ReadCpuFeatures() {
	CpuFeatures features;
	features.physical_cores     = cpuinfo_get_cores_count();
	features.logical_processors = cpuinfo_get_processors_count();
#if defined(__x86_64__) || defined(_M_X64)
	unsigned       eax      = 0;
	unsigned       ebx      = 0;
	unsigned       ecx      = 0;
	unsigned       edx      = 0;
	const unsigned max_leaf = __get_cpuid_max(0, nullptr);
	if (max_leaf >= 7 && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
		features.avx512_cpu = (ebx & (1u << 16u)) != 0;
	}
	constexpr unsigned OsXsave = 1u << 27u;
	if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0 && (ecx & OsXsave) != 0) {
		unsigned low  = 0;
		unsigned high = 0;
		// xgetbv(0); OSXSAVE says the instruction exists and XCR0 is readable.
		__asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
		features.xcr0 = (static_cast<uint64_t>(high) << 32u) | low;
		// Bits 5-7: opmask, upper halves of ZMM0-15, ZMM16-31.
		features.avx512_os = (features.xcr0 & 0xe0u) == 0xe0u;
		if (max_leaf >= 0xd && __get_cpuid_count(0xd, 0, &eax, &ebx, &ecx, &edx) != 0) {
			features.xsave_size = ebx;
		}
	}
#endif
	return features;
}

SystemInfo GetSystemInfo() {
	static const bool initialized = cpuinfo_initialize();
	EXIT_IF(!initialized);

	const auto* package = cpuinfo_get_package(0);
	EXIT_IF(package == nullptr);

	return {package->name, ReadCpuFeatures()};
}

std::string FormatCpuLine(const SystemInfo& info) {
	const auto& f    = info.Features;
	std::string text = info.ProcessorName;
	if (f.physical_cores != 0 && f.logical_processors != 0) {
		text += fmt::format(" ({} cores, {} threads)", f.physical_cores, f.logical_processors);
	}
	const char* avx512 = "no";
	if (f.avx512_cpu) {
		avx512 = f.avx512_os ? "yes" : "yes, not enabled by the OS";
	}
	text += fmt::format(", AVX-512: {}", avx512);
	if (f.xsave_size != 0) {
		text += fmt::format(", XSAVE area {} bytes, XCR0 0x{:x}", f.xsave_size, f.xcr0);
	}
	return text;
}

} // namespace Common
