#ifndef KYTY_COMMON_SYSTEM_INFO_H_
#define KYTY_COMMON_SYSTEM_INFO_H_

#include <cstdint>
#include <string>

namespace Common {

// What decides the size of an exception frame on the host: the XSAVE area of the state the OS
// enables (XCR0) is pushed below the stack pointer of a thread that takes an exception, so CPUs
// with and without AVX-512 state overwrite different amounts of a guest's red zone.
struct CpuFeatures {
	uint32_t physical_cores     = 0;     // 0 = unknown
	uint32_t logical_processors = 0;     // 0 = unknown
	bool     avx512_cpu         = false; // AVX-512F reported by CPUID
	bool     avx512_os          = false; // opmask, ZMM_Hi256 and Hi16_ZMM state enabled in XCR0
	uint64_t xcr0               = 0;     // 0 = OSXSAVE off or not an x86-64 host
	uint32_t xsave_size         = 0;     // CPUID.0xD.0:EBX, XSAVE area of the XCR0 state, bytes
};

struct SystemInfo {
	std::string ProcessorName;
	CpuFeatures Features;
};

[[nodiscard]] SystemInfo GetSystemInfo();

// "<name> (8 cores, 16 threads), AVX-512: no, XSAVE area 2696 bytes, XCR0 0x207".
[[nodiscard]] std::string FormatCpuLine(const SystemInfo& info);

} // namespace Common

#endif /* KYTY_COMMON_SYSTEM_INFO_H_ */
