#ifndef KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
#define KYTY_TESTS_FILE_TIME_TEST_UTIL_H_

#include <chrono>
#include <cstdint>
#include <filesystem>

// Sets a host file's write time to `unix_ns` nanoseconds since the Unix epoch.
inline void SetUnixWriteTime(const std::filesystem::path& path, int64_t unix_ns) {
#ifdef _MSVC_STL_VERSION
	// No file_clock::from_sys, and clock_cast goes through utc_clock and its leap second table,
	// which gave a time SetFileTime rejects on a Windows 11 host. This file_clock counts 100 ns
	// ticks from 1601-01-01, 11644473600 s before the Unix epoch.
	constexpr int64_t                      UNIX_EPOCH_TICKS = 11644473600ll * 10000000ll;
	const std::filesystem::file_time_type file_time {
	    std::filesystem::file_time_type::duration(UNIX_EPOCH_TICKS + unix_ns / 100)};
#else
	const std::chrono::sys_time<std::chrono::nanoseconds> time {std::chrono::nanoseconds(unix_ns)};
	const auto file_time = std::chrono::floor<std::filesystem::file_time_type::duration>(
	    std::chrono::file_clock::from_sys(time));
#endif
	std::filesystem::last_write_time(path, file_time);
}

#endif // KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
