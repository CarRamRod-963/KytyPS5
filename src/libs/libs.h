#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include <fmt/format.h>

namespace Libs {
// KYTY_PRINT_HLE=1 (diagnostic): every library's PRINT_NAME() on from the start (with a printf direction other than
// Silent), so the HLE calls before a crash are in the log; compared 28.10 with chenxiao's fork running Ghost of Yotei.
inline bool PrintHleNamesDefault() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PRINT_HLE");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}
} // namespace Libs

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLED g_print_name

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLE(flag) PRINT_NAME_ENABLED = flag;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_DEFINE(name) void name(Loader::SymbolDatabase* s)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_NAME(l, m)                                                                             \
	[[maybe_unused]] static thread_local bool PRINT_NAME_ENABLED = ::Libs::PrintHleNamesDefault();                          \
	static constexpr char                     g_library[]        = l;                              \
	static constexpr char                     g_module[]         = m;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_VERSION(l, lv, m, mv1, mv2)                                                            \
	LIB_NAME(l, m);                                                                                \
	static constexpr int g_library_version      = lv;                                              \
	static constexpr int g_module_version_major = mv1;                                             \
	static constexpr int g_module_version_minor = mv2;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_ADD(n, f, t)                                                                           \
	{                                                                                              \
		Loader::SymbolResolve sr {};                                                               \
		sr.name                 = n;                                                               \
		sr.library              = g_library;                                                       \
		sr.library_version      = g_library_version;                                               \
		sr.module               = g_module;                                                        \
		sr.module_version_major = g_module_version_major;                                          \
		sr.module_version_minor = g_module_version_minor;                                          \
		sr.type                 = t;                                                               \
		auto        func        = reinterpret_cast<uint64_t>(f);                                   \
		const char* dbg_name    = "" #f;                                                           \
		s->Add(sr, func, dbg_name);                                                                \
	}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_OBJECT(n, f) LIB_ADD(n, f, Loader::SymbolType::Object)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f) LIB_ADD(n, f, Loader::SymbolType::Func)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KYTY_HLE_CONCAT_IMPL(a, b) a##b
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KYTY_HLE_CONCAT(a, b) KYTY_HLE_CONCAT_IMPL(a, b)

// HLE_COUNT(): only the KYTY_TRACE_HLE_COUNT counter of PRINT_NAME (hot functions that do not log).
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define HLE_COUNT()                                                                                \
	Libs::HleScope KYTY_HLE_CONCAT(kyty_hle_scope_, __LINE__)(                                     \
	    Libs::HleCountEnabled() ? [fn = __func__]() -> Libs::HleCounter* {                         \
		    static Libs::HleCounter counter(g_library, fn);                                        \
		    return &counter;                                                                       \
	    }()                                                                                        \
	                            : nullptr)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	Libs::HleScope KYTY_HLE_CONCAT(kyty_hle_scope_, __LINE__)(                                     \
	    Libs::HleCountEnabled() ? [fn = __func__]() -> Libs::HleCounter* {                         \
		    static Libs::HleCounter counter(g_library, fn);                                        \
		    return &counter;                                                                       \
	    }()                                                                                        \
	                            : nullptr);                                                        \
	if (PRINT_NAME_ENABLED || Libs::TraceAllCalls()) {                                             \
		Libs::PrintName(g_library, g_module, __func__);                                              \
	}

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

// KYTY_TRACE_HLE_CALLS=<count>: the first <count> library calls of every thread are logged
// (PRINT_NAME otherwise covers only the thread that registered a library).
[[gnu::noinline]] inline bool TraceAllCalls() {
	static const int64_t budget = [] {
		const char* value = std::getenv("KYTY_TRACE_HLE_CALLS");
		return value != nullptr ? std::strtoll(value, nullptr, 10) : 0ll;
	}();
	if (budget <= 0) return false;
	static std::atomic<int64_t> used {0};
	return used.fetch_add(1, std::memory_order_relaxed) < budget;
}

// KYTY_TRACE_HLE_COUNT=1 (diagnostic): every PRINT_NAME function counts its calls and the time
// until it returns (all threads); every ~5 s the top functions by time and by calls are printed as
// HLE_TOP lines.
[[gnu::noinline]] inline bool HleCountEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_HLE_COUNT");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

struct HleCounter {
	HleCounter(const char* library, const char* function);
	const char*           library;
	const char*           function;
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> ns {0};
	uint64_t              last_calls = 0;
	uint64_t              last_ns    = 0;
	HleCounter*           next       = nullptr;
};

uint64_t HleNowNs();

class HleScope {
public:
	explicit HleScope(HleCounter* counter): m_counter(counter), m_start(counter != nullptr ? HleNowNs() : 0) {}
	~HleScope() {
		if (m_counter != nullptr) {
			m_counter->calls.fetch_add(1, std::memory_order_relaxed);
			m_counter->ns.fetch_add(HleNowNs() - m_start, std::memory_order_relaxed);
		}
	}
	HleScope(const HleScope&)            = delete;
	HleScope& operator=(const HleScope&) = delete;

private:
	HleCounter* m_counter;
	uint64_t    m_start;
};

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
[[gnu::noinline]] inline void PrintName(const char* library, const char* module, const char* function) {
	if (Log::GetDirection() != Log::Direction::Silent) {
		const auto elapsed_ms      = static_cast<uint64_t>(Loader::Timer::GetTimeMs());
		const auto print_name_time = fmt::format(
		    "{:02}:{:02}:{:02}.{:03}", elapsed_ms / 3600000, (elapsed_ms / 60000) % 60,
		    (elapsed_ms / 1000) % 60, elapsed_ms % 1000);
		LOGF_COLOR(Log::Color::Cyan, "[%d][%s] %s::%s::%s()\n",
		           Common::Thread::GetThreadIdUnique(), print_name_time.c_str(), library, module,
		           function);
	}
}

void InitAll(Loader::SymbolDatabase* s);

} // namespace Libs
#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_ */
