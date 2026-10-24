#ifndef KYTY_COMMON_ASSERT_H_
#define KYTY_COMMON_ASSERT_H_

#include "common/common.h"
#include "common/logging/log.h"

#include <string>
#include <string_view>

namespace Common {

// Thrown instead of ending the process by EXIT, EXIT_IF and EXIT_NOT_IMPLEMENTED on a thread
// that is inside a RecoverableFatalScope.
struct RecoverableFatal {
	std::string message;
};

// Makes the fatal checks of speculative work on this thread recoverable: a speculative shader
// translation may meet code the recompiler cannot handle, which no guest ever runs. Only for
// self-contained work whose partial state the unwinding discards; nests.
class RecoverableFatalScope {
public:
	RecoverableFatalScope();
	~RecoverableFatalScope();
	RecoverableFatalScope(const RecoverableFatalScope&)            = delete;
	RecoverableFatalScope& operator=(const RecoverableFatalScope&) = delete;
};

[[noreturn]] void DbgExitHandler(char const* file, int line, std::string_view text);
[[noreturn]] void DbgExitIfHandler(char const* expr, char const* file, int line);
[[noreturn]] void DbgNotImplementedHandler(char const* expr, char const* file, int line);

} // namespace Common

#ifndef KYTY_FINAL
#define EXIT_IF(x) ((x) ? Common::DbgExitIfHandler(#x, __FILE__, __LINE__) : (void)0)
#else
#define EXIT_IF(x) ((void)(false && (x)))
#endif
#define EXIT(...) Common::DbgExitHandler(__FILE__, __LINE__, ::fmt::sprintf(__VA_ARGS__))
#define EXIT_NOT_IMPLEMENTED(x)                                                                    \
	((x) ? Common::DbgNotImplementedHandler(#x, __FILE__, __LINE__) : (void)0)

#endif /* KYTY_COMMON_ASSERT_H_ */
