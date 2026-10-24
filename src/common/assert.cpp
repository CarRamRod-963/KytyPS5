#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

namespace Common {

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

static thread_local uint32_t g_recoverable_fatal_depth = 0;

RecoverableFatalScope::RecoverableFatalScope() {
	g_recoverable_fatal_depth++;
}

RecoverableFatalScope::~RecoverableFatalScope() {
	g_recoverable_fatal_depth--;
}

static void ThrowIfRecoverable(std::string_view text, const char* file, int line) {
	if (g_recoverable_fatal_depth != 0) {
		throw RecoverableFatal {fmt::format("{} in {}:{}", text, file, line)};
	}
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	ThrowIfRecoverable(text, file, line);
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	ThrowIfRecoverable(text, file, line);
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	ThrowIfRecoverable(text, file, line);
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
