#ifndef KYTY_COMMON_HOST_EXCEPTION_H_
#define KYTY_COMMON_HOST_EXCEPTION_H_

#include "common/common.h"

namespace Common::HostException {

enum class ExceptionType { Unknown, AccessViolation, IllegalInstruction };

enum class AccessViolationType { Unknown, Read, Write, Execute };

struct ExceptionInfo {
	ExceptionType       type                   = ExceptionType::Unknown;
	AccessViolationType access_violation_type  = AccessViolationType::Unknown;
	uint64_t            access_violation_vaddr = 0;
	uint64_t            exception_address      = 0;
	uint64_t            rax                    = 0;
	uint64_t            rbx                    = 0;
	uint64_t            rcx                    = 0;
	uint64_t            rdx                    = 0;
	uint64_t            rsi                    = 0;
	uint64_t            rdi                    = 0;
	uint64_t            rbp                    = 0;
	uint64_t            rsp                    = 0;
	uint64_t            r8                     = 0;
	uint64_t            r9                     = 0;
	uint64_t            r10                    = 0;
	uint64_t            r11                    = 0;
	uint64_t            r12                    = 0;
	uint64_t            r13                    = 0;
	uint64_t            r14                    = 0;
	uint64_t            r15                    = 0;
	uint32_t            native_code            = 0;
	// Platform-specific mutable context, valid only for the duration of the handler call.
	void* native_context = nullptr;
};

using Handler = bool (*)(const ExceptionInfo&);

bool InstallHandler(Handler handler);

// Reports an exception nobody handled; must not return.
using Reporter = void (*)(const ExceptionInfo&);

// Installs `report` for the access violations and illegal instructions that reach the end of the
// structured exception handler search (Windows only; elsewhere it returns false). The handler of
// InstallHandler runs first, for every exception, and leaves some of them to that search (see
// LeaveToModuleHandlers).
bool InstallUnhandledReporter(Reporter report);

// Whether a handler should leave an exception it cannot resolve to the structured exception
// handlers of the code that raised it instead of ending the process: a host thread that faults
// inside a loaded module other than the executable (a GPU driver, a system DLL) may be inside
// that module's own __try. Windows only, and only while an unhandled reporter is installed, so
// that an exception nobody handles is still reported.
bool LeaveToModuleHandlers(const ExceptionInfo& info, bool guest_thread);

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
bool InitializeThreadSignalStack();
// The host signal that delivers guest signals. Fault handling blocks it, so a guest handler runs
// only once the faulting instruction is resolved.
int GuestSignalDispatchSignal();
#endif

} // namespace Common::HostException

#endif /* KYTY_COMMON_HOST_EXCEPTION_H_ */
