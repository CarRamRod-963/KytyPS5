#include "loader/runtimeLinker.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/fpEnv.h"
#include "common/guestStoreRing.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/platform/sysDbg.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "loader/elf.h"
#include "loader/gamePatch.h"
#include "loader/guestFaultReport.h"
#include "loader/guestInstructionPatcher.h"
#include "loader/symbolDatabase.h"
#include "loader/x64InstructionEmulator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <memory>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#endif
#endif

namespace Libs::LibKernel {
void SetProgName(const std::string& name);
} // namespace Libs::LibKernel

namespace Loader {

Program::Program() = default;

Program::~Program() = default;

static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return alignment != 0 ? (value + alignment - 1) & ~(alignment - 1) : value;
}

using atexit_func_t = KYTY_SYSV_ABI void (*)();
using entry_func_t  = KYTY_SYSV_ABI void (*)(EntryParams* params, atexit_func_t atexit_func);
using module_ini_fini_func_t = KYTY_SYSV_ABI int (*)(size_t args, const void* argp,
                                                     module_func_t func);

enum class BindType { Unknown, Local, Global, Weak };

struct RelocationInfo {
	bool        resolved   = false;
	BindType    bind       = BindType::Unknown;
	SymbolType  type       = SymbolType::Unknown;
	uint64_t    value      = 0;
	uint64_t    vaddr      = 0;
	uint64_t    base_vaddr = 0;
	std::string name;
	std::string dbg_name;
	bool        bind_self = false;
};

struct StubbedImportRecord {
	uint32_t    index       = 0;
	uint64_t    patch_vaddr = 0;
	uint64_t    thunk_vaddr = 0;
	std::string name;
	SymbolType  type = SymbolType::Unknown;
	BindType    bind = BindType::Unknown;
	std::string program;
};

static std::vector<StubbedImportRecord> g_stubbed_imports;
static std::atomic_uint32_t             g_unresolved_stub_call_log_count {0};
static std::vector<uint64_t>            g_unresolved_stub_thunk_pages;
static uint64_t                         g_unresolved_stub_thunk_offset = 0;
static constexpr uint64_t               UNRESOLVED_STUB_PAGE_SIZE      = 4096;

static KYTY_SYSV_ABI uint64_t UnresolvedImportStub(uint64_t record_id);

static bool PatchGuestMemory64(uint64_t vaddr, uint64_t value) {
	auto* ptr     = reinterpret_cast<uint64_t*>(vaddr);
	bool  changed = (*ptr != value);
	std::memcpy(ptr, &value, sizeof(value));
	return changed;
}

static uint64_t AllocateUnresolvedImportThunk(uint64_t record_id) {
	constexpr uint64_t thunk_size = 34;

	if (g_unresolved_stub_thunk_pages.empty() ||
	    g_unresolved_stub_thunk_offset + thunk_size > UNRESOLVED_STUB_PAGE_SIZE) {
		auto page = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, UNRESOLVED_STUB_PAGE_SIZE, Common::VirtualMemory::Mode::ExecuteReadWrite,
		    "unresolved_import_thunk");
		EXIT_NOT_IMPLEMENTED(page == 0);
		g_unresolved_stub_thunk_pages.push_back(page);
		g_unresolved_stub_thunk_offset = 0;
	}

	auto* code = reinterpret_cast<uint8_t*>(g_unresolved_stub_thunk_pages.back() +
	                                        g_unresolved_stub_thunk_offset);
	g_unresolved_stub_thunk_offset += thunk_size;

	uint8_t bytes[thunk_size] = {
	    0x48, 0x83, 0xec, 0x08,               // sub rsp, 8
	    0x48, 0xbf, 0, 0, 0, 0, 0, 0, 0, 0, // mov rdi, record_id
	    0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, // mov rax, UnresolvedImportStub
	    0xff, 0xd0,                           // call rax
	    0x48, 0x83, 0xc4, 0x08,               // add rsp, 8
	    0x0f, 0x57, 0xc0,                     // xorps xmm0, xmm0
	    0xc3,                                 // ret
	};
	const auto target = reinterpret_cast<uint64_t>(UnresolvedImportStub);
	std::memcpy(bytes + 6, &record_id, sizeof(record_id));
	std::memcpy(bytes + 16, &target, sizeof(target));
	std::memcpy(code, bytes, sizeof(bytes));
	Common::VirtualMemory::FlushInstructionCache(reinterpret_cast<uint64_t>(code), thunk_size);
	return reinterpret_cast<uint64_t>(code);
}

static uint64_t RegisterStubbedImport(uint32_t index, const Program* program,
                                      const RelocationInfo& ri) {
	for (const auto& record: g_stubbed_imports) {
		if (record.patch_vaddr == ri.vaddr) {
			return record.thunk_vaddr;
		}
	}

	StubbedImportRecord record {};
	record.index       = index;
	record.patch_vaddr = ri.vaddr;
	record.name        = ri.name;
	record.type        = ri.type;
	record.bind        = ri.bind;
	record.program     = Common::PathToString(program->file_name);
	g_stubbed_imports.push_back(record);
	const auto record_id                     = g_stubbed_imports.size() - 1;
	const auto thunk                         = AllocateUnresolvedImportThunk(record_id);
	g_stubbed_imports[record_id].thunk_vaddr = thunk;
	return thunk;
}

static KYTY_SYSV_ABI uint64_t UnresolvedImportStub(uint64_t record_id) {
	Common::DebugCounters::Add(Common::DebugCounters::Counter::UnresolvedImportCalls);
	const auto log_index = g_unresolved_stub_call_log_count.fetch_add(1);
	if (log_index < 1024) {
		if (record_id < g_stubbed_imports.size()) {
			const auto& record = g_stubbed_imports[record_id];
			printf("Unresolved import stub called: %s\n", record.name.c_str());
			LOGF("Unresolved import stub called [%u]: patch_vaddr=0x%016" PRIx64
			     " jmprela_index=%" PRIu32 " symbol=%s type=%s bind=%s program=%s\n",
			     log_index, record.patch_vaddr, record.index, record.name.c_str(),
			     magic_enum::enum_name(record.type), magic_enum::enum_name(record.bind),
			     record.program.c_str());
		} else {
			printf("Unresolved import stub called: <bad-record>\n");
			LOGF("Unresolved import stub called [%u]: record_id=%" PRIu64 " symbol=<bad-record>\n",
			     log_index, record_id);
		}
	}
	return 0;
}

constexpr uint64_t SYSTEM_RESERVED  = 0x800000000u;
constexpr uint64_t CODE_BASE_INCR   = 0x010000000u;
constexpr uint64_t INVALID_OFFSET   = 0x040000000u;
constexpr uint64_t CODE_BASE_OFFSET = 0x100000000u;
constexpr uint64_t INVALID_MEMORY   = SYSTEM_RESERVED + INVALID_OFFSET;

static uint64_t g_desired_base_addr = SYSTEM_RESERVED + CODE_BASE_OFFSET;
static uint64_t g_invalid_memory    = 0;

static Program* g_tls_main_program = nullptr;

static KYTY_SYSV_ABI void RunEntry(uint64_t addr, EntryParams* params, atexit_func_t atexit_func,
                                   void* stack_top) {
#if defined(__x86_64__) || defined(_M_X64)
	auto* func = reinterpret_cast<entry_func_t>(addr);

	if (stack_top != nullptr) {
		const auto aligned_stack_top =
		    reinterpret_cast<uintptr_t>(stack_top) & ~static_cast<uintptr_t>(0x0f);
		const auto guest_rsp = aligned_stack_top - 2u * sizeof(uintptr_t);
		const auto guest_rbp = guest_rsp;

		auto* guest_root_frame = reinterpret_cast<uintptr_t*>(guest_rbp);
		guest_root_frame[0]    = 0;
		guest_root_frame[1]    = 0;

#if defined(__APPLE__)
		// Clang on macOS can allocate plain "r" inputs to r12/r13, which the template
		// clobbers before consuming them. Pin the inputs to registers the SysV guest
		// preserves without changing register allocation on Windows or Linux.
		register entry_func_t func_reg asm("rbx")      = func;
		register uintptr_t    guest_rsp_reg asm("r14") = guest_rsp;
		register uintptr_t    guest_rbp_reg asm("r15") = guest_rbp;
#endif

#if defined(__APPLE__)
		asm volatile(
		    "pushq %%r12\n\t"
		    "pushq %%r13\n\t"
		    "movq %%rsp, %%r12\n\t"
		    "movq %%rbp, %%r13\n\t"
		    "movq %[guest_rsp], %%rsp\n\t"
		    "movq %[guest_rbp], %%rbp\n\t"
		    "callq *%[func]\n\t"
		    "movq %%r13, %%rbp\n\t"
		    "movq %%r12, %%rsp\n\t"
		    "popq %%r13\n\t"
		    "popq %%r12\n\t"
		    :
		    : [func] "r"(func_reg), "D"(params),
		      "S"(atexit_func), [guest_rsp] "r"(guest_rsp_reg), [guest_rbp] "r"(guest_rbp_reg)
		    : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2",
		      "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
		      "xmm13", "xmm14", "xmm15");
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Windows stack probes use the TEB stack limits during the guest stack switch.
		// bounds, which describe the host stack and are invalid while RSP is in guest memory.
		register entry_func_t func_reg asm("rbx")     = func;
		register uintptr_t    guest_rsp_reg asm("r8") = guest_rsp;
		register uintptr_t    guest_rbp_reg asm("r9") = guest_rbp;
		asm volatile("pushq %%r12\n\t"
		             "pushq %%r13\n\t"
		             "pushq %%r14\n\t"
		             "pushq %%r15\n\t"
		             "movq %%gs:0x08, %%r14\n\t"
		             "movq %%gs:0x10, %%r15\n\t"
		             "xorq %%rcx, %%rcx\n\t"
		             "movq %%rcx, %%gs:0x08\n\t"
		             "movq %%rcx, %%gs:0x10\n\t"
		             "movq %%rsp, %%r12\n\t"
		             "movq %%rbp, %%r13\n\t"
		             "movq %[guest_rsp], %%rsp\n\t"
		             "movq %[guest_rbp], %%rbp\n\t"
		             "callq *%[func]\n\t"
		             "movq %%r13, %%rbp\n\t"
		             "movq %%r12, %%rsp\n\t"
		             "movq %%r14, %%gs:0x08\n\t"
		             "movq %%r15, %%gs:0x10\n\t"
		             "popq %%r15\n\t"
		             "popq %%r14\n\t"
		             "popq %%r13\n\t"
		             "popq %%r12\n\t"
		             : [guest_rsp] "+r"(guest_rsp_reg), [guest_rbp] "+r"(guest_rbp_reg)
		             : [func] "r"(func_reg), "D"(params), "S"(atexit_func)
		             : "cc", "memory", "rax", "rcx", "rdx", "r10", "r11", "xmm0", "xmm1", "xmm2",
		               "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",
		               "xmm12", "xmm13", "xmm14", "xmm15");
#else
		// Clobbers prevent inputs from being allocated to r12/r13.
		asm volatile("movq %%rsp, %%r12\n\t"
		             "movq %%rbp, %%r13\n\t"
		             "movq %[guest_rsp], %%rsp\n\t"
		             "movq %[guest_rbp], %%rbp\n\t"
		             "callq *%[func]\n\t"
		             "movq %%r13, %%rbp\n\t"
		             "movq %%r12, %%rsp\n\t"
		             :
		             : [func] "r"(func), "D"(params),
		               "S"(atexit_func), [guest_rsp] "r"(guest_rsp), [guest_rbp] "r"(guest_rbp)
		             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "r12", "r13",
		               "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8",
		               "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
#endif
		return;
	}

	uintptr_t guest_root_frame[2] = {};

#if defined(__APPLE__)
	register entry_func_t func_reg asm("rbx")      = func;
	register uintptr_t    guest_rbp_reg asm("r14") = reinterpret_cast<uintptr_t>(guest_root_frame);
#endif

#if defined(__APPLE__) || KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	asm volatile("pushq %%r12\n\t"
	             "pushq %%r13\n\t"
	             "movq %%rbp, %%r12\n\t"
	             "movq %[guest_rbp], %%rbp\n\t"
	             "callq *%[func]\n\t"
	             "movq %%r12, %%rbp\n\t"
	             "popq %%r13\n\t"
	             "popq %%r12\n\t"
	             :
#if defined(__APPLE__)
	             : [func] "r"(func_reg), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_rbp_reg)
#else
	             : [func] "r"(func), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_root_frame)
#endif
	             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1",
	               "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",
	               "xmm12", "xmm13", "xmm14", "xmm15");
#else
	// Keep inputs out of r12.
	asm volatile("movq %%rbp, %%r12\n\t"
	             "movq %[guest_rbp], %%rbp\n\t"
	             "callq *%[func]\n\t"
	             "movq %%r12, %%rbp\n\t"
	             :
	             : [func] "r"(func), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_root_frame)
	             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "r12", "xmm0",
	               "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
	               "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
#endif
#else
	(void)stack_top;
	reinterpret_cast<entry_func_t>(addr)(params, atexit_func);
#endif
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
struct MainEntryStackTestState {
	bool      called = false;
	uintptr_t rsp    = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t teb_stack_base  = UINTPTR_MAX;
	uintptr_t teb_stack_limit = UINTPTR_MAX;
#endif
};

static KYTY_SYSV_ABI void TestMainEntryStackCallback(EntryParams* params,
                                                     atexit_func_t /*atexit_func*/) {
	auto* state = reinterpret_cast<MainEntryStackTestState*>(const_cast<char*>(params->argv[0]));
	asm volatile("pushq %%r15\n\t"
	             "pushq %%r14\n\t"
	             "popq %%r14\n\t"
	             "popq %%r15\n\t"
	             :
	             :
	             : "memory");
	asm volatile("movq %%rsp, %0" : "=r"(state->rsp) : : "memory");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	asm volatile("movq %%gs:0x08, %0\n\t"
	             "movq %%gs:0x10, %1\n\t"
	             : "=r"(state->teb_stack_base), "=r"(state->teb_stack_limit)
	             :
	             : "memory");
#endif
	state->called = true;
}

bool TestMainEntryUsesGuestStack() {
	constexpr uint64_t stack_size = 0x10000;
	const auto         stack_base = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    0, stack_size, Common::VirtualMemory::Mode::ReadWrite, "main_entry_stack_test");
	if (stack_base == 0) {
		return false;
	}

	MainEntryStackTestState state {};
	EntryParams             params {};
	params.argv[0] = reinterpret_cast<const char*>(&state);
	std::memset(reinterpret_cast<void*>(stack_base), 0xcd, stack_size);
	auto* root_frame = reinterpret_cast<const uintptr_t*>(stack_base + stack_size) - 2;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t original_teb_stack_base  = 0;
	uintptr_t original_teb_stack_limit = 0;
	asm volatile("movq %%gs:0x08, %0\n\t"
	             "movq %%gs:0x10, %1\n\t"
	             : "=r"(original_teb_stack_base), "=r"(original_teb_stack_limit)
	             :
	             : "memory");
#endif

	RunEntry(reinterpret_cast<uint64_t>(TestMainEntryStackCallback), &params, nullptr,
	         reinterpret_cast<void*>(stack_base + stack_size));

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uintptr_t restored_teb_stack_base  = 0;
	uintptr_t restored_teb_stack_limit = 0;
	asm volatile("movq %%gs:0x08, %0\n\t"
	             "movq %%gs:0x10, %1\n\t"
	             : "=r"(restored_teb_stack_base), "=r"(restored_teb_stack_limit)
	             :
	             : "memory");
	const bool teb_ok = state.teb_stack_base == 0 && state.teb_stack_limit == 0 &&
	                    restored_teb_stack_base == original_teb_stack_base &&
	                    restored_teb_stack_limit == original_teb_stack_limit;
#else
	constexpr bool teb_ok = true;
#endif

	const bool rsp_ok  = state.rsp >= stack_base && state.rsp < stack_base + stack_size;
	const bool root_ok = root_frame[0] == 0 && root_frame[1] == 0;
	const bool freed   = Libs::LibKernel::Memory::FreeGuestMemory(stack_base, stack_size);
	return state.called && rsp_ok && root_ok && teb_ok && freed;
}

bool TestModuleRelocationUsesWritableHostMapping() {
	constexpr uint64_t page_size = 0x4000;
	constexpr uint64_t value     = 0x4b59545950415443;
	const auto         base      = Libs::LibKernel::Memory::AllocateProgramMemory(
	    0, page_size, Common::VirtualMemory::Mode::ReadWrite, "host_only_patch_test");
	if (base == 0) {
		return false;
	}
	Libs::LibKernel::Memory::SetProgramMemoryProtection(base, page_size,
	                                                    Common::VirtualMemory::Mode::Read);

	Libs::LibKernel::Memory::VirtualQueryInfo before {};
	Libs::LibKernel::Memory::VirtualQueryInfo after {};
	const bool                                before_ok =
	    Libs::LibKernel::Memory::KernelVirtualQuery(reinterpret_cast<const void*>(base), 0, &before,
	                                                sizeof(before)) == 0;
	const bool changed  = PatchGuestMemory64(base, value);
	const bool after_ok = Libs::LibKernel::Memory::KernelVirtualQuery(
	                          reinterpret_cast<const void*>(base), 0, &after, sizeof(after)) == 0;
	const bool value_ok = *reinterpret_cast<const uint64_t*>(base) == value;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION mbi {};
	const bool               host_mode_ok =
	    VirtualQuery(reinterpret_cast<const void*>(base), &mbi, sizeof(mbi)) != 0 &&
	    mbi.Protect == PAGE_READWRITE;
#else
	constexpr bool host_mode_ok = true;
#endif
	const bool freed = Libs::LibKernel::Memory::FreeGuestMemory(base, page_size);

	return before_ok && after_ok && changed && value_ok && host_mode_ok && freed &&
	       before.protection == after.protection;
}
#endif

static uint64_t GetAlignedSize(const Elf64_Phdr* p) {
	return (p->p_align != 0 ? (p->p_memsz + (p->p_align - 1)) & ~(p->p_align - 1) : p->p_memsz);
}

static void DbgDumpSymbols(const std::string& folder, Elf64_Sym* symbols, uint64_t size,
                           const char* names) {
	auto folder_str = Common::FixDirectorySlash(folder);

	Common::File::CreateDirectories(folder_str);

	Common::File f;
	f.Create(folder_str + "symbols.txt");

	for (auto* sym = symbols;
	     reinterpret_cast<uint8_t*>(sym) < reinterpret_cast<uint8_t*>(symbols) + size; sym++) {
		f.Printf("----\n");
		f.Printf("st_name = %" PRIu32 ", %s\n", sym->st_name, names + sym->st_name);
		f.Printf("st_info = 0x%02" PRIx8 "\n", sym->st_info);
		f.Printf("st_other = 0x%02" PRIx8 "\n", sym->st_other);
		f.Printf("st_shndx = 0x%04" PRIx16 "\n", sym->st_shndx);
		f.Printf("st_value = 0x%016" PRIx64 "\n", sym->st_value);
		f.Printf("st_size = %" PRIu64 "\n", sym->st_size);
	}

	f.Close();
}

static void DbgDumpRela(const std::string& folder, Elf64_Rela* records, uint64_t size,
                        const char* /*names*/, const char* file_name) {
	auto folder_str = Common::FixDirectorySlash(folder);

	Common::File::CreateDirectories(folder_str);

	Common::File f;
	f.Create(folder_str + file_name);

	for (auto* r = records;
	     reinterpret_cast<uint8_t*>(r) < reinterpret_cast<uint8_t*>(records) + size; r++) {
		f.Printf("----\n"
		         "r_offset = 0x%016" PRIx64 "\n"
		         "r_info = 0x%016" PRIx64 "\n"
		         "r_addend = %" PRId64 "\n",
		         r->r_offset, r->r_info, r->r_addend);
	}

	f.Close();
}

static Common::VirtualMemory::Mode GetMode(Elf64_Word flags) {
	switch (flags) {
		case PF_R: return Common::VirtualMemory::Mode::Read;
		case PF_W: return Common::VirtualMemory::Mode::Write;
		case PF_R | PF_W: return Common::VirtualMemory::Mode::ReadWrite;
		case PF_X: return Common::VirtualMemory::Mode::Execute;
		case PF_X | PF_R: return Common::VirtualMemory::Mode::ExecuteRead;
		case PF_X | PF_W: return Common::VirtualMemory::Mode::ExecuteWrite;
		case PF_X | PF_W | PF_R: return Common::VirtualMemory::Mode::ExecuteReadWrite;

		default: return Common::VirtualMemory::Mode::NoAccess;
	}
}

struct GuestStackFrame {
	GuestStackFrame* next;
	uintptr_t        return_address;
};

static bool IsReadableRange(uint64_t addr, uint64_t size);
static void AppendHostBacktrace(const Common::HostException::ExceptionInfo& info,
                                Loader::GuestFaultText*                     out);

static int WalkGuestStack(uint64_t rbp, uint64_t rsp, void** stack, int capacity) {
	constexpr uintptr_t STACK_SIZE = 1024u * 1024u;
	const uintptr_t     code_start = SYSTEM_RESERVED + CODE_BASE_OFFSET;
	const uintptr_t     stack_end  = rsp + STACK_SIZE;
	if (stack == nullptr || capacity <= 0 || rsp == 0 || rbp < rsp || stack_end < rsp ||
	    g_desired_base_addr <= code_start) {
		return 0;
	}

	auto* frame = reinterpret_cast<GuestStackFrame*>(rbp);

	int depth = 0;
	while (depth < capacity) {
		const auto frame_addr = reinterpret_cast<uintptr_t>(frame);
		if (frame_addr < rsp || frame_addr > stack_end - sizeof(GuestStackFrame) ||
		    !IsReadableRange(frame_addr, sizeof(GuestStackFrame)) ||
		    frame->return_address < code_start || frame->return_address >= g_desired_base_addr) {
			break;
		}
		stack[depth++] = reinterpret_cast<void*>(frame->return_address);
		if (reinterpret_cast<uintptr_t>(frame->next) <= frame_addr) {
			break;
		}
		frame = frame->next;
	}
	return depth;
}

// Probe diagnostic ranges without raising another fault.
static bool IsReadableRange(uint64_t addr, uint64_t size) {
	if (addr == 0 || size == 0) {
		return false;
	}

	const uint64_t end = addr + size;
	if (end < addr) {
		return false;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uint64_t current = addr;
	while (current < end) {
		MEMORY_BASIC_INFORMATION mbi {};
		if (VirtualQuery(reinterpret_cast<const void*>(current), &mbi, sizeof(mbi)) == 0 ||
		    mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
			return false;
		}
		const auto region_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
		if (region_end <= current) {
			return false;
		}
		current = std::min(region_end, end);
	}
#elif defined(__APPLE__)
	// Walk the Mach regions covering the range and require read permission. The fatal
	// report dumps memory behind raw register values, and a fault inside the reporter
	// re-enters the signal handler and wedges the reporting thread.
	uint64_t current = addr;
	while (current < end) {
		mach_vm_address_t              region_addr = current;
		mach_vm_size_t                 region_size = 0;
		vm_region_basic_info_data_64_t info {};
		mach_msg_type_number_t         count       = VM_REGION_BASIC_INFO_COUNT_64;
		mach_port_t                    object_name = MACH_PORT_NULL;
		if (mach_vm_region(mach_task_self(), &region_addr, &region_size, VM_REGION_BASIC_INFO_64,
		                   reinterpret_cast<vm_region_info_t>(&info), &count,
		                   &object_name) != KERN_SUCCESS ||
		    region_addr > current || (info.protection & VM_PROT_READ) == 0) {
			return false;
		}
		current = region_addr + region_size;
	}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	const auto page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
	if (page_size == 0) {
		return false;
	}

	for (uint64_t current = addr; current < end;) {
		uint8_t probe = 0;

		iovec local {&probe, sizeof(probe)};
		iovec remote {reinterpret_cast<void*>(current), sizeof(probe)};

		if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) !=
		    static_cast<ssize_t>(sizeof(probe))) {
			return false;
		}

		const uint64_t next = (current & ~(page_size - 1)) + page_size;
		if (next <= current) { // wrapped at the top of the address space
			break;
		}
		current = next;
	}
#else
	(void)end;
#endif
	return true;
}

// Host frames above the fault, as module+offset: a fault raised from inside the host (a
// software-originated exception in ntdll, a library call on a guest thread) names no guest
// code, and the unwinder walks through the exception dispatcher back to the frame that
// faulted first.
static void AppendHostBacktrace(const Common::HostException::ExceptionInfo& info,
                                Loader::GuestFaultText*                     out) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (info.native_context == nullptr) {
		return;
	}
	CONTEXT context = *static_cast<const CONTEXT*>(info.native_context);
	out->Append("host frames:\n");
	for (int depth = 0; depth < 48; depth++) {
		const auto pc                    = static_cast<uint64_t>(context.Rip);
		char       module_name[MAX_PATH] = "";
		void*      module_base           = nullptr;
		if (RtlPcToFileHeader(reinterpret_cast<void*>(pc), &module_base) != nullptr) {
			GetModuleFileNameA(static_cast<HMODULE>(module_base), module_name, sizeof(module_name));
		}
		const char* base_name = std::strrchr(module_name, '\\');
		base_name             = base_name != nullptr ? base_name + 1 : module_name;
		if (module_base != nullptr) {
			out->Append("  {:2} {}+0x{:x} (0x{:016x})\n", depth, base_name,
			            pc - reinterpret_cast<uint64_t>(module_base), pc);
		} else {
			out->Append("  {:2} 0x{:016x}\n", depth, pc);
		}
		DWORD64 image_base = 0;
		auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
		if (function == nullptr) {
			// A leaf function or code without unwind data (guest code): the return address
			// is at the stack pointer.
			if (!IsReadableRange(context.Rsp, sizeof(uint64_t))) {
				break;
			}
			context.Rip = *reinterpret_cast<const uint64_t*>(context.Rsp);
			context.Rsp += sizeof(uint64_t);
		} else {
			void*   handler_data = nullptr;
			DWORD64 establisher  = 0;
			RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
			                 &handler_data, &establisher, nullptr);
		}
		if (context.Rip == 0 || !IsReadableRange(context.Rip, 1)) {
			break;
		}
	}
#else
	(void)info;
	(void)out;
#endif
}

[[noreturn]] static void
ReportUnhandledHostException(const Common::HostException::ExceptionInfo& exception_info);

// Loaded guest modules for fault reports, which cannot take the linker's mutex.
static GuestModuleMap g_guest_modules;
// Guest threads, for the report of where the others were at a fatal fault.
static GuestThreadRegistry g_guest_threads;
// The faults this thread had resolved, for the report of a fatal fault that follows.
static thread_local HandledFaultRing g_handled_faults;

GuestThreadRegistry& GuestThreads() {
	return g_guest_threads;
}

static uint64_t MonotonicNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

static void RecordHandledFault(const Common::HostException::ExceptionInfo& info,
                               HandledFaultKind kind, uint64_t address) {
	g_handled_faults.Record({.address = address,
	                         .pc      = info.exception_address,
	                         .rsp     = info.rsp,
	                         .time_ns = MonotonicNs(),
	                         .kind    = kind,
	                         .stores  = Common::GuestStores().Count()});
}

// Copies the 16-byte rows of [start, start + rows * 16) that can be read, without faulting even
// when another thread changes a page's protection meanwhile, and returns the mask of rows read.
static uint64_t CopyReadableRows(uint64_t start, uint8_t* out, size_t rows) {
	uint64_t readable = 0;
	for (size_t row = 0; row < rows && row < 64; row++) {
		const uint64_t address = start + row * 16;
		if (address < start || !IsReadableRange(address, 16)) {
			continue;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SIZE_T copied = 0;
		if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address),
		                      out + row * 16, 16, &copied) == 0 ||
		    copied != 16) {
			continue;
		}
#else
		std::memcpy(out + row * 16, reinterpret_cast<const void*>(address), 16);
#endif
		readable |= uint64_t {1} << row;
	}
	return readable;
}

static bool MayBePointer(uint64_t value) {
	// Small integers and non-canonical values are no pointers.
	return value >= 0x10000 && value <= 0x00007fffffffffffull;
}

// The window of `before` bytes below and `after` bytes from `value`, when any of it is readable.
static GuestFaultMemory DumpAround(std::string_view name, uint64_t value, uint64_t before,
                                   uint64_t after, uint8_t* buffer) {
	GuestFaultMemory window;
	window.name  = name;
	window.value = value;
	if (!MayBePointer(value) || value < before) {
		return window;
	}
	window.start         = value - before;
	const size_t rows    = static_cast<size_t>((before + after) / 16);
	window.readable_rows = CopyReadableRows(window.start, buffer, rows);
	window.bytes         = {buffer, rows * 16};
	return window;
}

// Storage of the second part of a fatal report: static, so that a report from a thread near the
// end of its stack does not need tens of KiB of it. One report at a time.
struct GuestFaultMemoryStorage {
	static constexpr size_t Registers    = 16;
	static constexpr size_t WindowBytes  = GuestFaultPointerBefore + GuestFaultPointerAfter;
	static constexpr size_t ThreadsMax   = GuestThreadRegistry::Capacity;
	static constexpr size_t HostNameSize = 64;

	uint8_t              register_bytes[Registers][WindowBytes];
	uint8_t              stack_bytes[GuestFaultStackBefore + GuestFaultStackAfter];
	uint8_t              frame_bytes[GuestFaultFrameBefore + GuestFaultFrameAfter];
	static constexpr size_t InterestMax = 2 * (Registers + GuestFaultIndirectMax) + 1;
	static constexpr size_t CallerScan  = 0x1000;

	uint8_t              indirect_bytes[GuestFaultIndirectMax][GuestFaultIndirectAfter];
	std::array<char, GuestFaultIndirectName> indirect_names[GuestFaultIndirectMax];
	GuestFaultMemory     registers[Registers];
	GuestFaultMemory     stack[2];
	GuestFaultMemory     indirect[GuestFaultIndirectMax];
	GuestThreadState     threads[ThreadsMax];
	char                 thread_names[ThreadsMax][GuestThreadRegistry::NameSize];
	char                 host_modules[ThreadsMax][HostNameSize];
	uint8_t              caller_bytes[CallerScan];
	Common::GuestStore   stores[Common::GuestStoreRing::Capacity];
	GuestFaultInterest   interest[InterestMax];
	GuestFaultMemoryText text;
};
static GuestFaultMemoryStorage g_fault_memory;
static Loader::FaultReportGate g_fault_memory_gate;

// Pointers found in the first GuestFaultPointerScan bytes of the register dumps (first qword of
// each register, then the second, ...) and in the frame below rbp (nearest first), dumped once
// more when they point to readable memory no dump holds yet, off this thread's stack and outside
// the guest modules; then the pointers in the first GuestFaultPointerScan bytes of those dumps
// once more (a second level).
static size_t DumpIndirect(GuestFaultMemoryStorage* s, uint64_t rsp) {
	return CollectGuestFaultIndirect(
	    s->registers, s->stack, s->indirect, s->indirect_names,
	    [&](uint64_t target) {
		    return MayBePointer(target) && !(target + 0x1000 >= rsp && target < rsp + 0x100000) &&
		           g_guest_modules.Find(target).module.empty();
	    },
	    [&](std::string_view name, uint64_t target, size_t index) {
		    return DumpAround(name, target, 0, GuestFaultIndirectAfter, s->indirect_bytes[index]);
	    });
}

// The first return address into a guest module on the stack above rsp, for a thread that is in
// host code (an HLE function called by guest code keeps running on the guest stack).
[[maybe_unused]] static uint64_t FindGuestCaller(GuestFaultMemoryStorage* s, uint64_t rsp) {
	constexpr size_t rows     = GuestFaultMemoryStorage::CallerScan / 16;
	const uint64_t   readable = CopyReadableRows(rsp & ~uint64_t {15}, s->caller_bytes, rows);
	for (size_t row = 0; row < rows; row++) {
		if (((readable >> row) & 1u) == 0) {
			continue;
		}
		for (size_t word = 0; word < 2; word++) {
			uint64_t value = 0;
			std::memcpy(&value, s->caller_bytes + row * 16 + word * 8, sizeof(value));
			if (!g_guest_modules.Find(value).module.empty()) {
				return value;
			}
		}
	}
	return 0;
}

// Suspends the other registered guest threads together, reads where they are and resumes them
// before anything else happens (no lock is taken while they are suspended).
static size_t SnapshotGuestThreads(GuestFaultMemoryStorage* s, size_t* total) {
	size_t count = 0;
	*total       = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const uint64_t self = GetCurrentThreadId();
	g_guest_threads.ForEach([&](uint64_t id, std::string_view name) {
		if (id == self) {
			return;
		}
		(*total)++;
		if (count >= GuestFaultMemoryStorage::ThreadsMax) {
			return;
		}
		auto& state = s->threads[count];
		state       = {};
		const auto length =
		    std::min(name.size(), static_cast<size_t>(GuestThreadRegistry::NameSize - 1));
		std::memcpy(s->thread_names[count], name.data(), length);
		s->thread_names[count][length] = '\0';
		state.name                     = {s->thread_names[count], length};
		state.host_thread_id           = id;
		count++;
	});
	static HANDLE handles[GuestFaultMemoryStorage::ThreadsMax];
	static bool   suspended[GuestFaultMemoryStorage::ThreadsMax];
	for (size_t i = 0; i < count; i++) {
		handles[i]   = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE,
		                          static_cast<DWORD>(s->threads[i].host_thread_id));
		suspended[i] = handles[i] != nullptr && SuspendThread(handles[i]) != static_cast<DWORD>(-1);
	}
	for (size_t i = 0; i < count; i++) {
		CONTEXT context {};
		context.ContextFlags = CONTEXT_CONTROL;
		if (suspended[i] && GetThreadContext(handles[i], &context) != 0) {
			s->threads[i].rip  = context.Rip;
			s->threads[i].rsp  = context.Rsp;
			s->threads[i].read = true;
		}
	}
	for (size_t i = 0; i < count; i++) {
		if (suspended[i]) {
			ResumeThread(handles[i]);
		}
		if (handles[i] != nullptr) {
			CloseHandle(handles[i]);
		}
	}
	// Host code (an HLE function, the emulator, a system DLL) is named by its module.
	for (size_t i = 0; i < count; i++) {
		auto& state = s->threads[i];
		void* base  = nullptr;
		if (!state.read || !g_guest_modules.Find(state.rip).module.empty() ||
		    RtlPcToFileHeader(reinterpret_cast<void*>(state.rip), &base) == nullptr) {
			continue;
		}
		char path[MAX_PATH] = "";
		GetModuleFileNameA(static_cast<HMODULE>(base), path, sizeof(path));
		const char* file  = std::strrchr(path, '\\');
		file              = file != nullptr ? file + 1 : path;
		const auto length = std::min(
		    std::strlen(file), static_cast<size_t>(GuestFaultMemoryStorage::HostNameSize - 1));
		std::memcpy(s->host_modules[i], file, length);
		s->host_modules[i][length] = '\0';
		state.host_module          = {s->host_modules[i], length};
		state.host_offset          = state.rip - reinterpret_cast<uint64_t>(base);
		state.guest_caller         = FindGuestCaller(s, state.rsp);
	}
#else
	(void)s;
#endif
	return count;
}

// Second part of a fatal report: the memory the registers point to, the faults this thread had
// resolved before and where the other guest threads were. Together they tell a red zone or a
// saved value an exception frame overwrote from data the program got wrong, and show a thread
// that was working on the same data at the same moment.
static void ReportGuestFaultMemory(const Common::HostException::ExceptionInfo& info) {
	if (!g_fault_memory_gate.Begin()) {
		Log::WriteToConsoleAndLog("--- Guest memory at the registers: left out (another thread is "
		                          "reporting a fault) ---\n");
		// This thread goes on to end the process: two seconds at most for the other report.
		(void)g_fault_memory_gate.WaitForReport(
		    2000, [](uint32_t ms) { Common::Thread::SleepMicro(ms * 1000u); });
		return;
	}
	const uint64_t                         now_ns = MonotonicNs();
	auto*                                  s      = &g_fault_memory;
	const std::pair<const char*, uint64_t> registers[GuestFaultMemoryStorage::Registers] = {
	    {"fault address", info.access_violation_vaddr},
	    {"rax", info.rax},
	    {"rbx", info.rbx},
	    {"rcx", info.rcx},
	    {"rdx", info.rdx},
	    {"rsi", info.rsi},
	    {"rdi", info.rdi},
	    {"rbp", info.rbp},
	    {"r8", info.r8},
	    {"r9", info.r9},
	    {"r10", info.r10},
	    {"r11", info.r11},
	    {"r12", info.r12},
	    {"r13", info.r13},
	    {"r14", info.r14},
	    {"r15", info.r15},
	};
	for (size_t i = 0; i < GuestFaultMemoryStorage::Registers; i++) {
		s->registers[i] =
		    DumpAround(registers[i].first, registers[i].second, GuestFaultPointerBefore,
		               GuestFaultPointerAfter, s->register_bytes[i]);
	}
	s->stack[0] =
	    DumpAround("rsp", info.rsp, GuestFaultStackBefore, GuestFaultStackAfter, s->stack_bytes);
	s->stack[0].prefix   = "rsp";
	s->stack[0].red_zone = true;
	s->stack[1] =
	    DumpAround("rbp", info.rbp, GuestFaultFrameBefore, GuestFaultFrameAfter, s->frame_bytes);
	s->stack[1].prefix         = "rbp";
	const size_t indirect      = DumpIndirect(s, info.rsp);
	size_t       threads_total = 0;
	const size_t threads       = SnapshotGuestThreads(s, &threads_total);

	GuestFaultMemoryReport report;
	report.info          = &info;
	report.registers     = s->registers;
	report.stack         = s->stack;
	report.indirect      = {s->indirect, indirect};
	report.handled       = &g_handled_faults;
	report.modules       = &g_guest_modules;
	report.now_ns        = now_ns;
	report.threads       = {s->threads, threads};
	report.threads_total = threads_total;
	// The emulator's own stores into guest memory, newest last, and the ranges whose stores the
	// report lists: around the fault, the registers and the dumps.
	const auto& ring          = Common::GuestStores();
	report.store_tsc_now      = Common::GuestStoreClock();
	report.store_ticks_per_us = Common::GuestStoreClockTicksPerUs();
	report.stores_total       = ring.Count();
	report.stores   = {s->stores, ring.Snapshot(s->stores, Common::GuestStoreRing::Capacity)};
	report.interest = {s->interest,
	                   CollectGuestFaultInterest(info.access_violation_vaddr, s->registers,
	                                             report.indirect, s->interest)};
	FormatGuestFaultMemory(report, &s->text);
	Log::WriteToConsoleAndLog(s->text.View());
	g_fault_memory_gate.Finish();
}

bool KytyExceptionHandler(const Common::HostException::ExceptionInfo& exception_info) {
	const auto* info = &exception_info;

	if (info->type == Common::HostException::ExceptionType::IllegalInstruction &&
	    Loader::X64InstructionEmulator::TryEmulate(info->native_context)) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::EmulatedInstructions);
		RecordHandledFault(*info, HandledFaultKind::EmulatedInstruction, 0);
		return true;
	}

	if (info->type == Common::HostException::ExceptionType::AccessViolation) {
		using CoreAccess = Common::HostException::AccessViolationType;
		using GpuAccess  = Libs::Graphics::PageFaultAccess;
		GpuAccess access;
		switch (info->access_violation_type) {
			case CoreAccess::Read: access = GpuAccess::Read; break;
			case CoreAccess::Write: access = GpuAccess::Write; break;
			case CoreAccess::Execute: access = GpuAccess::Execute; break;
			case CoreAccess::Unknown: return false;
		}
		const auto access_kind = access == GpuAccess::Read    ? HandledFaultKind::Read
		                         : access == GpuAccess::Write ? HandledFaultKind::Write
		                                                      : HandledFaultKind::Execute;
		if (const auto length = Libs::Graphics::ShaderRecompiler::IR::SrtWalkerLoadLength(
		        info->exception_address)) {
			// The backing serves an SRT walker load, so false sharing with GPU-written bytes
			// needs no readback. Otherwise the load retries once its fault is handled, or
			// reads zero.
			uint32_t value = 0;
			if (!Libs::LibKernel::Memory::TryReadBufferBacking(info->r10, &value, sizeof(value)) &&
			    Libs::LibKernel::Memory::HandleGpuFault(access, info->access_violation_vaddr)) {
				Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuFaults);
				RecordHandledFault(*info, access_kind, info->access_violation_vaddr);
				return true;
			}
			Loader::X64InstructionEmulator::CompleteLoad(info->native_context, length, value);
			RecordHandledFault(*info, HandledFaultKind::SkippedLoad, info->access_violation_vaddr);
			return true;
		}
		if (Libs::LibKernel::Memory::HandleGpuFault(access, info->access_violation_vaddr)) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuFaults);
			RecordHandledFault(*info, access_kind, info->access_violation_vaddr);
			return true;
		}
	}
	// A host thread faulting inside a driver or system DLL may be inside that module's own __try;
	// the unhandled-exception filter reports the exception if nobody handles it.
	const bool guest_thread = Libs::LibKernel::PthreadSelfOrNull() != nullptr;
	if (Common::HostException::LeaveToModuleHandlers(*info, guest_thread)) {
		return false;
	}
	ReportUnhandledHostException(*info);
}

static void
ReportUnhandledHostException(const Common::HostException::ExceptionInfo& exception_info) {
	const auto* info = &exception_info;
	// Report whatever guest context can be read safely before terminating: which guest thread
	// faulted, the register file, the faulting code bytes and the top of its stack.
	{
		char thread_name[64] = "(host thread)";
		if (auto self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
			if (Libs::LibKernel::PthreadGetname(self, thread_name) != 0) {
				std::snprintf(thread_name, sizeof(thread_name), "(unnamed guest thread)");
			}
		}
		constexpr size_t         CodeBytes  = 2 * Loader::GuestFaultCodeBefore;
		constexpr size_t         StackWords = 32;
		constexpr int            MaxFrames  = 16;
		Loader::GuestFaultReport report;
		report.thread = thread_name;
		report.info   = info;
		if (IsReadableRange(info->exception_address - Loader::GuestFaultCodeBefore, CodeBytes)) {
			report.code = {reinterpret_cast<const uint8_t*>(info->exception_address -
			                                                Loader::GuestFaultCodeBefore),
			               CodeBytes};
		}
		if (IsReadableRange(info->rsp, StackWords * sizeof(uint64_t))) {
			report.stack = {reinterpret_cast<const uint64_t*>(info->rsp), StackWords};
		}
		void*     frames[MaxFrames] {};
		uint64_t  guest_frames[MaxFrames] {};
		const int depth = WalkGuestStack(info->rbp, info->rsp, frames, MaxFrames);
		for (int i = 0; i < depth; i++) {
			guest_frames[i] = reinterpret_cast<uint64_t>(frames[i]);
		}
		report.guest_frames = {guest_frames, static_cast<size_t>(std::max(depth, 0))};
		report.copy         = Loader::FindGuestFunctionCopy(info->exception_address);
		// Formatted without allocating (the fault may have hit inside the host heap). The log
		// file is what testers send: the report must not exist on the console only.
		Loader::GuestFaultText text;
		Loader::FormatGuestFaultReport(report, &text);
		AppendHostBacktrace(*info, &text);
		Log::WriteToConsoleAndLog(text.View());
	}
	ReportGuestFaultMemory(*info);
	EXIT("Unhandled host exception: type=%u code=%u pc=0x%016" PRIx64
	     " access=%u address=0x%016" PRIx64 "\n",
	     static_cast<unsigned>(info->type), info->native_code, info->exception_address,
	     static_cast<unsigned>(info->access_violation_type), info->access_violation_vaddr);
}

static void EncodeId64(uint16_t in_id, std::string* out_id) {
	static const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
	if (in_id < 0x40u) {
		*out_id += str[in_id];
	} else {
		if (in_id < 0x1000u) {
			*out_id += str[static_cast<uint16_t>(in_id >> 6u) & 0x3fu];
			*out_id += str[in_id & 0x3fu];
		} else {
			*out_id += str[static_cast<uint16_t>(in_id >> 12u) & 0x3fu];
			*out_id += str[static_cast<uint16_t>(in_id >> 6u) & 0x3fu];
			*out_id += str[in_id & 0x3fu];
		}
	}
}

template <class T>
static void GetDynDataOs(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = elf->GetDynamicData<T>(dyn->d_un.d_ptr);
	}
}

template <class T>
static void GetDynData(Elf64* elf, uint64_t base_vaddr, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = reinterpret_cast<T>(base_vaddr + dyn->d_un.d_ptr);
	}
}

template <class T>
static void GetDynValue(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = dyn->d_un.d_val;
	}
}

template <class T>
static void GetDynValues(Elf64* elf, T* out, Elf64_Sxword tag) {
	for (const auto* dyn: elf->GetDynList(tag)) {
		out->push_back(dyn->d_un.d_val);
	}
}

template <class T>
static void GetDynPtr(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = dyn->d_un.d_ptr;
	}
}

static void KYTY_SYSV_ABI ProgramExitHandler() {
	Common::Singleton<RuntimeLinker>::Instance()->StopAllModules();

	LOGF("exit!!!\n");
}

template <class T>
static void GetDynModules(Elf64* elf, T* out, const char* names, Elf64_Sxword tag) {
	std::vector<uint64_t> needed_modules;
	GetDynValues(elf, &needed_modules, tag);
	for (auto need: needed_modules) {
		ModuleId id {};
		EncodeId64(static_cast<uint16_t>((need >> 48u) & 0xffffu), &id.id);
		id.name = names + (need & 0xffffffff);
		out->push_back(id);
	}
}

template <class T>
static void GetDynLibs(Elf64* elf, T* out, const char* names, Elf64_Sxword tag) {
	std::vector<uint64_t> needed_modules;
	GetDynValues(elf, &needed_modules, tag);
	for (auto need: needed_modules) {
		LibraryId id {};
		EncodeId64(static_cast<uint16_t>((need >> 48u) & 0xffffu), &id.id);
		id.name = names + (need & 0xffffffff);
		out->push_back(id);
	}
}

static RelocationInfo GetRelocationInfo(Elf64_Rela* r, Program* program) {
	KYTY_PROFILER_FUNCTION();

	RelocationInfo ret;
	// SymbolRecord   sr {};

	auto         type    = r->GetType();
	auto         symbol  = r->GetSymbol();
	Elf64_Sxword addend  = r->r_addend;
	auto*        symbols = program->dynamic_info->symbol_table;
	auto*        names   = program->dynamic_info->str_table;
	ret.base_vaddr       = program->base_vaddr;
	ret.vaddr            = ret.base_vaddr + r->r_offset;
	ret.bind_self        = false;

	switch (type) {
		case R_X86_64_GLOB_DAT:
		case R_X86_64_JUMP_SLOT: addend = 0; [[fallthrough]];
		case R_X86_64_64: {
			auto         sym          = symbols[symbol];
			auto         bind         = sym.GetBind();
			auto         sym_type     = sym.GetType();
			uint64_t     symbol_vaddr = 0;
			SymbolRecord sr {};
			switch (sym_type) {
				case STT_NOTYPE: ret.type = SymbolType::NoType; break;
				case STT_FUNC: ret.type = SymbolType::Func; break;
				case STT_OBJECT: ret.type = SymbolType::Object; break;
				default: EXIT("unknown symbol type: %d\n", (int)sym_type);
			}
			switch (bind) {
				case STB_LOCAL:
					symbol_vaddr = ret.base_vaddr + sym.st_value;
					ret.bind     = BindType::Local;
					break;
				case STB_GLOBAL: ret.bind = BindType::Global; [[fallthrough]];
				case STB_WEAK: {
					ret.bind = (ret.bind == BindType::Unknown ? BindType::Weak : ret.bind);
					ret.name = names + sym.st_name;
					program->rt->Resolve(ret.name, ret.type, program, &sr, &ret.bind_self);
					symbol_vaddr = sr.vaddr;
				} break;
				default: EXIT("unknown bind: %d\n", (int)bind);
			}
			ret.resolved = (symbol_vaddr != 0);
			ret.value    = (ret.resolved ? symbol_vaddr + addend : 0);
			ret.name     = sr.name;
			ret.dbg_name = sr.dbg_name;
		} break;
		case R_X86_64_RELATIVE:
			ret.value    = ret.base_vaddr + addend;
			ret.resolved = true;
			break;
		case R_X86_64_DTPMOD64:
			ret.value    = reinterpret_cast<uint64_t>(program);
			ret.resolved = true;
			ret.type     = SymbolType::TlsModule;
			ret.bind     = BindType::Local;
			ret.dbg_name = Common::PathToString(program->file_name);
			break;
		default: EXIT("unknown type: %d\n", (int)type);
	}

	return ret;
}

static bool RelocateRecord(uint32_t index, Elf64_Rela* r, Program* program, bool jmprela_table) {
	KYTY_PROFILER_FUNCTION();

	const auto ri      = GetRelocationInfo(r, program);
	auto       value   = ri.value;
	bool       stubbed = false;
	if (!ri.resolved) {
		const bool weak = ri.bind == BindType::Weak || !program->fail_if_global_not_resolved;
		if (!weak) {
			LOGF("Stubbed: %s\n",
			     fmt::format("[{:016x}] <- {:016x}, {}, {}, {}, {}", ri.vaddr, ri.value, ri.name,
			                 magic_enum::enum_name(ri.type), magic_enum::enum_name(ri.bind),
			                 ri.dbg_name).c_str());
		}
		if (ri.type == SymbolType::Object) {
			value = g_invalid_memory;
		} else {
			value   = RegisterStubbedImport(index, program, ri);
			stubbed = true;
		}
	}

	const bool patched = PatchGuestMemory64(ri.vaddr, value);
	if (patched && stubbed) {
		LOGF("Relocate: unresolved %s import patched to stub [%u] [%016" PRIx64 "] <- %016" PRIx64
		     ", %s, %s, %s, %s\n",
		     jmprela_table ? "PLT" : "non-PLT", index, ri.vaddr, value, ri.name.c_str(),
		     magic_enum::enum_name(ri.type), magic_enum::enum_name(ri.bind),
		     Common::PathToString(program->file_name).c_str());
	}

	if (program->dbg_print_reloc && patched && !ri.bind_self &&
	    (ri.bind == BindType::Global || ri.bind == BindType::Weak ||
	     ri.type == SymbolType::TlsModule)) {
		LOGF("Relocate: %s\n",
		     fmt::format("[{:016x}] <- {:016x}, {}, {}, {}, {}", ri.vaddr, ri.value, ri.name,
		                 magic_enum::enum_name(ri.type), magic_enum::enum_name(ri.bind),
		                 ri.dbg_name).c_str());
	}
	return ri.resolved;
}

static void ForEachRelocation(Program* program, auto&& func) {
	const auto& info = *program->dynamic_info;
	for (uint32_t i = 0; i < info.rela_table_total_size / sizeof(Elf64_Rela); ++i) {
		func(info.rela_table + i, i, false);
	}
	for (uint32_t i = 0; i < info.jmprela_table_size / sizeof(Elf64_Rela); ++i) {
		func(info.jmprela_table + i, i, true);
	}
}

// Guest code reads its TCB from fs:[0], which the host owns, so PatchGuestCode rewrites those reads
// to gs:[TcbGsOffset()]. On Linux the gs base is the TCB, whose first qword points to itself;
// Windows (TEB TLS slots) and macOS (TSD slots) hold the TCB in a host slot at that offset.
struct TlsBlock {
	uint8_t* ptr  = nullptr;
	uint64_t size = 0;
};

struct Tcb {
	Tcb*                   self;
	std::vector<TlsBlock>* dtv; // indexed by ThreadLocalStorage::index
};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Allocated before main: host libraries loaded later can use up the slots at a fixed TEB offset.
static const DWORD g_tcb_slot = TlsAlloc();

static uint32_t TcbGsOffset() {
	return 0x1480 + 8 * g_tcb_slot;
}

void RuntimeLinker::SetThreadTcb(void* tcb) {
	EXIT_IF(g_tcb_slot >= 64); // later slots are not at a fixed TEB offset
	EXIT_IF(TlsSetValue(g_tcb_slot, tcb) == 0);
}
#elif defined(__APPLE__)
static pthread_key_t TcbKey() {
	static const pthread_key_t key = [] {
		pthread_key_t k {};
		EXIT_IF(pthread_key_create(&k, nullptr) != 0);
		return k;
	}();
	return key;
}

static uint32_t TcbGsOffset() {
	return 8 * TcbKey();
}

void RuntimeLinker::SetThreadTcb(void* tcb) {
	EXIT_IF(pthread_setspecific(TcbKey(), tcb) != 0);
}
#else
static uint32_t TcbGsOffset() {
	return 0;
}

void RuntimeLinker::SetThreadTcb(void* tcb) {
	EXIT_IF(syscall(SYS_arch_prctl, ARCH_SET_GS, tcb) != 0);
}
#endif

void* RuntimeLinker::GetThreadTcb() {
	void* tcb = nullptr;
	asm volatile("movq %%gs:(%1), %0" : "=r"(tcb) : "r"(static_cast<uint64_t>(TcbGsOffset())));
	return tcb;
}

static TlsBlock AllocateTlsBlock(const ThreadLocalStorage& tls, uint64_t size) {
	auto* ptr = reinterpret_cast<uint8_t*>(Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    0, size, Common::VirtualMemory::Mode::ReadWrite, "thread_local_storage"));
	EXIT_IF(ptr == nullptr);
	std::memcpy(ptr, reinterpret_cast<const void*>(tls.image_vaddr), tls.init_size);
	std::memset(ptr + tls.init_size, 0, size - tls.init_size);
	return {ptr, size};
}

void RuntimeLinker::CreateThreadTls() {
	constexpr uint64_t TCB_SIZE = 0x40;

	const auto tls =
	    g_tls_main_program != nullptr ? g_tls_main_program->tls : ThreadLocalStorage {};
	const auto block = AllocateTlsBlock(tls, tls.image_size + TCB_SIZE);
	auto*      tcb   = reinterpret_cast<Tcb*>(block.ptr + tls.image_size);
	tcb->self        = tcb;
	tcb->dtv         = new std::vector<TlsBlock> {block};
	SetThreadTcb(tcb);
}

void RuntimeLinker::DeleteThreadTls() {
	auto* dtv = static_cast<Tcb*>(GetThreadTcb())->dtv;
	for (const auto& block: *dtv) {
		EXIT_IF(block.ptr != nullptr && !Libs::LibKernel::Memory::FreeGuestMemory(
		                                    reinterpret_cast<uint64_t>(block.ptr), block.size));
	}
	delete dtv;
}

uint8_t* RuntimeLinker::TlsGetAddr(Program* program) {
	auto&      dtv   = *static_cast<Tcb*>(GetThreadTcb())->dtv;
	const auto index = program->tls.index;
	if (index >= dtv.size()) {
		dtv.resize(index + 1);
	}
	if (dtv[index].ptr == nullptr) {
		dtv[index] = AllocateTlsBlock(program->tls, program->tls.image_size);
	}
	return dtv[index].ptr;
}

void RuntimeLinker::PatchGuestCode(uint64_t address, uint64_t size) {
	if (size >= 12) {
		// Replace guest stack-canary/errno stores through fs:[0x28] with nops.
		// Windows x64 cannot host guest FS directly, and an unpatched shared-library access faults
		// at address 0x28.
		const uint8_t fs_store_pattern[8] = {0x64, 0xc7, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00};
		auto*         start_ptr           = reinterpret_cast<uint8_t*>(address);
		auto*         end_ptr             = start_ptr + size - 12;

		for (auto* ptr = start_ptr; ptr <= end_ptr; ptr++) {
			if (memcmp(ptr, fs_store_pattern, sizeof(fs_store_pattern)) == 0) {
				LOGF("Patch fs:[0x28] store at addr: [%016" PRIx64 "]\n",
				     reinterpret_cast<uint64_t>(ptr));
				if (ptr + 16 < start_ptr + size && ptr[12] == 0xcd && ptr[13] == 0x45 &&
				    ptr[14] == 0x90 && ptr[15] == 0x0f && ptr[16] == 0x0b) {
					ptr[0] = 0x5d; // pop rbp
					ptr[1] = 0xc3; // ret
					std::memset(ptr + 2, 0x90, 15);
				} else {
					std::memset(ptr, 0x90, 12);
				}
			}
		}
	}

	// mov r64, qword ptr fs:[0] -> mov r64, qword ptr gs:[TcbGsOffset()]
	const uint32_t offset  = TcbGsOffset();
	auto*          end_ptr = reinterpret_cast<uint8_t*>(address + size);
	for (auto* ptr = reinterpret_cast<uint8_t*>(address); ptr + 9 <= end_ptr; ptr++) {
		if (ptr[0] == 0x64 && (ptr[1] & 0xfbu) == 0x48 && ptr[2] == 0x8b &&
		    (ptr[3] & 0xc7u) == 0x04 && ptr[4] == 0x25 &&
		    (ptr[5] | ptr[6] | ptr[7] | ptr[8]) == 0) {
			ptr[0] = 0x65;
			std::memcpy(ptr + 5, &offset, sizeof(offset));
			ptr += 8;
		}
	}
}

struct GuestPatchOptions {
	bool     protect_memory_faults   = false;
	bool     emulate_amd             = false;
	bool     copy_red_zone_functions = false;
	uint64_t tsc_multiplier          = 0;

	[[nodiscard]] bool Any() const {
		return protect_memory_faults || emulate_amd || copy_red_zone_functions ||
		       tsc_multiplier != 0;
	}
};

static GuestPatchOptions GetGuestPatchOptions() {
	GuestPatchOptions options;
	options.emulate_amd = Config::AmdCpuEnabled();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	options.protect_memory_faults = Config::RedZoneProtectionEnabled();
	// Windows writes exception frames directly below the guest RSP, over the SysV red zone. How
	// many bytes below RSP survive depends on the host CPU's XSAVE layout and on which register
	// state is in use at the fault, so the protection is always on.
	options.copy_red_zone_functions = true;
#endif
	options.tsc_multiplier =
	    Config::GuestTscEnabled() ? Libs::LibKernel::KernelGuestTscMultiplier() : 0;
	return options;
}

// Function entries the unwind table does not list: exported functions and code addresses stored
// by relocations (vtables, function pointer tables).
static void CollectDynamicFunctionStarts(const Program* program, uint64_t code_start,
                                         uint64_t code_end, std::vector<uintptr_t>* starts) {
	const auto* info = program->dynamic_info.get();
	if (info == nullptr) {
		return;
	}
	const auto in_code = [&](uint64_t address) {
		return address >= code_start && address < code_end;
	};
	if (info->symbol_table != nullptr) {
		for (uint64_t i = 0; i < info->symbol_table_total_size / sizeof(Elf64_Sym); ++i) {
			const auto& symbol  = info->symbol_table[i];
			const auto  address = program->base_vaddr + symbol.st_value;
			if (symbol.GetType() == STT_FUNC && symbol.st_shndx != 0 && in_code(address)) {
				starts->push_back(address);
			}
		}
	}
	if (info->rela_table != nullptr) {
		for (uint64_t i = 0; i < info->rela_table_total_size / sizeof(Elf64_Rela); ++i) {
			const auto& relocation = info->rela_table[i];
			const auto  address    = program->base_vaddr + relocation.r_addend;
			if (relocation.GetType() == R_X86_64_RELATIVE && in_code(address)) {
				starts->push_back(address);
			}
		}
	}
}

// Applies the guest instruction patches to the executable segments of a loaded program, once its
// dynamic information is parsed.
static void PatchGuestInstructionsOf(Program* program) {
	const auto options = GetGuestPatchOptions();
	if (!options.Any() || program->instruction_trampoline_size == 0) {
		return;
	}
	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	std::vector<std::pair<uint64_t, uint64_t>> executable_segments;
	std::vector<std::pair<uint64_t, uint64_t>> readable_segments;
	uint64_t                                   eh_frame_header_addr = 0;
	uint64_t                                   eh_frame_header_size = 0;
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && phdr[i].p_type == PT_LOAD && (phdr[i].p_flags & PF_X) != 0) {
			executable_segments.emplace_back(phdr[i].p_vaddr + program->base_vaddr,
			                                 phdr[i].p_filesz);
		}
		if (phdr[i].p_filesz != 0 && phdr[i].p_type == PT_LOAD && (phdr[i].p_flags & PF_R) != 0) {
			readable_segments.emplace_back(phdr[i].p_vaddr + program->base_vaddr, phdr[i].p_filesz);
		}
		if (phdr[i].p_type == PT_GNU_EH_FRAME) {
			eh_frame_header_addr = phdr[i].p_vaddr + program->base_vaddr;
			eh_frame_header_size = phdr[i].p_memsz;
		}
	}

	const auto             patch_start = std::chrono::steady_clock::now();
	std::vector<uintptr_t> function_starts;
	const bool             have_function_starts =
	    DecodeEhFrameFunctionStarts(eh_frame_header_addr, eh_frame_header_size, &function_starts) &&
	    !function_starts.empty();
	const auto module_name = Common::PathToString(program->file_name.filename());
	if (options.copy_red_zone_functions) {
		for (const auto& [segment_addr, segment_size]: executable_segments) {
			CollectDynamicFunctionStarts(program, segment_addr, segment_addr + segment_size,
			                             &function_starts);
		}
	}
	if (!have_function_starts && !options.copy_red_zone_functions) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Guest instruction patches: {} not applied (function boundaries unavailable)\n",
		    module_name));
	}
	const bool                  protect_memory_faults = options.protect_memory_faults;
	const bool                  emulate_amd           = options.emulate_amd;
	GuestInstructionPatchResult totals {};
	for (const auto& [segment_addr, segment_size]: executable_segments) {
		const auto result = PatchGuestInstructions(
		    segment_addr, segment_size, function_starts, protect_memory_faults, emulate_amd,
		    GetGuestInstructionHostFeatures(), options.tsc_multiplier,
		    options.copy_red_zone_functions, readable_segments);
		totals.reciprocal_sqrt += result.reciprocal_sqrt;
		totals.extrq += result.extrq;
		totals.insertq += result.insertq;
		totals.rdpid += result.rdpid;
		totals.clwb += result.clwb;
		totals.rdtsc += result.rdtsc;
		totals.function_count += result.function_count;
		totals.discovered_function_count += result.discovered_function_count;
		totals.red_zone_copy_candidate_count += result.red_zone_copy_candidate_count;
		totals.rewritten_function_count += result.rewritten_function_count;
		totals.rewritten_function_bytes += result.rewritten_function_bytes;
		totals.partially_rewritten_function_count += result.partially_rewritten_function_count;
		totals.unprotected_instruction_count += result.unprotected_instruction_count;
		totals.copy_entry_conflict_count += result.copy_entry_conflict_count;
		totals.copy_unsupported_count += result.copy_unsupported_count;
		totals.copy_no_space_count += result.copy_no_space_count;
		if (protect_memory_faults) {
			LOGF("Windows guest red-zone patching: %s, functions=%" PRIu64 ", red_zone=%" PRIu64
			     ", memory=%" PRIu64 ", patched=%" PRIu64 ", short=%" PRIu64 ", stack=%" PRIu64
			     ", control=%" PRIu64 ", unrelocatable=%" PRIu64 "\n",
			     module_name.c_str(), result.function_count, result.red_zone_function_count,
			     result.memory_instruction_count, result.patched_memory_instruction_count,
			     result.short_memory_instruction_count,
			     result.stack_dependent_memory_instruction_count,
			     result.control_flow_memory_instruction_count,
			     result.unrelocatable_memory_instruction_count);
		}
	}
	if (options.copy_red_zone_functions) {
		const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		    std::chrono::steady_clock::now() - patch_start);
		Log::WriteToConsoleAndLog(fmt::format(
		    "Guest red-zone protection: {}: {} of {} functions copied ({} KiB; {} leave the copy "
		    "early, {} unprotected instructions; not copied: {} entry conflicts, {} unsupported, "
		    "{} out of space); {} functions analyzed ({} found by the code sweep) in {} ms\n",
		    module_name, totals.rewritten_function_count, totals.red_zone_copy_candidate_count,
		    totals.rewritten_function_bytes / 1024, totals.partially_rewritten_function_count,
		    totals.unprotected_instruction_count, totals.copy_entry_conflict_count,
		    totals.copy_unsupported_count, totals.copy_no_space_count, totals.function_count,
		    totals.discovered_function_count, elapsed.count()));
		const auto not_copied =
		    totals.red_zone_copy_candidate_count - totals.rewritten_function_count;
		if (not_copied != 0) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Warning: guest red-zone protection: {}: {} of {} red-zone functions are not "
			    "protected ({} entry conflicts, {} unsupported, {} out of space); a host exception "
			    "inside them can overwrite their red zone\n",
			    module_name, not_copied, totals.red_zone_copy_candidate_count,
			    totals.copy_entry_conflict_count, totals.copy_unsupported_count,
			    totals.copy_no_space_count));
		}
	}
	if (totals.rdtsc.found != 0) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Guest TSC: {} (RDTSC/RDTSCP: native={}, skipped={})\n", module_name,
		                totals.rdtsc.native, totals.rdtsc.Skipped()));
	}
	if (emulate_amd && have_function_starts) {
		InstructionPatchCounts combined {};
		std::string            details;
		for (const auto& [name, counts]: {std::pair {"VRSQRTPS", totals.reciprocal_sqrt},
		                                  {"EXTRQ", totals.extrq},
		                                  {"INSERTQ", totals.insertq},
		                                  {"RDPID", totals.rdpid},
		                                  {"CLWB", totals.clwb}}) {
			combined += counts;
			if (!details.empty()) details += "; ";
			details += fmt::format("{}: native={}, trapped={}, skipped={}", name, counts.native,
			                       counts.trapped, counts.Skipped());
		}
		const auto  found   = combined.found;
		const auto  skipped = combined.Skipped();
		const char* status  = found == 0         ? "no matching instructions"
		                      : skipped == found ? "not patched"
		                      : skipped != 0     ? "partially patched"
		                                         : "patched";
		Log::WriteToConsoleAndLog(
		    fmt::format("AMD CPU compatibility: {} {} ({})\n", module_name, status, details));
	}
}

uint64_t RuntimeLinker::GetEntry() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		if (p->elf != nullptr && !p->elf->IsShared()) {
			return p->elf->GetEntry() + p->base_vaddr;
		}
	}
	return 0;
}

uint64_t RuntimeLinker::GetProcParam() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		if (p->elf != nullptr && !p->elf->IsShared()) {
			return p->proc_param_vaddr;
		}
	}
	return 0;
}

void RuntimeLinker::DbgDump(const std::string& folder) {
	KYTY_PROFILER_FUNCTION();

	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		auto folder_str = Common::FixDirectorySlash(folder);
		folder_str += Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name));

		EXIT_IF(p->elf == nullptr);

		p->elf->DbgDump(folder_str);

		if (p->dynamic_info != nullptr) {
			EXIT_NOT_IMPLEMENTED(p->dynamic_info->symbol_table_entry_size != 0 &&
			                     p->dynamic_info->symbol_table_entry_size != sizeof(Elf64_Sym));
			EXIT_NOT_IMPLEMENTED(p->dynamic_info->rela_table_entry_size != 0 &&
			                     p->dynamic_info->rela_table_entry_size != sizeof(Elf64_Rela));
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->jmprela_table == nullptr);
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->rela_table == nullptr);
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->symbol_table == nullptr);

			if (p->dynamic_info->symbol_table != nullptr) {
				DbgDumpSymbols(folder_str, p->dynamic_info->symbol_table,
				               p->dynamic_info->symbol_table_total_size,
				               p->dynamic_info->str_table);
			}
			if (p->dynamic_info->jmprela_table != nullptr) {
				DbgDumpRela(folder_str, p->dynamic_info->jmprela_table,
				            p->dynamic_info->jmprela_table_size, p->dynamic_info->str_table,
				            "jmprela_table.txt");
			}
			if (p->dynamic_info->rela_table != nullptr) {
				DbgDumpRela(folder_str, p->dynamic_info->rela_table,
				            p->dynamic_info->rela_table_total_size, p->dynamic_info->str_table,
				            "rela_table.txt");
			}
		}

		if (p->export_symbols != nullptr) {
			p->export_symbols->DbgDump(folder_str, "export_symbols.txt");
		}
		if (p->import_symbols != nullptr) {
			p->import_symbols->DbgDump(folder_str, "import_symbols.txt");
		}
	}
}

void RuntimeLinker::RelocateAll() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		Relocate(p);
	}
}

void RuntimeLinker::RelocateProgram(Program* program) {
	Common::LockGuard lock(m_mutex);

	EXIT_IF(program == nullptr);
	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	RelocateAll();
	if (!GamePatch::ApplyPending(program)) {
		EXIT("Failed to apply pending game cheat\n");
	}
}

void RuntimeLinker::UnloadProgram(Program* program) {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	auto it = std::find(m_programs.begin(), m_programs.end(), program);
	EXIT_IF(it == m_programs.end());
	m_programs.erase(it);
	g_guest_modules.Remove(program->base_vaddr);
	g_guest_modules.Remove(program->base_vaddr + program->base_size_aligned);

	for (auto* importer: m_programs) {
		const auto& info = *importer->dynamic_info;
		const auto rela_count =
		    static_cast<uint32_t>(info.rela_table_total_size / sizeof(Elf64_Rela));
		ForEachRelocation(importer, [&](Elf64_Rela* r, uint32_t index, bool jmprela) {
			const auto bit_index = (jmprela ? rela_count : 0) + index;
			auto&      bits      = importer->rela_bits[bit_index >> 3u];
			const auto mask      = 1u << (bit_index & 7u);
			const auto type      = r->GetType();
			if ((bits & mask) == 0 || (type != R_X86_64_64 && type != R_X86_64_GLOB_DAT &&
			                           type != R_X86_64_JUMP_SLOT)) {
				return;
			}
			const auto& symbol = info.symbol_table[r->GetSymbol()];
			if (symbol.GetBind() != STB_GLOBAL && symbol.GetBind() != STB_WEAK) {
				return;
			}
			const auto value =
			    *reinterpret_cast<const uint64_t*>(importer->base_vaddr + r->r_offset);
			const auto address = value - (type == R_X86_64_64 ? r->r_addend : 0);
			if (address < program->base_vaddr ||
			    address >= program->base_vaddr + program->mapped_size) {
				return;
			}
			SymbolType symbol_type;
			switch (symbol.GetType()) {
				case STT_NOTYPE: symbol_type = SymbolType::NoType; break;
				case STT_FUNC: symbol_type = SymbolType::Func; break;
				case STT_OBJECT: symbol_type = SymbolType::Object; break;
				default: return;
			}
			const std::string name = info.str_table + symbol.st_name;
			if (program->export_symbols->FindByNid(name.substr(0, name.find('#')), symbol_type,
			                                       address) != nullptr) {
				bits &= ~mask;
				if (RelocateRecord(index, r, importer, jmprela)) {
					bits |= mask;
				}
			}
		});
	}
	DeleteProgram(program);
}

RuntimeLinker::RuntimeLinker(): m_symbols(std::make_unique<SymbolDatabase>()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RuntimeLinker::~RuntimeLinker() {
	Clear();
}

Program* RuntimeLinker::LoadProgram(const std::filesystem::path& elf_name) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	static int32_t id_seq = 0;

	LOGF("Loading: %s\n", Common::PathToString(elf_name).c_str());

	auto  program_owner = std::make_unique<Program>();
	auto* program       = program_owner.get();

	program->rt        = this;
	program->file_name = elf_name;
	program->unique_id = ++id_seq;

	program->elf = std::make_unique<Elf64>();
	program->elf->Open(elf_name);

	if (program->elf->IsValid()) {
		LoadProgramToMemory(program);
		ParseProgramDynamicInfo(program);
		PatchGuestInstructionsOf(program);
		CreateSymbolDatabase(program);
	} else {
		EXIT("elf is not valid: %s\n", Common::PathToString(elf_name).c_str());
	}

	m_programs.push_back(program_owner.release());

	if (!program->elf->IsShared()) {
		program->fail_if_global_not_resolved = false;
		Libs::LibKernel::SetProgName(elf_name.filename().string());
	}

	if (Common::ToLower(Common::DirectoryWithoutFilename(Common::PathToGenericString(elf_name)))
	        .ends_with("_module/")) {
		program->fail_if_global_not_resolved = false;
	}

	return program;
}

void RuntimeLinker::SaveMainProgram(const std::filesystem::path& elf_name) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		EXIT_IF(p->elf == nullptr);

		if (!p->elf->IsShared()) {
			p->elf->Save(elf_name);
			break;
		}
	}
}

void RuntimeLinker::SaveProgram(Program* program, const std::filesystem::path& elf_name) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	if (auto it = std::find(m_programs.begin(), m_programs.end(), program);
	    it != m_programs.end()) {
		EXIT_IF((*it)->elf == nullptr);

		(*it)->elf->Save(elf_name);
	} else {
		EXIT("program not found");
	}
}

void RuntimeLinker::Execute(const std::filesystem::path& game_patch) {
	KYTY_PROFILER_THREAD("Thread_Main");

	Libs::LibKernel::PthreadInitSelfForMainThread();
	auto* main_stack_top = Libs::LibKernel::PthreadCreateMainGuestStack();

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Guest code has no Windows stack probes and may jump over the guard page. Module
	// initializers execute on the host stack too, so grow it before calling any guest code.
	size_t expanded_size = 0;
	while (expanded_size < static_cast<size_t>(768) * 1024) {
		const auto stack = SysStackUsage();
		*reinterpret_cast<uint32_t*>(stack.guard_addr) = 0;
		expanded_size += stack.guard_size;
	}
#endif

	// The runtime automatically loads libc; other PRXs are requested by the application.
	const auto libc_path = Libs::LibKernel::FileSystem::GetRealFilename("/app0/sce_module/libc.prx");
	auto* libc = Common::File::IsFileExisting(libc_path) ? LoadProgram(libc_path) : nullptr;
	RelocateAll();

	if (!game_patch.empty()) {
		if (!GamePatch::Apply(game_patch, m_programs.empty() ? nullptr : m_programs.front(),
		                      m_programs)) {
			EXIT("Failed to apply game cheat\n");
		}
	}
	CreateThreadTls();
	Common::WriteFpEnv(Common::GuestFpEnv);
	if (libc != nullptr && libc->dynamic_info->init_vaddr != 0) {
		StartModule(libc, 0, nullptr, nullptr);
	}

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Execute: %s\n---\n", "Main");

	if (auto entry = GetEntry(); entry != 0) {
		const auto& arguments = Config::GetGuestArguments();
		const auto params_size = offsetof(EntryParams, argv) +
		                         (arguments.size() + 2) * sizeof(const char*);
		auto* params = reinterpret_cast<EntryParams*>(
		    (reinterpret_cast<uintptr_t>(main_stack_top) - params_size - 0x100u) &
		    ~static_cast<uintptr_t>(0x0f));
		std::memset(params, 0, params_size);
		for (const auto& argument: arguments) {
			params->argv[params->argc++] = argument.c_str();
		}

		LOGF("stack_addr = %" PRIx64 "\n", reinterpret_cast<uint64_t>(params));

		RunEntry(entry, params, ProgramExitHandler,
		         reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(params) - 0x1000u));
	}
}

void RuntimeLinker::Clear() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);
	GamePatch::Clear();

	for (auto* p: m_programs) {
		DeleteProgram(p);
	}
	m_programs.clear();
	g_guest_modules.Clear();
	for (const auto page: g_unresolved_stub_thunk_pages) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(page, UNRESOLVED_STUB_PAGE_SIZE));
	}
	g_unresolved_stub_thunk_pages.clear();
	g_unresolved_stub_thunk_offset = 0;
	g_stubbed_imports.clear();
	g_unresolved_stub_call_log_count.store(0);
	if (g_invalid_memory != 0) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(g_invalid_memory, 4096));
		g_invalid_memory = 0;
	}
	g_tls_main_program  = nullptr;
	g_desired_base_addr = SYSTEM_RESERVED + CODE_BASE_OFFSET;
	m_symbols.reset();
}

void RuntimeLinker::Resolve(const std::string& name, SymbolType type, Program* program,
                            SymbolRecord* out_info, bool* bind_self) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	EXIT_IF(out_info == nullptr);

	auto ids = Common::Split(name, '#');

	if (bind_self != nullptr) {
		*bind_self = false;
	}

	*out_info = {.name = name};
	if (ids.size() != 3) {
		return;
	}

	// The PS5 binds an import by NID, library name and module name.
	const LibraryId* l = FindLibrary(*program, ids.at(1));
	const ModuleId*  m = FindModule(*program, ids.at(2));
	EXIT_IF(l == nullptr || m == nullptr);

	auto find = [&](const SymbolResolve& key) {
		if (const auto* rec = m_symbols->Find(key); rec != nullptr) {
			*out_info = *rec;
			return true;
		}
		for (auto* p: m_programs) {
			if (const auto* rec = p->export_symbols->Find(key); rec != nullptr) {
				*out_info = *rec;
				if (bind_self != nullptr) {
					*bind_self = (p == program);
				}
				return true;
			}
		}
		return false;
	};

	const SymbolResolve sr {ids.at(0), l->name, m->name, type};
	// libc exports the LibcInternal functions under the same NIDs; it serves a LibcInternal import
	// nothing else exports.
	if (find(sr) || (l->name == "libSceLibcInternal" && find({sr.name, "libc", "libc", type}))) {
		return;
	}
	out_info->name = SymbolDatabase::GenerateName(sr);
}

uint64_t RuntimeLinker::ReadFromElf(Program* program, uint64_t vaddr) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->base_vaddr == 0 || program->base_size == 0);
	EXIT_IF(program->elf == nullptr);

	uint64_t ret = 0;

	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	EXIT_IF(phdr == nullptr || ehdr == nullptr);

	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t segment_addr      = phdr[i].p_vaddr + program->base_vaddr;
			uint64_t segment_file_size = phdr[i].p_filesz;

			if (vaddr >= segment_addr && vaddr < segment_addr + segment_file_size) {
				program->elf->LoadSegment(reinterpret_cast<uint64_t>(&ret),
				                          phdr[i].p_offset + vaddr - segment_addr, sizeof(ret));
				break;
			}
		}
	}

	return ret;
}

Program* RuntimeLinker::FindProgramById(int32_t id) {
	Common::LockGuard lock(m_mutex);

	// Id 0 is reserved for main program
	if (id == 0 && !m_programs.empty()) {
		return m_programs.front();
	}

	for (auto* p: m_programs) {
		if (p->unique_id == id) {
			return p;
		}
	}

	return nullptr;
}

std::vector<int32_t> RuntimeLinker::GetProgramIds() {
	Common::LockGuard    lock(m_mutex);
	std::vector<int32_t> ids(m_programs.size());
	std::ranges::transform(m_programs, ids.begin(), &Program::unique_id);
	return ids;
}

Program* RuntimeLinker::FindProgramByFileName(const std::filesystem::path& elf_name) {
	Common::LockGuard lock(m_mutex);

	auto fixed_name = Common::FixFilenameSlash(Common::PathToGenericString(elf_name));
	for (auto* p: m_programs) {
		if (Common::EqualNoCase(Common::FixFilenameSlash(Common::PathToGenericString(p->file_name)),
		                        fixed_name)) {
			return p;
		}
	}

	return nullptr;
}

Program* RuntimeLinker::FindProgramByAddr(uint64_t vaddr) {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		const auto* ehdr = p->elf->GetEhdr();
		const auto* phdr = p->elf->GetPhdr();

		EXIT_IF(phdr == nullptr || ehdr == nullptr);

		for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
			if (phdr[i].p_memsz != 0 &&
			    (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
				uint64_t segment_addr = phdr[i].p_vaddr + p->base_vaddr;
				uint64_t segment_size = GetAlignedSize(phdr + i);

				if (vaddr >= segment_addr && vaddr < segment_addr + segment_size) {
					return p;
				}
			}
		}
	}

	return nullptr;
}

void RuntimeLinker::StackTrace(uint64_t frame_ptr, uint64_t stack_ptr) {
	void* stack[20];
	int   depth = WalkGuestStack(frame_ptr, stack_ptr, stack, static_cast<int>(std::size(stack)));

	LOGF("Stack trace [thread = %d]:\n", Common::Thread::GetThreadIdUnique());

	for (int i = 0; i < depth; i++) {
		auto  vaddr = reinterpret_cast<uint64_t>(stack[i]);
		auto* p     = FindProgramByAddr(vaddr);
		LOGF("[%d] %016" PRIx64 ", off=%016" PRIx64 ", %s\n", i, vaddr,
		     (p == nullptr ? 0 : vaddr - p->base_vaddr),
		     (p == nullptr
		          ? "???"
		          : Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name))
		                .c_str()));
	}
}

void RuntimeLinker::StopAllModules() {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		if (p->elf->IsShared() && p->dynamic_info->fini_vaddr != 0) {
			StopModule(p, 0, nullptr, nullptr);
		}
	}
}

int RuntimeLinker::StartModule(Program* program, size_t args, const void* argp,
                               module_func_t func) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(!program->elf->IsShared());

	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Start module: %s\n---\n",
	           Common::PathToString(program->file_name).c_str());

	return reinterpret_cast<module_ini_fini_func_t>(program->dynamic_info->init_vaddr +
	                                                program->base_vaddr)(args, argp, func);
}

int RuntimeLinker::StopModule(Program* program, size_t args, const void* argp, module_func_t func) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(!program->elf->IsShared());

	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Stop module: %s\n---\n",
	           Common::PathToString(program->file_name).c_str());

	int result = reinterpret_cast<module_ini_fini_func_t>(program->dynamic_info->fini_vaddr +
	                                                      program->base_vaddr)(args, argp, func);

	Libs::LibKernel::PthreadDeleteStaticObjects(program);

	return result;
}

constexpr uint64_t GUEST_PAGE_SIZE = 0x4000;

static uint64_t CalcBaseSize(const Elf64_Ehdr* ehdr, const Elf64_Phdr* phdr) {
	uint64_t base_size = 0;
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t last_addr = phdr[i].p_vaddr + GetAlignedSize(phdr + i);
			if (last_addr > base_size) {
				base_size = last_addr;
			}
		}
	}
	return base_size;
}

static uint64_t CalcExecutableSize(const Elf64_Ehdr* ehdr, const Elf64_Phdr* phdr) {
	uint64_t size = 0;
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_type == PT_LOAD && (phdr[i].p_flags & PF_X) != 0) {
			size += phdr[i].p_filesz;
		}
	}
	return size;
}

// A module's E, R and RW LOAD segments use flexible memory in 16 KiB pages; GNU_RELRO adds none
// since it lies inside an RW LOAD.
uint64_t CalcFlexibleMemorySize(const Elf64_Phdr* phdr, Elf64_Half phnum) {
	uint64_t size = 0;
	for (Elf64_Half i = 0; i < phnum; i++) {
		const auto flags = phdr[i].p_flags;
		if (phdr[i].p_type == PT_LOAD &&
		    (flags == PF_X || flags == PF_R || flags == (PF_R | PF_W))) {
			size += AlignUp(phdr[i].p_memsz, GUEST_PAGE_SIZE);
		}
	}
	return size;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RuntimeLinker::LoadProgramToMemory(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr || program->base_vaddr != 0 || program->base_size != 0 ||
	        program->elf == nullptr);

	// static uint64_t desired_base_addr = DESIRED_BASE_ADDR;

	bool is_shared   = program->elf->IsShared();
	bool is_next_gen = program->elf->IsNextGen();

	EXIT_NOT_IMPLEMENTED(!is_shared && !is_next_gen);

	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	EXIT_IF(phdr == nullptr || ehdr == nullptr);

	program->base_size = CalcBaseSize(ehdr, phdr);
	EXIT_IF(program->base_size > UINT64_MAX - (GUEST_PAGE_SIZE - 1));
	program->base_size_aligned = AlignUp(program->base_size, GUEST_PAGE_SIZE);

	program->mapped_size = program->base_size_aligned;
	const auto patch_options = GetGuestPatchOptions();

	uint64_t instruction_trampoline_size = 8u * 1024u * 1024u;
	if (patch_options.copy_red_zone_functions) {
		// Room for copies of the red-zone functions of the executable segments.
		instruction_trampoline_size +=
		    AlignUp(CalcExecutableSize(ehdr, phdr) / 16, GUEST_PAGE_SIZE);
	}
	if (patch_options.Any()) {
		EXIT_IF(instruction_trampoline_size > UINT64_MAX - program->mapped_size);
		program->mapped_size += instruction_trampoline_size;
	}

	program->base_vaddr = Libs::LibKernel::Memory::AllocateProgramMemory(
	    g_desired_base_addr, program->mapped_size, Common::VirtualMemory::Mode::ExecuteReadWrite,
	    Common::PathToString(program->file_name.filename()).c_str());
	EXIT_IF(program->base_vaddr == 0);
	program->flexible_size = CalcFlexibleMemorySize(phdr, ehdr->e_phnum);
	Libs::LibKernel::Memory::ChargeFlexibleMemory(program->flexible_size);

	const auto module_name = Common::PathToString(program->file_name.filename());
	g_guest_modules.Add(program->base_vaddr, program->base_size_aligned, module_name);
	if (program->mapped_size > program->base_size_aligned) {
		// Red-zone function copies and instruction patches run from here.
		g_guest_modules.Add(program->base_vaddr + program->base_size_aligned,
		                    program->mapped_size - program->base_size_aligned,
		                    module_name + " patch area");
	}

	if (patch_options.Any()) {
		const auto trampoline_addr           = program->base_vaddr + program->base_size_aligned;
		program->instruction_trampoline_size = instruction_trampoline_size;
		RegisterGuestInstructionPatchModule(
		    reinterpret_cast<void*>(program->base_vaddr), program->base_size_aligned,
		    reinterpret_cast<void*>(trampoline_addr), program->instruction_trampoline_size);
	}

	g_desired_base_addr += CODE_BASE_INCR * (1 + program->mapped_size / CODE_BASE_INCR);

	EXIT_IF(program->base_size_aligned < program->base_size);
	LOGF("base_vaddr             = 0x%016" PRIx64 "\n"
	     "base_size              = 0x%016" PRIx64 "\n"
	     "base_size_aligned      = 0x%016" PRIx64 "\n"
	     "mapped_size            = 0x%016" PRIx64 "\n",
	     program->base_vaddr, program->base_size, program->base_size_aligned, program->mapped_size);

	if (!Common::HostException::InstallHandler(KytyExceptionHandler)) {
		EXIT("Failed to install the required vectored exception handler\n");
	}
	Common::HostException::InstallUnhandledReporter(ReportUnhandledHostException);

	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t segment_addr        = phdr[i].p_vaddr + program->base_vaddr;
			uint64_t segment_file_size   = phdr[i].p_filesz;
			uint64_t segment_memory_size = GetAlignedSize(phdr + i);
			auto     mode                = GetMode(phdr[i].p_flags);

			LOGF("[%d] addr        = 0x%016" PRIx64 "\n"
			     "[%d] file_size   = %" PRIu64 "\n"
			     "[%d] memory_size = %" PRIu64 "\n"
			     "[%d] mode        = %s\n",
			     i, segment_addr, i, segment_file_size, i, segment_memory_size, i,
			     magic_enum::enum_name(mode));

			program->elf->LoadSegment(segment_addr, phdr[i].p_offset, segment_file_size);

			bool skip_protect = (phdr[i].p_type == PT_LOAD && is_next_gen &&
			                     mode == Common::VirtualMemory::Mode::NoAccess);

			if (Common::VirtualMemory::IsExecute(mode)) {
				PatchGuestCode(segment_addr, segment_memory_size);
			}

			if (!skip_protect) {
				Libs::LibKernel::Memory::SetProgramMemoryProtection(segment_addr,
				                                                    segment_memory_size, mode);

				if (Common::VirtualMemory::IsExecute(mode)) {
					Common::VirtualMemory::FlushInstructionCache(segment_addr, segment_memory_size);
				}
			}
		}

		if (phdr[i].p_type == PT_TLS) {
			EXIT_IF(phdr[i].p_vaddr >= program->base_size);

			program->tls.image_vaddr = phdr[i].p_vaddr + program->base_vaddr;
			program->tls.init_size   = std::min(phdr[i].p_filesz, GetAlignedSize(phdr + i));
			program->tls.image_size  = GetAlignedSize(phdr + i);
			program->tls.index       = is_shared ? static_cast<uint32_t>(program->unique_id) : 0;

			LOGF("tls addr = 0x%016" PRIx64 "\n"
			     "tls init   = %" PRIu64 "\n"
			     "tls size   = %" PRIu64 "\n",
			     program->tls.image_vaddr, program->tls.init_size, program->tls.image_size);
		}

		if (phdr[i].p_type == PT_OS_PROCPARAM) {
			EXIT_IF(program->proc_param_vaddr != 0);
			EXIT_IF(phdr[i].p_vaddr >= program->base_size);

			program->proc_param_vaddr = phdr[i].p_vaddr + program->base_vaddr;
		}
	}

	if (!is_shared) {
		g_tls_main_program = program;
	}

	LOGF("entry = 0x%016" PRIx64 "\n", program->elf->GetEntry() + program->base_vaddr);
}

void RuntimeLinker::DeleteProgram(Program* p) {
	auto program = std::unique_ptr<Program>(p);
	if (g_tls_main_program == program.get()) {
		g_tls_main_program = nullptr;
	}
	for (auto& record: g_stubbed_imports) {
		if (record.patch_vaddr >= program->base_vaddr &&
		    record.patch_vaddr < program->base_vaddr + program->mapped_size) {
			record.patch_vaddr = 0;
		}
	}

	if (program->base_vaddr != 0 || program->mapped_size != 0) {
		EXIT_IF(program->base_vaddr == 0 || program->mapped_size == 0);
		if (program->instruction_trampoline_size != 0) {
			UnregisterGuestInstructionPatchModule(reinterpret_cast<void*>(program->base_vaddr));
		}
		EXIT_IF(
		    !Libs::LibKernel::Memory::FreeGuestMemory(program->base_vaddr, program->mapped_size));
		Libs::LibKernel::Memory::ReleaseFlexibleMemory(program->flexible_size);
	}
}

void RuntimeLinker::ParseProgramDynamicInfo(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(program->dynamic_info != nullptr);

	program->dynamic_info = std::make_unique<DynamicInfo>();

	auto* elf = program->elf.get();

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_HASH) && elf->HasDynValue(DT_HASH));
	GetDynDataOs(elf, &program->dynamic_info->hash_table, DT_OS_HASH);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->hash_table, DT_HASH);
	GetDynValue(elf, &program->dynamic_info->hash_table_size, DT_OS_HASHSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_STRTAB) && elf->HasDynValue(DT_STRTAB));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_STRSZ) && elf->HasDynValue(DT_STRSZ));
	GetDynDataOs(elf, &program->dynamic_info->str_table, DT_OS_STRTAB);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->str_table, DT_STRTAB);
	GetDynValue(elf, &program->dynamic_info->str_table_size, DT_OS_STRSZ);
	GetDynValue(elf, &program->dynamic_info->str_table_size, DT_STRSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_SYMTAB) && elf->HasDynValue(DT_SYMTAB));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_SYMENT) && elf->HasDynValue(DT_SYMENT));
	GetDynDataOs(elf, &program->dynamic_info->symbol_table, DT_OS_SYMTAB);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->symbol_table, DT_SYMTAB);
	GetDynValue(elf, &program->dynamic_info->symbol_table_total_size, DT_OS_SYMTABSZ);
	GetDynValue(elf, &program->dynamic_info->symbol_table_entry_size, DT_OS_SYMENT);
	GetDynValue(elf, &program->dynamic_info->symbol_table_entry_size, DT_SYMENT);

	GetDynPtr(elf, &program->dynamic_info->init_vaddr, DT_INIT);
	GetDynPtr(elf, &program->dynamic_info->fini_vaddr, DT_FINI);
	GetDynPtr(elf, &program->dynamic_info->init_array_vaddr, DT_INIT_ARRAY);
	GetDynPtr(elf, &program->dynamic_info->fini_array_vaddr, DT_FINI_ARRAY);
	GetDynPtr(elf, &program->dynamic_info->preinit_array_vaddr, DT_PREINIT_ARRAY);
	GetDynValue(elf, &program->dynamic_info->init_array_size, DT_INIT_ARRAYSZ);
	GetDynValue(elf, &program->dynamic_info->fini_array_size, DT_FINI_ARRAYSZ);
	GetDynValue(elf, &program->dynamic_info->preinit_array_size, DT_PREINIT_ARRAYSZ);

	Elf64_Sxword jmprel_type = 0;
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTREL) && elf->HasDynValue(DT_PLTREL));
	GetDynValue(elf, &jmprel_type, DT_OS_PLTREL);
	GetDynValue(elf, &jmprel_type, DT_PLTREL);

	EXIT_NOT_IMPLEMENTED(jmprel_type != DT_RELA);
	if (jmprel_type == DT_RELA) {
		EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_JMPREL) && elf->HasDynValue(DT_JMPREL));
		EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTRELSZ) && elf->HasDynValue(DT_PLTRELSZ));
		GetDynDataOs(elf, &program->dynamic_info->jmprela_table, DT_OS_JMPREL);
		GetDynData(elf, program->base_vaddr, &program->dynamic_info->jmprela_table, DT_JMPREL);
		GetDynValue(elf, &program->dynamic_info->jmprela_table_size, DT_OS_PLTRELSZ);
		GetDynValue(elf, &program->dynamic_info->jmprela_table_size, DT_PLTRELSZ);
	}

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_RELA) && elf->HasDynValue(DT_RELA));
	GetDynDataOs(elf, &program->dynamic_info->rela_table, DT_OS_RELA);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->rela_table, DT_RELA);
	GetDynValue(elf, &program->dynamic_info->rela_table_total_size, DT_OS_RELASZ);
	GetDynValue(elf, &program->dynamic_info->rela_table_total_size, DT_RELASZ);
	GetDynValue(elf, &program->dynamic_info->rela_table_entry_size, DT_OS_RELAENT);
	GetDynValue(elf, &program->dynamic_info->rela_table_entry_size, DT_RELAENT);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table_total_size % sizeof(Elf64_Rela) != 0 ||
	                     program->dynamic_info->jmprela_table_size % sizeof(Elf64_Rela) != 0);
	const auto rela_count = program->dynamic_info->rela_table_total_size / sizeof(Elf64_Rela);
	const auto jmprela_count = program->dynamic_info->jmprela_table_size / sizeof(Elf64_Rela);
	program->rela_bits.resize((rela_count + jmprela_count + 7u) / 8u);

	GetDynValue(elf, &program->dynamic_info->relative_count, DT_RELACOUNT);

	GetDynValue(elf, &program->dynamic_info->debug, DT_DEBUG);
	GetDynValue(elf, &program->dynamic_info->flags, DT_FLAGS);
	GetDynValue(elf, &program->dynamic_info->textrel, DT_TEXTREL);

	EXIT_NOT_IMPLEMENTED(program->dynamic_info->debug != 0);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->textrel != 0);

	uint64_t so_name = 0;
	GetDynValue(elf, &so_name, DT_SONAME);
	program->dynamic_info->so_name = program->dynamic_info->str_table + so_name;

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_NEEDED_MODULE) &&
	                     elf->HasDynValue(DT_OS_NEEDED_MODULE_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_MODULE_INFO) &&
	                     elf->HasDynValue(DT_OS_MODULE_INFO_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_IMPORT_LIB) &&
	                     elf->HasDynValue(DT_OS_IMPORT_LIB_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_EXPORT_LIB) &&
	                     elf->HasDynValue(DT_OS_EXPORT_LIB_1));
	GetDynModules(elf, &program->dynamic_info->import_modules, program->dynamic_info->str_table,
	              DT_OS_NEEDED_MODULE);
	GetDynModules(elf, &program->dynamic_info->import_modules, program->dynamic_info->str_table,
	              DT_OS_NEEDED_MODULE_1);
	GetDynModules(elf, &program->dynamic_info->export_modules, program->dynamic_info->str_table,
	              DT_OS_MODULE_INFO);
	GetDynModules(elf, &program->dynamic_info->export_modules, program->dynamic_info->str_table,
	              DT_OS_MODULE_INFO_1);
	GetDynLibs(elf, &program->dynamic_info->import_libs, program->dynamic_info->str_table,
	           DT_OS_IMPORT_LIB);
	GetDynLibs(elf, &program->dynamic_info->import_libs, program->dynamic_info->str_table,
	           DT_OS_IMPORT_LIB_1);
	GetDynLibs(elf, &program->dynamic_info->export_libs, program->dynamic_info->str_table,
	           DT_OS_EXPORT_LIB);
	GetDynLibs(elf, &program->dynamic_info->export_libs, program->dynamic_info->str_table,
	           DT_OS_EXPORT_LIB_1);
}

void RuntimeLinker::Relocate(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);

	if (g_invalid_memory == 0) {
		g_invalid_memory = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    INVALID_MEMORY, 4096, Common::VirtualMemory::Mode::NoAccess, "invalid_memory", true);
		EXIT_NOT_IMPLEMENTED(g_invalid_memory == 0);
	}

	LOGF_COLOR(Log::Color::White, "--- Relocate program: %s ---\n",
	           Common::PathToString(program->file_name).c_str());

	EXIT_NOT_IMPLEMENTED(program->dynamic_info->symbol_table_entry_size != sizeof(Elf64_Sym));
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table_entry_size != sizeof(Elf64_Rela));
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->jmprela_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->symbol_table == nullptr);

	const auto rela_count = static_cast<uint32_t>(
	    program->dynamic_info->rela_table_total_size / sizeof(Elf64_Rela));

	ForEachRelocation(program, [&](Elf64_Rela* r, uint32_t index, bool jmprela) {
		const auto bit_index = (jmprela ? rela_count : 0) + index;
		if ((program->rela_bits[bit_index >> 3u] & (1u << (bit_index & 7u))) == 0 &&
		    RelocateRecord(index, r, program, jmprela)) {
			program->rela_bits[bit_index >> 3u] |= 1u << (bit_index & 7u);
		}
	});
}

const ModuleId* RuntimeLinker::FindModule(const Program& program, const std::string& id) {
	const auto& import_modules = program.dynamic_info->import_modules;

	if (auto it = std::find_if(import_modules.begin(), import_modules.end(),
	                           [&id](const auto& module) { return module.id == id; });
	    it != import_modules.end()) {
		return &(*it);
	}

	const auto& export_modules = program.dynamic_info->export_modules;

	if (auto it = std::find_if(export_modules.begin(), export_modules.end(),
	                           [&id](const auto& module) { return module.id == id; });
	    it != export_modules.end()) {
		return &(*it);
	}

	return nullptr;
}

const LibraryId* RuntimeLinker::FindLibrary(const Program& program, const std::string& id) {
	const auto& import_libs = program.dynamic_info->import_libs;

	if (auto it = std::find_if(import_libs.begin(), import_libs.end(),
	                           [&id](const auto& lib) { return lib.id == id; });
	    it != import_libs.end()) {
		return &(*it);
	}

	const auto& export_libs = program.dynamic_info->export_libs;

	if (auto it = std::find_if(export_libs.begin(), export_libs.end(),
	                           [&id](const auto& lib) { return lib.id == id; });
	    it != export_libs.end()) {
		return &(*it);
	}

	return nullptr;
}

void RuntimeLinker::CreateSymbolDatabase(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);
	EXIT_IF(program->export_symbols != nullptr);
	EXIT_IF(program->import_symbols != nullptr);

	program->export_symbols = std::make_unique<SymbolDatabase>();
	program->import_symbols = std::make_unique<SymbolDatabase>();

	auto syms = [](Program* program, SymbolDatabase* symbols, bool is_export) {
		if (program->dynamic_info->symbol_table == nullptr ||
		    program->dynamic_info->str_table == nullptr) {
			return;
		}

		for (auto* sym = program->dynamic_info->symbol_table;
		     reinterpret_cast<uint8_t*>(sym) <
		     reinterpret_cast<uint8_t*>(program->dynamic_info->symbol_table) +
		         program->dynamic_info->symbol_table_total_size;
		     sym++) {
			std::string id   = std::string(program->dynamic_info->str_table + sym->st_name);
			auto        bind = sym->GetBind();
			auto        type = sym->GetType();
			auto        ids  = Common::Split(id, '#');

			if (ids.size() == 3) {
				const auto* l = FindLibrary(*program, ids.at(1));
				const auto* m = FindModule(*program, ids.at(2));

				if (l != nullptr && m != nullptr && (bind == STB_GLOBAL || bind == STB_WEAK) &&
				    (type == STT_FUNC || type == STT_OBJECT || type == STT_NOTYPE) &&
				    is_export == (sym->st_value != 0)) {
					SymbolResolve sr {};
					sr.name    = ids.at(0);
					sr.library = l->name;
					sr.module  = m->name;
					switch (type) {
						case STT_NOTYPE: sr.type = SymbolType::NoType; break;
						case STT_FUNC: sr.type = SymbolType::Func; break;
						case STT_OBJECT: sr.type = SymbolType::Object; break;
						default: sr.type = SymbolType::Unknown; break;
					}
					symbols->Add(sr, (is_export ? sym->st_value + program->base_vaddr : 0));
				}
			}
		}
	};

	syms(program, program->export_symbols.get(), true);
	syms(program, program->import_symbols.get(), false);
}

void RuntimeLinker::SetApplicationHeapApi(void* const api[10]) {
	Common::LockGuard lock(m_mutex);

	if (api == nullptr || api[0] == nullptr || api[1] == nullptr) {
		m_application_heap_malloc         = nullptr;
		m_application_heap_free           = nullptr;
		m_application_heap_posix_memalign = nullptr;
		return;
	}

	m_application_heap_malloc = reinterpret_cast<application_heap_malloc_func_t>(api[0]);
	m_application_heap_free   = reinterpret_cast<application_heap_free_func_t>(api[1]);
	m_application_heap_posix_memalign =
	    reinterpret_cast<application_heap_posix_memalign_func_t>(api[6]);
}

void* RuntimeLinker::ApplicationHeapMemalign(uint64_t alignment, uint64_t size) {
	Common::LockGuard lock(m_mutex);

	if (m_application_heap_posix_memalign != nullptr) {
		void* ptr = nullptr;
		return m_application_heap_posix_memalign(&ptr, alignment, size) == 0 ? ptr : nullptr;
	}

	return nullptr;
}

void* RuntimeLinker::ApplicationHeapMalloc(uint64_t size) {
	Common::LockGuard lock(m_mutex);

	return m_application_heap_malloc != nullptr ? m_application_heap_malloc(size) : nullptr;
}

} // namespace Loader
