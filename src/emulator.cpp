#include "emulator.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/startupTrace.h"
#include "common/stringUtils.h"
#include "common/subsystems.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/window.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "kytyGitVersion.h"
#include "libs/agc.h"
#include "libs/audio.h"
#include "libs/controller.h"
#include "libs/libs.h"
#include "libs/network.h"
#include "loader/additionalContent.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"
#include "loader/timer.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <thread>

namespace Emulator {

static void PrintSystemInfo() {
	const Common::SystemInfo info = Common::GetSystemInfo();

#if defined(__APPLE__)
	static constexpr auto platform_name = "macOS";
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static constexpr auto platform_name = "Windows";
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static constexpr auto platform_name = "Linux";
#else
	static constexpr auto platform_name = "Unknown";
#endif

	LOGF("Build\n"
	     "  version: %s\n\n"
	     "Host\n"
	     "  os:      %s\n"
	     "  cpu:     %s\n"
	     "  threads: %u\n\n",
	     KYTY_BUILD_LABEL, platform_name, info.ProcessorName.c_str(),
	     std::thread::hardware_concurrency());
}

static void KytyClose() {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	rt->Clear();

	LOGF("done!\n");

	Common::Subsystems::EmergencyShutdownActive();
}

static void MountOrCreateDir(const std::filesystem::path& dir, const std::string& point) {
	if (!Common::File::IsDirectoryExisting(dir)) {
		Common::File::CreateDirectories(dir);
	}

	EXIT_NOT_IMPLEMENTED(!Common::File::IsDirectoryExisting(dir));

	Libs::LibKernel::FileSystem::Mount(dir, point);
	auto dir_text = Common::PathToString(dir);
	LOGF("Mounted %s -> %s\n", point.c_str(), dir_text.c_str());
}

static void MountSandboxDirs() {
	std::string title_id;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
		title_id = "UNKNOWN";
	}

	MountOrCreateDir("_DownloadData/" + title_id, "/download0");
	MountOrCreateDir("_TempData/" + title_id, "/temp0");
	MountOrCreateDir("_TempData/" + title_id, "/temp");
}

static void Init(const Config::ConfigOptions& cfg, const std::filesystem::path& param_json,
                 Common::Subsystems& subsystems) {
	EXIT_IF(!Common::Thread::IsMainThread());

	// KYTY_TRACE_STARTUP: its reporter starts here, before anything it measures.
	(void)Common::StartupTrace::Enabled();
	Common::InstallAbortReporters();

	subsystems.Initialize<Config::Lifecycle>();
	Config::Load(cfg);
	subsystems.Initialize<Log::Lifecycle>();

	if (Common::File::IsFileExisting(param_json)) {
		Loader::SystemContentLoadParamSfo(param_json);
		// The game's flexible pool from param.json (Ghost of Yotei: 156 MiB); upstream 09753bf8f sizes it elsewhere.
		if (const auto flexible_memory_size = Loader::SystemContentGetMemoryParams().flexible_size;
		    flexible_memory_size != 0) {
			Libs::LibKernel::Memory::SetFlexibleMemorySize(flexible_memory_size);
		}
	}
	if (cfg.splash_enabled) {
		Libs::Graphics::InitializeSplashScreen(param_json.parent_path());
	}

	// Initialization order is explicit; destruction is automatic and reversed.
	subsystems.Initialize<Loader::Timer::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
	subsystems.Initialize<Profiler::Lifecycle>();
	subsystems.Initialize<Libs::Network::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::FileSystem::Lifecycle>();
	subsystems.Initialize<Libs::Controller::Lifecycle>();
	subsystems.Initialize<Libs::Audio::Lifecycle>();
	subsystems.Initialize<Libs::Graphics::Lifecycle>();
}

static void LoadElf(const std::filesystem::path& elf, bool dbg_print_reloc = false,
                    const std::filesystem::path& save_name = {}) {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	auto* program = rt->LoadProgram(
	    Libs::LibKernel::FileSystem::GetRealFilename(Common::PathToGenericString(elf)));

	if (dbg_print_reloc) {
		program->dbg_print_reloc = true;
	}

	if (!save_name.empty()) {
		rt->SaveProgram(program, Libs::LibKernel::FileSystem::GetRealFilename(
		                             Common::PathToGenericString(save_name)));
	}
}

// --precompile-only: set by "quit" or "stop" on stdin (the launcher's Stop), Ctrl+C or Ctrl+Break. The running
// compiles finish, the driver cache is saved, the process exits. The end of stdin is not a stop (a script may start it
// without input).
static std::atomic<bool> g_precompile_stop {false};

static void PrecompileStopSignal(int /*signal*/) {
	g_precompile_stop.store(true);
}

// --precompile-only (after Init: the window, here hidden, the Vulkan device and the renderer are made as for the game,
// so the pipelines are those the game asks for): the guest is not loaded; the recorded shader journals are compiled
// into the driver cache (PipelineCache::RunPrecompileOnly), which is saved; then the process exits with its code.
[[noreturn]] static void RunPrecompileOnly() {
	PrintSystemInfo();
	std::string title_id;
	if (Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) && !title_id.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("Title ID: {} (precompile only)\n", title_id));
	}
	EXIT_NOT_IMPLEMENTED(at_quick_exit(Common::Subsystems::EmergencyShutdownActive) != 0);
	std::signal(SIGINT, PrecompileStopSignal);
#ifdef SIGBREAK
	std::signal(SIGBREAK, PrecompileStopSignal);
#endif
	std::thread([] {
		char line[128];
		while (std::fgets(line, sizeof(line), stdin) != nullptr) {
			const std::string_view text(line);
			if (text.starts_with("quit") || text.starts_with("stop")) {
				g_precompile_stop.store(true);
				return;
			}
		}
	}).detach();
	const int code = Libs::Graphics::WindowPrecompileOnly(Config::GetPrecompileThreads(), g_precompile_stop);
	std::fflush(stdout);
	std::fflush(stderr);
	std::quick_exit(code);
}

static void Execute(const std::filesystem::path& game_patch) {
	auto           patch_path = game_patch;
	Common::Thread guest_thread(
	    [](void* param) {
		    auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
		    rt->Execute(*static_cast<const std::filesystem::path*>(param));
	    },
	    &patch_path);
	Libs::Graphics::WindowRun();
	std::quick_exit(0);
}

void Run(const RunOptions& options) {
	if (options.app0_dir.empty()) {
		EXIT("app0 directory is required\n");
	}

	if (options.elf.empty()) {
		EXIT("ELF is required\n");
	}

	const auto         param_json = options.app0_dir / "sce_sys" / "param.json";
	Common::Subsystems subsystems(true);
	Init(options.config, param_json, subsystems);

	if (options.config.precompile_only) {
		RunPrecompileOnly();
	}

	PrintSystemInfo();
	std::string title_id;
	Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id);
	if (title_id.empty()) {
		std::string content_id;
		if (Loader::SystemContentParamSfoGetString("CONTENT_ID", &content_id) &&
		    content_id.size() == 36 && content_id[6] == '-' && content_id[16] == '_') {
			title_id = content_id.substr(7, 9);
		}
	}
	if (!title_id.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("Title ID: {}\n", title_id));
	}

	int ok = atexit(KytyClose);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	// Guest threads are still running, so skip KytyClose() and only flush emergency state.
	ok = at_quick_exit(Common::Subsystems::EmergencyShutdownActive);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/app0");
	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/hostapp");

	MountSandboxDirs();

	Loader::AddContentLoad(options.addcont_dir, options.addcont, title_id);
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	Libs::InitAll(rt->Symbols());

	LoadElf(options.elf);

	Execute(options.game_patch);
}

} // namespace Emulator
