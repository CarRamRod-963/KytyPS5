#include "common/archive.h"
#include "common/common.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#if defined(KYTY_PGO_GENERATE)
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

// compiler-rt profile runtime, linked into instrumented builds (CMake KYTY_PGO_GENERATE).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

using namespace Common;
using namespace Emulator;

#if defined(KYTY_PGO_GENERATE)
// The profile runtime writes its counters from an atexit handler, which the emulator skips (it
// leaves with quick_exit, a fatal error, or a killed process). Instead a thread writes them every
// KYTY_PGO_DUMP_SECONDS (default 60) to KYTY_PGO_FILE (default kyty_emulator.profraw in the
// working directory), and quick_exit writes them once more. Each dump goes to <file>.tmp first and
// is renamed over <file>, so a stopped training run leaves its last complete profile behind.
static std::mutex  g_pgo_mutex;
static std::string g_pgo_file;

static void WritePgoProfile() {
	std::scoped_lock lock {g_pgo_mutex};
	const auto       temp = g_pgo_file + ".tmp";
	__llvm_profile_set_filename(temp.c_str());
	if (__llvm_profile_write_file() != 0) {
		::fprintf(stderr, "PGO: failed to write %s\n", temp.c_str());
		return;
	}
	std::error_code error;
	std::filesystem::rename(temp, g_pgo_file, error);
	if (error) {
		::fprintf(stderr, "PGO: failed to rename %s: %s\n", temp.c_str(), error.message().c_str());
	}
}

static void StartPgoDumps() {
	const char* file    = std::getenv("KYTY_PGO_FILE");
	g_pgo_file          = (file != nullptr && *file != '\0') ? file : "kyty_emulator.profraw";
	const char* text    = std::getenv("KYTY_PGO_DUMP_SECONDS");
	const int   seconds = (text != nullptr && std::atoi(text) > 0) ? std::atoi(text) : 60;
	::fprintf(stderr, "PGO: writing the profile to %s every %d s and at exit\n", g_pgo_file.c_str(),
	          seconds);
	at_quick_exit(WritePgoProfile);
	std::thread([seconds] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(seconds));
			WritePgoProfile();
		}
	}).detach();
}
#endif

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler = Debug::GetCompiler() + "-" + Debug::GetLinker();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --addcont-dir <dir>                  Add-ons (DLC) folder (default: none).\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf("  --keyboard-player <0-4>             Keyboard player; 0 shares player 1 with the gamepad. Default: 0.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --hdr                                Enable HDR10 output when supported.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf("  --no-splash                          Disable the startup splash screen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation                  Enable Vulkan validation.\n");
	::printf("  --network-connected                  Use the host local network.\n");
	::printf("  --gpu-assisted-validation            Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation                  Enable shader validation.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump                Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump                Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf                 Enable SPIR-V debug printf.\n");
	::printf("  --readback-linear-images             Read back writable linear images on submit.\n");
	::printf("  --readback-tiled-images              Read back small writable tiled images on submit.\n");
	::printf("  --sync-raw-image-buffers             Synchronize raw reads of GPU images.\n");
	::printf(
	    "  --no-async-pipelines                 Build missing pipelines optimized at once,\n"
	    "                                       not unoptimized first with an optimized build\n"
	    "                                       in the background.\n");
	::printf(
	    "  --skip-pending-draws                 Skip draws that only change the picture while\n"
	    "                                       their pipelines build in the background (at most\n"
	    "                                       2 s per pipeline).\n");
	::printf(
	    "  --precompile-pipelines               Build the compute pipelines recorded in earlier\n"
	    "                                       sessions into the driver cache on background\n"
	    "                                       threads while the game runs.\n");
	::printf(
	    "  --lookahead-threads <n>              Threads that evaluate the resources of upcoming\n"
	    "                                       draws and dispatches ahead of the GPU thread\n"
	    "                                       (0 = off, at most 8). Default: 2.\n");
	::printf(
	    "  --precompile-only                    Build the recorded compute pipelines into the\n"
	    "                                       driver cache, save it and exit.\n");
	::printf("  --serialize-dispatches               Order every guest compute dispatch after the\n"
	         "                                       previous one, also where the guest sends no\n"
	         "                                       flush between them.\n");
	::printf("  --no-trophy-notifications            Disable trophy unlock toasts and their sound.\n");
	::printf(
	    "  --trophy-duration <seconds>          Trophy toast duration, 5-10. Default: 6.\n");
	::printf("  --perf-overlay                       Show the performance panel (F2 or Cmd+P).\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf("  --skip-notice-screen                 Skip startup logos and notices in supported games.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

using Config::ConfigOptions;

struct Flag {
	std::string_view name;
	bool ConfigOptions::* field;
	bool                  value = true;
};

static constexpr Flag FLAGS[] = {
    {"--rd", &ConfigOptions::renderdoc_enabled},
    {"--fullscreen", &ConfigOptions::fullscreen_enabled},
    {"--hdr", &ConfigOptions::hdr_enabled},
    {"--hide-cursor", &ConfigOptions::hide_cursor_enabled},
    {"--no-splash", &ConfigOptions::splash_enabled, false},
    {"--vr", &ConfigOptions::vr_enabled},
    {"--amd-cpu", &ConfigOptions::amd_cpu_enabled},
    {"--network-connected", &ConfigOptions::network_connected},
    {"--vulkan-validation", &ConfigOptions::vulkan_validation_enabled},
    {"--gpu-assisted-validation", &ConfigOptions::gpu_assisted_validation_enabled},
    {"--shader-validation", &ConfigOptions::shader_validation_enabled},
    {"--tessellation", &ConfigOptions::tessellation_enabled},
    {"--command-buffer-dump", &ConfigOptions::command_buffer_dump_enabled},
    {"--graphics-debug-dump", &ConfigOptions::graphics_debug_dump_enabled},
    {"--profile", &ConfigOptions::profiler_enabled},
    {"--spirv-debug-printf", &ConfigOptions::spirv_debug_printf_enabled},
    {"--readback-linear-images", &ConfigOptions::readback_linear_images},
    {"--readback-tiled-images", &ConfigOptions::readback_tiled_images},
    {"--sync-raw-image-buffers", &ConfigOptions::sync_raw_image_buffers},
    {"--no-async-pipelines", &ConfigOptions::async_pipelines_enabled, false},
    {"--skip-pending-draws", &ConfigOptions::skip_pending_draws},
    {"--precompile-pipelines", &ConfigOptions::precompile_pipelines},
    {"--precompile-only", &ConfigOptions::precompile_only},
    {"--serialize-dispatches", &ConfigOptions::serialize_dispatches},
    {"--no-trophy-notifications", &ConfigOptions::trophy_enabled, false},
    {"--perf-overlay", &ConfigOptions::perf_overlay_enabled},
    {"--playgo-hack", &ConfigOptions::playgo_hack_enabled},
    {"--skip-notice-screen", &ConfigOptions::skip_notice_screen},
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
    {"--redzone", &ConfigOptions::red_zone_protection_enabled},
#endif
};

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (const auto* flag = std::ranges::find(FLAGS, arg, &Flag::name); flag != std::end(FLAGS)) {
			options.config.*flag->field = flag->value;
			continue;
		}

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--guest-exec") {
			options.elf = Common::PathFromUtf8(Common::FixFilenameSlash(value));
			options.config.guest_arguments.assign(argv + i + 1, argv + argc);
			break;
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--addcont-dir") {
			if (!options.addcont_dir.empty()) {
				::printf("--addcont-dir can only be specified once\n");
				return false;
			}
			options.addcont_dir = Common::PathFromUtf8(Common::FixFilenameSlash(value));
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--keyboard-player") {
			if (!ParseUint32(value, options.config.keyboard_player) ||
			    options.config.keyboard_player > static_cast<uint32_t>(Config::MAX_PLAYERS)) {
				::printf("invalid keyboard player: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--lookahead-threads") {
			if (!ParseUint32(value, options.config.lookahead_threads) ||
			    options.config.lookahead_threads > 8u) {
				::printf("invalid lookahead thread count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--trophy-duration") {
			if (!ParseUint32(value, options.config.trophy_duration_seconds)) {
				::printf("invalid trophy duration: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

static int Main(int argc, char* argv[]) {
	VirtualMemory::Init();
	InitializeThreads();
#if defined(KYTY_PGO_GENERATE)
	// After the address-space reservations: its stack must not land in the guest range.
	StartPgoDumps();
#endif

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
