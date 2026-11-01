#include "graphics/presentation/perfOverlay.h"

#include <SDL3/SDL.h>

#include "common/emulatorConfig.h"
#include "common/startupTrace.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "imgui.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#include <tlhelp32.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

using Clock   = std::chrono::steady_clock;
using Overlay = Common::OverlayStats::Counter;
using Startup = Common::StartupTrace::Counter;

constexpr auto     SAMPLE_PERIOD = std::chrono::milliseconds(500);
constexpr size_t   FRAME_RING    = 1024; // frame times kept (the graph shows the last GRAPH_FRAMES)
constexpr size_t   GRAPH_FRAMES  = 120;
constexpr size_t   LOW_FRAMES    = 600;  // 1% low and worst frame: over the last LOW_FRAMES frames
constexpr uint32_t TOP_THREADS   = 6;
constexpr double   GIB           = 1024.0 * 1024.0 * 1024.0;
constexpr double   MIB           = 1024.0 * 1024.0;

enum class Tone : uint8_t { Text, Dim, Good, Warn, Bad, Head };

struct Line {
	std::string text;
	Tone        tone = Tone::Text;
};

struct Column {
	const char*       title = "";
	std::vector<Line> lines;
};

// What the sampler built (read by the present thread).
struct Page {
	double                 fps      = -1.0; // < 0: no frame yet
	double                 frame_ms = 0.0;
	Line                   limit;
	Line                   precompile;
	std::array<Column, 8>  columns;
};

// F2 / Shift+F2.
std::atomic<bool> g_hidden {false};
std::atomic<bool> g_compact {false};

// Frames handed to the swapchain (present thread writes, the sampler and the draw read).
std::mutex                       g_frame_mutex;
std::array<float, FRAME_RING>    g_frame_ms {};
uint64_t                         g_frame_count = 0;
Clock::time_point                g_frame_last {};
uint32_t                         g_source_width  = 0;
uint32_t                         g_source_height = 0;
uint32_t                         g_swap_width    = 0;
uint32_t                         g_swap_height   = 0;
uint32_t                         g_swap_images   = 0;
bool                             g_swap_hdr      = false;
vk::PresentModeKHR               g_swap_mode     = vk::PresentModeKHR::eFifo;

// Vulkan side, read by the present thread every SAMPLE_PERIOD (the device outlives every present).
std::mutex                                               g_gpu_mutex;
std::string                                              g_device_name;
std::vector<GraphicContext::HeapBudget>                  g_heaps;
Clock::time_point                                        g_gpu_read {};

std::mutex                  g_page_mutex;
std::shared_ptr<const Page> g_page;

std::mutex              g_sampler_mutex;
std::condition_variable g_sampler_wake;
bool                    g_sampler_stop = false;
std::thread*            g_sampler = nullptr; // never destroyed while running (no std::terminate at exit)
std::once_flag          g_sampler_started;

std::string Rate(double value) {
	if (value >= 1e6) return fmt::format("{:.1f}M", value / 1e6);
	if (value >= 1e4) return fmt::format("{:.0f}k", value / 1e3);
	if (value >= 1e3) return fmt::format("{:.1f}k", value / 1e3);
	if (value >= 10.0 || value == 0.0) return fmt::format("{:.0f}", value);
	return fmt::format("{:.1f}", value);
}

// The difference of an only-growing counter since the previous sample.
struct Delta {
	uint64_t previous = 0;
	bool     primed   = false;
	uint64_t Take(uint64_t now) {
		const uint64_t delta = primed && now >= previous ? now - previous : 0;
		previous             = now;
		primed               = true;
		return delta;
	}
};

#if defined(_WIN32)
uint64_t FileTime100ns(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
}

std::string Utf8(const wchar_t* text) {
	const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
	if (size <= 1) return {};
	std::string result(static_cast<size_t>(size), '\0');
	WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
	result.resize(static_cast<size_t>(size - 1));
	return result;
}
#endif

class Sampler {
public:
	void Run() {
#if defined(_WIN32)
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
		Common::Thread::SetCurrentName("KytyPerfOverlay");
		auto previous = Clock::now();
		for (;;) {
			{
				std::unique_lock lock(g_sampler_mutex);
				if (g_sampler_wake.wait_for(lock, SAMPLE_PERIOD, [] { return g_sampler_stop; })) break;
			}
			if (g_hidden.load(std::memory_order_relaxed)) continue; // F2: nothing is sampled while hidden
			const auto now = Clock::now();
			const double dt = std::max(std::chrono::duration<double>(now - previous).count(), 1e-3);
			previous        = now;
			auto page       = std::make_shared<Page>();
			Sample(*page, dt);
			if (Common::OverlayStats::LogOnly() && now - m_last_log >= std::chrono::seconds(10)) {
				m_last_log = now;
				Log(*page);
			}
			std::scoped_lock lock(g_page_mutex);
			g_page = std::move(page);
		}
#if defined(_WIN32)
		for (auto& [id, thread]: m_threads) {
			if (thread.handle != nullptr) CloseHandle(thread.handle);
		}
		m_threads.clear();
#endif
	}

private:
	// KYTY_OVERLAY=2: the page as text lines in stderr.
	static void Log(const Page& page) {
		std::string text = fmt::format("PERF overlay fps={:.2f} frame_ms={:.1f} | {} | {}", page.fps, page.frame_ms,
		                               page.limit.text, page.precompile.text);
		for (const auto& column: page.columns) {
			if (column.lines.empty()) continue;
			text += fmt::format(" | {}:", column.title);
			for (const auto& line: column.lines) text += " " + line.text + ";";
		}
		std::fprintf(stderr, "%s\n", text.c_str());
	}
	Clock::time_point m_last_log {};

	struct ThreadState {
#if defined(_WIN32)
		HANDLE handle = nullptr;
#endif
		std::string name;
		uint64_t    cpu_100ns = 0;
		bool        seen      = false;
		bool        fresh     = true; // first sample: no difference yet
		double      pct       = 0.0;
	};
	struct ThreadRow {
		double      pct; // its work: CPU time less the busy-wait it reports (spin)
		std::string name;
		double      spin = 0.0;
	};

	std::array<Delta, static_cast<uint32_t>(Overlay::Count)>     m_overlay_count {};
	std::array<Delta, static_cast<uint32_t>(Overlay::Count)>     m_overlay_bytes {};
	std::array<Delta, static_cast<uint32_t>(Startup::Count)>     m_startup_count {};
	std::array<Delta, static_cast<uint32_t>(Startup::Count)>     m_startup_nanos {};
	std::array<Delta, static_cast<uint32_t>(Startup::Count)>     m_startup_bytes {};
	Delta                                                        m_unresolved;
	Libs::LibKernel::Memory::GuestMemoryStats                    m_guest {};
	bool                                                         m_guest_valid = false;
#if defined(_WIN32)
	Delta                                                        m_process_cpu;
	Delta                                                        m_system_idle;
	Delta                                                        m_system_total;
	Delta                                                        m_page_faults;
	Delta                                                        m_io_read;
	Delta                                                        m_io_write;
	std::unordered_map<uint32_t, ThreadState>                    m_threads;
#endif

	double OverlayRate(Overlay c, double dt) {
		return static_cast<double>(m_overlay_count[static_cast<uint32_t>(c)].Take(Common::OverlayStats::Count(c))) / dt;
	}
	double OverlayBytesRate(Overlay c, double dt) {
		return static_cast<double>(m_overlay_bytes[static_cast<uint32_t>(c)].Take(Common::OverlayStats::Bytes(c))) / dt;
	}
	// StartupTrace slot: count/s, ms/s and bytes/s since the previous sample (every slot is taken once per sample).
	struct StartupRate {
		double count = 0.0;
		double ms    = 0.0;
		double bytes = 0.0;
	};
	std::array<StartupRate, static_cast<uint32_t>(Startup::Count)> TakeStartup(double dt) {
		std::array<StartupRate, static_cast<uint32_t>(Startup::Count)> rates {};
		for (uint32_t i = 0; i < rates.size(); i++) {
			const auto& slot = Common::StartupTrace::g_slots[i];
			rates[i].count   = static_cast<double>(m_startup_count[i].Take(slot.count.load(std::memory_order_relaxed))) / dt;
			rates[i].ms = static_cast<double>(m_startup_nanos[i].Take(slot.nanos.load(std::memory_order_relaxed))) / 1e6 / dt;
			rates[i].bytes = static_cast<double>(m_startup_bytes[i].Take(slot.bytes.load(std::memory_order_relaxed))) / dt;
		}
		return rates;
	}
	static uint64_t Total(Startup c) {
		return Common::StartupTrace::g_slots[static_cast<uint32_t>(c)].count.load(std::memory_order_relaxed);
	}
	static double TotalMs(Startup c) {
		return static_cast<double>(
		           Common::StartupTrace::g_slots[static_cast<uint32_t>(c)].nanos.load(std::memory_order_relaxed)) /
		       1e6;
	}
	static double PeakMs(Startup c) {
		return static_cast<double>(
		           Common::StartupTrace::g_slots[static_cast<uint32_t>(c)].max_nanos.load(std::memory_order_relaxed)) /
		       1e6;
	}

	std::vector<ThreadRow> SampleThreads(double dt) {
		std::vector<ThreadRow> rows;
#if defined(_WIN32)
		for (auto& [id, thread]: m_threads) thread.seen = false;
		const DWORD self   = GetCurrentProcessId();
		HANDLE      handle = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (handle != INVALID_HANDLE_VALUE) {
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			for (BOOL ok = Thread32First(handle, &entry); ok; ok = Thread32Next(handle, &entry)) {
				if (entry.th32OwnerProcessID != self) continue;
				auto [it, inserted] = m_threads.try_emplace(entry.th32ThreadID);
				auto& thread        = it->second;
				if (inserted) {
					thread.handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
				}
				if (thread.handle == nullptr) continue;
				thread.seen = true;
				if (thread.name.empty()) {
					PWSTR description = nullptr;
					if (SUCCEEDED(GetThreadDescription(thread.handle, &description)) && description != nullptr) {
						thread.name = Utf8(description);
						LocalFree(description);
					}
				}
				FILETIME create {}, exit {}, kernel {}, user {};
				if (GetThreadTimes(thread.handle, &create, &exit, &kernel, &user) == 0) continue;
				const uint64_t cpu = FileTime100ns(kernel) + FileTime100ns(user);
				thread.pct         = thread.fresh || cpu < thread.cpu_100ns
				                         ? 0.0
				                         : static_cast<double>(cpu - thread.cpu_100ns) / 1e5 / dt;
				thread.cpu_100ns   = cpu;
				thread.fresh       = false;
			}
			CloseHandle(handle);
		}
		for (auto it = m_threads.begin(); it != m_threads.end();) {
			if (!it->second.seen) {
				if (it->second.handle != nullptr) CloseHandle(it->second.handle);
				it = m_threads.erase(it);
				continue;
			}
			if (it->second.pct > 0.0) {
				rows.push_back({it->second.pct, it->second.name.empty() ? fmt::format("tid {}", it->first)
				                                                        : it->second.name});
			}
			++it;
		}
		// The threads that busy-wait for work report the spin's time (OverlayStats::RecordSpin / PrepSpin): their CPU
		// time counts it as work, so it is taken off (split evenly between threads of the same name).
		for (const auto& [counter, name]: {std::pair {Overlay::RecordSpin, "Thread_VkRecord"},
		                                   std::pair {Overlay::PrepSpin, "Thread_PrepWorker"}}) {
			const double spin = OverlayBytesRate(counter, dt) / 1e7; // ns/s -> % of a core
			const auto   same = std::ranges::count_if(rows, [&](const ThreadRow& r) { return r.name == name; });
			if (same == 0) continue;
			for (auto& row: rows) {
				if (row.name != name) continue;
				row.spin = std::min(spin / static_cast<double>(same), row.pct);
				row.pct -= row.spin;
			}
		}
		std::ranges::sort(rows, [](const ThreadRow& a, const ThreadRow& b) { return a.pct > b.pct; });
		if (rows.size() > TOP_THREADS) rows.resize(TOP_THREADS);
#else
		(void)dt;
#endif
		return rows;
	}

	void Sample(Page& page, double dt) {
		auto& frames  = page.columns[0];
		auto& cpu     = page.columns[1];
		auto& gpu     = page.columns[2];
		auto& gpu_mem = page.columns[3];
		auto& guest   = page.columns[4];
		auto& host    = page.columns[5];
		auto& io      = page.columns[6];
		auto& threads = page.columns[7];
		frames.title  = "Frames";
		cpu.title     = "CPU";
		gpu.title     = "GPU";
		gpu_mem.title = "GPU memory";
		guest.title   = "PS5 memory";
		host.title    = "Emulator process";
		io.title      = "I/O & audio";
		threads.title = "Busiest threads (% of a core)";

		// Frames: FPS = frames of the last second of frame times; 1% low / worst over the last LOW_FRAMES.
		std::vector<float> recent;
		uint32_t           source_w = 0, source_h = 0, swap_w = 0, swap_h = 0, swap_images = 0;
		bool               swap_hdr = false;
		auto               mode     = vk::PresentModeKHR::eFifo;
		double             idle_s   = 0.0;
		{
			std::scoped_lock lock(g_frame_mutex);
			const size_t     n = static_cast<size_t>(std::min<uint64_t>(g_frame_count, LOW_FRAMES));
			recent.reserve(n);
			for (size_t i = 0; i < n; i++) recent.push_back(g_frame_ms[(g_frame_count - 1 - i) % FRAME_RING]);
			source_w    = g_source_width;
			source_h    = g_source_height;
			swap_w      = g_swap_width;
			swap_h      = g_swap_height;
			swap_images = g_swap_images;
			swap_hdr    = g_swap_hdr;
			mode        = g_swap_mode;
			if (g_frame_count != 0) idle_s = std::chrono::duration<double>(Clock::now() - g_frame_last).count();
		}
		if (!recent.empty()) {
			double sum = 0.0;
			size_t n   = 0;
			for (; n < recent.size() && sum < 1000.0; n++) sum += recent[n];
			if (idle_s > 1.0) {
				page.fps      = 0.0;
				page.frame_ms = idle_s * 1000.0;
			} else {
				page.fps      = sum > 0.0 ? 1000.0 * static_cast<double>(n) / sum : 0.0;
				page.frame_ms = n != 0 ? sum / static_cast<double>(n) : 0.0;
			}
		}
		if (recent.size() >= 100) {
			std::vector<float> sorted = recent;
			std::ranges::sort(sorted);
			const auto p99 = sorted[std::min(sorted.size() - 1, (sorted.size() * 99 + 99) / 100 - 1)];
			frames.lines.push_back({fmt::format("1% low  {:.1f} FPS", p99 > 0.0f ? 1000.0 / p99 : 0.0)});
			frames.lines.push_back({fmt::format("worst   {:.1f} ms  (last {} frames)", sorted.back(), sorted.size())});
		}
		if (source_w != 0) frames.lines.push_back({fmt::format("guest   {}x{}", source_w, source_h)});
		if (swap_w != 0) {
			frames.lines.push_back({fmt::format("window  {}x{} {}{} {} img", swap_w, swap_h, vk::to_string(mode),
			                                    swap_hdr ? " HDR" : "", swap_images)});
		}
		frames.lines.push_back({fmt::format("vblank  {} Hz (emulated)", Config::GetVblankFrequency())});
		const double flips      = OverlayRate(Overlay::FlipRequest, dt);
		const double queue_full = OverlayRate(Overlay::FlipQueueFull, dt);
		const double late       = OverlayRate(Overlay::LateVblank, dt);
		frames.lines.push_back({fmt::format("flips requested {}/s", Rate(flips))});
		frames.lines.push_back({fmt::format("flip queue full {}/s", Rate(queue_full)), queue_full > 0.0 ? Tone::Warn : Tone::Text});
		frames.lines.push_back({fmt::format("late vblanks    {}/s", Rate(late)), late > 0.0 ? Tone::Warn : Tone::Text});
		frames.lines.push_back({fmt::format("swapchain rebuilds {}", Common::OverlayStats::Count(Overlay::SwapchainRebuild))});

		// CPU (OS counters).
		const auto thread_rows = SampleThreads(dt);
#if defined(_WIN32)
		{
			const uint32_t logical = std::max<uint32_t>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS), 1u);
			FILETIME       create {}, exit {}, kernel {}, user {};
			if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user) != 0) {
				const double cores =
				    static_cast<double>(m_process_cpu.Take(FileTime100ns(kernel) + FileTime100ns(user))) / 1e7 / dt;
				cpu.lines.push_back({fmt::format("emulator {:.0f}% of {} logical CPUs ({:.1f} cores)",
				                                 100.0 * cores / logical, logical, cores)});
			}
			FILETIME idle {}, sys_kernel {}, sys_user {};
			if (GetSystemTimes(&idle, &sys_kernel, &sys_user) != 0) {
				const auto i = static_cast<double>(m_system_idle.Take(FileTime100ns(idle)));
				const auto t = static_cast<double>(m_system_total.Take(FileTime100ns(sys_kernel) + FileTime100ns(sys_user)));
				if (t > 0.0) cpu.lines.push_back({fmt::format("machine  {:.0f}%", 100.0 * (t - i) / t)});
			}
			PROCESS_MEMORY_COUNTERS_EX memory {};
			if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
			                            sizeof(memory)) != 0) {
				cpu.lines.push_back(
				    {fmt::format("page faults {}/s", Rate(static_cast<double>(m_page_faults.Take(memory.PageFaultCount)) / dt))});
				host.lines.push_back({fmt::format("footprint {:.2f} GB (private)", static_cast<double>(memory.PrivateUsage) / GIB)});
				host.lines.push_back({fmt::format("resident  {:.2f} GB (working set)", static_cast<double>(memory.WorkingSetSize) / GIB)});
			}
			MEMORYSTATUSEX status {};
			status.dwLength = sizeof(status);
			if (GlobalMemoryStatusEx(&status) != 0) {
				host.lines.push_back({"System memory", Tone::Head});
				const auto total = static_cast<double>(status.ullTotalPhys);
				const auto avail = static_cast<double>(status.ullAvailPhys);
				host.lines.push_back({fmt::format("used {:.1f} / {:.1f} GB", (total - avail) / GIB, total / GIB)});
				host.lines.push_back({fmt::format("free {:.1f} GB", avail / GIB)});
				host.lines.push_back({fmt::format("pressure: load {}%, commit {:.1f} / {:.1f} GB", status.dwMemoryLoad,
				                                  static_cast<double>(status.ullTotalPageFile - status.ullAvailPageFile) / GIB,
				                                  static_cast<double>(status.ullTotalPageFile) / GIB),
				                      status.dwMemoryLoad >= 90 ? Tone::Bad : Tone::Text});
			}
			IO_COUNTERS counters {};
			if (GetProcessIoCounters(GetCurrentProcess(), &counters) != 0) {
				io.lines.push_back({fmt::format("process I/O read {:.1f} MB/s, write {:.1f} MB/s",
				                                static_cast<double>(m_io_read.Take(counters.ReadTransferCount)) / MIB / dt,
				                                static_cast<double>(m_io_write.Take(counters.WriteTransferCount)) / MIB / dt)});
			}
		}
#endif
		const uint32_t unresolved = ::Loader::UnresolvedImportStubCalls();
		const double   missing    = static_cast<double>(m_unresolved.Take(unresolved)) / dt;
		cpu.lines.push_back({fmt::format("missing HLE calls {} ({}/s)", unresolved, Rate(missing)),
		                     missing > 0.0 ? Tone::Warn : Tone::Text});

		// GPU: the emulator's counters (per frame = per second / FPS).
		const double fps       = page.fps > 0.0 ? page.fps : 0.0;
		const auto   per_frame = [&](double rate) { return fps > 0.0 ? Rate(rate / fps) : std::string("-"); };
		{
			std::scoped_lock lock(g_gpu_mutex);
			if (!g_device_name.empty()) gpu.lines.push_back({g_device_name});
			uint64_t usage = 0, budget = 0;
			for (const auto& heap: g_heaps) {
				if (!heap.device_local) continue;
				usage += heap.usage;
				budget += heap.budget;
			}
			if (budget != 0) {
				gpu_mem.lines.push_back({fmt::format("emu VRAM {:.1f} / {:.1f} GB", static_cast<double>(usage) / GIB,
				                                     static_cast<double>(budget) / GIB),
				                         usage * 10 >= budget * 9 ? Tone::Warn : Tone::Text});
			}
			for (size_t i = 0; i < g_heaps.size(); i++) {
				const auto& heap = g_heaps[i];
				gpu_mem.lines.push_back({fmt::format("heap {} {} {:.2f} / {:.2f} GB ({} alloc, {} blocks)", i,
				                                     heap.device_local ? "VRAM" : "host", static_cast<double>(heap.usage) / GIB,
				                                     static_cast<double>(heap.budget) / GIB, heap.allocations, heap.blocks)});
			}
		}
		const double draws      = OverlayRate(Overlay::Draw, dt);
		const double instances  = OverlayRate(Overlay::Instance, dt);
		const double indirect   = OverlayRate(Overlay::IndirectDraw, dt);
		const double dispatches = OverlayRate(Overlay::Dispatch, dt);
		const double submits    = OverlayRate(Overlay::QueueSubmit, dt);
		const double faults     = OverlayRate(Overlay::GpuMemoryFault, dt);
		gpu.lines.push_back({fmt::format("draws      {}/frame  {}/s", per_frame(draws), Rate(draws))});
		gpu.lines.push_back({fmt::format("instances  {}/frame", per_frame(instances))});
		gpu.lines.push_back({fmt::format("indirect   {}/frame (packets)", per_frame(indirect))});
		gpu.lines.push_back({fmt::format("dispatches {}/frame  {}/s", per_frame(dispatches), Rate(dispatches))});
		gpu.lines.push_back({fmt::format("queue submits {}/frame", per_frame(submits))});
		gpu.lines.push_back({fmt::format("tracking faults {}/s", Rate(faults))});

		// The StartupTrace counters (unless its reporter takes them).
		double fence_share = -1.0;
		double gpu_busy    = -1.0; // the GPU thread's share of time in its work (waits for the host GPU included)
		if (Common::StartupTrace::g_reporter_running.load(std::memory_order_relaxed)) {
			gpu.lines.push_back({"(KYTY_TRACE_STARTUP takes the timing counters)", Tone::Dim});
		} else {
			const auto s   = TakeStartup(dt);
			const auto at  = [&](Startup c) -> const StartupRate& { return s[static_cast<uint32_t>(c)]; };
			fence_share    = at(Startup::GpuFence).ms / 1000.0;
			gpu_busy       = at(Startup::GpuBusy).ms / 1000.0;
			gpu.lines.push_back({fmt::format("GPU thread busy {:.0f}%", at(Startup::GpuBusy).ms / 10.0)});
			gpu.lines.push_back({fmt::format("GPU thread waiting on GPU {:.0f} ms/s, {} waits/s", at(Startup::GpuFence).ms,
			                                 Rate(at(Startup::GpuFence).count)),
			                     fence_share >= 0.35 ? Tone::Warn : Tone::Text});
			const auto avg = [](double ms, uint64_t n) { return n != 0 ? ms / static_cast<double>(n) : 0.0; };
			const uint64_t shaders = Total(Startup::Shader);
			gpu.lines.push_back({fmt::format("shaders {} (avg {:.1f}, peak {:.0f} ms) +{} bg", shaders,
			                                 avg(TotalMs(Startup::Shader), shaders), PeakMs(Startup::Shader),
			                                 Total(Startup::ShaderBackground))});
			const uint64_t pipes = Total(Startup::PipeGraphics) + Total(Startup::PipeCompute);
			gpu.lines.push_back({fmt::format("pipelines {} (avg {:.1f}, peak {:.0f} ms) +{} bg, {} slow", pipes,
			                                 avg(TotalMs(Startup::PipeGraphics) + TotalMs(Startup::PipeCompute), pipes),
			                                 std::max(PeakMs(Startup::PipeGraphics), PeakMs(Startup::PipeCompute)),
			                                 Total(Startup::PipeBackground), Total(Startup::PipeSlow))});
			if (Total(Startup::PrecompileWait) != 0) {
				gpu.lines.push_back({fmt::format("precompile waits {} ({:.0f} ms)", Total(Startup::PrecompileWait),
				                                 TotalMs(Startup::PrecompileWait))});
			}
			gpu_mem.lines.push_back({fmt::format("up   tex {:.1f} MB/s, buf {:.1f} MB/s", at(Startup::ImageUpload).bytes / MIB,
			                                     at(Startup::BufferUpload).bytes / MIB)});
			gpu_mem.lines.push_back({fmt::format("down readbacks {}/s ({:.0f} ms/s)", Rate(at(Startup::GpuReadback).count),
			                                     at(Startup::GpuReadback).ms)});
			io.lines.push_back({fmt::format("guest file read {:.1f} MB/s (kernel {:.1f}, APR {:.1f})",
			                                (at(Startup::KernelRead).bytes + at(Startup::AprRead).bytes) / MIB,
			                                at(Startup::KernelRead).bytes / MIB, at(Startup::AprRead).bytes / MIB)});
		}
		const double image_evict = OverlayRate(Overlay::ImageEvict, dt);
		const double image_bytes = OverlayBytesRate(Overlay::ImageEvict, dt);
		const double buf_evict   = OverlayRate(Overlay::BufferEvict, dt);
		const double buf_bytes   = OverlayBytesRate(Overlay::BufferEvict, dt);
		gpu_mem.lines.push_back({fmt::format("evictions tex {}/s ({:.1f} MB/s), buf {}/s ({:.1f} MB/s)", Rate(image_evict),
		                                     image_bytes / MIB, Rate(buf_evict), buf_bytes / MIB)});

		// PS5 memory.
		Libs::LibKernel::Memory::GuestMemoryStats stats {};
		if (Libs::LibKernel::Memory::TryGetGuestMemoryStats(&stats)) {
			m_guest       = stats;
			m_guest_valid = true;
		}
		if (m_guest_valid) {
			const auto& m = m_guest;
			guest.lines.push_back({fmt::format("direct   {:.2f} / {:.2f} GB", static_cast<double>(m.direct_allocated) / GIB,
			                                   static_cast<double>(m.direct_total) / GIB)});
			guest.lines.push_back({fmt::format("mapped   {:.2f} GB", static_cast<double>(m.direct_mapped) / GIB)});
			guest.lines.push_back({fmt::format("pool     {:.2f} committed / {:.2f} GB", static_cast<double>(m.pool_committed) / GIB,
			                                   static_cast<double>(m.pool_expanded) / GIB)});
			guest.lines.push_back({fmt::format("flexible {:.0f} / {:.0f} MB", static_cast<double>(m.flexible_used) / MIB,
			                                   static_cast<double>(m.flexible_total) / MIB)});
			guest.lines.push_back({fmt::format("page table cpu {} / {}, gpu {} / {}", m.page_table_cpu, m.page_table_total,
			                                   m.page_table_gpu, m.page_table_total)});
		}

		// I/O & audio.
		io.lines.push_back({fmt::format("guest file write {:.2f} MB/s ({}/s)",
		                                OverlayBytesRate(Overlay::GuestFileWrite, dt) / MIB,
		                                Rate(OverlayRate(Overlay::GuestFileWrite, dt)))});
		const double starved = OverlayRate(Overlay::AudioStarved, dt);
		io.lines.push_back({fmt::format("audio queue starved {} ({}/s)", Common::OverlayStats::Count(Overlay::AudioStarved),
		                                Rate(starved)),
		                    starved > 0.0 ? Tone::Warn : Tone::Text});

		// Busiest threads.
		for (const auto& row: thread_rows) {
			threads.lines.push_back({row.spin >= 0.5 ? fmt::format("{:4.0f}% {} (+{:.0f}% spin)", row.pct, row.name, row.spin)
			                                         : fmt::format("{:4.0f}% {}", row.pct, row.name),
			                         row.pct >= 90.0 ? Tone::Bad : row.pct >= 60.0 ? Tone::Warn : Tone::Text});
		}

		// What limits the speed (as scripts/perf_overlay.ps1 decided it).
		if (!thread_rows.empty() && thread_rows.front().pct >= 90.0) {
			page.limit = {fmt::format("Limit: CPU ({} {:.0f}%)", thread_rows.front().name, thread_rows.front().pct), Tone::Bad};
		} else if (fence_share >= 0.35) {
			page.limit = {fmt::format("Limit: GPU (fence wait {:.0f}% of the time)", 100.0 * fence_share), Tone::Head};
		} else if (gpu_busy >= 0.9) {
			page.limit = {fmt::format("Limit: GPU thread (busy {:.0f}%, of it waiting on GPU {:.0f}%)", 100.0 * gpu_busy,
			                          100.0 * std::max(fence_share, 0.0)),
			              Tone::Bad};
		} else {
			page.limit = {"Limit: -  (no thread >= 90%, GPU not waited for)", Tone::Dim};
		}

		// Background precompile.
		if (!ShaderPrecompiler::Enabled()) {
			page.precompile = {"Precompile: off", Tone::Dim};
		} else if (const auto progress = GetPrecompileProgress(); !progress.started) {
			page.precompile = {"Precompile: not started", Tone::Dim};
		} else if (progress.done) {
			page.precompile = {fmt::format("Precompile: done, {} compiled in {} s", progress.compiled, progress.seconds),
			                   Tone::Dim};
		} else {
			page.precompile = {fmt::format("Precompile: running, {} compiled, {} queued ({} thr)", progress.compiled,
			                               progress.queued, progress.threads),
			                   Tone::Warn};
		}
	}
};

void StartSampler() {
	std::call_once(g_sampler_started, [] {
		std::scoped_lock lock(g_sampler_mutex);
		if (!g_sampler_stop) g_sampler = new std::thread([] { Sampler().Run(); });
	});
}

ImU32 ToneColor(Tone tone) {
	switch (tone) {
		case Tone::Dim: return IM_COL32(150, 155, 165, 255);
		case Tone::Good: return IM_COL32(110, 220, 120, 255);
		case Tone::Warn: return IM_COL32(240, 200, 80, 255);
		case Tone::Bad: return IM_COL32(240, 90, 80, 255);
		case Tone::Head: return IM_COL32(130, 190, 255, 255);
		default: return IM_COL32(230, 232, 236, 255);
	}
}

void DrawLine(const Line& line) {
	ImGui::PushStyleColor(ImGuiCol_Text, ToneColor(line.tone));
	ImGui::TextUnformatted(line.text.c_str());
	ImGui::PopStyleColor();
}

// The last GRAPH_FRAMES frame times as bars: green up to 1.5x the median, yellow up to 2.5x, red above.
void DrawFrameGraph(float width, float height) {
	std::array<float, GRAPH_FRAMES> values {};
	size_t                          count = 0;
	{
		std::scoped_lock lock(g_frame_mutex);
		count = static_cast<size_t>(std::min<uint64_t>(g_frame_count, GRAPH_FRAMES));
		for (size_t i = 0; i < count; i++) {
			values[count - 1 - i] = g_frame_ms[(g_frame_count - 1 - i) % FRAME_RING];
		}
	}
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	ImGui::Dummy({width, height});
	auto* draw = ImGui::GetWindowDrawList();
	draw->AddRectFilled(origin, {origin.x + width, origin.y + height}, IM_COL32(0, 0, 0, 90), 3.0f);
	if (count == 0) return;
	std::array<float, GRAPH_FRAMES> sorted = values;
	std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(count));
	const float median = std::max(sorted[count / 2], 0.1f);
	const float top    = std::clamp(sorted[count - 1] * 1.15f, 34.0f, 250.0f);
	for (const float guide: {16.7f, 33.3f}) {
		if (guide >= top) continue;
		const float y = origin.y + height - height * guide / top;
		draw->AddLine({origin.x, y}, {origin.x + width, y}, IM_COL32(255, 255, 255, 40));
	}
	const float bar = width / static_cast<float>(GRAPH_FRAMES);
	for (size_t i = 0; i < count; i++) {
		const float ms    = values[i];
		const float h     = std::max(1.0f, height * std::min(ms, top) / top);
		const float x     = origin.x + width - bar * static_cast<float>(count - i);
		const ImU32 color = ms <= median * 1.5f   ? ToneColor(Tone::Good)
		                    : ms <= median * 2.5f ? ToneColor(Tone::Warn)
		                                          : ToneColor(Tone::Bad);
		draw->AddRectFilled({x, origin.y + height - h}, {x + std::max(bar - 1.0f, 1.0f), origin.y + height}, color);
	}
}

void ReadGpuInfo(const GraphicContext& graphics) {
	const auto now = Clock::now();
	std::scoped_lock lock(g_gpu_mutex);
	if (g_gpu_read != Clock::time_point {} && now - g_gpu_read < SAMPLE_PERIOD) return;
	g_gpu_read = now;
	if (g_device_name.empty()) g_device_name = graphics.physical_device_properties.deviceName.data();
	std::array<GraphicContext::HeapBudget, VK_MAX_MEMORY_HEAPS> heaps {};
	const uint32_t count = graphics.GetHeapBudgets(heaps.data(), static_cast<uint32_t>(heaps.size()));
	g_heaps.assign(heaps.begin(), heaps.begin() + count);
}

} // namespace

bool PerfOverlayVisible() noexcept {
	return PerfOverlayEnabled() && !Common::OverlayStats::LogOnly() && !g_hidden.load(std::memory_order_relaxed);
}

bool ProcessPerfOverlayKey(const SDL_Event& event) {
	if (!PerfOverlayEnabled() || (event.type != SDL_EVENT_KEY_DOWN && event.type != SDL_EVENT_KEY_UP) ||
	    event.key.key != SDLK_F2) {
		return false;
	}
	if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
		if ((event.key.mod & SDL_KMOD_SHIFT) != 0) {
			g_compact.store(!g_compact.load(std::memory_order_relaxed), std::memory_order_relaxed);
			g_hidden.store(false, std::memory_order_relaxed);
		} else {
			g_hidden.store(!g_hidden.load(std::memory_order_relaxed), std::memory_order_relaxed);
		}
	}
	return true;
}

void PerfOverlayNoteFrame(uint32_t width, uint32_t height) {
	if (!PerfOverlayEnabled()) return;
	StartSampler(); // KYTY_OVERLAY=2 has no overlay pass to start it (once; a no-op afterwards)
	const auto       now = Clock::now();
	std::scoped_lock lock(g_frame_mutex);
	if (g_frame_last != Clock::time_point {}) {
		g_frame_ms[g_frame_count % FRAME_RING] =
		    std::chrono::duration<float, std::milli>(now - g_frame_last).count();
		g_frame_count++;
	}
	g_frame_last    = now;
	g_source_width  = width;
	g_source_height = height;
}

void PerfOverlayNoteSwapchain(uint32_t width, uint32_t height, vk::PresentModeKHR mode, uint32_t images, bool hdr) {
	if (!PerfOverlayEnabled()) return;
	std::scoped_lock lock(g_frame_mutex);
	g_swap_width  = width;
	g_swap_height = height;
	g_swap_mode   = mode;
	g_swap_images = images;
	g_swap_hdr    = hdr;
}

void PerfOverlayTick(const GraphicContext& graphics) {
	if (!PerfOverlayEnabled()) return;
	StartSampler();
	ReadGpuInfo(graphics);
}

void DrawPerfOverlay(const GraphicContext& graphics, vk::Extent2D extent) {
	if (!PerfOverlayVisible()) return;
	std::shared_ptr<const Page> page;
	{
		std::scoped_lock lock(g_page_mutex);
		page = g_page;
	}
	const bool  compact = g_compact.load(std::memory_order_relaxed);
	const float scale   = std::clamp(static_cast<float>(extent.height) / 1080.0f, 0.75f, 2.0f);
	ImGui::SetNextWindowPos({12.0f * scale, 12.0f * scale}, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.62f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10.0f * scale, 8.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {6.0f * scale, 2.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f * scale);
	ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {8.0f * scale, 2.0f * scale});
	ImGui::PushFont(nullptr, 14.0f * scale);
	constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
	                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
	                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
	                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_AlwaysAutoResize;
	ImGui::Begin("##PerfOverlay", nullptr, flags);

	ImGui::PushFont(nullptr, 26.0f * scale);
	if (page == nullptr || page->fps < 0.0) {
		DrawLine({"-- FPS", Tone::Dim});
	} else {
		const Tone tone = page->fps >= 50.0 ? Tone::Good : page->fps >= 25.0 ? Tone::Warn : Tone::Bad;
		DrawLine({fmt::format("{:.1f} FPS  {:.1f} ms", page->fps, page->frame_ms), tone});
	}
	ImGui::PopFont();
	DrawFrameGraph(360.0f * scale, 56.0f * scale);
	if (page != nullptr) {
		DrawLine(page->limit);
		if (!compact) {
			DrawLine(page->precompile);
			constexpr ImGuiTableFlags table_flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings;
			if (ImGui::BeginTable("##PerfColumns", 4, table_flags)) {
				for (size_t row = 0; row < 2; row++) {
					ImGui::TableNextRow();
					for (size_t column = 0; column < 4; column++) {
						ImGui::TableSetColumnIndex(static_cast<int>(column));
						const auto& cell = page->columns[row * 4 + column];
						DrawLine({cell.title, Tone::Head});
						for (const auto& line: cell.lines) DrawLine(line);
					}
				}
				ImGui::EndTable();
			}
		}
	}
	DrawLine({compact ? "Shift+F2 full  F2 hide" : "Shift+F2 compact  F2 hide", Tone::Dim});
	ImGui::End();
	ImGui::PopFont();
	ImGui::PopStyleVar(4);
}

void StopPerfOverlay() {
	std::thread* sampler = nullptr;
	{
		std::scoped_lock lock(g_sampler_mutex);
		g_sampler_stop = true;
		sampler        = std::exchange(g_sampler, nullptr);
	}
	g_sampler_wake.notify_all();
	if (sampler != nullptr) {
		sampler->join();
		delete sampler;
	}
}

} // namespace Libs::Graphics
