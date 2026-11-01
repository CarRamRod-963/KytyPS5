// KYTY_TRACE_HLE_COUNT (libs.h HLE_COUNT): per-function call counts and time of HLE functions. In its own file so the
// unit tests that compile single library sources (ngs2, audio out 2, pad) link it without libs.cpp.
#include "libs/libs.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs {

namespace {

std::atomic<HleCounter*> g_hle_counters {nullptr};

// Prints the functions with the most time and the most calls since the last report.
void HleReportLoop() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		struct Row {
			HleCounter* counter;
			uint64_t    calls;
			uint64_t    ns;
		};
		std::vector<Row> rows;
		for (auto* c = g_hle_counters.load(std::memory_order_acquire); c != nullptr; c = c->next) {
			const auto calls = c->calls.load(std::memory_order_relaxed);
			const auto ns    = c->ns.load(std::memory_order_relaxed);
			rows.push_back({c, calls - c->last_calls, ns - c->last_ns});
			c->last_calls = calls;
			c->last_ns    = ns;
		}
		uint64_t total_calls = 0;
		uint64_t total_ns    = 0;
		for (const auto& r: rows) {
			total_calls += r.calls;
			total_ns += r.ns;
		}
		const auto print = [&](const char* kind, auto&& key) {
			std::sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) { return key(a) > key(b); });
			std::fprintf(stderr, "HLE_TOP by=%s window_s=5 total_calls=%llu total_ms=%.1f", kind,
			             static_cast<unsigned long long>(total_calls), static_cast<double>(total_ns) / 1e6);
			for (size_t i = 0; i < rows.size() && i < 25u; i++) {
				if (rows[i].calls == 0) break;
				std::fprintf(stderr, " [%s::%s n=%llu ms=%.1f]", rows[i].counter->library, rows[i].counter->function,
				             static_cast<unsigned long long>(rows[i].calls), static_cast<double>(rows[i].ns) / 1e6);
			}
			std::fprintf(stderr, "\n");
		};
		print("time", [](const Row& r) { return r.ns; });
		print("calls", [](const Row& r) { return r.calls; });
	}
}

} // namespace

uint64_t HleNowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

HleCounter::HleCounter(const char* library_name, const char* function_name): library(library_name), function(function_name) {
	auto* head = g_hle_counters.load(std::memory_order_relaxed);
	do {
		next = head;
	} while (!g_hle_counters.compare_exchange_weak(head, this, std::memory_order_release, std::memory_order_relaxed));
	static std::once_flag started;
	std::call_once(started, [] {
		std::fprintf(stderr, "TRACE_HLE_COUNT enabled\n");
		std::thread(HleReportLoop).detach();
	});
}

} // namespace Libs
