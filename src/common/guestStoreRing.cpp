#include "common/guestStoreRing.h"

#include <chrono>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace Common {

namespace {

GuestStoreRing g_guest_stores;

thread_local GuestStorePath g_path   = GuestStorePath::Backing;
thread_local uint64_t       g_source = 0;
thread_local uint32_t       g_thread = 0;

uint32_t CurrentThreadId() {
	if (g_thread == 0) {
#if defined(_WIN32)
		g_thread = static_cast<uint32_t>(GetCurrentThreadId());
#else
		g_thread = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_self()));
#endif
	}
	return g_thread;
}

uint64_t SteadyNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct ClockOrigin {
	uint64_t tsc = GuestStoreClock();
	uint64_t ns  = SteadyNs();
};

const ClockOrigin& Origin() {
	static const ClockOrigin origin;
	return origin;
}

} // namespace

GuestStoreRing& GuestStores() {
	return g_guest_stores;
}

GuestStorePath CurrentGuestStorePath() {
	return g_path;
}

uint64_t GuestStoreClock() {
	return __rdtsc();
}

uint64_t GuestStoreClockTicksPerUs() {
	const auto& origin = Origin();
	const auto  tsc    = GuestStoreClock();
	const auto  ns     = SteadyNs();
	if (ns <= origin.ns + 1000 || tsc <= origin.tsc) {
		return 0;
	}
	return (tsc - origin.tsc) * 1000 / (ns - origin.ns);
}

void RecordGuestStore(uint64_t address, uint64_t size, const void* data) {
	RecordGuestStore(g_path, address, size, data, g_source);
}

void RecordGuestStore(GuestStorePath path, uint64_t address, uint64_t size, const void* data,
                      uint64_t source) {
	(void)Origin();
	g_guest_stores.Record(path, address, size, data, CurrentThreadId(), GuestStoreClock(), source);
}

GuestStoreScope::GuestStoreScope(GuestStorePath path, uint64_t source)
    : m_previous(g_path), m_previous_source(g_source) {
	g_path   = path;
	g_source = source;
}

GuestStoreScope::~GuestStoreScope() {
	g_path   = m_previous;
	g_source = m_previous_source;
}

} // namespace Common
