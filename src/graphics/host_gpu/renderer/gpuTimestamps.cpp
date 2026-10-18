#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <chrono>
#include <cmath>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics {

namespace {

constexpr int64_t WINDOW_NS      = 1'000'000'000;
constexpr double  LOG_INTERVAL_S = 10.0;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// The calibrated host domain that std::chrono::steady_clock reads: QueryPerformanceCounter on
// Windows, CLOCK_MONOTONIC with libstdc++. Other platforms report GPU busy time only.
std::optional<vk::TimeDomainKHR> SteadyClockDomain() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return vk::TimeDomainKHR::eQueryPerformanceCounter;
#elif defined(__linux__)
	return vk::TimeDomainKHR::eClockMonotonic;
#else
	return std::nullopt;
#endif
}

PFN_vkGetCalibratedTimestampsKHR CalibratedTimestampsFunction() {
	const auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	return dispatcher.vkGetCalibratedTimestampsKHR != nullptr
	           ? dispatcher.vkGetCalibratedTimestampsKHR
	           : dispatcher.vkGetCalibratedTimestampsEXT;
}

bool SupportsTimeDomain(vk::PhysicalDevice device, vk::TimeDomainKHR domain) {
	const auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	auto        enumerate  = dispatcher.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR != nullptr
	                             ? dispatcher.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
	                             : dispatcher.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
	if (enumerate == nullptr) {
		return false;
	}
	uint32_t count = 0;
	if (enumerate(device, &count, nullptr) != VK_SUCCESS || count == 0) {
		return false;
	}
	std::vector<VkTimeDomainKHR> domains(count);
	if (enumerate(device, &count, domains.data()) != VK_SUCCESS) {
		return false;
	}
	bool has_device = false;
	bool has_domain = false;
	for (uint32_t i = 0; i < count; i++) {
		has_device = has_device || domains[i] == VK_TIME_DOMAIN_DEVICE_KHR;
		has_domain = has_domain || domains[i] == static_cast<VkTimeDomainKHR>(domain);
	}
	return has_device && has_domain;
}

} // namespace

GpuTimestamps::GpuTimestamps(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	uint32_t family_count = 0;
	graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
	const uint32_t valid_bits = graphics.queue_family < family_count
	                                ? families[graphics.queue_family].timestampValidBits
	                                : 0;
	const float    period     = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	if (valid_bits == 0 || !(period > 0.0f)) {
		LOGF("GpuTiming: the queue has no timestamps, GPU timing disabled\n");
		return;
	}
	m_valid_mask = valid_bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << valid_bits) - 1;
	m_period_ns  = period;

	vk::QueryPoolCreateInfo create {};
	create.queryType  = vk::QueryType::eTimestamp;
	create.queryCount = RING_SIZE * 2;
	if (graphics.device.createQueryPool(&create, nullptr, &m_pool) != vk::Result::eSuccess) {
		m_pool = nullptr;
		LOGF("GpuTiming: could not create a timestamp query pool, GPU timing disabled\n");
		return;
	}

	const auto domain = SteadyClockDomain();
	if (graphics.calibrated_timestamps_enabled && domain && CalibratedTimestampsFunction() &&
	    SupportsTimeDomain(graphics.physical_device, *domain)) {
		m_host_domain = *domain;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		LARGE_INTEGER frequency {};
		QueryPerformanceFrequency(&frequency);
		m_host_frequency = static_cast<uint64_t>(frequency.QuadPart);
#endif
		m_calibration = true;
		Calibrate();
	}
	LOGF("GpuTiming: %u-bit timestamps, %.3f ns/tick, %s\n", valid_bits, m_period_ns,
	     m_calibrated ? "calibrated to the host clock" : "durations only (no calibration)");
	m_window_start_ns = NowNs();
	m_window_flips    = Common::GpuTiming::GuestFlips();
	m_accumulator.SetCalibrated(m_calibrated);
}

GpuTimestamps::~GpuTimestamps() {
	if (m_pool != nullptr) {
		m_graphics.device.destroyQueryPool(m_pool, nullptr);
	}
}

void GpuTimestamps::Calibrate() {
	std::array<VkCalibratedTimestampInfoKHR, 2> infos {};
	infos[0].sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
	infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
	infos[1].sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
	infos[1].timeDomain = static_cast<VkTimeDomainKHR>(m_host_domain);
	std::array<uint64_t, 2> timestamps {};
	uint64_t                deviation = 0;
	if (CalibratedTimestampsFunction()(m_graphics.device, 2, infos.data(), timestamps.data(),
	                                   &deviation) != VK_SUCCESS) {
		return;
	}
	const uint64_t host = timestamps[1];
	if (m_host_frequency != 0) {
		const uint64_t whole = host / m_host_frequency;
		const uint64_t part  = host % m_host_frequency;
		m_cal_host_ns        = static_cast<int64_t>(whole * 1'000'000'000) +
		                       static_cast<int64_t>(static_cast<double>(part) * 1e9 /
		                                            static_cast<double>(m_host_frequency));
	} else {
		m_cal_host_ns = static_cast<int64_t>(host);
	}
	m_cal_device = timestamps[0];
	m_calibrated = true;
}

uint64_t GpuTimestamps::DeviceNs(uint64_t ticks) const {
	return static_cast<uint64_t>(static_cast<double>(ticks & m_valid_mask) * m_period_ns);
}

std::optional<int64_t> GpuTimestamps::HostNs(uint64_t device_ticks) const {
	if (!m_calibrated) {
		return std::nullopt;
	}
	// Signed distance from the calibration point within the valid timestamp bits.
	uint64_t delta = (device_ticks - m_cal_device) & m_valid_mask;
	if (m_valid_mask != ~uint64_t {0} && (delta & ((m_valid_mask >> 1u) + 1)) != 0) {
		delta |= ~m_valid_mask;
	}
	return m_cal_host_ns + static_cast<int64_t>(std::llround(
	                           static_cast<double>(static_cast<int64_t>(delta)) * m_period_ns));
}

void GpuTimestamps::Begin(vk::CommandBuffer buffer, uint64_t tick) {
	if (m_pool == nullptr) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	auto&            slot = SlotOf(tick);
	if (slot.tick > m_collected) {
		Collect();
		if (slot.tick > m_collected) {
			// More submissions in flight than the ring holds; leave this one untimed.
			return;
		}
	}
	slot             = {};
	slot.tick        = tick;
	const auto first = static_cast<uint32_t>(tick % RING_SIZE) * 2;
	buffer.resetQueryPool(m_pool, first, 2);
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, m_pool, first);
}

void GpuTimestamps::End(vk::CommandBuffer buffer, uint64_t tick) {
	if (m_pool == nullptr) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	auto&            slot = SlotOf(tick);
	if (slot.tick != tick) {
		return;
	}
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, m_pool,
	                       static_cast<uint32_t>(tick % RING_SIZE) * 2 + 1);
	slot.ended = true;
}

void GpuTimestamps::Submitted(uint64_t tick, int64_t submit_ns,
                              Common::GpuTiming::SyncReason reason, uint32_t commands) {
	if (m_pool == nullptr) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	auto&            slot = SlotOf(tick);
	if (slot.tick == tick) {
		slot.submit_ns = submit_ns;
		slot.submitted = true;
		slot.reason    = reason;
		slot.commands  = commands;
	}
	Collect();
	MaybeRotate(NowNs());
}

void GpuTimestamps::Collect() {
	const auto known = m_master.KnownGpuTick();
	for (uint64_t tick = m_collected + 1; tick <= known; tick++) {
		auto& slot = SlotOf(tick);
		if (slot.tick != tick || !slot.ended || !slot.submitted) {
			m_accumulator.AddUntimedSubmission();
			continue;
		}
		std::array<uint64_t, 2> data {};
		// The submission signalled its timeline value, so both queries are available; a
		// result that is not ready is dropped rather than waited for.
		const auto result = m_graphics.device.getQueryPoolResults(
		    m_pool, static_cast<uint32_t>(tick % RING_SIZE) * 2, 2, sizeof(data), data.data(),
		    sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		if (result != vk::Result::eSuccess) {
			m_accumulator.AddUntimedSubmission();
			continue;
		}
		const uint64_t begin = data[0] & m_valid_mask;
		const uint64_t span  = (data[1] - begin) & m_valid_mask;
		// Busy time needs a monotonic device clock: extend the valid bits across wraps.
		if (begin < m_last_raw && m_last_raw - begin > (m_valid_mask >> 1u)) {
			m_wrap_ns += DeviceNs(m_valid_mask) + static_cast<uint64_t>(m_period_ns);
		}
		m_last_raw = begin;
		Common::GpuTiming::Submission submission;
		submission.submit_ns     = slot.submit_ns;
		submission.gpu_begin_ns  = m_wrap_ns + DeviceNs(begin);
		submission.gpu_end_ns    = submission.gpu_begin_ns + DeviceNs(span);
		submission.host_begin_ns = HostNs(begin);
		submission.reason        = slot.reason;
		submission.commands      = slot.commands;
		slot.host_end_ns         = HostNs(data[1]);
		m_accumulator.AddSubmission(submission);
	}
	m_collected = std::max(m_collected, known);
}

void GpuTimestamps::RecordEndLatency(uint64_t tick, bool label, int64_t floor_ns) {
	if (m_pool == nullptr || !m_calibrated) {
		return;
	}
	const auto       now = NowNs();
	std::scoped_lock lock(m_mutex);
	Collect();
	const auto& slot = SlotOf(tick);
	if (slot.tick != tick || tick > m_collected || !slot.host_end_ns) {
		return;
	}
	const auto latency = now - std::max(*slot.host_end_ns, floor_ns);
	if (label) {
		m_accumulator.AddLabel(latency);
	} else {
		m_accumulator.AddWake(latency);
	}
}

void GpuTimestamps::RecordWake(uint64_t tick) {
	RecordEndLatency(tick, false);
}

void GpuTimestamps::RecordLabel(uint64_t tick, int64_t issued_ns) {
	RecordEndLatency(tick, true, issued_ns);
}

void GpuTimestamps::MaybeRotate(int64_t now_ns) {
	if (now_ns - m_window_start_ns < WINDOW_NS) {
		return;
	}
	const auto flips = Common::GpuTiming::GuestFlips();
	m_accumulator.AddFlips(flips - m_window_flips);
	m_window_flips    = flips;
	if (m_primary) {
		const auto sync = Common::GpuTiming::Sync().Snapshot();
		m_accumulator.SetSync(sync.Since(m_last_sync));
		m_last_sync = sync;
	}
	const auto window = m_accumulator.Take(static_cast<double>(now_ns - m_window_start_ns) / 1e9);
	m_window_start_ns = now_ns;
	if (m_primary) {
		Common::GpuTiming::Publish(window);
	}
	m_log_window.Merge(window);
	if (m_log_window.seconds >= LOG_INTERVAL_S) {
		LOGF("%s\n", Common::GpuTiming::FormatSummary(m_log_window).c_str());
		m_log_window = {};
	}
	// Device and host clocks drift apart slowly; recalibrate once per window.
	if (m_calibration) {
		Calibrate();
		m_accumulator.SetCalibrated(m_calibrated);
	}
}

} // namespace Libs::Graphics
