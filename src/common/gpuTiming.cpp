#include "common/gpuTiming.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>

namespace Common::GpuTiming {

namespace {

constexpr double BUCKETS_PER_OCTAVE = 4.0;
constexpr double BUCKET_BASE_NS     = 1000.0;

std::mutex             g_latest_mutex;
std::optional<Summary> g_latest;
std::atomic<uint64_t>  g_guest_flips {0};

double Ms(int64_t ns) {
	return static_cast<double>(ns) / 1e6;
}

std::string FormatLatency(const char* name, const LatencyHistogram& histogram, double seconds) {
	char buffer[160];
	if (histogram.Count() == 0) {
		std::snprintf(buffer, sizeof(buffer), "%s n=0", name);
	} else {
		std::snprintf(buffer, sizeof(buffer), "%s p50/p95 %.2f/%.2f ms (%.0f/s, %.0f ms/s)", name,
		              Ms(histogram.Quantile(0.5)), Ms(histogram.Quantile(0.95)),
		              seconds > 0.0 ? static_cast<double>(histogram.Count()) / seconds : 0.0,
		              seconds > 0.0 ? Ms(histogram.TotalNs()) / seconds : 0.0);
	}
	return buffer;
}

} // namespace

size_t LatencyHistogram::BucketOf(int64_t ns) noexcept {
	if (static_cast<double>(ns) < BUCKET_BASE_NS) {
		return 0;
	}
	const double octaves = std::log2(static_cast<double>(ns) / BUCKET_BASE_NS);
	const auto   bucket  = 1 + static_cast<size_t>(octaves * BUCKETS_PER_OCTAVE);
	return std::min(bucket, BUCKET_COUNT - 1);
}

int64_t LatencyHistogram::BucketUpperNs(size_t bucket) noexcept {
	return static_cast<int64_t>(
	    std::ceil(BUCKET_BASE_NS * std::exp2(static_cast<double>(bucket) / BUCKETS_PER_OCTAVE)));
}

void LatencyHistogram::Add(int64_t ns) noexcept {
	ns = std::max<int64_t>(ns, 0);
	m_buckets[BucketOf(ns)]++;
	m_count++;
	m_total_ns += ns;
}

void LatencyHistogram::Merge(const LatencyHistogram& other) noexcept {
	for (size_t i = 0; i < BUCKET_COUNT; i++) {
		m_buckets[i] += other.m_buckets[i];
	}
	m_count += other.m_count;
	m_total_ns += other.m_total_ns;
}

int64_t LatencyHistogram::Quantile(double q) const noexcept {
	if (m_count == 0) {
		return 0;
	}
	// Rank of the q-quantile, 1-based: the smallest sample with at least q of all at or below.
	const auto rank = std::max<uint64_t>(
	    1,
	    static_cast<uint64_t>(std::ceil(std::clamp(q, 0.0, 1.0) * static_cast<double>(m_count))));
	uint64_t seen = 0;
	for (size_t i = 0; i < BUCKET_COUNT; i++) {
		seen += m_buckets[i];
		if (seen >= rank) {
			return BucketUpperNs(i);
		}
	}
	return BucketUpperNs(BUCKET_COUNT - 1);
}

void Summary::Merge(const Summary& other) noexcept {
	seconds += other.seconds;
	submits += other.submits;
	timed += other.timed;
	flips += other.flips;
	busy_ns += other.busy_ns;
	calibrated = calibrated || other.calibrated;
	queue.Merge(other.queue);
	wake.Merge(other.wake);
	label.Merge(other.label);
}

double Summary::BusyMsPerSecond() const noexcept {
	return seconds > 0.0 ? static_cast<double>(busy_ns) / 1e6 / seconds : 0.0;
}

double Summary::SubmitsPerSecond() const noexcept {
	return seconds > 0.0 ? static_cast<double>(submits) / seconds : 0.0;
}

std::optional<double> Summary::SubmitsPerFlip() const noexcept {
	if (flips == 0) {
		return std::nullopt;
	}
	return static_cast<double>(submits) / static_cast<double>(flips);
}

void Accumulator::AddSubmission(const Submission& submission) noexcept {
	m_window.submits++;
	m_window.timed++;
	if (submission.gpu_end_ns > submission.gpu_begin_ns) {
		// The queue runs submissions in order but may overlap the tail of one with the head of
		// the next; count only the part past everything already counted.
		const auto start = std::max(submission.gpu_begin_ns, m_last_end_ns);
		if (submission.gpu_end_ns > start) {
			m_window.busy_ns += submission.gpu_end_ns - start;
		}
		m_last_end_ns = std::max(m_last_end_ns, submission.gpu_end_ns);
	}
	if (submission.host_begin_ns) {
		m_window.queue.Add(*submission.host_begin_ns - submission.submit_ns);
	}
}

Summary Accumulator::Take(double seconds) noexcept {
	Summary window      = m_window;
	window.seconds      = seconds;
	m_window            = {};
	m_window.calibrated = window.calibrated;
	return window;
}

std::string FormatSummary(const Summary& summary) {
	char       head[160];
	const auto per_flip = summary.SubmitsPerFlip();
	char       flips[32];
	if (per_flip) {
		std::snprintf(flips, sizeof(flips), "%.1f", *per_flip);
	} else {
		std::snprintf(flips, sizeof(flips), "n/a");
	}
	std::snprintf(head, sizeof(head),
	              "GpuTiming: busy %.0f ms/s, %.0f submits/s (%llu timed), %s submits/flip",
	              summary.BusyMsPerSecond(), summary.SubmitsPerSecond(),
	              static_cast<unsigned long long>(summary.timed), flips);
	std::string line = head;
	if (!summary.calibrated) {
		return line + ", latencies n/a (no calibrated timestamps)";
	}
	line += ", " + FormatLatency("submit->start", summary.queue, summary.seconds);
	line += ", " + FormatLatency("end->complete", summary.wake, summary.seconds);
	line += ", " + FormatLatency("end->label", summary.label, summary.seconds);
	return line;
}

void Publish(const Summary& summary) {
	std::scoped_lock lock(g_latest_mutex);
	g_latest = summary;
}

std::optional<Summary> Latest() {
	std::scoped_lock lock(g_latest_mutex);
	return g_latest;
}

void RecordGuestFlip() noexcept {
	g_guest_flips.fetch_add(1, std::memory_order_relaxed);
}

uint64_t GuestFlips() noexcept {
	return g_guest_flips.load(std::memory_order_relaxed);
}

} // namespace Common::GpuTiming
