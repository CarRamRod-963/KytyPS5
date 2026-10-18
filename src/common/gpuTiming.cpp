#include "common/gpuTiming.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <vector>

namespace Common::GpuTiming {

namespace {

constexpr double BUCKETS_PER_OCTAVE = 4.0;
constexpr double BUCKET_BASE_NS     = 1000.0;

std::mutex             g_latest_mutex;
std::optional<Summary> g_latest;
std::atomic<uint64_t>  g_guest_flips {0};
thread_local SyncReason g_sync_reason = SyncReason::Other;

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

const char* SyncReasonName(SyncReason reason) noexcept {
	switch (reason) {
		case SyncReason::Other: return "other";
		case SyncReason::QueueSlice: return "slice";
		case SyncReason::Label: return "label";
		case SyncReason::Interrupt: return "interrupt";
		case SyncReason::Flip: return "flip";
		case SyncReason::FrameBoundary: return "frame";
		case SyncReason::GuestWait: return "guest-wait";
		case SyncReason::FaultRead: return "fault-read";
		case SyncReason::CpuWrite: return "cpu-write";
		case SyncReason::BufferRead: return "buffer-read";
		case SyncReason::WalkRead: return "walk-read";
		case SyncReason::MetadataCheck: return "metadata";
		case SyncReason::StreamWrap: return "stream-wrap";
		case SyncReason::FaultBuffer: return "fault-buffer";
		case SyncReason::Unmap: return "unmap";
		case SyncReason::GarbageCollect: return "gc";
		case SyncReason::Present: return "present";
		case SyncReason::Completion: return "completion";
		case SyncReason::Count: break;
	}
	return "?";
}

const char* StallKindName(StallKind kind) noexcept {
	switch (kind) {
		case StallKind::WaitRegMem: return "wait-reg-mem";
		case StallKind::WaitCe: return "ce";
		case StallKind::WaitDeDiff: return "de-diff";
		case StallKind::Rewind: return "rewind";
		case StallKind::Predication: return "predication";
		case StallKind::Count: break;
	}
	return "?";
}

size_t SizeClassOf(uint32_t commands) noexcept {
	return commands <= 1 ? 0 : commands < 16 ? 1 : commands < 128 ? 2 : 3;
}

SyncReason CurrentSyncReason() noexcept {
	return g_sync_reason;
}

SyncScope::SyncScope(SyncReason reason) noexcept: m_previous(g_sync_reason) {
	if (g_sync_reason == SyncReason::Other) {
		g_sync_reason = reason;
	}
}

SyncScope::~SyncScope() {
	g_sync_reason = m_previous;
}

void SyncCounts::Merge(const SyncCounts& other) noexcept {
	for (size_t i = 0; i < SYNC_REASON_COUNT; i++) {
		submits[i] += other.submits[i];
		waits[i] += other.waits[i];
		wait_ns[i] += other.wait_ns[i];
		skipped[i] += other.skipped[i];
		served[i] += other.served[i];
		joined[i] += other.joined[i];
	}
	async_pages += other.async_pages;
	async_deferred += other.async_deferred;
	labels_direct += other.labels_direct;
	labels_gpu_thread += other.labels_gpu_thread;
	for (size_t i = 0; i < BARRIER_KIND_COUNT; i++) {
		barriers[i] += other.barriers[i];
	}
	barriers_unmerged += other.barriers_unmerged;
	gpu_thread_starved += other.gpu_thread_starved;
	gpu_thread_starved_ns += other.gpu_thread_starved_ns;
	gpu_thread_blocked += other.gpu_thread_blocked;
	gpu_thread_timeouts += other.gpu_thread_timeouts;
	gpu_thread_blocked_ns += other.gpu_thread_blocked_ns;
	for (size_t i = 0; i < STALL_KIND_COUNT; i++) {
		stalls[i] += other.stalls[i];
	}
}

SyncCounts SyncCounts::Since(const SyncCounts& earlier) const noexcept {
	SyncCounts delta;
	for (size_t i = 0; i < SYNC_REASON_COUNT; i++) {
		delta.submits[i] = submits[i] - earlier.submits[i];
		delta.waits[i]   = waits[i] - earlier.waits[i];
		delta.wait_ns[i] = wait_ns[i] - earlier.wait_ns[i];
		delta.skipped[i] = skipped[i] - earlier.skipped[i];
		delta.served[i]  = served[i] - earlier.served[i];
		delta.joined[i]  = joined[i] - earlier.joined[i];
	}
	delta.async_pages    = async_pages - earlier.async_pages;
	delta.async_deferred    = async_deferred - earlier.async_deferred;
	delta.labels_direct     = labels_direct - earlier.labels_direct;
	delta.labels_gpu_thread = labels_gpu_thread - earlier.labels_gpu_thread;
	for (size_t i = 0; i < BARRIER_KIND_COUNT; i++) {
		delta.barriers[i] = barriers[i] - earlier.barriers[i];
	}
	delta.barriers_unmerged     = barriers_unmerged - earlier.barriers_unmerged;
	delta.gpu_thread_starved    = gpu_thread_starved - earlier.gpu_thread_starved;
	delta.gpu_thread_starved_ns = gpu_thread_starved_ns - earlier.gpu_thread_starved_ns;
	delta.gpu_thread_blocked    = gpu_thread_blocked - earlier.gpu_thread_blocked;
	delta.gpu_thread_timeouts   = gpu_thread_timeouts - earlier.gpu_thread_timeouts;
	delta.gpu_thread_blocked_ns = gpu_thread_blocked_ns - earlier.gpu_thread_blocked_ns;
	for (size_t i = 0; i < STALL_KIND_COUNT; i++) {
		delta.stalls[i] = stalls[i] - earlier.stalls[i];
	}
	return delta;
}

uint64_t SyncCounts::TotalSubmits() const noexcept {
	return std::accumulate(submits.begin(), submits.end(), uint64_t {0});
}

uint64_t SyncCounts::TotalWaits() const noexcept {
	return std::accumulate(waits.begin(), waits.end(), uint64_t {0});
}

uint64_t SyncCounts::TotalSkipped() const noexcept {
	return std::accumulate(skipped.begin(), skipped.end(), uint64_t {0});
}

uint64_t SyncCounts::TotalServed() const noexcept {
	return std::accumulate(served.begin(), served.end(), uint64_t {0});
}

uint64_t SyncCounts::TotalJoined() const noexcept {
	return std::accumulate(joined.begin(), joined.end(), uint64_t {0});
}

SyncCounters& Sync() noexcept {
	static SyncCounters counters;
	return counters;
}

SyncCounts SyncCounters::Snapshot() const noexcept {
	SyncCounts counts;
	for (size_t i = 0; i < SYNC_REASON_COUNT; i++) {
		counts.submits[i] = m_submits[i].load(std::memory_order_relaxed);
		counts.waits[i]   = m_waits[i].load(std::memory_order_relaxed);
		counts.wait_ns[i] = m_wait_ns[i].load(std::memory_order_relaxed);
		counts.skipped[i] = m_skipped[i].load(std::memory_order_relaxed);
		counts.served[i]  = m_served[i].load(std::memory_order_relaxed);
		counts.joined[i]  = m_joined[i].load(std::memory_order_relaxed);
	}
	counts.async_pages    = m_async_pages.load(std::memory_order_relaxed);
	counts.async_deferred    = m_async_deferred.load(std::memory_order_relaxed);
	counts.labels_direct     = m_labels_direct.load(std::memory_order_relaxed);
	counts.labels_gpu_thread = m_labels_gpu_thread.load(std::memory_order_relaxed);
	for (size_t i = 0; i < BARRIER_KIND_COUNT; i++) {
		counts.barriers[i] = m_barriers[i].load(std::memory_order_relaxed);
	}
	counts.barriers_unmerged     = m_barriers_unmerged.load(std::memory_order_relaxed);
	counts.gpu_thread_starved    = m_gpu_thread_starved.load(std::memory_order_relaxed);
	counts.gpu_thread_starved_ns = m_gpu_thread_starved_ns.load(std::memory_order_relaxed);
	counts.gpu_thread_blocked    = m_gpu_thread_blocked.load(std::memory_order_relaxed);
	counts.gpu_thread_timeouts   = m_gpu_thread_timeouts.load(std::memory_order_relaxed);
	counts.gpu_thread_blocked_ns = m_gpu_thread_blocked_ns.load(std::memory_order_relaxed);
	for (size_t i = 0; i < STALL_KIND_COUNT; i++) {
		counts.stalls[i] = m_stalls[i].load(std::memory_order_relaxed);
	}
	return counts;
}

std::vector<SyncReason> BusiestReasons(const SyncCounts& counts, size_t limit) {
	std::vector<SyncReason> order;
	for (size_t i = 0; i < SYNC_REASON_COUNT; i++) {
		if (counts.submits[i] != 0 || counts.waits[i] != 0) {
			order.push_back(static_cast<SyncReason>(i));
		}
	}
	std::stable_sort(order.begin(), order.end(), [&](SyncReason a, SyncReason b) {
		const auto i = static_cast<size_t>(a);
		const auto j = static_cast<size_t>(b);
		return counts.submits[i] + counts.waits[i] > counts.submits[j] + counts.waits[j];
	});
	if (order.size() > limit) {
		order.resize(limit);
	}
	return order;
}

std::string FormatSyncCounts(const SyncCounts& counts, double seconds, size_t limit) {
	const double scale = seconds > 0.0 ? 1.0 / seconds : 0.0;
	std::string  line;
	for (const auto reason: BusiestReasons(counts, limit)) {
		const auto i = static_cast<size_t>(reason);
		char       item[96];
		std::snprintf(item, sizeof(item), "%s%s %.0f/%.0f/%.1f", line.empty() ? "" : ", ",
		              SyncReasonName(reason), static_cast<double>(counts.submits[i]) * scale,
		              static_cast<double>(counts.waits[i]) * scale,
		              static_cast<double>(counts.wait_ns[i]) / 1e6 * scale);
		line += item;
	}
	return line.empty() ? "none" : line;
}

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
	sync.Merge(other.sync);
	for (size_t i = 0; i < SYNC_REASON_COUNT; i++) {
		span_ns_by_reason[i] += other.span_ns_by_reason[i];
	}
	for (size_t i = 0; i < SIZE_CLASS_COUNT; i++) {
		span_ns_by_size[i] += other.span_ns_by_size[i];
		submits_by_size[i] += other.submits_by_size[i];
	}
	launch_gap_ns += other.launch_gap_ns;
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
	const auto span   = submission.gpu_end_ns > submission.gpu_begin_ns
	                        ? submission.gpu_end_ns - submission.gpu_begin_ns
	                        : uint64_t {0};
	const auto reason = static_cast<size_t>(submission.reason);
	m_window.span_ns_by_reason[reason < SYNC_REASON_COUNT ? reason : 0] += span;
	const auto size_class = SizeClassOf(submission.commands);
	m_window.span_ns_by_size[size_class] += span;
	m_window.submits_by_size[size_class]++;
	if (submission.host_begin_ns) {
		const auto host_begin = *submission.host_begin_ns;
		m_window.queue.Add(host_begin - submission.submit_ns);
		const auto ready = std::max(submission.submit_ns, m_last_host_end_ns);
		if (host_begin > ready) {
			m_window.launch_gap_ns += static_cast<uint64_t>(host_begin - ready);
		}
		m_last_host_end_ns = std::max(m_last_host_end_ns, host_begin + static_cast<int64_t>(span));
	}
}

Summary Accumulator::Take(double seconds) noexcept {
	Summary window      = m_window;
	window.seconds      = seconds;
	m_window            = {};
	m_window.calibrated = window.calibrated;
	return window;
}

namespace {

std::string SyncSuffix(const Summary& summary) {
	const double scale = summary.seconds > 0.0 ? 1.0 / summary.seconds : 0.0;
	char         tail[400];
	std::snprintf(tail, sizeof(tail),
	              ", empty flushes skipped %.0f/s, readbacks served by submitted downloads %.0f/s, "
	              "readbacks joined to another's submission %.0f/s, async download pages %.0f/s, "
	              "skipped over cap %.0f/s, labels on completion %.0f/s, via the GPU thread %.0f/s",
	              static_cast<double>(summary.sync.TotalSkipped()) * scale,
	              static_cast<double>(summary.sync.TotalServed()) * scale,
	              static_cast<double>(summary.sync.TotalJoined()) * scale,
	              static_cast<double>(summary.sync.async_pages) * scale,
	              static_cast<double>(summary.sync.async_deferred) * scale,
	              static_cast<double>(summary.sync.labels_direct) * scale,
	              static_cast<double>(summary.sync.labels_gpu_thread) * scale);
	char        barriers[200];
	const auto& kinds = summary.sync.barriers;
	std::snprintf(barriers, sizeof(barriers),
	              ", dispatch barriers per s: compute->compute %.0f, all->compute %.0f, "
	              "compute->all %.0f, all->all %.0f (unmerged %.0f)",
	              static_cast<double>(kinds[0]) * scale, static_cast<double>(kinds[1]) * scale,
	              static_cast<double>(kinds[2]) * scale, static_cast<double>(kinds[3]) * scale,
	              static_cast<double>(summary.sync.barriers_unmerged) * scale);
	const auto& sync = summary.sync;
	char        thread[256];
	std::snprintf(thread, sizeof(thread),
	              ", GPU thread starved %.0f ms/s (%.0f/s), blocked %.0f ms/s (%.0f/s, %.0f "
	              "timeouts/s), stalls per s:",
	              static_cast<double>(sync.gpu_thread_starved_ns) / 1e6 * scale,
	              static_cast<double>(sync.gpu_thread_starved) * scale,
	              static_cast<double>(sync.gpu_thread_blocked_ns) / 1e6 * scale,
	              static_cast<double>(sync.gpu_thread_blocked) * scale,
	              static_cast<double>(sync.gpu_thread_timeouts) * scale);
	std::string line =
	    ", sync submits/waits/wait-ms per s: " + FormatSyncCounts(sync, summary.seconds) + tail +
	    barriers + thread;
	for (size_t i = 0; i < STALL_KIND_COUNT; i++) {
		char item[48];
		std::snprintf(item, sizeof(item), "%s %s %.0f", i == 0 ? "" : ",",
		              StallKindName(static_cast<StallKind>(i)),
		              static_cast<double>(sync.stalls[i]) * scale);
		line += item;
	}
	line += ", GPU span ms/s by submit reason:";
	bool any = false;
	for (const auto reason: BusiestReasons(sync)) {
		const auto i = static_cast<size_t>(reason);
		if (summary.span_ns_by_reason[i] == 0) {
			continue;
		}
		char item[48];
		std::snprintf(item, sizeof(item), "%s %s %.1f", any ? "," : "", SyncReasonName(reason),
		              static_cast<double>(summary.span_ns_by_reason[i]) / 1e6 * scale);
		line += item;
		any = true;
	}
	static constexpr std::array<const char*, SIZE_CLASS_COUNT> SizeNames = {"<=1", "2-15", "16-127",
	                                                                        ">=128"};
	line += ", by commands (submits/s, ms/s):";
	for (size_t i = 0; i < SIZE_CLASS_COUNT; i++) {
		char item[64];
		std::snprintf(item, sizeof(item), "%s %s %.0f/%.1f", i == 0 ? "" : ",", SizeNames[i],
		              static_cast<double>(summary.submits_by_size[i]) * scale,
		              static_cast<double>(summary.span_ns_by_size[i]) / 1e6 * scale);
		line += item;
	}
	char gap[64];
	std::snprintf(gap, sizeof(gap), ", launch gaps %.1f ms/s",
	              static_cast<double>(summary.launch_gap_ns) / 1e6 * scale);
	return line + gap;
}

} // namespace

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
		return line + ", latencies n/a (no calibrated timestamps)" + SyncSuffix(summary);
	}
	line += ", " + FormatLatency("submit->start", summary.queue, summary.seconds);
	line += ", " + FormatLatency("end->complete", summary.wake, summary.seconds);
	line += ", " + FormatLatency("end->label", summary.label, summary.seconds);
	return line + SyncSuffix(summary);
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
