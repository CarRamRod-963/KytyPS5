#ifndef KYTY_COMMON_GPU_TIMING_H_
#define KYTY_COMMON_GPU_TIMING_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Host GPU timing per queue submission: how long the GPU executed, how long work waited before
// it started, and how long the CPU took to notice a finished submission. The renderer feeds the
// accumulator from timestamp queries; the overlay and the log read one-second summaries.
namespace Common::GpuTiming {

// Approximate quantiles from quarter-octave buckets starting at 1 us (about 19% resolution).
class LatencyHistogram {
public:
	static constexpr size_t BUCKET_COUNT = 128;

	void Add(int64_t ns) noexcept;
	void Merge(const LatencyHistogram& other) noexcept;

	[[nodiscard]] uint64_t Count() const noexcept { return m_count; }
	[[nodiscard]] int64_t  TotalNs() const noexcept { return m_total_ns; }
	// Upper bound of the bucket that holds the q-quantile (0 < q <= 1); 0 when empty.
	[[nodiscard]] int64_t Quantile(double q) const noexcept;

	[[nodiscard]] static size_t  BucketOf(int64_t ns) noexcept;
	[[nodiscard]] static int64_t BucketUpperNs(size_t bucket) noexcept;

private:
	std::array<uint32_t, BUCKET_COUNT> m_buckets {};
	uint64_t                           m_count    = 0;
	int64_t                            m_total_ns = 0;
};

// Why a queue submission or a blocking wait on the GPU timeline happened. The renderer sets the
// reason with a SyncScope around code that may submit or wait; Submit and the semaphore wait
// count under the reason of the calling thread.
enum class SyncReason : uint8_t {
	Other,
	QueueSlice,     // End of a processed slice of a guest command buffer.
	Label,          // End-of-pipe label written by the guest command stream.
	Interrupt,      // End-of-pipe interrupt without a label.
	Flip,           // Guest flip.
	FrameBoundary,  // Guest suspend point (frame end).
	GuestWait,      // The command stream waits for the GPU (GDS read, explicit idle).
	FaultRead,      // Guest CPU read fault on GPU-written memory.
	CpuWrite,       // Guest CPU write into GPU-written memory (read back first).
	BufferRead,     // Other host read of GPU-written buffer memory.
	WalkRead,       // Resource walk reading GPU-written bytes.
	MetadataCheck,  // Color-clear metadata written by compute and read by the texture cache.
	StreamWrap,     // Stream or upload buffer full: wait for its oldest user.
	FaultBuffer,    // Device fault buffer processing.
	Unmap,          // Guest unmap of memory the caches own.
	GarbageCollect, // Cache eviction of GPU-written buffers.
	Present,        // Presentation queue work.
	Completion,     // Completion thread waiting to run labels and published readbacks.
	Count
};

inline constexpr size_t SYNC_REASON_COUNT = static_cast<size_t>(SyncReason::Count);

[[nodiscard]] const char* SyncReasonName(SyncReason reason) noexcept;

// The calling thread's current reason; Other outside any scope.
[[nodiscard]] SyncReason CurrentSyncReason() noexcept;

// Sets the calling thread's reason for its lifetime. The outermost scope wins, so a generic
// helper (a buffer read) called for a specific purpose (a resource walk) keeps the purpose.
class SyncScope {
public:
	explicit SyncScope(SyncReason reason) noexcept;
	~SyncScope();
	SyncScope(const SyncScope&)            = delete;
	SyncScope& operator=(const SyncScope&) = delete;

private:
	SyncReason m_previous;
};

// Pipeline barriers recorded around guest dispatches and flushes.
enum class BarrierKind : uint8_t { ComputeToCompute, AllToCompute, ComputeToAll, AllToAll, Count };

inline constexpr size_t BARRIER_KIND_COUNT = static_cast<size_t>(BarrierKind::Count);

struct SyncCounts {
	std::array<uint64_t, SYNC_REASON_COUNT> submits {};
	std::array<uint64_t, SYNC_REASON_COUNT> waits {}; // Blocking waits on the GPU timeline.
	std::array<uint64_t, SYNC_REASON_COUNT> wait_ns {};
	// Flushes that found nothing to submit and kept the command buffer open.
	std::array<uint64_t, SYNC_REASON_COUNT> skipped {};
	// Readbacks served by waiting for an already submitted asynchronous download.
	std::array<uint64_t, SYNC_REASON_COUNT> served {};
	// Readbacks that rode the submission of another readback served at the same time.
	std::array<uint64_t, SYNC_REASON_COUNT> joined {};
	// Hot pages whose asynchronous download rode a submission, and hot pages whose GPU writes
	// were left for a later submission because the downloads of one submission are capped.
	uint64_t async_pages    = 0;
	uint64_t async_deferred = 0;
	// End-of-pipe labels written by the completion thread, and labels it handed to the GPU
	// thread (watched pages, or behind such a label).
	uint64_t labels_direct     = 0;
	uint64_t labels_gpu_thread = 0;
	// Pipeline barriers that order guest dispatches and flushes, by kind (BarrierKind), and the
	// number a barrier before every writing dispatch, one after every dispatch and one per guest
	// flush would have been.
	std::array<uint64_t, BARRIER_KIND_COUNT> barriers {};
	uint64_t                                 barriers_unmerged = 0;

	void                     Merge(const SyncCounts& other) noexcept;
	[[nodiscard]] SyncCounts Since(const SyncCounts& earlier) const noexcept;
	[[nodiscard]] uint64_t   TotalSubmits() const noexcept;
	[[nodiscard]] uint64_t   TotalWaits() const noexcept;
	[[nodiscard]] uint64_t   TotalSkipped() const noexcept;
	[[nodiscard]] uint64_t   TotalServed() const noexcept;
	[[nodiscard]] uint64_t   TotalJoined() const noexcept;
};

// Running totals per reason, written from any thread with one relaxed increment.
class SyncCounters {
public:
	void AddSubmit(SyncReason reason) noexcept {
		m_submits[Index(reason)].fetch_add(1, std::memory_order_relaxed);
	}
	void AddWait(SyncReason reason, int64_t ns) noexcept {
		m_waits[Index(reason)].fetch_add(1, std::memory_order_relaxed);
		m_wait_ns[Index(reason)].fetch_add(static_cast<uint64_t>(ns > 0 ? ns : 0),
		                                   std::memory_order_relaxed);
	}
	void AddSkippedFlush(SyncReason reason) noexcept {
		m_skipped[Index(reason)].fetch_add(1, std::memory_order_relaxed);
	}
	void AddServedBySubmitted(SyncReason reason) noexcept {
		m_served[Index(reason)].fetch_add(1, std::memory_order_relaxed);
	}
	void AddJoinedReadback(SyncReason reason) noexcept {
		m_joined[Index(reason)].fetch_add(1, std::memory_order_relaxed);
	}
	void AddAsyncDownloads(uint64_t recorded, uint64_t deferred) noexcept {
		m_async_pages.fetch_add(recorded, std::memory_order_relaxed);
		m_async_deferred.fetch_add(deferred, std::memory_order_relaxed);
	}
	void AddLabel(bool direct) noexcept {
		(direct ? m_labels_direct : m_labels_gpu_thread).fetch_add(1, std::memory_order_relaxed);
	}
	void AddBarrier(BarrierKind kind) noexcept {
		const auto index = static_cast<size_t>(kind);
		m_barriers[index < BARRIER_KIND_COUNT ? index : 0].fetch_add(1, std::memory_order_relaxed);
	}
	void AddUnmergedBarriers(uint64_t count) noexcept {
		m_barriers_unmerged.fetch_add(count, std::memory_order_relaxed);
	}
	[[nodiscard]] SyncCounts Snapshot() const noexcept;

private:
	static size_t Index(SyncReason reason) noexcept {
		const auto index = static_cast<size_t>(reason);
		return index < SYNC_REASON_COUNT ? index : 0;
	}

	std::array<std::atomic<uint64_t>, SYNC_REASON_COUNT> m_submits {};
	std::array<std::atomic<uint64_t>, SYNC_REASON_COUNT> m_waits {};
	std::array<std::atomic<uint64_t>, SYNC_REASON_COUNT> m_wait_ns {};
	std::array<std::atomic<uint64_t>, SYNC_REASON_COUNT> m_skipped {};
	std::array<std::atomic<uint64_t>, SYNC_REASON_COUNT> m_served {};
	std::array<std::atomic<uint64_t>, SYNC_REASON_COUNT> m_joined {};
	std::atomic<uint64_t>                                m_async_pages {};
	std::atomic<uint64_t>                                m_async_deferred {};
	std::atomic<uint64_t>                                m_labels_direct {};
	std::atomic<uint64_t>                                m_labels_gpu_thread {};
	std::array<std::atomic<uint64_t>, BARRIER_KIND_COUNT> m_barriers {};
	std::atomic<uint64_t>                                 m_barriers_unmerged {};
};

// Process-wide counters: every scheduler's submits and semaphore waits.
[[nodiscard]] SyncCounters& Sync() noexcept;

// Reasons with any submit or wait, most submits + waits first; at most `limit`.
[[nodiscard]] std::vector<SyncReason> BusiestReasons(const SyncCounts& counts,
                                                     size_t            limit = SYNC_REASON_COUNT);

// "slice 600/0/0.0, label 330/0/0.0, ..." per second (submits/waits/wait ms), busiest first, at
// most `limit` reasons; "none" when nothing happened.
[[nodiscard]] std::string FormatSyncCounts(const SyncCounts& counts, double seconds,
                                           size_t limit = SYNC_REASON_COUNT);

// One finished submission. Device times are in nanoseconds of the device clock; host times use
// std::chrono::steady_clock and are present only when the device clock is calibrated.
struct Submission {
	int64_t                submit_ns    = 0;
	uint64_t               gpu_begin_ns = 0;
	uint64_t               gpu_end_ns   = 0;
	std::optional<int64_t> host_begin_ns;
};

struct Summary {
	double   seconds    = 0.0;
	uint64_t submits    = 0;
	uint64_t timed      = 0; // Submissions with timestamps.
	uint64_t flips      = 0; // Guest flips in the window.
	uint64_t busy_ns    = 0; // Union of the GPU execution intervals.
	bool     calibrated = false;
	// Submitted on the CPU -> started on the GPU.
	LatencyHistogram queue;
	// GPU finished -> a CPU thread blocked on the submission woke up.
	LatencyHistogram wake;
	// GPU finished -> the guest end-of-pipe label of the submission was written.
	LatencyHistogram label;
	// Submissions and blocking waits per reason in the window.
	SyncCounts sync;

	void Merge(const Summary& other) noexcept;

	[[nodiscard]] double                BusyMsPerSecond() const noexcept;
	[[nodiscard]] double                SubmitsPerSecond() const noexcept;
	[[nodiscard]] std::optional<double> SubmitsPerFlip() const noexcept;
};

// Single-consumer accumulator. Submissions must arrive in submission (queue) order so that
// overlapping execution intervals are counted once.
class Accumulator {
public:
	void AddSubmission(const Submission& submission) noexcept;
	void AddUntimedSubmission() noexcept { m_window.submits++; }
	void AddWake(int64_t end_to_wake_ns) noexcept { m_window.wake.Add(end_to_wake_ns); }
	void AddLabel(int64_t end_to_label_ns) noexcept { m_window.label.Add(end_to_label_ns); }
	void AddFlips(uint64_t flips) noexcept { m_window.flips += flips; }
	void SetCalibrated(bool calibrated) noexcept { m_window.calibrated = calibrated; }
	void SetSync(const SyncCounts& sync) noexcept { m_window.sync = sync; }

	// Ends the current window and starts the next one; execution overlap is tracked across.
	[[nodiscard]] Summary Take(double seconds) noexcept;

private:
	Summary  m_window;
	uint64_t m_last_end_ns = 0;
};

// "GpuTiming: busy 412 ms/s, 980 submits/s (...), end->complete p50/p95 ..." (no newline).
[[nodiscard]] std::string FormatSummary(const Summary& summary);

// Latest one-second summary, for the performance overlay.
void                                 Publish(const Summary& summary);
[[nodiscard]] std::optional<Summary> Latest();

// Guest flips presented on the main bus; the renderer reads deltas per window.
void                   RecordGuestFlip() noexcept;
[[nodiscard]] uint64_t GuestFlips() noexcept;

} // namespace Common::GpuTiming

#endif /* KYTY_COMMON_GPU_TIMING_H_ */
