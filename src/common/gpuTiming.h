#ifndef KYTY_COMMON_GPU_TIMING_H_
#define KYTY_COMMON_GPU_TIMING_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

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
