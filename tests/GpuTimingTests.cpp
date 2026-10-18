#include "common/gpuTiming.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using Common::GpuTiming::Accumulator;
using Common::GpuTiming::LatencyHistogram;
using Common::GpuTiming::Submission;
using Common::GpuTiming::Summary;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "GpuTimingTests: failed: %s\n", message);
    std::abort();
  }
}

constexpr int64_t US = 1000;
constexpr int64_t MS = 1000 * US;

void TestBuckets() {
  Check(LatencyHistogram::BucketOf(-5) == 0,
        "negative latency is not in the first bucket");
  Check(LatencyHistogram::BucketOf(999) == 0,
        "sub-microsecond latency is not in bucket 0");
  Check(LatencyHistogram::BucketOf(1000) == 1, "1 us is not in bucket 1");
  Check(LatencyHistogram::BucketUpperNs(0) == 1000,
        "bucket 0 does not end at 1 us");
  Check(LatencyHistogram::BucketUpperNs(4) == 2000,
        "bucket 4 does not end at 2 us");
  for (int64_t ns :
       {int64_t{1}, 1500 * int64_t{1}, 37 * US, 2 * MS, 950 * MS, 4000 * MS}) {
    const auto bucket = LatencyHistogram::BucketOf(ns);
    Check(ns < LatencyHistogram::BucketUpperNs(bucket),
          "value at or above its bucket end");
    Check(bucket == 0 || ns >= LatencyHistogram::BucketUpperNs(bucket - 1),
          "value below its bucket start");
  }
  Check(LatencyHistogram::BucketOf(INT64_MAX) ==
            LatencyHistogram::BUCKET_COUNT - 1,
        "huge latency is not clamped to the last bucket");
}

void TestQuantiles() {
  LatencyHistogram histogram;
  Check(histogram.Quantile(0.5) == 0, "empty histogram has a quantile");
  // 90 samples at 100 us, 10 at 10 ms: the median is ~100 us, p95 ~10 ms.
  for (int i = 0; i < 90; i++) {
    histogram.Add(100 * US);
  }
  for (int i = 0; i < 10; i++) {
    histogram.Add(10 * MS);
  }
  Check(histogram.Count() == 100, "count");
  Check(histogram.TotalNs() == 90 * 100 * US + 10 * 10 * MS, "total");
  const auto p50 = histogram.Quantile(0.5);
  const auto p90 = histogram.Quantile(0.9);
  const auto p95 = histogram.Quantile(0.95);
  Check(p50 > 100 * US && p50 <= 120 * US, "p50 is not the 100 us bucket");
  Check(p90 == p50, "p90 must still fall on the 90th sample");
  Check(p95 > 10 * MS && p95 <= 12 * MS, "p95 is not the 10 ms bucket");
  Check(histogram.Quantile(1.0) == p95, "max is not the 10 ms bucket");

  LatencyHistogram other;
  other.Add(10 * MS);
  histogram.Merge(other);
  Check(histogram.Count() == 101 &&
            histogram.TotalNs() == 90 * 100 * US + 11 * 10 * MS,
        "merge");
}

Submission Make(int64_t submit, uint64_t begin, uint64_t end) {
  Submission submission;
  submission.submit_ns = submit;
  submission.gpu_begin_ns = begin;
  submission.gpu_end_ns = end;
  submission.host_begin_ns = static_cast<int64_t>(begin);
  return submission;
}

void TestBusyUnion() {
  Accumulator accumulator;
  accumulator.SetCalibrated(true);
  accumulator.AddSubmission(Make(0, 1 * MS, 3 * MS)); // 2 ms.
  accumulator.AddSubmission(Make(0, 2 * MS, 4 * MS)); // Overlaps: +1 ms.
  accumulator.AddSubmission(Make(0, 5 * MS, 6 * MS)); // Gap, then +1 ms.
  accumulator.AddSubmission(Make(0, 5 * MS, 5 * MS)); // Empty interval.
  accumulator.AddSubmission(
      Make(0, 4 * MS, 5 * MS)); // Ends before the last end: +0.
  accumulator.AddUntimedSubmission();
  accumulator.AddFlips(2);
  const auto window = accumulator.Take(0.5);
  Check(window.busy_ns == static_cast<uint64_t>(4 * MS),
        "busy time is not the union");
  Check(window.submits == 6 && window.timed == 5, "submission counts");
  Check(window.SubmitsPerSecond() == 12.0, "submits per second");
  Check(window.BusyMsPerSecond() == 8.0, "busy ms per second");
  Check(window.SubmitsPerFlip() && *window.SubmitsPerFlip() == 3.0,
        "submits per flip");
  Check(window.queue.Count() == 5, "queue latency samples");
  Check(window.calibrated, "calibration flag lost");

  // The overlap state survives the window: a submission ending inside the
  // counted range adds nothing, and the next window starts empty.
  accumulator.AddSubmission(Make(0, 5 * MS, 6 * MS));
  const auto next = accumulator.Take(1.0);
  Check(next.busy_ns == 0 && next.submits == 1,
        "overlap across windows counted twice");
  Check(!next.SubmitsPerFlip(), "submits per flip without flips");
  Check(next.calibrated, "calibration flag not carried to the next window");
}

void TestLatencies() {
  Accumulator accumulator;
  accumulator.SetCalibrated(true);
  accumulator.AddSubmission(Make(1 * MS, 3 * MS, 4 * MS)); // Queued 2 ms.
  accumulator.AddWake(200 * US);
  accumulator.AddLabel(900 * US);
  accumulator.AddLabel(1100 * US);
  const auto window = accumulator.Take(1.0);
  Check(window.queue.Count() == 1 && window.queue.Quantile(0.5) > 2 * MS &&
            window.queue.Quantile(0.5) <= 2400 * US,
        "queue latency");
  Check(window.wake.Count() == 1 && window.wake.TotalNs() == 200 * US,
        "wake latency");
  Check(window.label.Count() == 2 && window.label.TotalNs() == 2 * MS,
        "label latency");

  Summary total;
  total.Merge(window);
  total.Merge(window);
  Check(total.seconds == 2.0 && total.label.Count() == 4 && total.submits == 2,
        "summary merge");

  const auto line = Common::GpuTiming::FormatSummary(total);
  Check(line.rfind("GpuTiming: busy 1 ms/s, 1 submits/s", 0) == 0,
        "summary line head");
  Check(line.find("end->label p50/p95") != std::string::npos,
        "summary line lacks labels");
  Check(line.find("n/a submits/flip") != std::string::npos,
        "summary line lacks flips");

  Summary uncalibrated;
  uncalibrated.seconds = 1.0;
  Check(Common::GpuTiming::FormatSummary(uncalibrated).find("latencies n/a") !=
            std::string::npos,
        "uncalibrated summary reports latencies");
}

void TestPublish() {
  Summary summary;
  summary.seconds = 1.0;
  summary.submits = 7;
  Common::GpuTiming::Publish(summary);
  const auto latest = Common::GpuTiming::Latest();
  Check(latest && latest->submits == 7, "published summary");
  const auto flips = Common::GpuTiming::GuestFlips();
  Common::GpuTiming::RecordGuestFlip();
  Check(Common::GpuTiming::GuestFlips() == flips + 1, "guest flip counter");
}

void TestSyncReasons() {
  using Common::GpuTiming::CurrentSyncReason;
  using Common::GpuTiming::SyncCounters;
  using Common::GpuTiming::SyncReason;
  using Common::GpuTiming::SyncScope;
  Check(CurrentSyncReason() == SyncReason::Other, "reason outside a scope");
  {
    const SyncScope outer(SyncReason::WalkRead);
    {
      // A generic helper inside a specific purpose keeps the purpose.
      const SyncScope inner(SyncReason::BufferRead);
      Check(CurrentSyncReason() == SyncReason::WalkRead,
            "inner scope replaced the outer reason");
    }
    Check(CurrentSyncReason() == SyncReason::WalkRead, "outer reason lost");
  }
  Check(CurrentSyncReason() == SyncReason::Other, "scope not restored");

  SyncCounters counters;
  const auto before = counters.Snapshot();
  for (int i = 0; i < 30; i++) {
    counters.AddSubmit(SyncReason::Label);
  }
  for (int i = 0; i < 10; i++) {
    counters.AddSubmit(SyncReason::WalkRead);
    counters.AddWait(SyncReason::WalkRead, 2 * MS);
  }
  counters.AddWait(SyncReason::Completion, 1 * MS);
  counters.AddAsyncDownloads(12, 4);
  for (int i = 0; i < 6; i++) {
    counters.AddLabel(true);
  }
  counters.AddLabel(false);
  counters.AddLabel(false);
  counters.AddBarrier(Common::GpuTiming::BarrierKind::ComputeToCompute);
  counters.AddBarrier(Common::GpuTiming::BarrierKind::ComputeToCompute);
  counters.AddBarrier(Common::GpuTiming::BarrierKind::AllToAll);
  counters.AddUnmergedBarriers(7);
  const auto delta = counters.Snapshot().Since(before);
  Check(delta.barriers[0] == 2 && delta.barriers[1] == 0 &&
            delta.barriers[3] == 1 && delta.barriers_unmerged == 7,
        "dispatch barrier counts");
  Check(delta.async_pages == 12 && delta.async_deferred == 4,
        "async download page counts");
  Check(delta.labels_direct == 6 && delta.labels_gpu_thread == 2,
        "label delivery counts");
  Check(delta.TotalSubmits() == 40 && delta.TotalWaits() == 11, "sync totals");
  Check(delta.wait_ns[static_cast<size_t>(SyncReason::WalkRead)] ==
            static_cast<uint64_t>(20 * MS),
        "walk-read wait time");
  const auto order = Common::GpuTiming::BusiestReasons(delta);
  Check(order.size() == 3 && order[0] == SyncReason::Label &&
            order[1] == SyncReason::WalkRead &&
            order[2] == SyncReason::Completion,
        "busiest reasons order");
  Check(Common::GpuTiming::BusiestReasons(delta, 1).size() == 1,
        "busiest reasons limit");
  // Over 2 s: label 15 submits/s; walk-read 5 submits, 5 waits, 10 ms per s.
  const auto line = Common::GpuTiming::FormatSyncCounts(delta, 2.0);
  Check(line == "label 15/0/0.0, walk-read 5/5/10.0, completion 0/0/0.5",
        "sync count format");
  Check(Common::GpuTiming::FormatSyncCounts({}, 1.0) == "none",
        "empty sync counts");

  Summary summary;
  summary.seconds = 2.0;
  summary.sync = delta;
  Check(Common::GpuTiming::FormatSummary(summary).find(
            "sync submits/waits/wait-ms per s: label 15/0/0.0") !=
            std::string::npos,
        "summary line carries the sync counts");
  Check(Common::GpuTiming::FormatSummary(summary).find(
            "async download pages 6/s, skipped over cap 2/s") !=
            std::string::npos,
        "summary line carries the async download pages");
  Check(Common::GpuTiming::FormatSummary(summary).find(
            "labels on completion 3/s, via the GPU thread 1/s") !=
            std::string::npos,
        "summary line carries the label delivery");
  Summary merged;
  merged.Merge(summary);
  merged.Merge(summary);
  Check(merged.sync.TotalSubmits() == 80 && merged.sync.async_deferred == 8 &&
            merged.sync.labels_direct == 12,
        "merged sync counts");
}

} // namespace

int main() {
  TestBuckets();
  TestQuantiles();
  TestBusyUnion();
  TestLatencies();
  TestPublish();
  TestSyncReasons();
  std::printf("GpuTimingTests: all passed\n");
  return 0;
}
