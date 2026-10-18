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

} // namespace

int main() {
  TestBuckets();
  TestQuantiles();
  TestBusyUnion();
  TestLatencies();
  TestPublish();
  std::printf("GpuTimingTests: all passed\n");
  return 0;
}
