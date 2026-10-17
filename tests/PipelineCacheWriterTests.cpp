#include "graphics/host_gpu/renderer/pipeline/pipelineCacheWriter.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>

namespace {

using Libs::Graphics::PipelineCacheWriter;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "PipelineCacheWriterTests: failed: %s\n", text);
    std::abort();
  }
}

// Holds a write until released, so that a test can observe it while it runs.
class Gate {
public:
  void Release() {
    {
      std::lock_guard lock(m_mutex);
      m_open = true;
    }
    m_cv.notify_all();
  }
  void Wait() {
    std::unique_lock lock(m_mutex);
    m_cv.wait(lock, [this] { return m_open; });
  }

private:
  std::mutex m_mutex;
  std::condition_variable m_cv;
  bool m_open = false;
};

void TestOneWriteAtATime() {
  PipelineCacheWriter writer;
  Check(!writer.Busy(), "a new writer is idle");
  writer.Wait();

  Gate gate;
  std::atomic<int> writes = 0;
  Check(writer.TryStart([&] {
    gate.Wait();
    writes++;
  }),
        "an idle writer starts a write");
  Check(writer.Busy(), "the writer is busy while the write runs");
  Check(!writer.TryStart([&] { writes += 100; }),
        "a second write does not start meanwhile");
  gate.Release();
  writer.Wait();
  Check(!writer.Busy(), "the writer is idle after Wait");
  Check(writes == 1, "only the first write ran");
  writer.Wait();

  Check(writer.TryStart([&] { writes++; }),
        "a finished writer starts the next write");
  writer.Wait();
  Check(writes == 2, "the next write ran");
}

void TestNoOverlap() {
  std::atomic<int> active = 0;
  std::atomic<int> overlap = 0;
  std::atomic<int> writes = 0;
  int started = 0;
  {
    PipelineCacheWriter writer;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (std::chrono::steady_clock::now() < deadline) {
      if (writer.TryStart([&] {
            if (active.fetch_add(1) != 0) {
              overlap++;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            active--;
            writes++;
          })) {
        started++;
      }
    }
    // The destructor waits for the last write.
  }
  Check(started > 1, "writes started again after the previous ones finished");
  Check(overlap == 0, "two writes never ran at the same time");
  Check(writes == started,
        "every started write ran to completion before destruction");
}

void TestDestructorWaits() {
  std::atomic<bool> done = false;
  {
    auto writer = std::make_unique<PipelineCacheWriter>();
    writer->TryStart([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      done = true;
    });
  }
  Check(done, "destroying the writer waits for the running write");
}

} // namespace

int main() {
  TestOneWriteAtATime();
  TestNoOverlap();
  TestDestructorWaits();
  std::printf("PipelineCacheWriterTests: all tests passed\n");
  return 0;
}
