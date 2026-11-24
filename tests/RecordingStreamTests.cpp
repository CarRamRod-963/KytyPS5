#include "graphics/host_gpu/recordingStream.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::Recording::ChunkBytes;
using Libs::Graphics::Recording::ChunkCount;
using Libs::Graphics::Recording::ChunkMaxCalls;
using Libs::Graphics::Recording::LargeCopyBytes;
using Libs::Graphics::Recording::Stream;
using Libs::Graphics::Recording::Writer;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "RecordingStreamTests: failed: %s\n", text);
    std::abort();
  }
}

// Calls run in the order they were queued, across many trips around the ring.
void TestOrderAcrossTheRing() {
  std::vector<uint32_t> seen; // Written by the stream's thread only.
  constexpr uint32_t Calls = ChunkCount * ChunkMaxCalls * 3 + 17;
  {
    Stream stream;
    for (uint32_t index = 0; index < Calls; index++) {
      Check(stream.Enqueue([&](Writer &writer) {
        writer.Call([&seen, index] { seen.push_back(index); });
      }),
            "a small call is queued");
    }
    stream.Drain();
    Check(!stream.Pending(), "nothing is pending after a drain");
    Check(seen.size() == Calls, "every call ran by the drain");
    for (uint32_t index = 0; index < Calls; index++) {
      Check(seen[index] == index, "calls run in queue order");
    }
    Check(stream.QueuedCalls() == Calls, "queued calls are counted");
  }
}

uint8_t Pattern(uint32_t call, size_t byte) {
  return static_cast<uint8_t>(call * 31u + byte * 7u);
}

// Copied arguments of every size live until their call ran, even when the
// producer reuses its own memory at once; large ones live in blocks of their
// own.
void TestCopiedArgumentsOutliveTheSource() {
  std::atomic<uint32_t> bad{0};
  std::atomic<uint32_t> ran{0};
  std::vector<uint8_t> source(ChunkBytes * 3);
  const size_t sizes[] = {0,
                          1,
                          3,
                          64,
                          4096,
                          LargeCopyBytes,
                          LargeCopyBytes + 1,
                          ChunkBytes - 512,
                          ChunkBytes,
                          ChunkBytes * 2 + 5};
  {
    Stream stream;
    for (uint32_t call = 0; call < 400; call++) {
      const size_t size = sizes[call % std::size(sizes)];
      for (size_t byte = 0; byte < size; byte++) {
        source[byte] = Pattern(call, byte);
      }
      Check(stream.Enqueue([&](Writer &writer) {
        const auto *copy =
            static_cast<const uint8_t *>(writer.CopyBytes(source.data(), size));
        Check((copy == nullptr) == (size == 0),
              "a copy exists exactly for a non-empty argument");
        writer.Call([copy, size, call, &bad, &ran] {
          for (size_t byte = 0; byte < size; byte++) {
            if (copy[byte] != Pattern(call, byte)) {
              bad++;
              break;
            }
          }
          ran++;
        });
      }),
            "an argument of any size is queued");
      // The producer's memory changes before the call ran.
      std::memset(source.data(), 0xcd, source.size());
    }
  }
  Check(ran == 400, "the stream's end ran every queued call");
  Check(bad == 0, "every call read its own copy of its argument");
}

struct Counted {
  explicit Counted(std::atomic<int> *alive) : alive(alive) { ++*alive; }
  Counted(const Counted &other) : alive(other.alive) { ++*alive; }
  ~Counted() { --*alive; }
  std::atomic<int> *alive;
};

// A call that holds resources is destroyed after it ran; one that is not queued
// is never run and leaves nothing behind.
void TestCallLifetimes() {
  std::atomic<int> alive{0};
  std::atomic<int> ran{0};
  Stream stream;
  {
    auto shared = std::make_shared<Counted>(&alive);
    for (int index = 0; index < 1000; index++) {
      Check(stream.Enqueue([&](Writer &writer) {
        writer.Call([shared, &ran] { ran++; });
      }),
            "a call holding a resource is queued");
    }
  }
  stream.Drain();
  Check(ran == 1000, "every call holding a resource ran");
  Check(alive == 0, "every call was destroyed after it ran");

  const auto queued_before = stream.QueuedCalls();
  Check(!stream.Enqueue([&](Writer &writer) {
    const char text[] = "dropped";
    (void)writer.CopyBytes(text, sizeof(text));
    std::vector<uint8_t> large(LargeCopyBytes * 2);
    (void)writer.CopyBytes(large.data(), large.size());
    writer.Fail();
    writer.Call([&ran] { ran += 100; });
  }),
        "a failed call is not queued");
  Check(!stream.Enqueue([](Writer &) {}),
        "a capture without a call is not queued");
  stream.Drain();
  Check(ran == 1000 && stream.QueuedCalls() == queued_before,
        "a call that was not queued never runs");
}

// A call whose stored arguments exceed a chunk is refused; a call that only
// does not fit into the rest of a chunk moves into the next one; the stream
// works on after both.
void TestChunkLimits() {
  std::atomic<int> ran{0};
  Stream stream;
  struct Huge {
    std::array<std::byte, ChunkBytes> bytes;
  };
  auto huge = std::make_unique<Huge>();
  Check(!stream.Enqueue([&](Writer &writer) {
    writer.Call(
        [copy = *huge, &ran] { ran += static_cast<int>(copy.bytes[0]); });
  }),
        "a call larger than a chunk is refused");

  struct Half {
    std::array<std::byte, ChunkBytes / 2 + 64> bytes{};
  };
  for (int index = 0; index < 7; index++) {
    Check(stream.Enqueue([&](Writer &writer) {
      writer.Call([half = Half{}, &ran] {
        ran += 1 + static_cast<int>(half.bytes[0]);
      });
    }),
          "calls larger than half a chunk each take a chunk");
  }
  stream.Drain();
  Check(ran == 7, "the stream runs on after a refused call");
}

// Published calls run without a drain; the stream's thread is not the producer
// and ran the start function first.
void TestPublishAndThread() {
  std::atomic<std::thread::id> started{};
  std::atomic<std::thread::id> ran_on{};
  std::atomic<bool> ran{false};
  Stream stream([&] { started = std::this_thread::get_id(); }, 0);
  Check(stream.Enqueue([&](Writer &writer) {
    writer.Call([&] {
      ran_on = std::this_thread::get_id();
      ran = true;
    });
  }),
        "a call is queued");
  Check(stream.Pending(), "a queued call is pending");
  stream.Publish();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!ran && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Check(ran,
        "a published call runs without a drain (a blocked thread is woken)");
  Check(ran_on.load() == stream.ThreadId() &&
            ran_on.load() != std::this_thread::get_id(),
        "calls run on the stream's thread");
  Check(started.load() == stream.ThreadId(),
        "the start function runs on the stream's thread");
  stream.Drain();
  Check(!stream.Pending(), "nothing is pending after a drain");
}

// A full ring makes the producer wait for the stream's thread instead of
// overwriting chunks.
void TestFullRingWaits() {
  std::atomic<bool> release{false};
  std::atomic<uint32_t> ran{0};
  Stream stream;
  Check(stream.Enqueue([&](Writer &writer) {
    writer.Call([&] {
      while (!release) {
        std::this_thread::yield();
      }
      ran++;
    });
  }),
        "a blocking call is queued");
  stream.Publish();
  std::thread unblock([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    release = true;
  });
  for (uint32_t index = 0; index < ChunkCount * ChunkMaxCalls * 2; index++) {
    Check(
        stream.Enqueue([&](Writer &writer) { writer.Call([&ran] { ran++; }); }),
        "calls are queued while the ring is full");
  }
  stream.Drain();
  unblock.join();
  Check(ran == ChunkCount * ChunkMaxCalls * 2 + 1,
        "no call was lost to a full ring");
  Check(stream.RoomWaits() != 0, "the producer waited for room");
}

} // namespace

int main() {
  TestOrderAcrossTheRing();
  TestCopiedArgumentsOutliveTheSource();
  TestCallLifetimes();
  TestChunkLimits();
  TestPublishAndThread();
  TestFullRingWaits();
  std::printf("RecordingStreamTests: all passed\n");
  return 0;
}
