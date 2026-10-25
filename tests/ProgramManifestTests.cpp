#include "graphics/host_gpu/renderer/pipeline/programManifest.h"
#include "graphics/host_gpu/renderer/pipeline/programPrecompileQueue.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::ComputeProgramRecord;
using Libs::Graphics::PrecompileOutcome;
using Libs::Graphics::PrecompileProgress;
using Libs::Graphics::ProgramManifest;
using Libs::Graphics::ProgramPrecompileQueue;
using Specialization =
    Libs::Graphics::ShaderRecompiler::IR::ResourceSpecialization;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ProgramManifestTests: failed: %s\n", text);
    std::abort();
  }
}

std::filesystem::path TempPath(const char *name) {
  auto dir =
      std::filesystem::temp_directory_path() / "kyty_program_manifest_tests";
  std::filesystem::create_directories(dir);
  auto path = dir / name;
  std::filesystem::remove(path);
  return path;
}

std::shared_ptr<const std::vector<uint32_t>> Code(uint32_t seed,
                                                  uint32_t words) {
  auto code = std::make_shared<std::vector<uint32_t>>(words);
  for (uint32_t i = 0; i < words; i++) {
    (*code)[i] = seed * 0x9e3779b9u + i;
  }
  return code;
}

ComputeProgramRecord
MakeRecord(uint32_t seed, std::shared_ptr<const std::vector<uint32_t>> code) {
  ComputeProgramRecord record;
  record.shader_hash = 0x1000u + seed;
  record.code = std::move(code);
  record.user_data_count = 12;
  record.input.threads_num[0] = 64;
  record.input.threads_num[1] = 2;
  record.input.threads_num[2] = 1;
  record.input.lds_size_dwords = 128u * seed;
  record.input.scratch_size_dwords = 16;
  record.input.wave_size = 32;
  record.input.async_compute = true;
  record.input.float_mode = 0xc0;
  record.input.group_id[1] = true;
  record.input.thread_ids_num = 3;
  record.input.workgroup_register = 12;
  record.input.tg_size_en = true;
  record.push_data_cursor = seed;
  Specialization::Buffer buffer;
  buffer.packed_stride = 1;
  buffer.descriptor_swizzle = 516;
  buffer.unbound = true;
  record.specialization.buffers = {buffer, buffer};
  record.specialization.buffers[1].indirect_root = 0;
  Specialization::Image image;
  image.mip_count = seed;
  image.cube = true;
  image.bindless = (seed & 1u) != 0;
  image.heap = (seed & 2u) != 0;
  record.specialization.images = {image};
  record.specialization.samplers.resize(3);
  record.specialization.samplers[2].indirect_search_iterations = 7;
  record.spirv_hash = 0xabcdef00u + seed;
  return record;
}

bool SameRecord(const ComputeProgramRecord &a, const ComputeProgramRecord &b) {
  return a.shader_hash == b.shader_hash && a.code != nullptr &&
         b.code != nullptr && *a.code == *b.code &&
         a.spirv_hash == b.spirv_hash && a.specialization == b.specialization &&
         ProgramManifest::EncodeComputeBody(a) ==
             ProgramManifest::EncodeComputeBody(b);
}

std::vector<uint8_t> ReadFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

void WriteFile(const std::filesystem::path &path,
               const std::vector<uint8_t> &data) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char *>(data.data()),
             static_cast<std::streamsize>(data.size()));
}

void TestRoundTrip() {
  const auto path = TempPath("round_trip.programs");
  const auto code = Code(1, 300);
  const auto a = MakeRecord(1, code);
  const auto b = MakeRecord(2, Code(2, 40));
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() == 0 && !manifest.Loaded().rewritten,
          "a new manifest is empty");
    Check(manifest.Add(a) && manifest.Add(b), "records are added");
  }
  ProgramManifest manifest(path);
  const auto records = manifest.Records();
  Check(records.size() == 2, "records survive a reload");
  Check(SameRecord(*records[0], a) && SameRecord(*records[1], b),
        "a reloaded record equals the recorded one, in order");
  Check(records[0]->Key() == a.Key(), "the key is stable across a reload");
  Check(!manifest.Loaded().rewritten && manifest.Loaded().dropped_bytes == 0,
        "an intact file is not rewritten");
}

void TestKey() {
  auto a = MakeRecord(3, Code(3, 64));
  const auto key = a.Key();
  a.spirv_hash++;
  Check(a.Key() == key, "the module hash is not part of the key");
  a.input.workgroup_counts[0] = 99;
  a.input.indirect_args_address = 0x1234;
  a.input.host_subgroup_size = 32;
  a.input.lds_storage = true;
  Check(a.Key() == key, "runtime and host fields are not part of the key");
  a.specialization.images[0].mip_count++;
  Check(a.Key() != key, "the specialization is part of the key");
  auto b = MakeRecord(3, Code(3, 64));
  b.input.wave_size = 64;
  Check(b.Key() != key, "static dispatch state is part of the key");
  const auto static_input = ComputeProgramRecord::StaticInput(a.input);
  Check(static_input.workgroup_counts[0] == 0 &&
            static_input.indirect_args_address == 0 &&
            static_input.threads_num[0] == 64 && static_input.tg_size_en,
        "StaticInput keeps exactly the stored fields");
}

void TestDedupe() {
  const auto path = TempPath("dedupe.programs");
  const auto code = Code(4, 2000);
  {
    ProgramManifest manifest(path);
    auto a = MakeRecord(4, code);
    Check(manifest.Add(a), "first add");
    a.spirv_hash++;
    Check(!manifest.Add(a),
          "a program recorded by another recompiler build is not added again");
    auto b = MakeRecord(5, code);
    b.shader_hash = a.shader_hash;
    Check(manifest.Add(b), "another program of the same shader is added");
    Check(manifest.Size() == 2, "two programs");
  }
  // The shared code is stored once: two code copies would exceed this size.
  Check(std::filesystem::file_size(path) < 2000u * sizeof(uint32_t) * 2u,
        "the code of a shader is stored once");
  ProgramManifest manifest(path);
  Check(manifest.Size() == 2, "deduplicated programs reload");
  Check(manifest.Records()[0]->code == manifest.Records()[1]->code,
        "programs of one shader share their code after a reload");
  Check(!manifest.Add(MakeRecord(4, Code(4, 2000))),
        "a reloaded program is known");
}

void TestVersion() {
  const auto path = TempPath("version.programs");
  {
    ProgramManifest manifest(path);
    Check(manifest.Add(MakeRecord(6, Code(6, 50))), "add");
  }
  auto data = ReadFile(path);
  Check(data.size() > 12, "file has a header");
  // The format version follows the 8-byte magic.
  data[8] ^= 0xffu;
  WriteFile(path, data);
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() == 0 && manifest.Loaded().rewritten,
          "a manifest of another format version is dropped");
    Check(manifest.Add(MakeRecord(7, Code(7, 50))),
          "a dropped manifest accepts records");
  }
  ProgramManifest manifest(path);
  Check(manifest.Size() == 1 && !manifest.Loaded().rewritten,
        "the rewritten manifest has the current version");
}

// Version 2 stored no key map header flags, version 3 no heap flags. An older
// manifest is read while the current one is missing: a record without
// GPU-selected resources reads the same in every version and is kept, the
// current file is written, and the previous one is left to the builds using it.
void TestPreviousVersion(uint8_t version) {
  const auto previous = TempPath("previous.programs");
  const auto path = TempPath("previous.v4.programs");
  std::filesystem::remove(path);
  std::filesystem::remove(previous);
  {
    ProgramManifest manifest(previous);
    auto record = MakeRecord(8, Code(8, 50));
    record.specialization = {};
    Check(manifest.Add(record), "add");
  }
  auto data = ReadFile(previous);
  Check(data.size() > 12 && data[8] == ProgramManifest::FormatVersion,
        "file has a header");
  data[8] = version;
  WriteFile(previous, data);
  {
    ProgramManifest manifest(path, previous);
    Check(manifest.Size() == 1 && manifest.Loaded().rewritten,
          "an older manifest lost its records or was not upgraded");
    Check(manifest.Add(MakeRecord(9, Code(9, 50))), "add after the upgrade");
  }
  Check(ReadFile(previous) == data,
        "the upgrade changed the previous version's file");
  ProgramManifest manifest(path, previous);
  Check(manifest.Size() == 2 && !manifest.Loaded().rewritten,
        "the upgraded manifest does not have the current version");
}

void TestCorruption() {
  const auto path = TempPath("corrupt.programs");
  {
    ProgramManifest manifest(path);
    for (uint32_t seed = 10; seed < 13; seed++) {
      Check(manifest.Add(MakeRecord(seed, Code(seed, 100))), "add");
    }
  }
  const auto intact = ReadFile(path);

  // A torn last entry (the session ended while writing it).
  auto torn = intact;
  torn.resize(torn.size() - 5);
  WriteFile(path, torn);
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() == 2 && manifest.Loaded().rewritten &&
              manifest.Loaded().dropped_bytes > 0,
          "a torn entry is dropped, the entries before it are kept");
    Check(manifest.Add(MakeRecord(20, Code(20, 100))),
          "records are appended after a torn tail");
  }
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() == 3 && !manifest.Loaded().rewritten,
          "records appended after a repaired tail are readable");
  }

  // A flipped byte inside the second program's code entry.
  auto flipped = intact;
  flipped[flipped.size() / 2] ^= 0x40u;
  WriteFile(path, flipped);
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() < 3 && manifest.Loaded().rewritten,
          "a corrupt entry and everything after it are dropped");
    const auto records = manifest.Records();
    for (const auto &record : records) {
      Check(record->code != nullptr && record->code->size() == 100,
            "kept records are whole");
    }
  }

  // Garbage and empty files.
  WriteFile(path, std::vector<uint8_t>(37, 0x5au));
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() == 0 && manifest.Loaded().rewritten,
          "a garbage file is replaced");
  }
  WriteFile(path, {});
  {
    ProgramManifest manifest(path);
    Check(manifest.Size() == 0, "an empty file is a new manifest");
    Check(manifest.Add(MakeRecord(30, Code(30, 8))), "add to a replaced file");
  }
  ProgramManifest manifest(path);
  Check(manifest.Size() == 1, "a replaced file accepts records");
}

void TestMemoryOnly() {
  ProgramManifest manifest;
  Check(manifest.Add(MakeRecord(40, Code(40, 8))) &&
            !manifest.Add(MakeRecord(40, Code(40, 8))),
        "an in-memory manifest deduplicates");
  ComputeProgramRecord empty;
  Check(!manifest.Add(empty), "a record without code is rejected");
}

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

void TestPrecompileQueue() {
  ProgramPrecompileQueue queue(1);
  Gate gate;
  std::atomic<bool> blocker_started{false};
  std::atomic<int> ran_a{0};
  std::atomic<int> ran_b{0};
  std::thread::id a_thread;
  // The single worker is held by the first job, so the others stay queued.
  Check(queue.Enqueue(1,
                      [&] {
                        blocker_started = true;
                        gate.Wait();
                      }),
        "queue the blocker");
  Check(queue.Enqueue(2,
                      [&] {
                        a_thread = std::this_thread::get_id();
                        ran_a++;
                      }),
        "queue a");
  Check(queue.Enqueue(3, [&] { ran_b++; }), "queue b");
  Check(!queue.Enqueue(2, [&] { ran_a++; }), "a key is queued once");
  Check(queue.Queued() == 3, "three jobs queued");
  while (!blocker_started) {
    std::this_thread::yield();
  }

  // Expediting a queued program runs it on the caller.
  queue.Expedite(2);
  Check(ran_a == 1 && a_thread == std::this_thread::get_id(),
        "an expedited job runs on the calling thread");
  queue.Expedite(2);
  Check(ran_a == 1, "expediting a finished job does nothing");
  queue.Expedite(99);
  Check(queue.Remaining() == 2, "the blocker and b remain");

  // Expediting the running job waits for it.
  std::thread releaser([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    gate.Release();
  });
  queue.Expedite(1);
  releaser.join();
  queue.WaitIdle();
  Check(ran_b == 1 && queue.Remaining() == 0, "every job ran once");
  Check(!queue.Enqueue(3, [&] { ran_b++; }),
        "a finished key is not queued again");

  // Stop drops jobs that have not started.
  ProgramPrecompileQueue stopped(1);
  Gate hold;
  std::atomic<bool> hold_started{false};
  std::atomic<int> dropped_ran{0};
  (void)stopped.Enqueue(1, [&] {
    hold_started = true;
    hold.Wait();
  });
  (void)stopped.Enqueue(2, [&] { dropped_ran++; });
  while (!hold_started) {
    std::this_thread::yield();
  }
  std::thread release_hold([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    hold.Release();
  });
  stopped.Stop();
  release_hold.join();
  Check(dropped_ran == 0, "a job that had not started is dropped on stop");
  stopped.Expedite(2);
  Check(dropped_ran == 0, "a stopped queue runs nothing");
  Check(!stopped.Enqueue(5, [] {}), "a stopped queue takes no jobs");

  // Programs start by order; an abandoned program counts as finished and its
  // worker is replaced, so the others still run.
  auto stuck_gate = std::make_shared<Gate>();
  auto stuck_started = std::make_shared<std::atomic<bool>>(false);
  {
    ProgramPrecompileQueue ordered(1, true);
    std::vector<int> ran;
    std::mutex ran_mutex;
    (void)ordered.Enqueue(10, [stuck_gate, stuck_started] {
      *stuck_started = true;
      stuck_gate->Wait();
    });
    while (!*stuck_started) {
      std::this_thread::yield();
    }
    const auto record = [&](int id) {
      return [&, id] {
        std::lock_guard lock(ran_mutex);
        ran.push_back(id);
      };
    };
    (void)ordered.Enqueue(11, record(1), 7);
    (void)ordered.Enqueue(12, record(2), 3);
    (void)ordered.Enqueue(13, record(3), 7);
    Check(ordered.QueuedBelow(7) == 1, "one program is queued before order 7");
    Check(!ordered.Abandon(11), "a queued program is not abandoned");
    Check(ordered.Abandon(10), "the running program is abandoned");
    ordered.WaitIdle();
    Check(ran == std::vector<int>{2, 1, 3} && ordered.Remaining() == 0,
          "the programs ran by order after the abandoned one's worker was "
          "replaced");
    ordered.Expedite(10);
    ordered.SetActiveThreads(1);
    Check(ordered.ActiveThreads() == 1 && ordered.ThreadCount() == 1,
          "the thread counts are reported");
  }
  // The queue is gone; the abandoned job returns later on its own thread.
  stuck_gate->Release();
}

void TestPrecompileProgress() {
  {
    PrecompileProgress progress;
    progress.BeginQueueing();
    Check(progress.EndQueueing(),
          "a run without jobs completes when queueing ends");
    const auto counts = progress.Get();
    Check(counts.queued == 0 && counts.finished == 0,
          "an empty run counts nothing");
  }
  {
    PrecompileProgress progress;
    progress.BeginQueueing();
    progress.Queued();
    Check(!progress.Finish(PrecompileOutcome::Cached, 0),
          "a job finishing while others are queued does not complete the run");
    progress.Queued();
    progress.Queued();
    progress.Rejected();
    Check(!progress.EndQueueing(), "a run with a running job is not complete");
    Check(progress.Finish(PrecompileOutcome::Compiled, 1500),
          "the last job completes the run");
    const auto counts = progress.Get();
    Check(counts.queued == 2 && counts.finished == 2 && counts.cached == 1 &&
              counts.compiled == 1 && counts.skipped == 0 &&
              counts.compile_ms == 1500,
          "outcomes are counted and a rejected job is not");
  }
  {
    // The pattern of PipelineCache::QueuePrecompile on a real queue: duplicate
    // keys are rejected, jobs finish while queueing goes on, and exactly one
    // caller learns that the run is complete, after every job has finished.
    constexpr uint64_t Jobs = 400;
    PrecompileProgress progress;
    std::atomic<int> completions{0};
    std::atomic<uint64_t> finished_at_completion{0};
    ProgramPrecompileQueue queue(4);
    const auto complete = [&] {
      completions++;
      finished_at_completion = progress.Get().finished;
    };
    progress.BeginQueueing();
    for (uint64_t i = 0; i < Jobs; i++) {
      const uint64_t key = 1 + i / 2; // every key twice
      progress.Queued();
      const auto outcome = static_cast<PrecompileOutcome>(i % 3);
      const bool queued = queue.Enqueue(key, [&, outcome] {
        if (progress.Finish(outcome, 1)) {
          complete();
        }
      });
      if (!queued) {
        progress.Rejected();
      }
    }
    if (progress.EndQueueing()) {
      complete();
    }
    queue.WaitIdle();
    const auto counts = progress.Get();
    Check(completions == 1, "a run completes exactly once");
    Check(counts.queued == Jobs / 2 && counts.finished == Jobs / 2 &&
              counts.cached + counts.compiled + counts.skipped == Jobs / 2,
          "every accepted job is counted once");
    Check(finished_at_completion == Jobs / 2,
          "the run completes only after its last job");
    Check(queue.Remaining() == 0, "the queue is drained");
  }
}
} // namespace

void TestCreatedShaders() {
  const auto directory = TempPath("created").parent_path();
  const auto created = ProgramManifest::CreatedPathFor(directory, "TITLE00001");
  Check(created != ProgramManifest::PathFor(directory, "TITLE00001") &&
            created !=
                ProgramManifest::PreviousPathFor(directory, "TITLE00001"),
        "created shaders live in their own file, which no format reads");
  std::filesystem::remove(created);
  // A created shader has its static input and no specialization; the same
  // shader created again (a level reloaded) is not recorded twice.
  auto shader = MakeRecord(9, Code(9, 96));
  shader.specialization = {};
  shader.spirv_hash = 0;
  {
    ProgramManifest manifest(created);
    Check(manifest.Add(shader) && !manifest.Add(shader),
          "a created shader is recorded once");
  }
  ProgramManifest manifest(created);
  const auto records = manifest.Records();
  Check(records.size() == 1 && SameRecord(*records[0], shader) &&
            records[0]->specialization.buffers.empty() &&
            records[0]->specialization.images.empty() &&
            records[0]->specialization.samplers.empty(),
        "a created shader survives a reload without a specialization");
}

void TestPrecompileAttempts() {
  using Libs::Graphics::PrecompileAttempts;
  const auto path = TempPath("precompile.attempts");
  {
    PrecompileAttempts attempts(path);
    Check(attempts.FailedCount() == 0, "a new attempts file has no failures");
    attempts.Begin(0x1234u);
    attempts.End(0x1234u);
    // The process ends in this build.
    attempts.Begin(0xabcdef0123456789u);
  }
  {
    // A torn last event is ignored.
    std::ofstream file(path, std::ios::app);
    file << "B";
  }
  {
    PrecompileAttempts attempts(path);
    Check(attempts.Retried(0xabcdef0123456789u) &&
              !attempts.Failed(0xabcdef0123456789u) &&
              !attempts.Retried(0x1234u) && attempts.RetriedCount() == 1,
          "an unfinished build is retried once, a finished one is not");
    // The retry ends the process again.
    attempts.Begin(0xabcdef0123456789u);
    attempts.Begin(0x55u);
    attempts.End(0x55u);
  }
  {
    PrecompileAttempts attempts(path);
    Check(attempts.Failed(0xabcdef0123456789u) && !attempts.Failed(0x55u) &&
              attempts.FailedCount() == 1,
          "a build that ended two processes is a failure");
    // Builds that run while the process ends on purpose.
    attempts.Begin(0x77u);
    attempts.Interrupt(0x77u);
    attempts.Begin(0x88u);
  }
  {
    PrecompileAttempts attempts(path);
    Check(!attempts.Retried(0x77u) && !attempts.Failed(0x77u) &&
              attempts.Retried(0x88u),
          "an interrupted build is not counted as one that ended the process");
    // Retried, and interrupted this time: the earlier unfinished begin stays.
    attempts.Begin(0x88u);
    attempts.Interrupt(0x88u);
  }
  PrecompileAttempts attempts(path);
  Check(attempts.Retried(0x88u) && !attempts.Failed(0x88u) &&
            attempts.Failed(0xabcdef0123456789u),
        "an interrupt takes back only its own process's begin");
  PrecompileAttempts memory;
  memory.Begin(1u);
  Check(!memory.Failed(1u), "attempts without a file record nothing");
}

int main() {
  TestRoundTrip();
  TestKey();
  TestDedupe();
  TestVersion();
  TestPreviousVersion(2u);
  TestPreviousVersion(3u);
  TestCorruption();
  TestMemoryOnly();
  TestPrecompileQueue();
  TestPrecompileProgress();
  TestCreatedShaders();
  TestPrecompileAttempts();
  std::printf("ProgramManifestTests: all tests passed\n");
  return 0;
}
