#include "graphics/host_gpu/memoryPriority.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

namespace {

using Libs::Graphics::MemoryPriority::Class;
using Libs::Graphics::MemoryPriority::Desired;
using Libs::Graphics::MemoryPriority::InUse;
using Libs::Graphics::MemoryPriority::Level;
using Libs::Graphics::MemoryPriority::Object;
using Libs::Graphics::MemoryPriority::Policy;
using Libs::Graphics::MemoryPriority::Tracker;
using Libs::Graphics::MemoryPriority::Value;

constexpr uint64_t MiB = 1024ull * 1024;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryPriorityTests: failed: %s\n", text);
    std::abort();
  }
}

// Records the priority calls the tracker makes, as the driver would see them.
struct Driver {
  std::map<uint64_t, float> priority;
  std::vector<uint64_t> calls;

  void operator()(uint64_t handle, float value) {
    priority[handle] = value;
    calls.push_back(handle);
  }
};

void TestLevels() {
  Check(Value(Level::Low) < Value(Level::Normal) &&
            Value(Level::Normal) < Value(Level::High),
        "levels are not ordered");
  Check(Value(Level::Normal) == 0.5f,
        "normal is not Vulkan's default priority");
  Check(InUse(Class::Cache) == Level::Normal,
        "cached textures do not default to normal");
  Check(InUse(Class::Target) == Level::High &&
            InUse(Class::Pinned) == Level::High,
        "targets and pinned memory are not high");

  Check(Desired(0, 0, 10, 42, 32) == Level::Normal,
        "a texture block used 32 frames ago is cold");
  Check(Desired(0, 0, 10, 43, 32) == Level::Low,
        "a texture block unused for 33 frames is not cold");
  Check(Desired(0, 1, 10, 42, 32) == Level::High,
        "a recently used target block is not high");
  Check(Desired(0, 1, 10, 43, 32) == Level::Low,
        "a cold target block is not low");
  Check(Desired(1, 0, 0, 1'000'000, 32) == Level::High, "pinned memory ages");
  Check(Desired(0, 0, 50, 40, 32) == Level::Normal,
        "a use stamped after the frame underflows");
}

void TestAging() {
  Tracker tracker(Policy{.cold_frames = 4, .objects_per_frame = 8});
  Driver driver;

  auto *texture = tracker.Acquire(1, 256 * MiB, Class::Cache, Level::Normal);
  auto *target = tracker.Acquire(2, 64 * MiB, Class::Target, Level::High);
  auto *ring = tracker.Acquire(3, 64 * MiB, Class::Pinned, Level::High);
  Check(tracker.ObjectCount() == 3, "objects are not registered");
  Check(tracker.GetStats().bytes[static_cast<size_t>(Level::Normal)] ==
            256 * MiB,
        "initial levels are not counted");

  // Used every frame: nothing changes, and no priority call is made.
  for (int frame = 0; frame < 10; frame++) {
    tracker.Use(texture);
    tracker.Use(target);
    tracker.Tick(driver);
  }
  Check(driver.calls.empty(), "used memory changed priority");

  // Unused: the texture and the target age to low; the ring stays high.
  for (int frame = 0; frame < 4; frame++) {
    tracker.Tick(driver);
  }
  Check(texture->CurrentLevel() == Level::Low, "the texture did not age");
  Check(target->CurrentLevel() == Level::Low, "the target did not age");
  Check(ring->CurrentLevel() == Level::High, "pinned memory aged");
  Check(driver.priority[1] == Value(Level::Low) &&
            driver.priority[2] == Value(Level::Low) &&
            !driver.priority.contains(3),
        "the driver did not see the demotions");
  const auto stats = tracker.GetStats();
  Check(stats.bytes[static_cast<size_t>(Level::Low)] == 320 * MiB &&
            stats.bytes[static_cast<size_t>(Level::High)] == 64 * MiB &&
            stats.bytes[static_cast<size_t>(Level::Normal)] == 0,
        "per-level bytes are wrong after aging");

  // A use promotes at the next tick, back to the class's level.
  tracker.Use(texture);
  tracker.Use(target);
  tracker.Tick(driver);
  Check(texture->CurrentLevel() == Level::Normal,
        "a used texture was not promoted");
  Check(target->CurrentLevel() == Level::High,
        "a used target was not promoted");
}

void TestBoundedTick() {
  Tracker tracker(Policy{.cold_frames = 1, .objects_per_frame = 4});
  Driver driver;
  for (uint64_t handle = 1; handle <= 64; handle++) {
    (void)tracker.Acquire(handle, MiB, Class::Cache, Level::Normal);
  }
  // Every object is cold after two frames, but only four are re-evaluated per
  // tick.
  tracker.Tick(driver);
  Check(driver.calls.empty(), "an object used in the last frame aged");
  tracker.Tick(driver);
  Check(driver.calls.size() == 4,
        "a tick visited more objects than the policy allows");
  for (int frame = 0; frame < 15; frame++) {
    tracker.Tick(driver);
  }
  Check(driver.calls.size() == 64,
        "round robin did not reach every object in 17 ticks");
  Check(tracker.GetStats().bytes[static_cast<size_t>(Level::Low)] == 64 * MiB,
        "not every object aged");
}

void TestPromotionIsQueued() {
  // Promotions do not wait for the round robin.
  Tracker tracker(Policy{.cold_frames = 1, .objects_per_frame = 1});
  Driver driver;
  std::vector<Object *> objects;
  for (uint64_t handle = 1; handle <= 32; handle++) {
    objects.push_back(
        tracker.Acquire(handle, MiB, Class::Cache, Level::Normal));
  }
  for (int frame = 0; frame < 40; frame++) {
    tracker.Tick(driver);
  }
  Check(objects[20]->CurrentLevel() == Level::Low, "the object did not age");
  tracker.Use(objects[20]);
  tracker.Use(objects[20]);
  driver.calls.clear();
  tracker.Tick(driver);
  Check(objects[20]->CurrentLevel() == Level::Normal,
        "a used object waited for the round robin");
  Check(driver.calls.size() == 1 && driver.calls[0] == 21,
        "a promotion was not applied exactly once");
}

void TestSharedObjectAndRelease() {
  Tracker tracker(Policy{.cold_frames = 2, .objects_per_frame = 8});
  Driver driver;

  // Two textures in one VMA block, then a target that fell back into it.
  auto *block = tracker.Acquire(7, 256 * MiB, Class::Cache, Level::Normal);
  Check(tracker.Acquire(7, 256 * MiB, Class::Cache, Level::Normal) == block,
        "a shared block is registered twice");
  (void)tracker.Acquire(7, 256 * MiB, Class::Target, Level::Normal);
  Check(tracker.ObjectCount() == 1, "a shared block is counted twice");
  tracker.Tick(driver);
  Check(block->CurrentLevel() == Level::High,
        "a block holding a target did not rise to high");

  tracker.Use(block);
  tracker.Release(block, Class::Target, driver);
  tracker.Tick(driver);
  Check(block->CurrentLevel() == Level::Normal,
        "a block without targets stayed high");

  for (int frame = 0; frame < 4; frame++) {
    tracker.Tick(driver);
  }
  Check(block->CurrentLevel() == Level::Low, "the block did not age");

  // The allocator may keep an empty block: the last release restores its
  // initial priority so the next allocation finds what the allocator set.
  tracker.Release(block, Class::Cache, driver);
  Check(tracker.ObjectCount() == 1, "the block left while still in use");
  tracker.Release(block, Class::Cache, driver);
  Check(tracker.ObjectCount() == 0, "the empty block stayed registered");
  Check(driver.priority[7] == Value(Level::Normal),
        "the empty block's initial priority was not restored");
  const auto stats = tracker.GetStats();
  Check(stats.bytes[0] == 0 && stats.bytes[1] == 0 && stats.bytes[2] == 0,
        "a released block is still counted");

  // A release of a class the object does not hold is ignored.
  auto *other = tracker.Acquire(8, MiB, Class::Cache, Level::Normal);
  tracker.Release(other, Class::Pinned, driver);
  Check(tracker.ObjectCount() == 1, "a mismatched release freed the object");
  tracker.Release(nullptr, Class::Cache, driver);
  tracker.Use(nullptr);
}

void TestReleaseWhileQueued() {
  Tracker tracker(Policy{.cold_frames = 1, .objects_per_frame = 8});
  Driver driver;
  auto *object = tracker.Acquire(9, MiB, Class::Cache, Level::Normal);
  auto *keep = tracker.Acquire(10, MiB, Class::Cache, Level::Normal);
  tracker.Tick(driver);
  tracker.Tick(driver);
  Check(object->CurrentLevel() == Level::Low, "the object did not age");
  tracker.Use(object);
  tracker.Release(object, Class::Cache, driver);
  // The queued promotion must not touch the released object.
  driver.calls.clear();
  tracker.Use(keep);
  tracker.Tick(driver);
  Check(driver.calls.size() == 1 && driver.calls[0] == 10,
        "a released object was promoted");
}

} // namespace

int main() {
  TestLevels();
  TestAging();
  TestBoundedTick();
  TestPromotionIsQueued();
  TestSharedObjectAndRelease();
  TestReleaseWhileQueued();
  std::printf("MemoryPriorityTests: all tests passed\n");
  return 0;
}
