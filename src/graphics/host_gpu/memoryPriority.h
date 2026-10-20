#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Residency priorities of device-memory objects (VK_EXT_memory_priority and
// VK_EXT_pageable_device_local_memory). When device memory is over budget, the OS moves whole
// memory objects to system memory, lowest priority first. Without priorities it may pick a block
// that every frame reads or a driver surface (a demoted local-memory surface for spills made each
// spill cross the bus). Cached textures nobody sampled for a while are what should go first.
//
// Vulkan-free so the policy can be unit tested: memory objects are opaque 64-bit handles and the
// priority change is a callback.
namespace Libs::Graphics::MemoryPriority {

// What an allocation holds, which decides the priority of the memory object it lives in.
enum class Class : uint8_t {
	// Sampled images of the texture cache: normal priority while used, low once cold.
	Cache,
	// Render targets, depth and storage images: high priority while used, low once cold.
	Target,
	// Memory that shaders may reach without a cache lookup marking it used (buffers reached
	// through device addresses, rings, descriptor tables, swapchain images): always high.
	Pinned,
};

enum class Level : uint8_t { Low, Normal, High };

// 0.5 is the default priority of every allocation without VK_EXT_memory_priority, which is also
// what the driver's own allocations get.
[[nodiscard]] constexpr float Value(Level level) {
	switch (level) {
		case Level::Low: return 0.0f;
		case Level::Normal: return 0.5f;
		case Level::High: return 1.0f;
	}
	return 0.5f;
}

// The level of an allocation of the class while it is in use.
[[nodiscard]] constexpr Level InUse(Class memory_class) {
	return memory_class == Class::Cache ? Level::Normal : Level::High;
}

struct Policy {
	// A memory object unused for more frames than this is cold. The texture cache keeps images
	// for 80 frames and more; a block unused for this long is the first to give up.
	uint64_t cold_frames = 32;
	// Memory objects re-evaluated per frame, round robin. Bounds the per-frame cost; promotions of
	// cold objects that are used again are applied at the next frame regardless.
	uint32_t objects_per_frame = 16;
};

// The level a memory object should have.
[[nodiscard]] constexpr Level Desired(uint32_t pinned, uint32_t targets, uint64_t last_use,
                                      uint64_t frame, uint64_t cold_frames) {
	if (pinned != 0) {
		return Level::High;
	}
	if (frame - std::min(frame, last_use) > cold_frames) {
		return Level::Low;
	}
	return targets != 0 ? Level::High : Level::Normal;
}

// One device-memory object (a VMA block or a dedicated allocation) and the allocations in it.
class Object {
public:
	[[nodiscard]] uint64_t Handle() const noexcept { return m_handle; }
	[[nodiscard]] uint64_t Size() const noexcept { return m_size; }
	[[nodiscard]] Level    CurrentLevel() const noexcept {
		return m_level.load(std::memory_order_relaxed);
	}

private:
	friend class Tracker;

	uint64_t              m_handle  = 0;
	uint64_t              m_size    = 0;
	Level                 m_initial = Level::Normal; // What the allocator gave the object.
	std::atomic<Level>    m_level {Level::Normal};
	std::atomic<uint64_t> m_last_use {0};
	std::atomic<bool>     m_queued {false};
	uint32_t              m_counts[3] {}; // Allocations per Class.
	size_t                m_index = 0;    // Position in Tracker::m_objects.
};

// Tracks the memory objects that hold the emulator's allocations and moves each to the level its
// contents need. Use() is lock-free and O(1); Tick() is O(Policy::objects_per_frame) plus the
// promotions that Use() queued since the last tick.
class Tracker {
public:
	struct Stats {
		uint64_t bytes[3] {}; // Per Level.
		uint64_t changes = 0; // Priority changes applied so far.
	};

	explicit Tracker(Policy policy = {}): m_policy(policy) {}
	Tracker(const Tracker&)            = delete;
	Tracker& operator=(const Tracker&) = delete;

	// Registers an allocation of the class in the memory object. `initial` is the priority the
	// allocator gave the object when it created it; it is restored when the last allocation
	// leaves, because the allocator may keep the empty object and fill it with other classes.
	Object* Acquire(uint64_t handle, uint64_t size, Class memory_class, Level initial) {
		std::scoped_lock lock(m_mutex);
		auto&            slot = m_by_handle[handle];
		if (slot == nullptr) {
			slot            = std::make_unique<Object>();
			slot->m_handle  = handle;
			slot->m_size    = size;
			slot->m_initial = initial;
			slot->m_level.store(initial, std::memory_order_relaxed);
			slot->m_index = m_objects.size();
			m_objects.push_back(slot.get());
			m_stats.bytes[static_cast<size_t>(initial)] += size;
		}
		auto* object = slot.get();
		object->m_counts[static_cast<size_t>(memory_class)]++;
		object->m_last_use.store(m_frame.load(std::memory_order_relaxed),
		                         std::memory_order_relaxed);
		return object;
	}

	// Unregisters an allocation; call it before the allocation is freed. When it was the last
	// one in the object, the object's initial priority is restored through `set`.
	template <typename SetPriority>
	void Release(Object* object, Class memory_class, SetPriority&& set) {
		if (object == nullptr) {
			return;
		}
		std::scoped_lock lock(m_mutex);
		auto&            count = object->m_counts[static_cast<size_t>(memory_class)];
		if (count == 0) {
			return;
		}
		count--;
		if (object->m_counts[0] != 0 || object->m_counts[1] != 0 || object->m_counts[2] != 0) {
			return;
		}
		Apply(*object, object->m_initial, set);
		m_stats.bytes[static_cast<size_t>(object->m_initial)] -= object->m_size;
		std::erase(m_promotions, object);
		auto* last                 = m_objects.back();
		last->m_index              = object->m_index;
		m_objects[object->m_index] = last;
		m_objects.pop_back();
		if (m_cursor > m_objects.size()) {
			m_cursor = 0;
		}
		m_by_handle.erase(object->m_handle);
	}

	// Marks the object used in the current frame. A low object is queued for promotion at the
	// next tick.
	void Use(Object* object) {
		if (object == nullptr) {
			return;
		}
		const auto frame = m_frame.load(std::memory_order_relaxed);
		if (object->m_last_use.load(std::memory_order_relaxed) != frame) {
			object->m_last_use.store(frame, std::memory_order_relaxed);
		}
		if (object->m_level.load(std::memory_order_relaxed) == Level::Low &&
		    !object->m_queued.exchange(true, std::memory_order_relaxed)) {
			std::scoped_lock lock(m_mutex);
			m_promotions.push_back(object);
		}
	}

	// Once per frame: advances the frame, applies queued promotions and re-evaluates up to
	// Policy::objects_per_frame objects.
	template <typename SetPriority>
	void Tick(SetPriority&& set) {
		const auto       frame = m_frame.fetch_add(1, std::memory_order_relaxed) + 1;
		std::scoped_lock lock(m_mutex);
		for (auto* object: m_promotions) {
			object->m_queued.store(false, std::memory_order_relaxed);
			Evaluate(*object, frame, set);
		}
		m_promotions.clear();
		const auto visits = std::min<size_t>(m_policy.objects_per_frame, m_objects.size());
		for (size_t i = 0; i < visits; i++) {
			if (m_cursor >= m_objects.size()) {
				m_cursor = 0;
			}
			Evaluate(*m_objects[m_cursor++], frame, set);
		}
	}

	[[nodiscard]] Stats GetStats() const {
		std::scoped_lock lock(m_mutex);
		return m_stats;
	}
	[[nodiscard]] size_t ObjectCount() const {
		std::scoped_lock lock(m_mutex);
		return m_objects.size();
	}
	[[nodiscard]] uint64_t Frame() const noexcept {
		return m_frame.load(std::memory_order_relaxed);
	}

private:
	template <typename SetPriority>
	void Evaluate(Object& object, uint64_t frame, SetPriority& set) {
		const auto desired =
		    Desired(object.m_counts[static_cast<size_t>(Class::Pinned)],
		            object.m_counts[static_cast<size_t>(Class::Target)],
		            object.m_last_use.load(std::memory_order_relaxed), frame, m_policy.cold_frames);
		Apply(object, desired, set);
	}

	template <typename SetPriority>
	void Apply(Object& object, Level level, SetPriority& set) {
		const auto current = object.m_level.load(std::memory_order_relaxed);
		if (current == level) {
			return;
		}
		set(object.m_handle, Value(level));
		object.m_level.store(level, std::memory_order_relaxed);
		m_stats.bytes[static_cast<size_t>(current)] -= object.m_size;
		m_stats.bytes[static_cast<size_t>(level)] += object.m_size;
		m_stats.changes++;
	}

	Policy                                                m_policy;
	mutable std::mutex                                    m_mutex;
	std::atomic<uint64_t>                                 m_frame {0};
	std::unordered_map<uint64_t, std::unique_ptr<Object>> m_by_handle;
	std::vector<Object*>                                  m_objects;
	std::vector<Object*>                                  m_promotions;
	size_t                                                m_cursor = 0;
	Stats                                                 m_stats;
};

} // namespace Libs::Graphics::MemoryPriority
