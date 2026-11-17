#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEREBUILD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEREBUILD_H_

#include <cstdint>

namespace Libs::Graphics {

// When a session rebuilds its driver cache in the background (PipelineCache::EnableStaleRebuild).
// A stale entry (DriverCacheHistory) compiles in every session until the cache is rebuilt from
// empty, so a session rebuilds once one is on record: found by earlier sessions or by this one.
// It rebuilds at most once, starts only while no other precompile run is open (they share the
// precompile queue), and replaces the loaded cache once every build of its run has finished.
// Vulkan-free; used on one thread.
class DriverCacheRebuildPolicy {
public:
	enum class State : uint8_t {
		// No rebuild in this session (disabled, or no driver cache).
		Off,
		// Waiting for a stale entry.
		Watching,
		// The new cache is being filled.
		Running,
		// The new cache replaced the loaded one, or the rebuild could not start.
		Done,
	};

	void Enable() {
		if (m_state == State::Off) m_state = State::Watching;
	}
	[[nodiscard]] State GetState() const { return m_state; }
	[[nodiscard]] bool  Running() const { return m_state == State::Running; }

	// Whether to start now: `stale` counts the stale misses on record (earlier sessions' and
	// this one's), `run_open` tells whether another precompile run has unfinished builds.
	[[nodiscard]] bool ShouldStart(uint64_t stale, bool run_open) const {
		return m_state == State::Watching && stale != 0 && !run_open;
	}
	void Started() { m_state = State::Running; }
	// Whether the new cache replaces the loaded one now: every build of its run has finished.
	[[nodiscard]] bool ShouldReplace(bool run_complete) const {
		return m_state == State::Running && run_complete;
	}
	// The cache was replaced, or the rebuild cannot start (no new cache): no other this session.
	void Finished() { m_state = State::Done; }

private:
	State m_state = State::Off;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEREBUILD_H_ */
