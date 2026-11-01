#include "graphics/host_gpu/renderer/cmdSignature.h"

#include "graphics/host_gpu/renderer/debugSample.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <xxhash.h>

// KYTY_TRACE_CMD_SIGNATURE (see cmdSignature.h).
namespace Libs::Graphics::CmdSignature {

namespace {

constexpr uint64_t kSeed = 0x9e3779b97f4a7c15ull;

uint64_t Mix(uint64_t hash, uint64_t value) {
	return XXH3_64bits_withSeed(&value, sizeof(value), hash);
}

struct Pass {
	std::string name;
	uint64_t    hash   = kSeed;
	uint32_t    events = 0;
};

struct State {
	// The current frame (presented-frame counter when it started).
	bool     frame_open = false;
	uint32_t frame      = 0;
	uint64_t frame_hash = kSeed;
	uint32_t events     = 0;
	uint32_t draws      = 0;
	uint32_t dispatches = 0;
	uint32_t outside    = 0; // folds made outside any event
	// Passes of the frame in first-appearance order.
	std::vector<Pass>                       passes;
	std::unordered_map<std::string, size_t> pass_index;
	// The current event.
	uint32_t  event_depth = 0;
	EventKind event_kind  = EventKind::Draw;
	uint64_t  event_hash  = kSeed;
	// KYTY_TRACE_CMD_SIGNATURE_FILE.
	FILE*    file        = nullptr;
	uint32_t file_first  = 0;
	uint32_t file_last   = UINT32_MAX;
};

thread_local State* t_state = nullptr;

const char* EventKindName(EventKind kind) {
	switch (kind) {
		case EventKind::Draw: return "draw";
		case EventKind::DrawIndirect: return "draw_indirect";
		case EventKind::Dispatch: return "dispatch";
		case EventKind::DispatchIndirect: return "dispatch_indirect";
	}
	return "?";
}

Pass& PassOf(State& s, const std::string& name) {
	if (const auto found = s.pass_index.find(name); found != s.pass_index.end()) {
		return s.passes[found->second];
	}
	s.pass_index.emplace(name, s.passes.size());
	s.passes.push_back({name, kSeed, 0});
	return s.passes.back();
}

void FlushFrame(State& s) {
	if (!s.frame_open) {
		return;
	}
	std::fprintf(stderr,
	             "CMD_SIGNATURE frame=%u events=%u draws=%u dispatches=%u outside=%u passes=%zu sig=%016llx\n", s.frame,
	             s.events, s.draws, s.dispatches, s.outside, s.passes.size(),
	             static_cast<unsigned long long>(s.frame_hash));
	if (Mode() >= 2) {
		for (const auto& pass: s.passes) {
			std::fprintf(stderr, "CMD_SIGNATURE_PASS frame=%u pass=%s events=%u sig=%016llx\n", s.frame,
			             pass.name.c_str(), pass.events, static_cast<unsigned long long>(pass.hash));
		}
	}
	if (s.file != nullptr) {
		std::fflush(s.file);
	}
	s.frame_open = false;
	s.frame_hash = kSeed;
	s.events     = 0;
	s.draws      = 0;
	s.dispatches = 0;
	s.outside    = 0;
	s.passes.clear();
	s.pass_index.clear();
}

// A new frame starts when the presented-frame counter moved since the current one started.
void SyncFrame(State& s) {
	const uint32_t present = DebugPresentCount().load(std::memory_order_relaxed);
	if (s.frame_open && present == s.frame) {
		return;
	}
	FlushFrame(s);
	s.frame_open = true;
	s.frame      = present;
}

bool PassesWanted(const State& s) {
	return Mode() >= 2 || s.file != nullptr;
}

} // namespace

uint32_t ModeInit() {
	const char* value = std::getenv("KYTY_TRACE_CMD_SIGNATURE");
	const auto  mode  = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	if (mode != 0) {
		std::fprintf(stderr, "TRACE_CMD_SIGNATURE enabled mode=%u\n", mode);
	}
	return mode;
}

void MarkGpuThread() {
	if (!Enabled() || t_state != nullptr) {
		return;
	}
	auto* state = new State {}; // kept for the life of the thread
	if (const char* path = std::getenv("KYTY_TRACE_CMD_SIGNATURE_FILE"); path != nullptr && path[0] != '\0') {
		state->file = std::fopen(path, "a");
		if (state->file == nullptr) {
			std::fprintf(stderr, "TRACE_CMD_SIGNATURE cannot open %s\n", path);
		}
		if (const char* frames = std::getenv("KYTY_TRACE_CMD_SIGNATURE_FRAMES"); frames != nullptr) {
			unsigned first = 0, last = 0;
			if (std::sscanf(frames, "%u-%u", &first, &last) == 2 && first <= last) {
				state->file_first = first;
				state->file_last  = last;
			}
		}
	}
	t_state = state;
}

void Fold(Kind kind, const void* data, size_t size) {
	State* const s = t_state;
	if (s == nullptr) {
		return;
	}
	if (s->event_depth != 0) {
		s->event_hash = XXH3_64bits_withSeed(data, size, Mix(s->event_hash, static_cast<uint64_t>(kind)));
		return;
	}
	// Outside a draw/dispatch: straight into the frame (and the pass).
	SyncFrame(*s);
	s->outside++;
	const uint64_t hash = XXH3_64bits_withSeed(data, size, static_cast<uint64_t>(kind));
	s->frame_hash       = Mix(s->frame_hash, hash);
	if (PassesWanted(*s)) {
		Pass& pass = PassOf(*s, DebugMarkerPath());
		pass.hash  = Mix(pass.hash, hash);
	}
}

bool EventBegin(EventKind kind) {
	State* const s = t_state;
	if (s == nullptr) {
		return false;
	}
	if (s->event_depth++ == 0) {
		SyncFrame(*s);
		s->event_kind = kind;
		s->event_hash = Mix(kSeed, static_cast<uint64_t>(kind));
	}
	return true;
}

void EventEnd() {
	State* const s = t_state;
	if (s == nullptr || s->event_depth == 0 || --s->event_depth != 0) {
		return;
	}
	const uint32_t event = s->events++;
	if (s->event_kind == EventKind::Dispatch || s->event_kind == EventKind::DispatchIndirect) {
		s->dispatches++;
	} else {
		s->draws++;
	}
	s->frame_hash = Mix(s->frame_hash, s->event_hash);
	if (PassesWanted(*s)) {
		const std::string marker = DebugMarkerPath();
		Pass&             pass   = PassOf(*s, marker);
		pass.hash                = Mix(pass.hash, s->event_hash);
		pass.events++;
		if (s->file != nullptr && s->frame >= s->file_first && s->frame <= s->file_last) {
			std::fprintf(s->file, "%u %u %s %016llx %s\n", s->frame, event, EventKindName(s->event_kind),
			             static_cast<unsigned long long>(s->event_hash), marker.c_str());
		}
	}
}

} // namespace Libs::Graphics::CmdSignature
