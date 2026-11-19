#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CANDIDATESELECTION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CANDIDATESELECTION_H_

#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace Libs::Graphics {

// Which candidates of the GPU-selected image tables shaders actually select, as the selection
// feedback reports it frame by frame (SelectionFeedback), and which candidates a stage may
// therefore bind as null descriptors.
//
// A program enumerates every texture its table could select and binds each of them, so a table
// that keeps descriptors of textures it no longer uses (Saros lists thousands of past transient
// placements in one table its draw never even samples) keeps an image alive per descriptor. A
// candidate is demoted - bound as a null descriptor, neither looked up nor touched - once the
// shader went QuietFrames observed frames without selecting it, or without selecting anything
// of its table. Only frames in which the candidate was bound with feedback count, so a program
// that does not run makes no candidate quiet. A demoted candidate is still reported when it is
// selected: that is a miss (the access read the null descriptor), after which the table binds
// every candidate for the rest of the session. Tables a shader writes are never demoted: a store
// to a null descriptor is dropped (the caller passes `writes`).
//
// Not thread-safe: the GPU thread prepares the bindings and reads the feedback.
class CandidateSelection {
public:
	static constexpr uint32_t QuietFrames = 64;
	// Candidates not bound for this many frames are forgotten; bound again they start quiet-free.
	static constexpr uint64_t ForgetFrames = 1024;

	struct Candidate {
		uint64_t last_bound = 0; // Frame of the last binding.
		uint64_t observed   = 0; // Last frame whose feedback counted for it.
		uint64_t selected   = 0; // Last frame it was selected in.
		uint32_t quiet      = 0; // Observed frames since its last selection.
	};

	struct Table {
		uint64_t key        = 0;
		uint64_t last_bound = 0;
		uint64_t observed   = 0;
		uint64_t selected   = 0;
		uint32_t quiet      = 0; // Observed frames in which nothing of the table was selected.
		bool     eager      = false;
		std::unordered_map<uint64_t, Candidate> candidates;
	};

	Table& FindTable(uint64_t key, uint64_t frame) {
		auto& table      = m_tables[key];
		table.key        = key;
		table.last_bound = frame;
		return table;
	}

	static Candidate& FindCandidate(Table& table, uint64_t key, uint64_t frame) {
		auto& candidate      = table.candidates[key];
		candidate.last_bound = frame;
		return candidate;
	}

	[[nodiscard]] static bool Demotes(const Table& table, const Candidate& candidate, bool writes) {
		return !writes && !table.eager &&
		       (table.quiet >= QuietFrames || candidate.quiet >= QuietFrames);
	}

	// Feedback of a decoded frame: first every selection of the frame (Selected), then every
	// candidate bound in it (Observed). `candidate` is null for the root's own ordinal and spares.
	static void Selected(Table& table, Candidate* candidate, uint64_t frame) {
		table.selected = frame;
		if (candidate != nullptr) {
			candidate->selected = frame;
		}
	}
	static void Observed(Table& table, Candidate* candidate, uint64_t frame) {
		if (table.observed != frame) {
			table.observed = frame;
			table.quiet    = table.selected == frame ? 0u : Saturated(table.quiet);
		}
		if (candidate != nullptr && candidate->observed != frame) {
			candidate->observed = frame;
			candidate->quiet    = candidate->selected == frame ? 0u : Saturated(candidate->quiet);
		}
	}
	// A demoted candidate was selected. Returns true when this turned the table eager.
	bool Missed(Table& table) {
		m_misses++;
		if (table.eager) return false;
		table.eager = true;
		m_eager_tables++;
		return true;
	}

	// Drops candidates not bound since `frame` - ForgetFrames; `frame` must be no later than the
	// oldest frame whose feedback is still to be read, so no pending record names them.
	void Forget(uint64_t frame) {
		if (frame <= ForgetFrames) return;
		const auto limit = frame - ForgetFrames;
		for (auto& [key, table]: m_tables) {
			std::erase_if(table.candidates,
			              [&](const auto& entry) { return entry.second.last_bound < limit; });
		}
	}

	[[nodiscard]] size_t   TableCount() const noexcept { return m_tables.size(); }
	[[nodiscard]] size_t   EagerTables() const noexcept { return m_eager_tables; }
	[[nodiscard]] uint64_t Misses() const noexcept { return m_misses; }

private:
	static uint32_t Saturated(uint32_t quiet) { return std::min(quiet + 1u, QuietFrames * 4u); }

	std::unordered_map<uint64_t, Table> m_tables;
	size_t                              m_eager_tables = 0;
	uint64_t                            m_misses       = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CANDIDATESELECTION_H_
