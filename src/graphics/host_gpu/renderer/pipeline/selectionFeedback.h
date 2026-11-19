#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SELECTIONFEEDBACK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SELECTIONFEEDBACK_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/candidateSelection.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace ShaderRecompiler::IR {
struct CompiledShaderInfo;
struct ResourceSnapshot;
} // namespace ShaderRecompiler::IR

class Buffer;
class CommandScheduler;
struct GraphicContext;

// The renderer side of the selection feedback (IR::Bindings::selection_dword): every binding of a
// program with enumerated image tables gets a region of a device ring, one word per candidate, in
// which the shader stores the frame's stamp at the ordinals it selects. At the end of a frame the
// frame's regions are copied to host memory; once the copy completed, CandidateSelection learns
// which candidates were selected, and the stage bindings of later frames bind the long unselected
// ones as null descriptors (CandidateSelection::Demotes). Images no binding touches any more age
// out under the texture cache's ordinary rules.
//
// Used only on the GPU thread.
class SelectionFeedback {
public:
	SelectionFeedback(GraphicContext& graphics, CommandScheduler& scheduler);
	~SelectionFeedback();
	KYTY_CLASS_NO_COPY(SelectionFeedback);

	// For a stage about to be bound: `words` receives the program's selection dwords (the
	// header and a region base per table, see IR::Bindings), and demoted[i] is set for every
	// image resource i the stage binds as a null descriptor. Leaves both empty for a program
	// without enumerated tables.
	void PrepareStage(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                  const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                  std::vector<uint32_t>& words, std::vector<uint8_t>& demoted);

	// At a frame boundary: copies the frame's regions for the host and reads the feedback of
	// every earlier frame whose copy completed.
	void EndFrame();

	[[nodiscard]] const CandidateSelection& Selection() const noexcept { return m_selection; }
	[[nodiscard]] uint64_t                  Frame() const noexcept { return m_frame; }
	// Frames whose feedback was read.
	[[nodiscard]] uint64_t DecodedFrame() const noexcept { return m_decoded; }

	static constexpr uint32_t RingWords = 1u << 20u;

private:
	struct Entry {
		CandidateSelection::Candidate* candidate = nullptr;
		bool                           demoted   = false;
	};
	struct Record {
		CandidateSelection::Table* table  = nullptr;
		uint32_t                   base   = 0;
		uint32_t                   first  = 0; // In PendingFrame::entries.
		uint32_t                   count  = 0;
		uint64_t                   shader = 0;
		uint32_t                   root   = 0;
	};
	struct PendingFrame {
		uint64_t            frame    = 0;
		uint32_t            stamp    = 0;
		uint64_t            tick     = 0;
		uint32_t            start    = 0;
		uint32_t            end      = 0;
		uint32_t            wrap_end = 0; // Ring end of the part before a wrap, 0 without one.
		std::vector<Record> records;
		std::vector<Entry>  entries;
	};

	bool                          Create();
	uint32_t                      Allocate(uint32_t words);
	void                          Decode(PendingFrame& frame);
	void                          Report();
	[[nodiscard]] static uint32_t Stamp(uint64_t frame) {
		return static_cast<uint32_t>(frame % 0xffffffffu) + 1u;
	}

	GraphicContext&          m_graphics;
	CommandScheduler&        m_scheduler;
	CandidateSelection       m_selection;
	std::unique_ptr<Buffer>  m_ring;
	std::unique_ptr<Buffer>  m_readback;
	bool                     m_failed  = false;
	uint64_t                 m_frame   = 1;
	uint64_t                 m_decoded = 0;
	PendingFrame             m_current;
	std::deque<PendingFrame> m_pending;
	uint32_t                 m_head = 0;
	// Regions of identical bindings within the frame: content hash -> base.
	std::unordered_map<uint64_t, uint32_t> m_regions;
	std::vector<Entry>                     m_scratch;
	// Report counters since the last report.
	uint64_t                              m_report_frames  = 0;
	uint64_t                              m_report_binds   = 0;
	uint64_t                              m_report_demoted = 0;
	uint64_t                              m_report_bound   = 0;
	uint64_t                              m_report_full    = 0;
	std::chrono::steady_clock::time_point m_report_time;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SELECTIONFEEDBACK_H_
