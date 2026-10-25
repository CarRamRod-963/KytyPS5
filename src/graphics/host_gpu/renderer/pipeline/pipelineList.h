#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIST_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/host_gpu/renderer/pipeline/programManifest.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// A shareable list of the pipelines a title needs: the records of the program and pipeline
// manifests with every shader named by its content (ShaderCodeId) instead of its code, so that it
// holds no game code. A player's emulator resolves the names against the shaders its own game
// created (ShaderCodeStore) or that earlier sessions recorded, and precompiles what resolves
// (--import-pipeline-list). Entries are checksummed like the manifests; a damaged list is read
// up to its last intact entry.
struct PipelineListContents {
	struct Compute {
		ShaderCodeId         code;
		ComputeProgramRecord record; // code is null
		// A created shader without a recorded program: built for its code-only specialization.
		bool code_only = false;
	};
	std::string          title_id;
	std::vector<Compute> compute;
	// Code ids set, code null.
	std::vector<GraphicsStageRecord>    stages;
	std::vector<GraphicsPipelineRecord> pipelines;
};

class PipelineList {
public:
	static constexpr uint32_t FormatVersion = 1;

	struct ReadStats {
		bool   valid         = false; // A list of this format.
		bool   title_matches = false;
		size_t dropped_bytes = 0; // Unreadable tail.
		size_t rejected      = 0; // Intact entries of a layout this build cannot use.
	};

	// Writes the list; only stages that a pipeline uses are listed, and only pipelines whose
	// stages are listed.
	static bool Write(const std::filesystem::path& path, const PipelineListContents& contents);
	// Reads a list. Entries of another title are not returned (title_matches is false).
	static PipelineListContents Read(const std::filesystem::path& path, const std::string& title_id,
	                                 ReadStats* stats = nullptr);
	// The contents of a title's manifests.
	static PipelineListContents FromManifests(const std::string&      title_id,
	                                          const ProgramManifest*  programs,
	                                          const ProgramManifest*  created,
	                                          const PipelineManifest* pipelines);
};

// Resolves a list's shader names to code: up front from what the emulator has, then as the
// guest creates shaders. A compute entry is complete when its code is known; a pipeline when the
// code of all of its stages is. Each entry is returned complete exactly once.
//
// Thread-safe.
class PipelineListResolver {
public:
	using Lookup = std::function<ShaderCode(const ShaderCodeId&)>;

	struct Resolved {
		std::vector<std::shared_ptr<const ComputeProgramRecord>> compute;
		std::vector<bool>                                        compute_code_only;
		struct Pipeline {
			std::shared_ptr<const GraphicsPipelineRecord>           record;
			std::vector<std::shared_ptr<const GraphicsStageRecord>> vertex_stages;
			std::shared_ptr<const GraphicsStageRecord>              pixel_stage;
		};
		std::vector<Pipeline> pipelines;
		[[nodiscard]] bool    Empty() const { return compute.empty() && pipelines.empty(); }
	};

	struct Counts {
		size_t compute            = 0;
		size_t compute_resolved   = 0;
		size_t pipelines          = 0;
		size_t pipelines_resolved = 0;
		// Pipelines that name a stage the list does not hold.
		size_t pipelines_broken = 0;
		// Shader names not resolved yet.
		size_t shaders_pending = 0;
	};

	explicit PipelineListResolver(PipelineListContents contents);
	KYTY_CLASS_NO_COPY(PipelineListResolver);

	// Resolves every pending name `lookup` knows.
	Resolved Resolve(const Lookup& lookup);
	// A shader with this code exists now (the guest created it).
	Resolved             OnCode(std::span<const uint32_t> code);
	[[nodiscard]] Counts GetCounts() const;
	[[nodiscard]] bool   Pending() const;

private:
	struct StageState {
		GraphicsStageRecord                        record;
		ShaderCode                                 front;
		ShaderCode                                 back;
		uint32_t                                   missing = 0;
		std::shared_ptr<const GraphicsStageRecord> done;
		std::vector<size_t>                        pipelines;
	};
	struct PipelineState {
		std::shared_ptr<const GraphicsPipelineRecord> record;
		std::vector<size_t>                           stages; // vertex stages, then pixel
		uint32_t                                      missing = 0;
		bool                                          done    = false;
	};
	struct ComputeState {
		PipelineListContents::Compute entry;
		bool                          done = false;
	};
	struct Waiters {
		std::vector<size_t> compute;
		std::vector<size_t> stages;
	};

	void Supply(const ShaderCodeId& id, const ShaderCode& code, Resolved& out);

	mutable std::mutex                                          m_mutex;
	std::vector<ComputeState>                                   m_compute;
	std::vector<StageState>                                     m_stages;
	std::vector<PipelineState>                                  m_pipelines;
	std::unordered_map<ShaderCodeId, Waiters, ShaderCodeIdHash> m_waiting;
	size_t                                                      m_broken             = 0;
	size_t                                                      m_compute_resolved   = 0;
	size_t                                                      m_pipelines_resolved = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIST_H_
