#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEMANIFEST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEMANIFEST_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/manifestFormat.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineState.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// Guest code by content: the XXH3-64 of its bytes and its length in words. Names a shader without
// its code, so that records can be shared without the game's code.
struct ShaderCodeId {
	uint64_t hash  = 0;
	uint32_t words = 0;

	[[nodiscard]] static ShaderCodeId Of(std::span<const uint32_t> code);
	[[nodiscard]] explicit            operator bool() const { return words != 0; }
	bool                              operator==(const ShaderCodeId&) const = default;
};

struct ShaderCodeIdHash {
	size_t operator()(const ShaderCodeId& id) const {
		return static_cast<size_t>(id.hash ^ (uint64_t {id.words} << 32u));
	}
};

using ShaderCode = std::shared_ptr<const std::vector<uint32_t>>;

// A graphics shader stage program as the pipeline cache translated it (ProgramCache::Get): the
// guest code, the static input the translation reads and the resource specialization the module
// was compiled for. Like ComputeProgramRecord, it re-translates with any later recompiler.
struct GraphicsStageRecord {
	// The logical stage: Vertex, Mesh, Local, TessellationControl, TessellationEvaluation or
	// Pixel.
	ShaderType stage = ShaderType::Unknown;
	// The program hash the pipeline cache keys the program by.
	uint64_t shader_hash = 0;
	// The code by content (part of the key), and the code itself when it is known (SetCode).
	ShaderCodeId code_id;
	ShaderCodeId back_code_id;
	ShaderCode   code;
	// The second half of a merged program (ShaderParams::back_code), else null.
	ShaderCode back_code;
	uint32_t   user_data_count  = 0;
	uint32_t   push_data_cursor = 0;
	// The input of the stage (vertex for every stage but Pixel), with the runtime fields
	// (addresses, record counts, stage handles) cleared (StaticInput).
	ShaderVertexInputInfo                        vertex;
	ShaderPixelInputInfo                         pixel;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	// Hash of the SPIR-V the recording build emitted; not part of the key.
	uint64_t spirv_hash = 0;

	// Sets the code and the ids that name it.
	void                                       SetCode(ShaderCode front, ShaderCode back);
	[[nodiscard]] uint64_t                     Key() const;
	[[nodiscard]] static ShaderVertexInputInfo StaticInput(const ShaderVertexInputInfo& input);
	[[nodiscard]] static ShaderPixelInputInfo  StaticInput(const ShaderPixelInputInfo& input);
	[[nodiscard]] static bool                  IsGraphicsStage(ShaderType stage);
};

// A graphics pipeline as PipelineCache::FindGraphicsPipeline keys it, with its stages named by
// GraphicsStageRecord::Key.
struct GraphicsPipelineRecord {
	// One vertex-side stage, or three for tessellation (local, control, evaluation).
	std::array<uint64_t, 3> vertex_stages {};
	uint32_t                vertex_stage_count = 0;
	// Zero without a pixel stage.
	uint64_t                 pixel_stage = 0;
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	[[nodiscard]] uint64_t                  Key() const;
	[[nodiscard]] std::span<const uint64_t> VertexStages() const {
		return std::span(vertex_stages).first(vertex_stage_count);
	}
};

// The serialized records, shared by PipelineManifest and PipelineList. Decoding validates every
// field against the values the pipeline cache can produce, so that a damaged or foreign record
// is rejected instead of reaching the recompiler or the driver.
namespace PipelineRecordFormat {
// Changes with the stored fields of any record.
constexpr uint32_t LayoutVersion = 1;
void               PutStage(ManifestFormat::Writer& w, const GraphicsStageRecord& record);
// The record's code stays null; its code ids are read.
bool GetStage(ManifestFormat::Reader& r, GraphicsStageRecord& record);
void PutPipeline(ManifestFormat::Writer& w, const GraphicsPipelineRecord& record);
bool GetPipeline(ManifestFormat::Reader& r, GraphicsPipelineRecord& record);
} // namespace PipelineRecordFormat

// The graphics pipelines a title has needed and their stage programs, persisted next to its
// driver pipeline cache (_PipelineCache/<title>.v5.pipelines), so that the pipelines can be
// built ahead of their first use (--precompile-pipelines, --precompile-only). Append-only like
// ProgramManifest: checksummed entries, guest code once per shader, a torn or corrupt tail or
// another format version rewrites the file with the readable entries. A pipeline is kept only
// when its stages are. Records keep the order they were first added in.
//
// Thread-safe.
class PipelineManifest {
public:
	static constexpr uint32_t FormatVersion = 1;

	// An empty path keeps the manifest in memory only.
	explicit PipelineManifest(std::filesystem::path path = {});
	~PipelineManifest();
	KYTY_CLASS_NO_COPY(PipelineManifest);

	static std::filesystem::path PathFor(const std::filesystem::path& directory,
	                                     const std::string&           title_id);

	struct LoadStats {
		size_t stages        = 0;
		size_t pipelines     = 0;
		size_t code_entries  = 0;
		size_t dropped_bytes = 0;
		bool   rewritten     = false;
	};
	[[nodiscard]] LoadStats Loaded() const { return m_loaded; }

	// Adds a stage (with its code) unless its key is known. Returns whether it was new.
	bool AddStage(const GraphicsStageRecord& record);
	// Adds a pipeline whose stages were added. Returns whether it was new.
	bool AddPipeline(const GraphicsPipelineRecord& record);
	[[nodiscard]] std::shared_ptr<const GraphicsStageRecord> FindStage(uint64_t key) const;
	[[nodiscard]] bool                                       HasPipeline(uint64_t key) const;
	[[nodiscard]] std::vector<std::shared_ptr<const GraphicsStageRecord>>    Stages() const;
	[[nodiscard]] std::vector<std::shared_ptr<const GraphicsPipelineRecord>> Pipelines() const;
	// Code of a stored stage, by content.
	[[nodiscard]] ShaderCode                   FindCode(const ShaderCodeId& id) const;
	[[nodiscard]] const std::filesystem::path& Path() const { return m_path; }

private:
	bool       Parse(std::span<const uint8_t> payload);
	ShaderCode StoreCode(const ShaderCode& code, std::vector<std::vector<uint8_t>>* out);
	bool       Rewrite();

	std::filesystem::path                                                    m_path;
	mutable std::mutex                                                       m_mutex;
	ManifestFormat::EntryAppender                                            m_file;
	LoadStats                                                                m_loaded;
	std::unordered_map<ShaderCodeId, ShaderCode, ShaderCodeIdHash>           m_code;
	std::unordered_map<uint64_t, std::shared_ptr<const GraphicsStageRecord>> m_stages;
	std::vector<std::shared_ptr<const GraphicsStageRecord>>                  m_stage_order;
	std::unordered_set<uint64_t>                                             m_pipeline_keys;
	std::vector<std::shared_ptr<const GraphicsPipelineRecord>>               m_pipelines;
};

// The code of every shader the guest created (sceAgcCreateShader), of every stage, by content
// (_PipelineCache/<title>.v5.created). The shareable pipeline list names shaders by content
// only; their code comes from here, so a list resolves against what this player's game created.
// Append-only, checksummed, code once per content.
//
// Thread-safe.
class ShaderCodeStore {
public:
	static constexpr uint32_t FormatVersion = 1;

	explicit ShaderCodeStore(std::filesystem::path path = {});
	~ShaderCodeStore();
	KYTY_CLASS_NO_COPY(ShaderCodeStore);

	static std::filesystem::path PathFor(const std::filesystem::path& directory,
	                                     const std::string&           title_id);

	// Adds the code of a created shader of the given ShaderBinaryType unless its content is
	// known. Returns whether it was new.
	bool                     Add(uint8_t binary_type, std::span<const uint32_t> code);
	[[nodiscard]] ShaderCode Find(const ShaderCodeId& id) const;
	[[nodiscard]] size_t     Size() const;
	// Entries per ShaderBinaryType (0..15).
	[[nodiscard]] std::array<size_t, 16> CountByType() const;
	[[nodiscard]] size_t                 LoadedDroppedBytes() const { return m_dropped; }

private:
	std::filesystem::path                                          m_path;
	mutable std::mutex                                             m_mutex;
	ManifestFormat::EntryAppender                                  m_file;
	std::unordered_map<ShaderCodeId, ShaderCode, ShaderCodeIdHash> m_code;
	std::array<size_t, 16>                                         m_by_type {};
	size_t                                                         m_dropped = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEMANIFEST_H_
