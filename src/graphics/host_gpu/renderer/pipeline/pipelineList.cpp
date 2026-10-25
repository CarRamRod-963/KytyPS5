#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <algorithm>
#include <array>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics {

namespace {

using ManifestFormat::Reader;
using ManifestFormat::Writer;

constexpr ManifestFormat::Magic ListMagic    = {'K', 'y', 'P', 'l', 'L', 'i', 's', 't'};
constexpr uint8_t               KindHeader   = 1;
constexpr uint8_t               KindCompute  = 2;
constexpr uint8_t               KindStage    = 3;
constexpr uint8_t               KindPipeline = 4;
constexpr uint32_t              MaxTitle     = 64;
// Kind, code id, code-only flag.
constexpr size_t ComputePrefix = 1 + sizeof(uint64_t) + sizeof(uint32_t) + 1;

} // namespace

PipelineListContents PipelineList::FromManifests(const std::string&      title_id,
                                                 const ProgramManifest*  programs,
                                                 const ProgramManifest*  created,
                                                 const PipelineManifest* pipelines) {
	PipelineListContents contents;
	contents.title_id = title_id;
	std::unordered_set<uint64_t> recorded_shaders;
	const auto add_compute = [&](const ComputeProgramRecord& record, bool code_only) {
		PipelineListContents::Compute entry {
		    .code = ShaderCodeId::Of(*record.code), .record = record, .code_only = code_only};
		entry.record.code = nullptr;
		contents.compute.push_back(std::move(entry));
	};
	if (programs != nullptr) {
		for (const auto& record: programs->Records()) {
			add_compute(*record, false);
			recorded_shaders.insert(record->shader_hash);
		}
	}
	// The created shaders the precompile builds code-only: those without a recorded program.
	if (created != nullptr) {
		for (const auto& record: created->Records()) {
			if (!recorded_shaders.contains(record->shader_hash)) {
				add_compute(*record, true);
			}
		}
	}
	if (pipelines != nullptr) {
		std::unordered_set<uint64_t> used;
		for (const auto& pipeline: pipelines->Pipelines()) {
			contents.pipelines.push_back(*pipeline);
			used.insert(pipeline->VertexStages().begin(), pipeline->VertexStages().end());
			if (pipeline->pixel_stage != 0) {
				used.insert(pipeline->pixel_stage);
			}
		}
		for (const auto& stage: pipelines->Stages()) {
			if (used.contains(stage->Key())) {
				auto entry      = *stage;
				entry.code      = nullptr;
				entry.back_code = nullptr;
				contents.stages.push_back(std::move(entry));
			}
		}
	}
	return contents;
}

bool PipelineList::Write(const std::filesystem::path& path, const PipelineListContents& contents) {
	std::vector<std::vector<uint8_t>> payloads;
	{
		std::vector<uint8_t> header;
		Writer               w(header);
		w.Put(KindHeader);
		w.Put(static_cast<uint32_t>(contents.title_id.size()));
		for (const char c: contents.title_id)
			w.Put(c);
		w.Put(ProgramManifest::FormatVersion);
		w.Put(PipelineRecordFormat::LayoutVersion);
		payloads.push_back(std::move(header));
	}
	for (const auto& entry: contents.compute) {
		std::vector<uint8_t> payload;
		Writer               w(payload);
		w.Put(KindCompute);
		w.Put(entry.code.hash);
		w.Put(entry.code.words);
		w.PutBool(entry.code_only);
		// The body carries the code's length only; the code itself is not listed.
		auto record     = entry.record;
		record.code     = std::make_shared<const std::vector<uint32_t>>(entry.code.words);
		const auto body = ProgramManifest::EncodeComputeBody(record);
		payload.insert(payload.end(), body.begin(), body.end());
		payloads.push_back(std::move(payload));
	}
	for (const auto& stage: contents.stages) {
		std::vector<uint8_t> payload;
		Writer               w(payload);
		w.Put(KindStage);
		PipelineRecordFormat::PutStage(w, stage);
		w.Put(stage.spirv_hash);
		payloads.push_back(std::move(payload));
	}
	for (const auto& pipeline: contents.pipelines) {
		std::vector<uint8_t> payload;
		Writer               w(payload);
		w.Put(KindPipeline);
		PipelineRecordFormat::PutPipeline(w, pipeline);
		payloads.push_back(std::move(payload));
	}
	return ManifestFormat::WriteEntryFile(path, ListMagic, FormatVersion, payloads);
}

PipelineListContents PipelineList::Read(const std::filesystem::path& path,
                                        const std::string& title_id, ReadStats* stats) {
	PipelineListContents contents;
	ReadStats            local;
	auto&                s           = stats != nullptr ? *stats : local;
	bool                 have_header = false;
	bool                 usable      = false;
	s                                = {};
	const auto parse                 = [&](std::span<const uint8_t> payload, uint32_t version) {
		if (version != FormatVersion) {
			return false;
		}
		Reader  r(payload);
		uint8_t kind = 0;
		if (!r.Get(kind)) {
			return false;
		}
		if (!have_header) {
			// The first entry names the title and the record layouts.
			uint32_t size = 0;
			if (kind != KindHeader || !r.Get(size) || size > MaxTitle) {
				return false;
			}
			std::string title(size, '?');
			for (auto& c: title) {
				if (!r.Get(c)) return false;
			}
			uint32_t compute_layout = 0;
			uint32_t record_layout  = 0;
			if (!r.Get(compute_layout) || !r.Get(record_layout) || !r.AtEnd()) {
				return false;
			}
			have_header       = true;
			contents.title_id = title;
			s.title_matches   = title == title_id;
			usable = s.title_matches && compute_layout == ProgramManifest::FormatVersion &&
			         record_layout == PipelineRecordFormat::LayoutVersion;
			return true;
		}
		if (!usable) {
			s.rejected++;
			return true;
		}
		switch (kind) {
			case KindCompute: {
				PipelineListContents::Compute entry;
				uint32_t                      words = 0;
				if (!r.Get(entry.code.hash) || !r.Get(entry.code.words) || !entry.code ||
				    !r.GetBool(entry.code_only) ||
				    !ProgramManifest::DecodeComputeBody(payload.subspan(ComputePrefix),
				                                        entry.record, words) ||
				    words != entry.code.words) {
					return false;
				}
				contents.compute.push_back(std::move(entry));
				return true;
			}
			case KindStage: {
				GraphicsStageRecord stage;
				if (!PipelineRecordFormat::GetStage(r, stage) || !r.Get(stage.spirv_hash) ||
				    !r.AtEnd()) {
					return false;
				}
				contents.stages.push_back(std::move(stage));
				return true;
			}
			case KindPipeline: {
				GraphicsPipelineRecord record;
				if (!PipelineRecordFormat::GetPipeline(r, record) || !r.AtEnd()) {
					return false;
				}
				contents.pipelines.push_back(record);
				return true;
			}
			default: return false;
		}
	};
	const auto read = ManifestFormat::ReadEntryFile(path, ListMagic, parse);
	s.valid         = read.valid && read.version == FormatVersion && have_header;
	s.dropped_bytes = read.valid ? read.size - read.intact : read.size;
	if (!usable) {
		contents.compute.clear();
		contents.stages.clear();
		contents.pipelines.clear();
	}
	return contents;
}

PipelineListResolver::PipelineListResolver(PipelineListContents contents) {
	m_compute.reserve(contents.compute.size());
	const auto code_only = static_cast<size_t>(
	    std::ranges::count_if(contents.compute, [](const auto& entry) { return entry.code_only; }));
	const std::array<size_t, 2> kind_counts {contents.compute.size() - code_only, code_only};
	std::array<size_t, 2>       kind_ranks {};
	for (auto& entry: contents.compute) {
		const auto kind     = entry.code_only ? 1u : 0u;
		const auto position = FirstSeenPosition(kind_ranks[kind]++, kind_counts[kind]);
		m_waiting[entry.code].compute.push_back(m_compute.size());
		m_compute.push_back({.entry = std::move(entry), .position = position});
	}
	std::unordered_map<uint64_t, size_t> stage_index;
	m_stages.reserve(contents.stages.size());
	for (auto& record: contents.stages) {
		const auto key = record.Key();
		if (!stage_index.try_emplace(key, m_stages.size()).second) {
			continue;
		}
		StageState state {.record = std::move(record)};
		state.missing    = 1u + (state.record.back_code_id ? 1u : 0u);
		const auto index = m_stages.size();
		m_waiting[state.record.code_id].stages.push_back(index);
		if (state.record.back_code_id) {
			m_waiting[state.record.back_code_id].stages.push_back(index);
		}
		m_stages.push_back(std::move(state));
	}
	m_pipelines.reserve(contents.pipelines.size());
	for (auto& record: contents.pipelines) {
		PipelineState state;
		bool          complete = true;
		const auto    add      = [&](uint64_t key) {
			const auto found = stage_index.find(key);
			if (found == stage_index.end()) {
				complete = false;
				return;
			}
			state.stages.push_back(found->second);
		};
		for (const auto key: record.VertexStages())
			add(key);
		if (record.pixel_stage != 0) {
			add(record.pixel_stage);
		}
		if (!complete) {
			m_broken++;
			continue;
		}
		const auto index = m_pipelines.size();
		for (const auto stage: state.stages) {
			m_stages[stage].pipelines.push_back(index);
		}
		state.missing = static_cast<uint32_t>(state.stages.size());
		state.record  = std::make_shared<const GraphicsPipelineRecord>(record);
		m_pipelines.push_back(std::move(state));
	}
	for (size_t i = 0; i < m_pipelines.size(); i++) {
		m_pipelines[i].position = FirstSeenPosition(i, m_pipelines.size());
	}
}

void PipelineListResolver::Supply(const ShaderCodeId& id, const ShaderCode& code, Resolved& out) {
	const auto waiting = m_waiting.find(id);
	if (waiting == m_waiting.end() || code == nullptr || ShaderCodeId::Of(*code) != id) {
		return;
	}
	const auto waiters = std::move(waiting->second);
	m_waiting.erase(waiting);
	for (const auto index: waiters.compute) {
		auto& state = m_compute[index];
		if (state.done) continue;
		state.done   = true;
		auto record  = std::make_shared<ComputeProgramRecord>(state.entry.record);
		record->code = code;
		out.compute.push_back(std::move(record));
		out.compute_code_only.push_back(state.entry.code_only);
		out.compute_position.push_back(state.position);
		m_compute_resolved++;
	}
	for (const auto index: waiters.stages) {
		auto& stage = m_stages[index];
		if (stage.record.code_id == id) stage.front = code;
		if (stage.record.back_code_id == id) stage.back = code;
		if (stage.missing == 0 || --stage.missing != 0) {
			continue;
		}
		auto done       = std::make_shared<GraphicsStageRecord>(stage.record);
		done->code      = stage.front;
		done->back_code = stage.back;
		stage.done      = std::move(done);
		for (const auto pipeline_index: stage.pipelines) {
			auto& pipeline = m_pipelines[pipeline_index];
			if (pipeline.done || --pipeline.missing != 0) {
				continue;
			}
			pipeline.done = true;
			Resolved::Pipeline resolved {.record = pipeline.record, .position = pipeline.position};
			const auto         vertex_count = pipeline.record->vertex_stage_count;
			for (uint32_t i = 0; i < pipeline.stages.size(); i++) {
				const auto& stage_record = m_stages[pipeline.stages[i]].done;
				if (i < vertex_count) {
					resolved.vertex_stages.push_back(stage_record);
				} else {
					resolved.pixel_stage = stage_record;
				}
			}
			out.pipelines.push_back(std::move(resolved));
			m_pipelines_resolved++;
		}
	}
}

PipelineListResolver::Resolved PipelineListResolver::Resolve(const Lookup& lookup) {
	std::lock_guard           lock(m_mutex);
	Resolved                  out;
	std::vector<ShaderCodeId> ids;
	ids.reserve(m_waiting.size());
	for (const auto& [id, waiters]: m_waiting) {
		(void)waiters;
		ids.push_back(id);
	}
	for (const auto& id: ids) {
		if (auto code = lookup(id); code != nullptr) {
			Supply(id, code, out);
		}
	}
	return out;
}

PipelineListResolver::Resolved PipelineListResolver::OnCode(std::span<const uint32_t> code) {
	const auto      id = ShaderCodeId::Of(code);
	std::lock_guard lock(m_mutex);
	Resolved        out;
	if (m_waiting.contains(id)) {
		Supply(id, std::make_shared<const std::vector<uint32_t>>(code.begin(), code.end()), out);
	}
	return out;
}

PipelineListResolver::Counts PipelineListResolver::GetCounts() const {
	std::lock_guard lock(m_mutex);
	return {.compute            = m_compute.size(),
	        .compute_resolved   = m_compute_resolved,
	        .pipelines          = m_pipelines.size(),
	        .pipelines_resolved = m_pipelines_resolved,
	        .pipelines_broken   = m_broken,
	        .shaders_pending    = m_waiting.size()};
}

bool PipelineListResolver::Pending() const {
	std::lock_guard lock(m_mutex);
	return !m_waiting.empty();
}

std::unordered_set<uint64_t> PipelineListResolver::PendingShaderHashes() const {
	std::lock_guard              lock(m_mutex);
	std::unordered_set<uint64_t> hashes;
	for (const auto& state: m_compute) {
		if (!state.done) hashes.insert(state.entry.record.shader_hash);
	}
	for (const auto& stage: m_stages) {
		if (stage.missing != 0) hashes.insert(stage.record.shader_hash);
	}
	return hashes;
}

} // namespace Libs::Graphics
