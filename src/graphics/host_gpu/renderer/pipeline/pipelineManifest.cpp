#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"

#include <algorithm>
#include <bit>
#include <system_error>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

using ManifestFormat::Reader;
using ManifestFormat::Writer;

// Serialized field by field; these trip when a stored struct changes so that the record layout
// is revisited (a new field must be stored and PipelineRecordFormat::LayoutVersion bumped).
static_assert(sizeof(ShaderVertexInputInfo) == 2112);
static_assert(sizeof(ShaderPixelInputInfo) == 232);
static_assert(sizeof(PipelineStaticParameters) == 114);
static_assert(sizeof(PipelineRenderingState) == 44);
static_assert(sizeof(PipelineVertexInputState) == 516);

constexpr ManifestFormat::Magic ManifestMagic   = {'K', 'y', 'G', 'f', 'x', 'P', 'l', 'n'};
constexpr ManifestFormat::Magic CodeStoreMagic  = {'K', 'y', 'S', 'h', 'C', 'o', 'd', 'e'};
constexpr uint8_t               KindCode        = 1;
constexpr uint8_t               KindStage       = 2;
constexpr uint8_t               KindPipeline    = 3;
constexpr uint8_t               KindCreatedCode = 4;
constexpr uint32_t              MaxUserData     = 64;
constexpr uint32_t              MaxCodeWords    = 16u << 20u;
constexpr int                   ResMax          = ShaderVertexInputInfo::RES_MAX;

bool IsVertexSideStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex:
		case ShaderType::Mesh:
		case ShaderType::Local:
		case ShaderType::TessellationControl:
		case ShaderType::TessellationEvaluation: return true;
		default: return false;
	}
}

bool ValidWaveSize(uint32_t wave_size) {
	return wave_size == 32u || wave_size == 64u;
}

// Core formats, and the extension formats (1000xxxyyy).
bool ValidFormat(vk::Format format) {
	const auto raw = static_cast<uint32_t>(format);
	return raw < 1000u || (raw >= 1000000000u && raw < 1001000000u);
}

void PutFloat(Writer& w, float value) {
	w.Put(std::bit_cast<uint32_t>(value));
}

bool GetFloat(Reader& r, float& value) {
	uint32_t raw = 0;
	if (!r.Get(raw)) return false;
	value = std::bit_cast<float>(raw);
	return true;
}

bool GetInt(Reader& r, int& value, int min, int max) {
	int32_t raw = 0;
	if (!r.Get(raw) || raw < min || raw > max) return false;
	value = raw;
	return true;
}

// Layout 1 stored a native indirect draw's vertex stage as one word: bit 31 set for such a draw,
// bits 0-7 and 8-15 the vertex and instance offset slots plus one (zero for none).
constexpr uint32_t IndirectDrawEnabled = 1u << 31u;

uint32_t EncodeIndirectDrawWord(const ShaderVertexInputInfo& info) {
	if (!info.indirect_draw) return 0;
	const auto field = [](uint8_t slot) {
		return slot == ShaderIndirectOffsets::None ? 0u : uint32_t {slot} + 1u;
	};
	return IndirectDrawEnabled | field(info.indirect_offsets.vertex) |
	       (field(info.indirect_offsets.instance) << 8u);
}

bool DecodeIndirectDrawWord(uint32_t word, ShaderVertexInputInfo& info) {
	if ((word & ~(IndirectDrawEnabled | 0xffffu)) != 0u ||
	    ((word & IndirectDrawEnabled) == 0u && word != 0u)) {
		return false;
	}
	const auto slot = [](uint32_t field) {
		return field == 0u ? ShaderIndirectOffsets::None : static_cast<uint8_t>(field - 1u);
	};
	info.indirect_draw    = word != 0u;
	info.indirect_offsets = {slot(word & 0xffu), slot((word >> 8u) & 0xffu)};
	return true;
}

void PutVertexInput(Writer& w, const ShaderVertexInputInfo& info, uint32_t layout) {
	w.PutEnum(info.logical_stage);
	w.Put(static_cast<int32_t>(info.resources_num));
	w.Put(static_cast<int32_t>(info.fetch_attrib_reg));
	w.Put(static_cast<int32_t>(info.fetch_buffer_reg));
	w.Put(static_cast<int32_t>(info.buffers_num));
	for (int i = 0; i < info.resources_num; i++) {
		for (const auto field: info.resources[i].fields)
			w.Put(field);
		const auto& dst = info.resources_dst[i];
		w.Put(static_cast<int32_t>(dst.register_start));
		w.Put(static_cast<int32_t>(dst.registers_num));
		w.Put(static_cast<int32_t>(dst.attr_id));
		w.Put(dst.fetch_index);
		w.Put(static_cast<int32_t>(dst.buffer_index));
	}
	for (int i = 0; i < info.buffers_num; i++) {
		w.Put(info.buffers[i].stride);
		w.Put(info.buffers[i].fetch_index);
	}
	w.Put(info.wave_size);
	w.Put(info.scratch_size_dwords);
	w.Put(info.lds_size_dwords);
	w.Put(info.pa_cl_vs_out_cntl);
	w.Put(info.user_clip_planes);
	w.PutBool(info.clip_space.enabled);
	for (const auto value: info.clip_space.scale)
		PutFloat(w, value);
	for (const auto value: info.clip_space.offset)
		PutFloat(w, value);
	for (const auto value: info.clip_space.half_extent)
		PutFloat(w, value);
	const auto& mesh = info.mesh;
	for (const auto value: mesh.threads_num)
		w.Put(value);
	w.Put(mesh.lds_size_dwords);
	w.Put(mesh.scratch_size_dwords);
	w.Put(mesh.host_subgroup_size);
	w.Put(mesh.wave_size);
	w.Put(mesh.input_primitive);
	w.Put(mesh.primitives_per_group);
	w.Put(mesh.vertices_per_group);
	w.Put(mesh.max_vertices);
	w.Put(mesh.max_primitives);
	w.Put(mesh.provoking_vertex);
	w.PutBool(mesh.fast_launch);
	const auto& tess = info.tess;
	for (const auto value:
	     {tess.input_control_points, tess.output_control_points, tess.ls_stride, tess.hs_stride,
	      tess.patch_base, tess.patch_size, tess.domain, tess.partitioning, tess.output_topology})
		w.Put(value);
	if (layout == 1) {
		// fetch_external, removed upstream; the indirect draw as one word.
		w.PutBool(false);
		w.PutBool(info.has_dynamic_vertex_fetch);
		w.Put(EncodeIndirectDrawWord(info));
		return;
	}
	w.PutBool(info.has_dynamic_vertex_fetch);
	w.PutBool(info.indirect_draw);
	w.Put(info.indirect_offsets.vertex);
	w.Put(info.indirect_offsets.instance);
}

bool GetVertexInput(Reader& r, ShaderVertexInputInfo& info, uint32_t layout) {
	info = {};
	if (!r.GetEnum(info.logical_stage, static_cast<uint32_t>(ShaderType::TessellationEvaluation)) ||
	    !IsVertexSideStage(info.logical_stage) || !GetInt(r, info.resources_num, 0, ResMax) ||
	    !GetInt(r, info.fetch_attrib_reg, 0, 255) || !GetInt(r, info.fetch_buffer_reg, 0, 255) ||
	    !GetInt(r, info.buffers_num, 0, ResMax)) {
		return false;
	}
	for (int i = 0; i < info.resources_num; i++) {
		for (auto& field: info.resources[i].fields) {
			if (!r.Get(field)) return false;
		}
		auto& dst = info.resources_dst[i];
		if (!GetInt(r, dst.register_start, 0, 255) || !GetInt(r, dst.registers_num, 0, 4) ||
		    !GetInt(r, dst.attr_id, -1, 255) || !r.Get(dst.fetch_index) ||
		    !GetInt(r, dst.buffer_index, 0, ResMax - 1)) {
			return false;
		}
	}
	for (int i = 0; i < info.buffers_num; i++) {
		if (!r.Get(info.buffers[i].stride) || !r.Get(info.buffers[i].fetch_index)) return false;
	}
	if (!r.Get(info.wave_size) || !ValidWaveSize(info.wave_size) ||
	    !r.Get(info.scratch_size_dwords) || !r.Get(info.lds_size_dwords) ||
	    !r.Get(info.pa_cl_vs_out_cntl) || !r.Get(info.user_clip_planes) ||
	    !r.GetBool(info.clip_space.enabled)) {
		return false;
	}
	for (auto* values:
	     {info.clip_space.scale, info.clip_space.offset, info.clip_space.half_extent}) {
		if (!GetFloat(r, values[0]) || !GetFloat(r, values[1])) return false;
	}
	auto& mesh = info.mesh;
	for (auto& value: mesh.threads_num) {
		if (!r.Get(value)) return false;
	}
	if (!r.Get(mesh.lds_size_dwords) || !r.Get(mesh.scratch_size_dwords) ||
	    !r.Get(mesh.host_subgroup_size) || !r.Get(mesh.wave_size) || !r.Get(mesh.input_primitive) ||
	    !r.Get(mesh.primitives_per_group) || !r.Get(mesh.vertices_per_group) ||
	    !r.Get(mesh.max_vertices) || !r.Get(mesh.max_primitives) || !r.Get(mesh.provoking_vertex) ||
	    !r.GetBool(mesh.fast_launch)) {
		return false;
	}
	if (info.logical_stage == ShaderType::Mesh && !ValidWaveSize(mesh.wave_size)) {
		return false;
	}
	auto& tess = info.tess;
	for (auto* value: {&tess.input_control_points, &tess.output_control_points, &tess.ls_stride,
	                   &tess.hs_stride, &tess.patch_base, &tess.patch_size, &tess.domain,
	                   &tess.partitioning, &tess.output_topology}) {
		if (!r.Get(*value)) return false;
	}
	if (tess.input_control_points > 32u) {
		return false;
	}
	if (layout == 1) {
		bool     fetch_external = false;
		uint32_t indirect_word  = 0;
		return r.GetBool(fetch_external) && r.GetBool(info.has_dynamic_vertex_fetch) &&
		       r.Get(indirect_word) && DecodeIndirectDrawWord(indirect_word, info);
	}
	return r.GetBool(info.has_dynamic_vertex_fetch) && r.GetBool(info.indirect_draw) &&
	       r.Get(info.indirect_offsets.vertex) && r.Get(info.indirect_offsets.instance);
}

void PutPixelInput(Writer& w, const ShaderPixelInputInfo& info, uint32_t layout) {
	w.Put(info.input_num);
	for (uint32_t i = 0; i < info.input_num; i++)
		w.Put(info.interpolator_settings[i]);
	w.Put(info.wave_size);
	w.Put(info.ps_system_input_base);
	w.Put(info.custom_interpolation_mask);
	w.Put(info.ps_perspective_center_vgpr);
	w.Put(info.ps_perspective_sample_vgpr);
	w.Put(info.ps_perspective_centroid_vgpr);
	for (const auto mode: info.target_output_mode)
		w.Put(mode);
	w.Put(info.target_shader_mask);
	for (const auto mapping: info.target_export_mapping)
		w.Put(mapping.packed);
	w.Put(info.target_packed_unorm_mask);
	w.Put(info.target_round_mode);
	if (layout >= 2) {
		w.Put(info.target_uint_mask);
		w.Put(info.target_sint_mask);
	}
	w.Put(info.scratch_size_dwords);
	w.Put(info.lds_size_dwords);
	for (const bool value:
	     {info.ps_pos_x, info.ps_pos_y, info.ps_pos_z, info.ps_pos_w, info.ps_front_face,
	      info.ps_front_face_all_bits, info.ps_ancillary, info.ps_no_perspective})
		w.PutBool(value);
	w.PutEnum(info.parameter_mode);
	for (const bool value:
	     {info.ps_pixel_kill_enable, info.ps_depth_export_enable, info.ps_sample_mask_export_enable,
	      info.ps_sample_shading, info.dual_source_blending, info.object_primitive_id})
		w.PutBool(value);
	w.PutEnum(info.alpha_blend_source);
	w.PutBool(info.ps_early_z);
	w.PutBool(info.ps_execute_on_noop);
}

bool GetPixelInput(Reader& r, ShaderPixelInputInfo& info, uint32_t layout) {
	info = {};
	if (!r.Get(info.input_num) || info.input_num > std::size(info.interpolator_settings)) {
		return false;
	}
	for (uint32_t i = 0; i < info.input_num; i++) {
		if (!r.Get(info.interpolator_settings[i])) return false;
	}
	if (!r.Get(info.wave_size) || !ValidWaveSize(info.wave_size) ||
	    !r.Get(info.ps_system_input_base) || !r.Get(info.custom_interpolation_mask) ||
	    !r.Get(info.ps_perspective_center_vgpr) || !r.Get(info.ps_perspective_sample_vgpr) ||
	    !r.Get(info.ps_perspective_centroid_vgpr)) {
		return false;
	}
	for (auto& mode: info.target_output_mode) {
		if (!r.Get(mode) || mode > 15u) return false;
	}
	if (!r.Get(info.target_shader_mask)) return false;
	for (auto& mapping: info.target_export_mapping) {
		if (!r.Get(mapping.packed)) return false;
	}
	if (!r.Get(info.target_packed_unorm_mask) || !r.Get(info.target_round_mode) ||
	    (layout >= 2 && (!r.Get(info.target_uint_mask) || !r.Get(info.target_sint_mask))) ||
	    !r.Get(info.scratch_size_dwords) || !r.Get(info.lds_size_dwords)) {
		return false;
	}
	for (auto* value:
	     {&info.ps_pos_x, &info.ps_pos_y, &info.ps_pos_z, &info.ps_pos_w, &info.ps_front_face,
	      &info.ps_front_face_all_bits, &info.ps_ancillary, &info.ps_no_perspective}) {
		if (!r.GetBool(*value)) return false;
	}
	if (!r.GetEnum(info.parameter_mode,
	               static_cast<uint32_t>(ShaderPixelParameterMode::Rectangle))) {
		return false;
	}
	for (auto* value: {&info.ps_pixel_kill_enable, &info.ps_depth_export_enable,
	                   &info.ps_sample_mask_export_enable, &info.ps_sample_shading,
	                   &info.dual_source_blending, &info.object_primitive_id}) {
		if (!r.GetBool(*value)) return false;
	}
	return r.GetEnum(info.alpha_blend_source,
	                 static_cast<uint32_t>(ShaderAlphaBlendSource::SourceAlphaZero)) &&
	       r.GetBool(info.ps_early_z) && r.GetBool(info.ps_execute_on_noop);
}

void PutCodeId(Writer& w, const ShaderCodeId& id) {
	w.Put(id.hash);
	w.Put(id.words);
}

bool GetCodeId(Reader& r, ShaderCodeId& id) {
	return r.Get(id.hash) && r.Get(id.words) && id.words <= MaxCodeWords;
}

std::vector<uint8_t> EncodeStageBody(const GraphicsStageRecord& record) {
	std::vector<uint8_t> out;
	out.reserve(512);
	Writer w(out);
	w.Put(KindStage);
	PipelineRecordFormat::PutStage(w, record);
	return out;
}

std::vector<uint8_t> EncodePipelineBody(const GraphicsPipelineRecord& record) {
	std::vector<uint8_t> out;
	out.reserve(768);
	Writer w(out);
	w.Put(KindPipeline);
	PipelineRecordFormat::PutPipeline(w, record);
	return out;
}

std::vector<uint8_t> EncodeCode(uint8_t kind, uint8_t binary_type, std::span<const uint32_t> code) {
	std::vector<uint8_t> out;
	out.reserve(24 + code.size_bytes());
	Writer w(out);
	w.Put(kind);
	if (kind == KindCreatedCode) {
		w.Put(binary_type);
	}
	PutCodeId(w, ShaderCodeId::Of(code));
	w.PutWords(code);
	return out;
}

// Reads a code entry's id and words; the id must match the words.
bool GetCode(Reader& r, ShaderCodeId& id, std::vector<uint32_t>& code) {
	return GetCodeId(r, id) && id.words != 0 && r.GetWords(code, id.words) && r.AtEnd() &&
	       ShaderCodeId::Of(code) == id;
}

} // namespace

ShaderCodeId ShaderCodeId::Of(std::span<const uint32_t> code) {
	return {.hash  = XXH3_64bits(code.data(), code.size_bytes()),
	        .words = static_cast<uint32_t>(code.size())};
}

void GraphicsStageRecord::SetCode(ShaderCode front, ShaderCode back) {
	code_id      = front != nullptr ? ShaderCodeId::Of(*front) : ShaderCodeId {};
	back_code_id = back != nullptr ? ShaderCodeId::Of(*back) : ShaderCodeId {};
	code         = std::move(front);
	back_code    = std::move(back);
}

bool GraphicsStageRecord::IsGraphicsStage(ShaderType stage) {
	return stage == ShaderType::Pixel || IsVertexSideStage(stage);
}

ShaderVertexInputInfo GraphicsStageRecord::StaticInput(const ShaderVertexInputInfo& input) {
	auto out  = input;
	out.stage = {};
	for (auto& resource: out.resources) {
		// The base address and record count are runtime values; the pipeline's attribute offsets
		// are in GraphicsPipelineRecord::vertex_input.
		resource.fields[0] = 0;
		resource.fields[1] &= 0xffff0000u;
		resource.fields[2] = 0;
	}
	for (auto& buffer: out.buffers) {
		buffer.addr        = 0;
		buffer.num_records = 0;
	}
	// Entries past the counts are not stored.
	for (int i = std::max(input.resources_num, 0); i < ShaderVertexInputInfo::RES_MAX; i++) {
		out.resources[i]     = {};
		out.resources_dst[i] = {};
	}
	for (int i = std::max(input.buffers_num, 0); i < ShaderVertexInputInfo::RES_MAX; i++) {
		out.buffers[i] = {};
	}
	return out;
}

ShaderPixelInputInfo GraphicsStageRecord::StaticInput(const ShaderPixelInputInfo& input) {
	auto out  = input;
	out.stage = {};
	for (uint32_t i = std::min<uint32_t>(input.input_num, 32u); i < 32u; i++) {
		out.interpolator_settings[i] = 0;
	}
	return out;
}

uint64_t GraphicsStageRecord::Key() const {
	const auto body = EncodeStageBody(*this);
	return XXH3_64bits(body.data(), body.size());
}

uint64_t GraphicsPipelineRecord::Key() const {
	const auto body = EncodePipelineBody(*this);
	return XXH3_64bits(body.data(), body.size());
}

void PipelineRecordFormat::PutStage(Writer& w, const GraphicsStageRecord& record, uint32_t layout) {
	w.PutEnum(record.stage);
	w.Put(record.shader_hash);
	PutCodeId(w, record.code_id);
	PutCodeId(w, record.back_code_id);
	w.Put(record.user_data_count);
	w.Put(record.push_data_cursor);
	if (record.stage == ShaderType::Pixel) {
		PutPixelInput(w, record.pixel, layout);
	} else {
		PutVertexInput(w, record.vertex, layout);
	}
	ManifestFormat::PutSpecialization(w, record.specialization);
}

bool PipelineRecordFormat::GetStage(Reader& r, GraphicsStageRecord& record, uint32_t layout) {
	if (!r.GetEnum(record.stage, static_cast<uint32_t>(ShaderType::TessellationEvaluation)) ||
	    !GraphicsStageRecord::IsGraphicsStage(record.stage) || !r.Get(record.shader_hash) ||
	    !GetCodeId(r, record.code_id) || !record.code_id || !GetCodeId(r, record.back_code_id) ||
	    !r.Get(record.user_data_count) || record.user_data_count > MaxUserData ||
	    !r.Get(record.push_data_cursor)) {
		return false;
	}
	if (record.stage == ShaderType::Pixel) {
		if (!GetPixelInput(r, record.pixel, layout)) return false;
	} else if (!GetVertexInput(r, record.vertex, layout) ||
	           record.vertex.logical_stage != record.stage) {
		return false;
	}
	return ManifestFormat::GetSpecialization(r, record.specialization);
}

void PipelineRecordFormat::PutPipeline(Writer& w, const GraphicsPipelineRecord& record,
                                       uint32_t layout) {
	w.Put(record.vertex_stage_count);
	for (const auto key: record.VertexStages())
		w.Put(key);
	w.Put(record.pixel_stage);
	const auto& rendering = record.rendering;
	w.Put(rendering.color_count);
	for (const auto format: rendering.color_formats)
		w.PutEnum(format);
	w.PutEnum(rendering.depth_format);
	w.PutEnum(rendering.stencil_format);
	const auto& input = record.vertex_input;
	w.Put(input.binding_count);
	w.Put(input.attribute_count);
	for (uint32_t i = 0; i < input.binding_count; i++) {
		w.Put(input.bindings[i].stride);
		w.PutBool(input.bindings[i].instance);
	}
	for (uint32_t i = 0; i < input.attribute_count; i++) {
		w.Put(input.attributes[i].offset);
		w.Put(input.attributes[i].binding);
	}
	const auto& p = record.static_params;
	for (const bool value: {p.negative_one_to_one, p.depth_clip_enable, p.primitive_restart_enable,
	                        p.sample_shading_enable, p.alpha_to_coverage_enable})
		w.PutBool(value);
	if (layout == 1) {
		// Cull mode and front face, dynamic state since upstream's 1ec69f8e7.
		for (uint32_t i = 0; i < 3; i++)
			w.PutBool(false);
	}
	w.PutBool(p.provoking_vtx_last);
	w.PutEnum(p.topology);
	w.Put(p.samples);
	w.PutEnum(p.polygon_mode);
	for (uint32_t i = 0; i < RENDER_COLOR_ATTACHMENTS_MAX; i++) {
		w.Put(p.color_mask[i]);
		w.Put(p.color_srcblend[i]);
		w.Put(p.color_comb_fcn[i]);
		w.Put(p.color_destblend[i]);
		w.Put(p.alpha_srcblend[i]);
		w.Put(p.alpha_comb_fcn[i]);
		w.Put(p.alpha_destblend[i]);
		w.PutBool(p.separate_alpha_blend[i]);
		w.PutBool(p.blend_enable[i]);
	}
}

bool PipelineRecordFormat::GetPipeline(Reader& r, GraphicsPipelineRecord& record, uint32_t layout) {
	record = {};
	if (!r.Get(record.vertex_stage_count) ||
	    (record.vertex_stage_count != 1u && record.vertex_stage_count != 3u)) {
		return false;
	}
	for (uint32_t i = 0; i < record.vertex_stage_count; i++) {
		if (!r.Get(record.vertex_stages[i]) || record.vertex_stages[i] == 0) return false;
	}
	auto& rendering = record.rendering;
	if (!r.Get(record.pixel_stage) || !r.Get(rendering.color_count) ||
	    rendering.color_count > RENDER_COLOR_ATTACHMENTS_MAX) {
		return false;
	}
	for (auto& format: rendering.color_formats) {
		if (!r.GetEnum(format) || !ValidFormat(format)) return false;
	}
	if (!r.GetEnum(rendering.depth_format) || !ValidFormat(rendering.depth_format) ||
	    !r.GetEnum(rendering.stencil_format) || !ValidFormat(rendering.stencil_format)) {
		return false;
	}
	auto& input = record.vertex_input;
	if (!r.Get(input.binding_count) || input.binding_count > ResMax ||
	    !r.Get(input.attribute_count) || input.attribute_count > ResMax) {
		return false;
	}
	for (uint32_t i = 0; i < input.binding_count; i++) {
		if (!r.Get(input.bindings[i].stride) || !r.GetBool(input.bindings[i].instance))
			return false;
	}
	for (uint32_t i = 0; i < input.attribute_count; i++) {
		if (!r.Get(input.attributes[i].offset) || !r.Get(input.attributes[i].binding) ||
		    input.attributes[i].binding >= input.binding_count) {
			return false;
		}
	}
	auto& p = record.static_params;
	for (auto* value: {&p.negative_one_to_one, &p.depth_clip_enable, &p.primitive_restart_enable,
	                   &p.sample_shading_enable, &p.alpha_to_coverage_enable}) {
		bool raw = false;
		if (!r.GetBool(raw)) return false;
		*value = raw;
	}
	if (layout == 1) {
		for (uint32_t i = 0; i < 3; i++) {
			bool cull = false;
			if (!r.GetBool(cull)) return false;
		}
	}
	bool provoking_vtx_last = false;
	if (!r.GetBool(provoking_vtx_last)) return false;
	p.provoking_vtx_last = provoking_vtx_last;
	vk::PrimitiveTopology topology {};
	vk::PolygonMode       polygon_mode {};
	uint32_t              samples = 0;
	if (!r.GetEnum(topology, static_cast<uint32_t>(vk::PrimitiveTopology::ePatchList)) ||
	    !r.Get(samples) || samples == 0 || samples > 16u || !std::has_single_bit(samples) ||
	    !r.GetEnum(polygon_mode, static_cast<uint32_t>(vk::PolygonMode::ePoint))) {
		return false;
	}
	p.topology     = topology;
	p.samples      = samples;
	p.polygon_mode = polygon_mode;
	for (uint32_t i = 0; i < RENDER_COLOR_ATTACHMENTS_MAX; i++) {
		uint32_t color_mask   = 0;
		bool     separate     = false;
		bool     blend_enable = false;
		if (!r.Get(color_mask) || color_mask > 0xfu || !r.Get(p.color_srcblend[i]) ||
		    !r.Get(p.color_comb_fcn[i]) || !r.Get(p.color_destblend[i]) ||
		    !r.Get(p.alpha_srcblend[i]) || !r.Get(p.alpha_comb_fcn[i]) ||
		    !r.Get(p.alpha_destblend[i]) || !r.GetBool(separate) || !r.GetBool(blend_enable)) {
			return false;
		}
		p.color_mask[i]           = color_mask;
		p.separate_alpha_blend[i] = separate;
		p.blend_enable[i]         = blend_enable;
	}
	return true;
}

std::filesystem::path PipelineManifest::PathFor(const std::filesystem::path& directory,
                                                const std::string&           title_id) {
	return directory / (title_id + ".v6.pipelines");
}

std::filesystem::path PipelineManifest::PreviousPathFor(const std::filesystem::path& directory,
                                                        const std::string&           title_id) {
	return directory / (title_id + ".v5.pipelines");
}

PipelineManifest::PipelineManifest(std::filesystem::path path, std::filesystem::path previous)
    : m_path(std::move(path)) {
	if (m_path.empty()) {
		return;
	}
	const auto read = ManifestFormat::ReadEntryFile(
	    m_path, ManifestMagic, [this](std::span<const uint8_t> payload, uint32_t version) {
		    return version == FormatVersion && Parse(payload, PipelineRecordFormat::LayoutVersion);
	    });
	std::error_code error;
	if (read.size == 0 && !previous.empty() && std::filesystem::exists(previous, error)) {
		// The previous file (record layout 1) is read once and left as it is for older builds.
		(void)ManifestFormat::ReadEntryFile(
		    previous, ManifestMagic, [this](std::span<const uint8_t> payload, uint32_t version) {
			    return version == FormatVersion && Parse(payload, 1);
		    });
		m_legacy_stage_keys.clear();
		m_loaded.imported = !m_stage_order.empty();
	}
	m_loaded.stages       = m_stage_order.size();
	m_loaded.pipelines    = m_pipelines.size();
	m_loaded.code_entries = m_code.size();
	if (!read.Clean() || read.version != FormatVersion || m_loaded.imported) {
		m_loaded.dropped_bytes = read.valid ? read.size - read.intact : read.size;
		m_loaded.rewritten     = read.size != 0;
		// A file that cannot be rewritten is not appended to; the manifest then lives in memory.
		if (!Rewrite()) {
			return;
		}
	}
	m_file.Open(m_path);
}

PipelineManifest::~PipelineManifest() = default;

bool PipelineManifest::Parse(std::span<const uint8_t> payload, uint32_t layout) {
	Reader  r(payload);
	uint8_t kind = 0;
	if (!r.Get(kind)) {
		return false;
	}
	if (kind == KindCode) {
		ShaderCodeId id;
		auto         code = std::make_shared<std::vector<uint32_t>>();
		if (!GetCode(r, id, *code)) {
			return false;
		}
		m_code.try_emplace(id, std::move(code));
		return true;
	}
	if (kind == KindStage) {
		auto record = std::make_shared<GraphicsStageRecord>();
		if (!PipelineRecordFormat::GetStage(r, *record, layout) || !r.Get(record->spirv_hash) ||
		    !r.AtEnd()) {
			return false;
		}
		if (layout != PipelineRecordFormat::LayoutVersion) {
			// Pipelines of another layout name their stages by the key of that layout's body:
			// the payload without its SPIR-V hash.
			const auto body = payload.first(payload.size() - sizeof(record->spirv_hash));
			m_legacy_stage_keys.emplace(XXH3_64bits(body.data(), body.size()), record->Key());
		}
		const auto front = m_code.find(record->code_id);
		const auto back  = record->back_code_id ? m_code.find(record->back_code_id) : m_code.end();
		// A stage whose code entry was lost cannot be translated; the entry itself is intact.
		if (front == m_code.end() || (record->back_code_id && back == m_code.end())) {
			return true;
		}
		record->code      = front->second;
		record->back_code = record->back_code_id ? back->second : nullptr;
		const auto key    = record->Key();
		if (m_stages.try_emplace(key, record).second) {
			m_stage_order.push_back(std::move(record));
		}
		return true;
	}
	if (kind != KindPipeline) {
		return false;
	}
	auto record = std::make_shared<GraphicsPipelineRecord>();
	if (!PipelineRecordFormat::GetPipeline(r, *record, layout) || !r.AtEnd()) {
		return false;
	}
	if (layout != PipelineRecordFormat::LayoutVersion) {
		const auto current = [this](uint64_t& key) {
			if (key == 0) return;
			const auto found = m_legacy_stage_keys.find(key);
			key              = found != m_legacy_stage_keys.end() ? found->second : 1u;
		};
		for (uint32_t i = 0; i < record->vertex_stage_count; i++)
			current(record->vertex_stages[i]);
		current(record->pixel_stage);
	}
	const auto known = [this](uint64_t key) { return m_stages.contains(key); };
	if (!std::ranges::all_of(record->VertexStages(), known) ||
	    (record->pixel_stage != 0 && !known(record->pixel_stage))) {
		return true;
	}
	if (m_pipeline_keys.insert(record->Key()).second) {
		m_pipelines.push_back(std::move(record));
	}
	return true;
}

ShaderCode PipelineManifest::StoreCode(const ShaderCode&                  code,
                                       std::vector<std::vector<uint8_t>>* out) {
	if (code == nullptr) {
		return nullptr;
	}
	const auto id = ShaderCodeId::Of(*code);
	if (const auto found = m_code.find(id); found != m_code.end()) {
		return found->second;
	}
	m_code.emplace(id, code);
	auto entry = EncodeCode(KindCode, 0, *code);
	if (out != nullptr) {
		out->push_back(std::move(entry));
	} else {
		m_file.Append(entry);
	}
	return code;
}

bool PipelineManifest::Rewrite() {
	std::vector<std::vector<uint8_t>> payloads;
	m_code.clear();
	for (const auto& stage: m_stage_order) {
		(void)StoreCode(stage->code, &payloads);
		(void)StoreCode(stage->back_code, &payloads);
		auto body = EncodeStageBody(*stage);
		Writer(body).Put(stage->spirv_hash);
		payloads.push_back(std::move(body));
	}
	for (const auto& pipeline: m_pipelines) {
		payloads.push_back(EncodePipelineBody(*pipeline));
	}
	return ManifestFormat::WriteEntryFile(m_path, ManifestMagic, FormatVersion, payloads);
}

bool PipelineManifest::AddStage(const GraphicsStageRecord& record) {
	if (record.code == nullptr || record.code->empty() || record.user_data_count > MaxUserData ||
	    !GraphicsStageRecord::IsGraphicsStage(record.stage)) {
		return false;
	}
	auto stored = std::make_shared<GraphicsStageRecord>(record);
	stored->SetCode(record.code, record.back_code);
	const auto      key = stored->Key();
	std::lock_guard lock(m_mutex);
	if (m_stages.contains(key)) {
		return false;
	}
	stored->code      = StoreCode(record.code, nullptr);
	stored->back_code = StoreCode(record.back_code, nullptr);
	auto body         = EncodeStageBody(*stored);
	Writer(body).Put(stored->spirv_hash);
	m_file.Append(body);
	m_stages.emplace(key, stored);
	m_stage_order.push_back(std::move(stored));
	return true;
}

bool PipelineManifest::AddPipeline(const GraphicsPipelineRecord& record) {
	const auto      body = EncodePipelineBody(record);
	const auto      key  = XXH3_64bits(body.data(), body.size());
	std::lock_guard lock(m_mutex);
	const auto      known = [this](uint64_t stage) { return m_stages.contains(stage); };
	if ((record.vertex_stage_count != 1u && record.vertex_stage_count != 3u) ||
	    !std::ranges::all_of(record.VertexStages(), known) ||
	    (record.pixel_stage != 0 && !known(record.pixel_stage)) ||
	    !m_pipeline_keys.insert(key).second) {
		return false;
	}
	m_file.Append(body);
	m_pipelines.push_back(std::make_shared<GraphicsPipelineRecord>(record));
	return true;
}

std::shared_ptr<const GraphicsStageRecord> PipelineManifest::FindStage(uint64_t key) const {
	std::lock_guard lock(m_mutex);
	const auto      found = m_stages.find(key);
	return found != m_stages.end() ? found->second : nullptr;
}

bool PipelineManifest::HasPipeline(uint64_t key) const {
	std::lock_guard lock(m_mutex);
	return m_pipeline_keys.contains(key);
}

std::vector<std::shared_ptr<const GraphicsStageRecord>> PipelineManifest::Stages() const {
	std::lock_guard lock(m_mutex);
	return m_stage_order;
}

std::vector<std::shared_ptr<const GraphicsPipelineRecord>> PipelineManifest::Pipelines() const {
	std::lock_guard lock(m_mutex);
	return m_pipelines;
}

ShaderCode PipelineManifest::FindCode(const ShaderCodeId& id) const {
	std::lock_guard lock(m_mutex);
	const auto      found = m_code.find(id);
	return found != m_code.end() ? found->second : nullptr;
}

std::filesystem::path ShaderCodeStore::PathFor(const std::filesystem::path& directory,
                                               const std::string&           title_id) {
	return directory / (title_id + ".v5.created");
}

ShaderCodeStore::ShaderCodeStore(std::filesystem::path path): m_path(std::move(path)) {
	if (m_path.empty()) {
		return;
	}
	std::vector<std::vector<uint8_t>> kept;
	const auto                        read = ManifestFormat::ReadEntryFile(
	    m_path, CodeStoreMagic, [&](std::span<const uint8_t> payload, uint32_t version) {
		    Reader       r(payload);
		    uint8_t      kind = 0;
		    uint8_t      type = 0;
		    ShaderCodeId id;
		    auto         code = std::make_shared<std::vector<uint32_t>>();
		    if (version != FormatVersion || !r.Get(kind) || kind != KindCreatedCode ||
		        !r.Get(type) || type >= m_by_type.size() || !GetCode(r, id, *code)) {
			    return false;
		    }
		    if (m_code.try_emplace(id, std::move(code)).second) {
			    m_by_type[type]++;
			    kept.emplace_back(payload.begin(), payload.end());
		    }
		    return true;
	    });
	if (!read.Clean() || read.version != FormatVersion) {
		m_dropped = read.valid ? read.size - read.intact : read.size;
		if (!ManifestFormat::WriteEntryFile(m_path, CodeStoreMagic, FormatVersion, kept)) {
			return;
		}
	}
	m_file.Open(m_path);
}

ShaderCodeStore::~ShaderCodeStore() = default;

bool ShaderCodeStore::Add(uint8_t binary_type, std::span<const uint32_t> code) {
	if (code.empty() || code.size() > MaxCodeWords || binary_type >= m_by_type.size()) {
		return false;
	}
	const auto      id = ShaderCodeId::Of(code);
	std::lock_guard lock(m_mutex);
	if (m_code.contains(id)) {
		return false;
	}
	m_code.emplace(id, std::make_shared<const std::vector<uint32_t>>(code.begin(), code.end()));
	m_by_type[binary_type]++;
	m_file.Append(EncodeCode(KindCreatedCode, binary_type, code));
	return true;
}

ShaderCode ShaderCodeStore::Find(const ShaderCodeId& id) const {
	std::lock_guard lock(m_mutex);
	const auto      found = m_code.find(id);
	return found != m_code.end() ? found->second : nullptr;
}

size_t ShaderCodeStore::Size() const {
	std::lock_guard lock(m_mutex);
	return m_code.size();
}

std::array<size_t, 16> ShaderCodeStore::CountByType() const {
	std::lock_guard lock(m_mutex);
	return m_by_type;
}

} // namespace Libs::Graphics
