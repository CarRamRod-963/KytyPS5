#include "graphics/host_gpu/renderer/pipeline/manifestFormat.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/host_gpu/renderer/pipeline/programManifest.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <xxhash.h>

namespace {

using Libs::Graphics::ComputeProgramRecord;
using Libs::Graphics::GraphicsPipelineRecord;
using Libs::Graphics::GraphicsStageRecord;
using Libs::Graphics::PipelineList;
using Libs::Graphics::PipelineListContents;
using Libs::Graphics::PipelineListResolver;
using Libs::Graphics::PipelineManifest;
using Libs::Graphics::ShaderIndirectOffsets;
using Libs::Graphics::PrecompileAttempts;
using Libs::Graphics::ProgramManifest;
using Libs::Graphics::ShaderCode;
using Libs::Graphics::ShaderCodeId;
using Libs::Graphics::ShaderCodeStore;
using Libs::Graphics::ShaderType;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "PipelineListTests: failed: %s\n", text);
    std::abort();
  }
}

std::filesystem::path TempPath(const char *name) {
  auto dir = std::filesystem::temp_directory_path() / "kyty_pipeline_list_tests";
  std::filesystem::create_directories(dir);
  auto path = dir / name;
  std::filesystem::remove(path);
  return path;
}

ShaderCode Code(uint32_t seed, uint32_t words) {
  auto code = std::make_shared<std::vector<uint32_t>>(words);
  for (uint32_t i = 0; i < words; i++) {
    (*code)[i] = seed * 0x9e3779b9u + i;
  }
  return code;
}

GraphicsStageRecord VertexStage(uint32_t seed, ShaderCode code) {
  GraphicsStageRecord record;
  record.stage = ShaderType::Vertex;
  record.shader_hash = 0x5000u + seed;
  record.user_data_count = 16;
  record.push_data_cursor = 4;
  auto &vertex = record.vertex;
  vertex.logical_stage = ShaderType::Vertex;
  vertex.resources_num = 2;
  vertex.buffers_num = 1;
  vertex.resources[0].fields[3] = 0x12345u;
  vertex.resources[1].fields[3] = 0x54321u;
  vertex.resources_dst[0].registers_num = 3;
  vertex.resources_dst[1].registers_num = 4;
  vertex.resources_dst[1].attr_id = 1;
  vertex.buffers[0].stride = 32;
  vertex.pa_cl_vs_out_cntl = seed;
  record.vertex = GraphicsStageRecord::StaticInput(vertex);
  record.specialization.buffers.resize(2);
  record.specialization.buffers[1].packed_stride = 16;
  record.spirv_hash = 0xabcdu + seed;
  record.SetCode(std::move(code), nullptr);
  return record;
}

GraphicsStageRecord PixelStage(uint32_t seed, ShaderCode code) {
  GraphicsStageRecord record;
  record.stage = ShaderType::Pixel;
  record.shader_hash = 0x6000u + seed;
  record.user_data_count = 8;
  record.pixel.input_num = 3;
  record.pixel.interpolator_settings[2] = seed;
  record.pixel.target_output_mode[0] = 4;
  record.pixel.ps_pos_z = true;
  record.specialization.images.resize(1);
  record.specialization.images[0].mip_count = 7;
  record.SetCode(std::move(code), nullptr);
  return record;
}

GraphicsPipelineRecord Pipeline(const GraphicsStageRecord &vs,
                                const GraphicsStageRecord *ps,
                                uint32_t seed) {
  GraphicsPipelineRecord record;
  record.vertex_stage_count = 1;
  record.vertex_stages[0] = vs.Key();
  record.pixel_stage = ps != nullptr ? ps->Key() : 0;
  record.rendering.color_count = 1;
  record.rendering.color_formats[0] = vk::Format::eR8G8B8A8Unorm;
  record.rendering.depth_format = vk::Format::eD32Sfloat;
  record.vertex_input.binding_count = 1;
  record.vertex_input.attribute_count = 2;
  record.vertex_input.bindings[0].stride = 32;
  record.vertex_input.attributes[1].offset = 12;
  record.static_params.topology = vk::PrimitiveTopology::eTriangleList;
  record.static_params.samples = 1;
  record.static_params.color_mask[0] = 0xf;
  record.static_params.blend_enable[0] = (seed & 1u) != 0;
  record.static_params.color_srcblend[0] = static_cast<uint8_t>(seed);
  return record;
}

void Append(const std::filesystem::path &path, const std::string &bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::app);
  file << bytes;
}

void TestStageKeys() {
  auto code = Code(1, 64);
  auto a = VertexStage(1, code);
  // Runtime fields are not part of a stage: the vertex buffer address and
  // record count change from draw to draw.
  auto with_address = a.vertex;
  with_address.resources[0].fields[0] = 0x12340000u;
  with_address.resources[0].fields[1] |= 0x9u;
  with_address.resources[0].fields[2] = 77;
  with_address.buffers[0].addr = 0x900000000ull;
  auto b = a;
  b.vertex = GraphicsStageRecord::StaticInput(with_address);
  Check(a.Key() == b.Key(), "runtime vertex fields do not change the key");
  auto c = a;
  c.vertex.resources_dst[1].registers_num = 2;
  Check(a.Key() != c.Key(), "a static input field changes the key");
  auto d = a;
  d.specialization.buffers[1].packed_stride = 8;
  Check(a.Key() != d.Key(), "the specialization changes the key");
  auto e = a;
  e.spirv_hash = 1;
  Check(a.Key() == e.Key(), "the SPIR-V hash is not part of the key");
  auto f = a;
  f.SetCode(Code(2, 64), nullptr);
  Check(a.Key() != f.Key(), "the code changes the key");
}

void TestManifestRoundTrip() {
  const auto path = TempPath("round.v5.pipelines");
  const auto vs_code = Code(1, 100);
  const auto vs = VertexStage(1, vs_code);
  const auto vs2 = VertexStage(2, vs_code);
  const auto ps = PixelStage(3, Code(3, 50));
  {
    PipelineManifest manifest(path);
    Check(manifest.Loaded().stages == 0, "a new manifest is empty");
    Check(manifest.AddStage(vs) && manifest.AddStage(vs2) &&
              manifest.AddStage(ps),
          "stages are added");
    Check(!manifest.AddStage(vs), "a known stage is not added again");
    Check(manifest.AddPipeline(Pipeline(vs, &ps, 1)) &&
              manifest.AddPipeline(Pipeline(vs2, nullptr, 2)),
          "pipelines are added");
    Check(!manifest.AddPipeline(Pipeline(vs, &ps, 1)),
          "a known pipeline is not added again");
    auto unknown = Pipeline(vs, &ps, 3);
    unknown.pixel_stage = 0x1234u;
    Check(!manifest.AddPipeline(unknown),
          "a pipeline of unknown stages is refused");
  }
  PipelineManifest manifest(path);
  const auto loaded = manifest.Loaded();
  Check(loaded.stages == 3 && loaded.pipelines == 2 && !loaded.rewritten,
        "the manifest reads back");
  Check(loaded.code_entries == 2, "code is stored once per content");
  const auto stages = manifest.Stages();
  Check(stages[0]->Key() == vs.Key() && stages[1]->Key() == vs2.Key() &&
            stages[2]->Key() == ps.Key(),
        "stages keep their order and keys");
  Check(stages[0]->code == stages[1]->code, "stages share their code");
  Check(*stages[2]->code == *ps.code && stages[2]->spirv_hash == ps.spirv_hash,
        "the stage code and SPIR-V hash read back");
  const auto pipelines = manifest.Pipelines();
  Check(pipelines[0]->Key() == Pipeline(vs, &ps, 1).Key() &&
            pipelines[1]->Key() == Pipeline(vs2, nullptr, 2).Key(),
        "pipelines keep their order and keys");
  Check(manifest.FindStage(ps.Key()) != nullptr &&
            manifest.HasPipeline(pipelines[1]->Key()),
        "lookups find the records");
  Check(manifest.FindCode(ShaderCodeId::Of(*vs_code)) != nullptr,
        "code is found by content");
}

void TestManifestDamage() {
  const auto path = TempPath("damage.v5.pipelines");
  const auto vs = VertexStage(1, Code(1, 100));
  const auto ps = PixelStage(2, Code(2, 40));
  {
    PipelineManifest manifest(path);
    manifest.AddStage(vs);
    manifest.AddStage(ps);
    manifest.AddPipeline(Pipeline(vs, &ps, 1));
  }
  const auto intact = std::filesystem::file_size(path);
  // A torn entry after the intact ones.
  Append(path, std::string("\x40\x00\x00\x00garbage", 11));
  {
    PipelineManifest manifest(path);
    Check(manifest.Loaded().pipelines == 1 && manifest.Loaded().rewritten &&
              manifest.Loaded().dropped_bytes == 11,
          "a torn tail is dropped");
  }
  Check(std::filesystem::file_size(path) == intact,
        "the rewrite keeps the intact entries");
  // A flipped byte in the last entry (the pipeline) drops it.
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(static_cast<std::streamoff>(intact - 5));
    file.put('\x7f');
  }
  {
    PipelineManifest manifest(path);
    Check(manifest.Loaded().stages == 2 && manifest.Loaded().pipelines == 0,
          "a corrupt entry is dropped");
  }
  // Garbage of another format.
  std::filesystem::remove(path);
  Append(path, "not a manifest at all");
  {
    PipelineManifest manifest(path);
    Check(manifest.Loaded().stages == 0 && manifest.Loaded().rewritten,
          "a foreign file is replaced");
    manifest.AddStage(vs);
  }
  PipelineManifest manifest(path);
  Check(manifest.Loaded().stages == 1, "the replaced file is appended to");
}

// A layout-1 file (<title>.v5.pipelines) is imported once into the current file: its stages keep
// their fields, its pipelines name the stages by their new keys, and the old file stays.
void TestManifestImport() {
  namespace RecordFormat = Libs::Graphics::PipelineRecordFormat;
  using Libs::Graphics::ManifestFormat::Writer;
  const auto previous = TempPath("import.v5.pipelines");
  const auto path = TempPath("import.v6.pipelines");
  auto vs = VertexStage(4, Code(4, 80));
  vs.vertex.has_dynamic_vertex_fetch = true;
  vs.vertex.indirect_draw = true;
  vs.vertex.indirect_offsets = {3, ShaderIndirectOffsets::None};
  const auto ps = PixelStage(5, Code(5, 30));
  const auto code_entry = [](const ShaderCode &code) {
    std::vector<uint8_t> out;
    Writer w(out);
    w.Put(uint8_t{1});
    const auto id = ShaderCodeId::Of(*code);
    w.Put(id.hash);
    w.Put(id.words);
    w.PutWords(*code);
    return out;
  };
  std::vector<std::vector<uint8_t>> payloads{code_entry(vs.code), code_entry(ps.code)};
  std::vector<uint64_t> old_keys;
  for (const GraphicsStageRecord *stage : {static_cast<const GraphicsStageRecord *>(&vs), &ps}) {
    std::vector<uint8_t> body;
    Writer w(body);
    w.Put(uint8_t{2});
    RecordFormat::PutStage(w, *stage, 1);
    old_keys.push_back(XXH3_64bits(body.data(), body.size()));
    w.Put(stage->spirv_hash);
    payloads.push_back(std::move(body));
  }
  auto pipeline = Pipeline(vs, &ps, 1);
  pipeline.vertex_stages[0] = old_keys[0];
  pipeline.pixel_stage = old_keys[1];
  {
    std::vector<uint8_t> body;
    Writer w(body);
    w.Put(uint8_t{3});
    RecordFormat::PutPipeline(w, pipeline, 1);
    payloads.push_back(std::move(body));
  }
  Check(old_keys[0] != vs.Key(), "the layouts encode a stage differently");
  Check(Libs::Graphics::ManifestFormat::WriteEntryFile(
            previous, {'K', 'y', 'G', 'f', 'x', 'P', 'l', 'n'},
            PipelineManifest::FormatVersion, payloads),
        "the layout-1 file is written");
  const auto previous_size = std::filesystem::file_size(previous);
  const auto ps2 = [] {
    auto stage = PixelStage(6, Code(6, 20));
    stage.pixel.target_uint_mask = 2;
    stage.pixel.target_sint_mask = 4;
    return stage;
  }();
  {
    PipelineManifest manifest(path, previous);
    const auto loaded = manifest.Loaded();
    Check(loaded.imported && loaded.stages == 2 && loaded.pipelines == 1,
          "the layout-1 records are imported");
    const auto imported = manifest.FindStage(vs.Key());
    Check(imported != nullptr && imported->vertex.indirect_draw &&
              imported->vertex.indirect_offsets == vs.vertex.indirect_offsets &&
              imported->vertex.has_dynamic_vertex_fetch,
          "a layout-1 stage keeps its indirect draw and fetch fields");
    const auto pipelines = manifest.Pipelines();
    Check(pipelines[0]->vertex_stages[0] == vs.Key() &&
              pipelines[0]->pixel_stage == ps.Key() &&
              pipelines[0]->Key() == Pipeline(vs, &ps, 1).Key(),
          "an imported pipeline names its stages by their current keys");
    Check(manifest.AddStage(ps2), "the imported manifest is appended to");
  }
  Check(std::filesystem::file_size(previous) == previous_size,
        "the layout-1 file is left as it is");
  PipelineManifest manifest(path, previous);
  const auto loaded = manifest.Loaded();
  Check(!loaded.imported && !loaded.rewritten && loaded.stages == 3 && loaded.pipelines == 1,
        "the current file is read without importing again");
  const auto masks = manifest.FindStage(ps2.Key());
  Check(masks != nullptr && masks->pixel.target_uint_mask == 2 &&
            masks->pixel.target_sint_mask == 4,
        "the integer target masks read back");
}

void TestFieldValidation() {
  using Libs::Graphics::ManifestFormat::Reader;
  using Libs::Graphics::ManifestFormat::Writer;
  namespace Format = Libs::Graphics::PipelineRecordFormat;
  const auto vs = VertexStage(1, Code(1, 10));
  const auto good = Pipeline(vs, nullptr, 1);
  const auto decode = [](const GraphicsPipelineRecord &record) {
    std::vector<uint8_t> bytes;
    Writer w(bytes);
    Format::PutPipeline(w, record);
    Reader r(bytes);
    GraphicsPipelineRecord out;
    return Format::GetPipeline(r, out) && r.AtEnd() && out.Key() == record.Key();
  };
  Check(decode(good), "a valid pipeline decodes to itself");
  auto bad = good;
  bad.static_params.topology = static_cast<vk::PrimitiveTopology>(99);
  Check(!decode(bad), "an invalid topology is rejected");
  bad = good;
  bad.static_params.samples = 3;
  Check(!decode(bad), "an invalid sample count is rejected");
  bad = good;
  bad.rendering.color_count = 9;
  Check(!decode(bad), "too many color targets are rejected");
  bad = good;
  bad.vertex_input.attributes[1].binding = 1;
  Check(!decode(bad), "an attribute of a missing binding is rejected");
  bad = good;
  bad.rendering.depth_format = static_cast<vk::Format>(0x7fffffff);
  Check(!decode(bad), "an unknown format is rejected");

  const auto decode_stage = [](const GraphicsStageRecord &record) {
    std::vector<uint8_t> bytes;
    Writer w(bytes);
    Format::PutStage(w, record);
    Reader r(bytes);
    GraphicsStageRecord out;
    return Format::GetStage(r, out) && r.AtEnd() && out.Key() == record.Key();
  };
  Check(decode_stage(vs), "a valid stage decodes to itself");
  auto bad_stage = vs;
  bad_stage.vertex.wave_size = 48;
  Check(!decode_stage(bad_stage), "an invalid wave size is rejected");
  bad_stage = vs;
  bad_stage.vertex.resources_dst[0].registers_num = 5;
  Check(!decode_stage(bad_stage), "an invalid register count is rejected");
  bad_stage = vs;
  bad_stage.stage = ShaderType::Compute;
  Check(!decode_stage(bad_stage), "a compute stage is rejected");
  auto bad_pixel = PixelStage(2, Code(2, 10));
  bad_pixel.pixel.parameter_mode =
      static_cast<Libs::Graphics::ShaderPixelParameterMode>(7);
  Check(!decode_stage(bad_pixel), "an invalid parameter mode is rejected");
}

void TestCodeStore() {
  const auto path = TempPath("store.v5.created");
  const auto a = Code(1, 30);
  const auto b = Code(2, 31);
  {
    ShaderCodeStore store(path);
    Check(store.Add(1, *a) && store.Add(0, *b), "codes are added");
    Check(!store.Add(1, *a), "known code is not added again");
  }
  Append(path, "torn");
  ShaderCodeStore store(path);
  Check(store.Size() == 2 && store.LoadedDroppedBytes() == 4,
        "the store reads back without its torn tail");
  Check(store.CountByType()[1] == 1 && store.CountByType()[0] == 1,
        "codes are counted by type");
  const auto found = store.Find(ShaderCodeId::Of(*a));
  Check(found != nullptr && *found == *a, "code is found by content");
  Check(store.Find({.hash = 1, .words = 30}) == nullptr,
        "unknown content is not found");
}

ComputeProgramRecord Compute(uint32_t seed, ShaderCode code) {
  ComputeProgramRecord record;
  record.shader_hash = ShaderCodeId::Of(*code).hash;
  record.code = std::move(code);
  record.user_data_count = 4;
  record.input.threads_num[0] = 64;
  record.input.threads_num[1] = 1;
  record.input.threads_num[2] = 1;
  record.push_data_cursor = seed;
  return record;
}

struct ListFixture {
  ShaderCode cs_code = Code(10, 80);
  ShaderCode created_code = Code(11, 90);
  ShaderCode vs_code = Code(12, 70);
  ShaderCode ps_code = Code(13, 60);
  ShaderCode ps2_code = Code(14, 60);
  GraphicsStageRecord vs = VertexStage(1, vs_code);
  GraphicsStageRecord ps = PixelStage(2, ps_code);
  GraphicsStageRecord ps2 = PixelStage(3, ps2_code);

  PipelineListContents Contents() {
    ProgramManifest programs;
    programs.Add(Compute(1, cs_code));
    ProgramManifest created;
    created.Add(Compute(0, cs_code));
    created.Add(Compute(0, created_code));
    PipelineManifest pipelines;
    pipelines.AddStage(vs);
    pipelines.AddStage(ps);
    pipelines.AddStage(ps2);
    pipelines.AddPipeline(Pipeline(vs, &ps, 1));
    pipelines.AddPipeline(Pipeline(vs, &ps2, 2));
    pipelines.AddPipeline(Pipeline(vs, nullptr, 3));
    return PipelineList::FromManifests("PPSA00001", &programs, &created,
                                       &pipelines);
  }
};

void TestListRoundTrip() {
  ListFixture fixture;
  const auto contents = fixture.Contents();
  Check(contents.compute.size() == 2 && contents.compute[1].code_only,
        "recorded programs and created shaders without one are listed");
  Check(contents.stages.size() == 3 && contents.pipelines.size() == 3,
        "stages and pipelines are listed");
  const auto path = TempPath("round.plist");
  Check(PipelineList::Write(path, contents), "the list is written");
  // No code: no 16-byte window of any shader appears in the file.
  {
    std::ifstream file(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), {});
    for (const auto &code : {fixture.cs_code, fixture.created_code,
                             fixture.vs_code, fixture.ps_code}) {
      const std::string window(reinterpret_cast<const char *>(code->data()) + 8,
                               16);
      Check(bytes.find(window) == std::string::npos,
            "the list holds no shader code");
    }
  }
  PipelineList::ReadStats stats;
  const auto read = PipelineList::Read(path, "PPSA00001", &stats);
  Check(stats.valid && stats.title_matches && stats.dropped_bytes == 0 &&
            stats.rejected == 0,
        "the list reads back cleanly");
  Check(read.compute.size() == 2 && read.stages.size() == 3 &&
            read.pipelines.size() == 3,
        "every entry reads back");
  Check(read.compute[0].code == ShaderCodeId::Of(*fixture.cs_code) &&
            read.compute[0].record.push_data_cursor == 1 &&
            !read.compute[0].code_only && read.compute[1].code_only,
        "compute entries read back");
  Check(read.stages[0].Key() == fixture.vs.Key() &&
            read.stages[2].Key() == fixture.ps2.Key(),
        "stage keys survive without code");
  Check(read.pipelines[1].Key() == Pipeline(fixture.vs, &fixture.ps2, 2).Key(),
        "pipelines read back");

  const auto other = PipelineList::Read(path, "PPSA00002", &stats);
  Check(stats.valid && !stats.title_matches && other.compute.empty() &&
            other.pipelines.empty(),
        "a list of another title gives nothing");

  // Damage in the last entry keeps the intact prefix.
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(-3, std::ios::end);
    file.put('\x55');
  }
  const auto damaged = PipelineList::Read(path, "PPSA00001", &stats);
  Check(stats.dropped_bytes != 0 && damaged.pipelines.size() == 2 &&
            damaged.stages.size() == 3,
        "a damaged list is read up to its last intact entry");
  std::filesystem::remove(path);
  Append(path, "garbage");
  const auto garbage = PipelineList::Read(path, "PPSA00001", &stats);
  Check(!stats.valid && garbage.compute.empty(), "garbage is not a list");
}

void TestResolver() {
  ListFixture fixture;
  PipelineListResolver resolver(fixture.Contents());
  auto counts = resolver.GetCounts();
  Check(counts.compute == 2 && counts.pipelines == 3 &&
            counts.pipelines_broken == 0 && counts.shaders_pending == 5,
        "the resolver starts with every name pending");
  const auto created_hash = ShaderCodeId::Of(*fixture.created_code).hash;
  auto hashes = resolver.PendingShaderHashes();
  Check(hashes.size() == 5 && hashes.contains(fixture.vs.shader_hash) &&
            hashes.contains(fixture.ps2.shader_hash) &&
            hashes.contains(created_hash),
        "every pending entry names its program hash");
  // The local shaders: everything but the second pixel shader and the created
  // compute shader.
  const auto lookup = [&](const ShaderCodeId &id) -> ShaderCode {
    for (const auto &code :
         {fixture.cs_code, fixture.vs_code, fixture.ps_code}) {
      if (ShaderCodeId::Of(*code) == id)
        return code;
    }
    return nullptr;
  };
  auto resolved = resolver.Resolve(lookup);
  Check(resolved.compute.size() == 1 &&
            *resolved.compute[0]->code == *fixture.cs_code &&
            !resolved.compute_code_only[0],
        "a compute entry resolves with its code");
  Check(resolved.pipelines.size() == 2, "pipelines of known shaders resolve");
  // Positions keep first-seen order, recorded and code-only compute entries
  // each among their own kind.
  Check(resolved.compute_position.size() == 1 &&
            resolved.compute_position[0] == 0 &&
            std::min(resolved.pipelines[0].position,
                     resolved.pipelines[1].position) == 0 &&
            std::max(resolved.pipelines[0].position,
                     resolved.pipelines[1].position) ==
                Libs::Graphics::FirstSeenPosition(2, 3),
        "resolved entries carry their first-seen positions");
  Check(Libs::Graphics::FirstSeenPosition(1, 2) == (uint64_t{1} << 31u) &&
            Libs::Graphics::FirstSeenPosition(5, 10) ==
                Libs::Graphics::FirstSeenPosition(1, 2) &&
            Libs::Graphics::FirstSeenPosition(0, 0) == 0,
        "positions of different list lengths compare by fraction");
  for (const auto &pipeline : resolved.pipelines) {
    Check(pipeline.vertex_stages.size() == 1 &&
              pipeline.vertex_stages[0]->code != nullptr &&
              pipeline.vertex_stages[0]->Key() == fixture.vs.Key(),
          "a resolved pipeline carries its stages with code");
  }
  Check(resolver.Resolve(lookup).Empty(), "entries resolve only once");
  counts = resolver.GetCounts();
  Check(counts.shaders_pending == 2 && counts.pipelines_resolved == 2 &&
            resolver.Pending(),
        "missing shaders stay pending");
  // A shader the list does not name changes nothing.
  Check(resolver.OnCode(*Code(99, 10)).Empty(), "unknown code is ignored");
  // The guest creates the missing shaders.
  hashes = resolver.PendingShaderHashes();
  Check(hashes.size() == 2 && hashes.contains(fixture.ps2.shader_hash) &&
            hashes.contains(created_hash),
        "resolved entries leave the pending hashes");
  resolved = resolver.OnCode(*fixture.ps2_code);
  Check(resolved.pipelines.size() == 1 &&
            resolved.pipelines[0].pixel_stage->Key() == fixture.ps2.Key() &&
            resolved.pipelines[0].position ==
                Libs::Graphics::FirstSeenPosition(1, 3),
        "a pipeline resolves when its last shader is created");
  resolved = resolver.OnCode(*fixture.created_code);
  Check(resolved.compute.size() == 1 && resolved.compute_code_only[0] &&
            resolved.compute_position[0] == 0,
        "a created compute entry resolves when the shader is created");
  Check(!resolver.Pending() && resolver.GetCounts().compute_resolved == 2 &&
            resolver.PendingShaderHashes().empty(),
        "everything resolved");

  // A pipeline whose stage entry is missing from the list.
  auto contents = fixture.Contents();
  contents.stages.pop_back();
  PipelineListResolver broken(std::move(contents));
  Check(broken.GetCounts().pipelines_broken == 1 &&
            broken.GetCounts().pipelines == 2,
        "a pipeline without its stage is dropped");
}

void TestAttempts() {
  const auto previous = TempPath("title.v4.attempts");
  const auto path = TempPath("title.v5.attempts");
  {
    std::ofstream file(previous);
    file << "B 11\nB 22\nE 22\nB 33\n";
  }
  {
    PrecompileAttempts attempts(path, previous);
    Check(attempts.Retried(0x11u) && attempts.Retried(0x33u) &&
              !attempts.Failed(0x11u) && !attempts.Retried(0x22u),
          "a build that ended one process is retried, not quarantined");
    Check(attempts.RetriedCount() == 2 && attempts.FailedCount() == 0,
          "the previous format's attempts are read");
    // The retry of 0x11 ends this process too; 0x33 finishes; 0x44 times out.
    attempts.Begin(0x11u);
    attempts.Begin(0x33u);
    attempts.End(0x33u);
    attempts.Begin(0x44u);
    attempts.TimeOut(0x44u);
  }
  {
    PrecompileAttempts attempts(path, previous);
    Check(attempts.Failed(0x11u) && !attempts.Retried(0x11u),
          "a build that ended two processes is quarantined");
    Check(!attempts.Failed(0x33u) && !attempts.Retried(0x33u),
          "a finished retry is cleared");
    Check(attempts.Failed(0x44u) && attempts.TimedOut(0x44u),
          "a timed-out build is quarantined");
    Check(attempts.FailedCount() == 2, "two builds are quarantined");
  }
  PrecompileAttempts attempts(path, previous);
  Check(attempts.Failed(0x11u) && attempts.TimedOut(0x44u) &&
            attempts.FailedCount() == 2,
        "quarantines survive the rewrite");
}

} // namespace

int main() {
  TestStageKeys();
  TestManifestRoundTrip();
  TestManifestDamage();
  TestManifestImport();
  TestFieldValidation();
  TestCodeStore();
  TestListRoundTrip();
  TestResolver();
  TestAttempts();
  std::printf("PipelineListTests: all tests passed\n");
  return 0;
}
