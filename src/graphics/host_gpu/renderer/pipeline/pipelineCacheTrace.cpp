#include "graphics/host_gpu/renderer/pipeline/pipelineCacheTrace.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/shaderEnv.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <fmt/format.h>
#include <spirv-tools/libspirv.hpp>

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <xxhash.h>

// The shader dumps and traces of the pipeline cache's compile path: every one is off unless its name is set (a compile is
// no hot path; the gates are read where the tool runs).
namespace Libs::Graphics::PipelineTrace {

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	// KYTY_TRACE_SHADER_CODE_DIR=<dir> (diagnostic): raw guest code of each compiled shader.
	if (const char* dir = Common::EnvRead::ShaderCodeDir(); dir != nullptr && !code.empty()) {
		const auto name = fmt::format("{}/{}_{:016x}.bin", dir, stage_name, shader_hash);
		if (FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(code.data(), 1, code.size_bytes(), f);
			std::fclose(f);
		}
		// KYTY_TRACE_DISASM=<hex,...>: also the decoded RDNA2 text of the listed shaders, written here
		// so it does not depend on a pipeline cache miss: <dir>/<stage>_<hash>.asm.
		if (const char* list = std::getenv("KYTY_TRACE_DISASM");
		    list != nullptr && std::string_view(list).find(fmt::format("{:016x}", shader_hash)) != std::string_view::npos) {
			ShaderRecompiler::Decoder::Program decoded;
			ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
			const auto text = ShaderRecompiler::Decoder::ProgramToString(decoded);
			const auto asm_name = fmt::format("{}/{}_{:016x}.asm", dir, stage_name, shader_hash);
			if (FILE* f = std::fopen(asm_name.c_str(), "w")) {
				std::fwrite(text.data(), 1, text.size(), f);
				std::fclose(f);
			}
		}
	}
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	// KYTY_TRACE_SPIRV_VALIDATE=1 (diagnostic, default off): every translated module is validated as with
	// --shader-validation (the test runners pass it off); a failure names the shader and stops before the driver sees it.
	static const bool trace_validate = [] {
		const char* value = std::getenv("KYTY_TRACE_SPIRV_VALIDATE");
		return value != nullptr && value[0] == '1';
	}();
	if (!Config::ShaderValidationEnabled() && !trace_validate) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	std::fprintf(stderr, "SPIRV_VALIDATE_FAIL %s hash=0x%016llx: %s", label,
	             static_cast<unsigned long long>(shader_hash), messages.c_str());
	return false;
}

// KYTY_TRACE_SPECIALIZATION=<hex hash,...> (diagnostic, M115): a new permutation of a listed shader prints what
// differs from its previous permutation (push-data start and every buffer, image and sampler specialization field),
// or all of it for the first one: which resource property gives a shader a new permutation (and a cold compile).
void TraceSpecialization(uint64_t hash, const char* stage_name, const ShaderRecompiler::IR::ResourceSpecialization* prev,
                         size_t permutations, const ShaderRecompiler::IR::ResourceSpecialization& spec, uint32_t push_data_cursor) {
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_TRACE_SPECIALIZATION");
		return value != nullptr ? std::string(value) : std::string();
	}();
	// =all: every shader, from its second permutation on (only what differs).
	if (list.empty() || (list == "all" ? permutations == 0 : list.find(fmt::format("{:016x}", hash)) == std::string::npos))
		return;
	std::string out = fmt::format("SPECIALIZATION hash={:016x} stage={} permutation={} push_cursor={} buffers={} images={} samplers={}",
	                              hash, stage_name, permutations + 1, push_data_cursor, spec.buffers.size(),
	                              spec.images.size(), spec.samplers.size());
	if (prev != nullptr && (prev->buffers.size() != spec.buffers.size() || prev->images.size() != spec.images.size() ||
	                        prev->samplers.size() != spec.samplers.size()))
		out += fmt::format(" (prev counts {}/{}/{})", prev->buffers.size(), prev->images.size(), prev->samplers.size());
	const auto field = [&](const char* kind, size_t index, const char* name, uint64_t now, std::optional<uint64_t> before) {
		if (before.has_value() && *before == now) return;
		out += before.has_value() ? fmt::format(" {}{}.{}={}->{}", kind, index, name, *before, now)
		                          : fmt::format(" {}{}.{}={}", kind, index, name, now);
	};
	for (size_t i = 0; i < spec.buffers.size(); i++) {
		const auto& b = spec.buffers[i];
		const auto* p = prev != nullptr && i < prev->buffers.size() ? &prev->buffers[i] : nullptr;
		const auto  was = [&](uint64_t v) { return p != nullptr ? std::optional<uint64_t>(v) : std::nullopt; };
		field("b", i, "stride", b.packed_stride, was(p ? p->packed_stride : 0));
		field("b", i, "format", static_cast<uint64_t>(b.descriptor_format), was(p ? static_cast<uint64_t>(p->descriptor_format) : 0));
		field("b", i, "swizzle", b.descriptor_swizzle, was(p ? p->descriptor_swizzle : 0));
		field("b", i, "zero_stride_oob", b.zero_stride_oob ? 1 : 0, was(p ? (p->zero_stride_oob ? 1 : 0) : 0));
		field("b", i, "indirect_root", b.indirect_root, was(p ? p->indirect_root : 0));
		field("b", i, "indirect_offset", b.indirect_mapping_offset, was(p ? p->indirect_mapping_offset : 0));
		field("b", i, "indirect_iterations", b.indirect_search_iterations, was(p ? p->indirect_search_iterations : 0));
	}
	for (size_t i = 0; i < spec.images.size(); i++) {
		const auto& m = spec.images[i];
		const auto* p = prev != nullptr && i < prev->images.size() ? &prev->images[i] : nullptr;
		const auto  was = [&](uint64_t v) { return p != nullptr ? std::optional<uint64_t>(v) : std::nullopt; };
		field("i", i, "class", static_cast<uint64_t>(m.numeric_class), was(p ? static_cast<uint64_t>(p->numeric_class) : 0));
		field("i", i, "dim", static_cast<uint64_t>(m.dimension), was(p ? static_cast<uint64_t>(p->dimension) : 0));
		field("i", i, "mips", m.mip_count, was(p ? p->mip_count : 0));
		field("i", i, "conversion", static_cast<uint64_t>(m.conversion_format), was(p ? static_cast<uint64_t>(p->conversion_format) : 0));
		field("i", i, "swizzle", m.shader_swizzle, was(p ? p->shader_swizzle : 0));
		field("i", i, "indirect_root", m.indirect_root, was(p ? p->indirect_root : 0));
		field("i", i, "indirect_offset", m.indirect_mapping_offset, was(p ? p->indirect_mapping_offset : 0));
		field("i", i, "indirect_iterations", m.indirect_search_iterations, was(p ? p->indirect_search_iterations : 0));
		field("i", i, "cube", m.cube ? 1 : 0, was(p ? (p->cube ? 1 : 0) : 0));
		field("i", i, "fmask", m.fmask ? 1 : 0, was(p ? (p->fmask ? 1 : 0) : 0));
		field("i", i, "bindless", m.bindless ? 1 : 0, was(p ? (p->bindless ? 1 : 0) : 0));
		field("i", i, "bindless_stride", m.bindless_stride_dwords, was(p ? p->bindless_stride_dwords : 0));
	}
	for (size_t i = 0; i < spec.samplers.size(); i++) {
		const auto& s = spec.samplers[i];
		const auto* p = prev != nullptr && i < prev->samplers.size() ? &prev->samplers[i] : nullptr;
		const auto  was = [&](uint64_t v) { return p != nullptr ? std::optional<uint64_t>(v) : std::nullopt; };
		field("s", i, "indirect_root", s.indirect_root, was(p ? p->indirect_root : 0));
		field("s", i, "indirect_offset", s.indirect_mapping_offset, was(p ? p->indirect_mapping_offset : 0));
		field("s", i, "indirect_iterations", s.indirect_search_iterations, was(p ? p->indirect_search_iterations : 0));
	}
	std::fprintf(stderr, "%s\n", out.c_str());
}

void DumpModule(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv) {
	// KYTY_TRACE_SPIRV_ALL_DIR=<dir> (diagnostic): every emitted module, <stage>_<hash>_<SPIR-V hash>.spv (one file
	// per distinct module), for comparing two builds or switch settings offline.
	if (const char* dir = Common::EnvRead::SpirvAllDir(); dir != nullptr && !spirv.empty()) {
		const auto spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t));
		const auto name = fmt::format("{}/{}_{:016x}_{:016x}.spv", dir, stage_name, shader_hash, spirv_hash);
		if (FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), f);
			std::fclose(f);
		}
	}
	// KYTY_TRACE_SHADER_CODE_DIR: SPIR-V of oversized modules (> 400k words) for inspection.
	if (const char* dir = Common::EnvRead::ShaderCodeDir(); dir != nullptr && spirv.size() > 400000u) {
		const auto name = fmt::format("{}/big_{:016x}_{}.spv", dir, shader_hash, spirv.size());
		if (FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), f);
			std::fclose(f);
		}
	}
}

void ReplaceSpirv(uint64_t shader_hash, std::vector<uint32_t>& spirv) {
	// KYTY_DIAG_SPIRV_REPLACE=<hex hash>=<file.spvasm>: the listed shader's SPIR-V is replaced by an
	// edited disassembly (measurement probes). Logged; the edited module is still validated.
	if (const char* replace = Common::ShaderEnv::Read("KYTY_DIAG_SPIRV_REPLACE"); replace != nullptr) {
		const std::string text(replace);
		const auto        separator = text.find('=');
		if (separator != std::string::npos &&
		    std::strtoull(text.substr(0, separator).c_str(), nullptr, 16) == shader_hash) {
			std::string source;
			if (FILE* f = std::fopen(text.substr(separator + 1).c_str(), "rb")) {
				char   chunk[65536];
				size_t read = 0;
				while ((read = std::fread(chunk, 1, sizeof(chunk), f)) > 0) source.append(chunk, read);
				std::fclose(f);
			}
			const auto env = source.find("; Version: 1.3") != std::string::npos ? SPV_ENV_VULKAN_1_1
			               : source.find("; Version: 1.5") != std::string::npos ? SPV_ENV_VULKAN_1_2
			                                                                    : SPV_ENV_VULKAN_1_3;
			spvtools::SpirvTools  assembler(env);
			std::vector<uint32_t> words;
			std::string           messages;
			assembler.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position,
			                                 const char* message) {
				messages += fmt::format("line {}: {}; ", position.line, message);
			});
			if (!source.empty() && assembler.Assemble(source, &words)) {
				std::fprintf(stderr, "DIAG_SPIRV_REPLACE APPROXIMATION hash=0x%016llx words=%zu (was %zu)\n",
				             static_cast<unsigned long long>(shader_hash), words.size(),
				             spirv.size());
				spirv = std::move(words);
			} else {
				std::fprintf(stderr, "DIAG_SPIRV_REPLACE failed hash=0x%016llx bytes=%zu %s\n",
				             static_cast<unsigned long long>(shader_hash), source.size(),
				             messages.c_str());
			}
		}
	}
}

void DumpSpirvText(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv) {
	// KYTY_TRACE_SPIRV_TEXT=<hex,...> (with KYTY_TRACE_SHADER_CODE_DIR): disassembled SPIR-V of the
	// listed shaders, for reading what a pass computes.
	if (const char* list = std::getenv("KYTY_TRACE_SPIRV_TEXT"); list != nullptr) {
		const auto hash_text = fmt::format("{:016x}", shader_hash);
		const char* dir      = Common::EnvRead::ShaderCodeDir();
		if (dir != nullptr && std::string(list).find(hash_text) != std::string::npos) {
			spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
			std::string          text;
			disassembler.Disassemble(spirv, &text,
			                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
			                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT));
			const auto name = fmt::format("{}/{}_{}_{}.spvasm", dir, stage_name, hash_text, spirv.size());
			if (FILE* f = std::fopen(name.c_str(), "wb")) {
				std::fwrite(text.data(), 1, text.size(), f);
				std::fclose(f);
			}
		}
	}
}

void TracePermutation(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSpecialization* previous, size_t permutations,
                      const ShaderRecompiler::IR::ResourceSpecialization& next) {
	// KYTY_TRACE_PERMUTATIONS=1: why a shader that already has a permutation compiles another.
	static const bool trace_permutations = std::getenv("KYTY_TRACE_PERMUTATIONS") != nullptr;
	if (trace_permutations && previous != nullptr) {
		std::string reason = fmt::format("buffers={}/{} images={}/{} samplers={}/{}", previous->buffers.size(),
		                                 next.buffers.size(), previous->images.size(), next.images.size(),
		                                 previous->samplers.size(), next.samplers.size());
		for (size_t i = 0; i < std::min(previous->images.size(), next.images.size()); ++i) {
			const auto& a = previous->images[i];
			const auto& b = next.images[i];
			if (a == b) continue;
			reason += fmt::format(" img{}: class {}/{} dim {}/{} mips {}/{} conv {}/{} swz {:x}/{:x} root {}/{} map {}/{} iter {}/{} cube {}/{}",
			                      i, static_cast<int>(a.numeric_class), static_cast<int>(b.numeric_class),
			                      static_cast<int>(a.dimension), static_cast<int>(b.dimension), a.mip_count, b.mip_count,
			                      static_cast<int>(a.conversion_format), static_cast<int>(b.conversion_format),
			                      a.shader_swizzle, b.shader_swizzle, a.indirect_root, b.indirect_root,
			                      a.indirect_mapping_offset, b.indirect_mapping_offset,
			                      a.indirect_search_iterations, b.indirect_search_iterations, a.cube, b.cube);
			break;
		}
		for (size_t i = 0; i < std::min(previous->buffers.size(), next.buffers.size()); ++i) {
			const auto& a = previous->buffers[i];
			const auto& b = next.buffers[i];
			if (a == b) continue;
			reason += fmt::format(" buf{}: stride {:x}/{:x} fmt {}/{} swz {:x}/{:x} oob {}/{} root {}/{} map {}/{} iter {}/{}",
			                      i, a.packed_stride, b.packed_stride, static_cast<int>(a.descriptor_format),
			                      static_cast<int>(b.descriptor_format), a.descriptor_swizzle, b.descriptor_swizzle,
			                      a.zero_stride_oob, b.zero_stride_oob, a.indirect_root, b.indirect_root,
			                      a.indirect_mapping_offset, b.indirect_mapping_offset,
			                      a.indirect_search_iterations, b.indirect_search_iterations);
			break;
		}
		// Every image table of the new permutation: root index, child count, b = bindless heap,
		// @ = mapping offset in the flattened SRT.
		reason += " roots:";
		for (size_t i = 0; i < next.images.size(); ++i) {
			if (next.images[i].indirect_root != i) continue;
			size_t children = 0;
			for (size_t j = 0; j < next.images.size(); ++j)
				children += j != i && next.images[j].indirect_root == i ? 1u : 0u;
			reason += fmt::format(" {}:{}{}@{}", i, children, next.images[i].bindless ? "b" : "",
			                      next.images[i].indirect_mapping_offset);
		}
		std::fprintf(stderr, "PERMUTATION hash=0x%016llx count=%zu %s\n",
		             static_cast<unsigned long long>(shader_hash), permutations,
		             reason.c_str());
	}
}

} // namespace Libs::Graphics::PipelineTrace
