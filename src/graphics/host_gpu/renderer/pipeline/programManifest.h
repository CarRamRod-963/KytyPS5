#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMMANIFEST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMMANIFEST_H_

#include "common/common.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// A compute program as the pipeline cache translated it, in guest terms: the guest code, the
// static dispatch state the translation reads (the program key) and the resource
// specialization the module was compiled for. Translating it again gives the module the
// pipeline cache would build, from whichever recompiler version reads it, without the guest.
// Runtime values (user data, workgroup counts, addresses) only feed the resource walk, whose
// result is the specialization, so they are not stored.
struct ComputeProgramRecord {
	uint64_t                                     shader_hash = 0;
	std::shared_ptr<const std::vector<uint32_t>> code;
	uint32_t                                     user_data_count = 0;
	// The static fields BuildStageStaticKey reads. Host fields (host_subgroup_size,
	// lds_storage) are derived from the device when the record is translated.
	ShaderComputeInputInfo                       input;
	uint32_t                                     push_data_cursor = 0;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	// Hash of the SPIR-V the recording build emitted. Not part of the key: a recompiler change
	// gives the same program another module.
	uint64_t spirv_hash = 0;

	// Identifies the program independently of the recompiler version.
	[[nodiscard]] uint64_t Key() const;
	// The fields of `input` a record stores; the others are zero.
	[[nodiscard]] static ShaderComputeInputInfo StaticInput(const ShaderComputeInputInfo& input);
};

// The compute programs a title has needed, persisted next to its driver pipeline cache
// (_PipelineCache/<title>.programs) so that their pipelines can be built ahead of their first
// use: in a later session, after the driver cache lost them, or after a recompiler change gave
// them new modules. The file is append-only; every entry is checksummed, and a file that ends
// in a torn or corrupt entry, or that has another format version, is rewritten with the entries
// that are still readable. Guest code is stored once per shader.
//
// Thread-safe.
class ProgramManifest {
public:
	// Bump when the stored fields change; the static_asserts in programManifest.cpp trip when
	// a serialized struct changes size.
	static constexpr uint32_t FormatVersion = 4;

	// Loads the manifest at `path`, creating it when missing. An empty path keeps the manifest
	// in memory only. While `path` is missing, the manifest of the previous format at
	// `previous` is read (and left as it is, for the builds that still use it).
	explicit ProgramManifest(std::filesystem::path path = {}, std::filesystem::path previous = {});

	// Where a title's manifest of the current format lives, and the previous format's.
	static std::filesystem::path PathFor(const std::filesystem::path& directory,
	                                     const std::string&           title_id);
	static std::filesystem::path PreviousPathFor(const std::filesystem::path& directory,
	                                             const std::string&           title_id);
	// Where a title's created compute shaders live: records with the static input a shader's
	// header describes and no specialization, for shaders no dispatch may have used yet.
	static std::filesystem::path CreatedPathFor(const std::filesystem::path& directory,
	                                            const std::string&           title_id);
	~ProgramManifest();
	KYTY_CLASS_NO_COPY(ProgramManifest);

	struct LoadStats {
		size_t records       = 0;
		size_t code_entries  = 0;
		size_t dropped_bytes = 0;     // Unreadable tail (torn or corrupt entries).
		bool   rewritten     = false; // Format version mismatch or corrupt tail.
	};
	[[nodiscard]] LoadStats Loaded() const { return m_loaded; }

	// Adds a record unless one with the same key is known and appends it to the file. Returns
	// whether it was new.
	bool Add(const ComputeProgramRecord& record);
	// The records in the order they were first added.
	[[nodiscard]] std::vector<std::shared_ptr<const ComputeProgramRecord>> Records() const;
	[[nodiscard]] size_t                                                   Size() const;
	// Whether a record of the shader is known.
	[[nodiscard]] bool                         HasShader(uint64_t shader_hash) const;
	[[nodiscard]] const std::filesystem::path& Path() const { return m_path; }

	// The serialized body of a record (everything but the SPIR-V hash), exposed for tests.
	static std::vector<uint8_t> EncodeComputeBody(const ComputeProgramRecord& record);
	// Reads a body of EncodeComputeBody; the record's code stays null and `code_words` receives
	// the length of its code.
	static bool DecodeComputeBody(std::span<const uint8_t> body, ComputeProgramRecord& record,
	                              uint32_t& code_words);

private:
	struct CodeKey {
		uint64_t hash                             = 0;
		uint32_t words                            = 0;
		bool     operator==(const CodeKey&) const = default;
	};
	struct CodeKeyHash {
		size_t operator()(const CodeKey& key) const {
			return static_cast<size_t>(key.hash ^ (uint64_t {key.words} << 32u));
		}
	};

	void                        Load();
	bool                        ParseEntry(std::span<const uint8_t> payload, uint32_t version);
	bool                        Rewrite();
	void                        AppendEntry(std::span<const uint8_t> payload);
	static std::vector<uint8_t> EncodeCode(uint64_t hash, std::span<const uint32_t> code);

	std::filesystem::path                                    m_path;
	std::filesystem::path                                    m_previous;
	mutable std::mutex                                       m_mutex;
	std::ofstream                                            m_file;
	LoadStats                                                m_loaded;
	std::vector<std::shared_ptr<const ComputeProgramRecord>> m_records;
	std::unordered_set<uint64_t>                             m_keys;
	std::unordered_set<uint64_t>                             m_shaders;
	std::unordered_map<CodeKey, std::shared_ptr<const std::vector<uint32_t>>, CodeKeyHash> m_code;
};

// Precompile builds that started and never finished. The recompiler may end the process on code
// it cannot translate (a created shader no dispatch has used), a driver may never finish a
// compile, and Windows or a user may end the process during any build. A build that ended two
// processes, or that ran past the precompile time budget once (TimeOut), is quarantined: later
// runs skip it, so that it cannot end or block every later precompile run. A build that ended
// one process only is retried once, since that process may have ended for another reason.
// The file (_PipelineCache/<title>.v5.attempts) holds one line per event, "B <key>" before a
// build, "E <key>" after it and "T <key>" when it timed out, appended and flushed; loading keeps
// the keys of unfinished builds and rewrites the file with only those. While the file is missing,
// the attempts of the previous format (B and E only) are read.
//
// Thread-safe.
class PrecompileAttempts {
public:
	// An empty path keeps the attempts in memory only.
	explicit PrecompileAttempts(std::filesystem::path path     = {},
	                            std::filesystem::path previous = {});
	~PrecompileAttempts();
	KYTY_CLASS_NO_COPY(PrecompileAttempts);

	static std::filesystem::path PathFor(const std::filesystem::path& directory,
	                                     const std::string&           title_id);
	static std::filesystem::path PreviousPathFor(const std::filesystem::path& directory,
	                                             const std::string&           title_id);

	// Whether the key's build is quarantined: it timed out, or it began and did not finish in
	// two earlier processes.
	[[nodiscard]] bool   Failed(uint64_t key) const;
	// Whether the key's build timed out in an earlier process.
	[[nodiscard]] bool TimedOut(uint64_t key) const;
	// Whether the key's build did not finish in exactly one earlier process (it is retried).
	[[nodiscard]] bool   Retried(uint64_t key) const;
	[[nodiscard]] size_t FailedCount() const;
	[[nodiscard]] size_t RetriedCount() const;
	void                 Begin(uint64_t key);
	void                 End(uint64_t key);
	// The build has run past the time budget; it is quarantined in later processes.
	void TimeOut(uint64_t key);

private:
	struct State {
		uint32_t unfinished = 0;
		bool     timed_out  = false;
	};
	void Append(char event, uint64_t key);

	std::filesystem::path               m_path;
	mutable std::mutex                  m_mutex;
	std::ofstream                       m_file;
	std::unordered_map<uint64_t, State> m_loaded;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMMANIFEST_H_
