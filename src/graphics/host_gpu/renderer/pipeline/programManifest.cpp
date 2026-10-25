#include "graphics/host_gpu/renderer/pipeline/programManifest.h"

#include "graphics/host_gpu/renderer/pipeline/manifestFormat.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

using Specialization = ShaderRecompiler::IR::ResourceSpecialization;

// Serialized field by field; these trip when a stored struct changes so that the format version
// is revisited (a new field must be stored, or the version bumped).
static_assert(sizeof(ShaderComputeInputInfo) == 112);

constexpr std::array<char, 8> FileMagic          = {'K', 'y', 'P', 'r', 'o', 'g', 'M', 'f'};
constexpr uint8_t             KindCode           = 1;
constexpr uint8_t             KindComputeProgram = 2;
// An entry is a u32 payload size, the payload's XXH3 and the payload.
constexpr size_t EntryHeaderSize = sizeof(uint32_t) + sizeof(uint64_t);
// Bounds that no real entry reaches; a larger size field is corruption.
constexpr uint32_t MaxPayloadSize = 64u << 20u;
constexpr uint32_t MaxUserData    = 64;

using ManifestFormat::Reader;
using ManifestFormat::Writer;

void PutInput(Writer& w, const ShaderComputeInputInfo& input) {
	for (const auto value: input.threads_num)
		w.Put(value);
	w.Put(input.lds_size_dwords);
	w.Put(input.scratch_size_dwords);
	w.Put(input.wave_size);
	w.PutBool(input.async_compute);
	w.Put(input.float_mode);
	for (const auto value: input.group_id)
		w.PutBool(value);
	w.PutBool(input.dispatch_thread_dimensions);
	w.Put(static_cast<int32_t>(input.thread_ids_num));
	w.Put(static_cast<int32_t>(input.workgroup_register));
	w.PutBool(input.tg_size_en);
}

bool GetInput(Reader& r, ShaderComputeInputInfo& input) {
	for (auto& value: input.threads_num) {
		if (!r.Get(value)) return false;
	}
	if (!r.Get(input.lds_size_dwords) || !r.Get(input.scratch_size_dwords) ||
	    !r.Get(input.wave_size) || !r.GetBool(input.async_compute) || !r.Get(input.float_mode)) {
		return false;
	}
	for (auto& value: input.group_id) {
		if (!r.GetBool(value)) return false;
	}
	int32_t thread_ids_num     = 0;
	int32_t workgroup_register = 0;
	if (!r.GetBool(input.dispatch_thread_dimensions) || !r.Get(thread_ids_num) ||
	    !r.Get(workgroup_register) || !r.GetBool(input.tg_size_en)) {
		return false;
	}
	input.thread_ids_num     = thread_ids_num;
	input.workgroup_register = workgroup_register;
	return true;
}

// Version 2 stored no mapping-header flags (every key map offset was a module constant),
// version 3 no resident image heap flags (there were no heaps).
constexpr uint32_t FirstHeaderVersion = 3;
constexpr uint32_t FirstHeapVersion   = 4;

std::vector<uint8_t> FileHeader(uint32_t version = ProgramManifest::FormatVersion) {
	std::vector<uint8_t> header(FileMagic.begin(), FileMagic.end());
	Writer               writer(header);
	writer.Put(version);
	return header;
}

// Whether a specialization reads a key map at an offset baked into the module, which the
// materialization no longer produces: such a module can never be used again.
bool HasConstantKeyMap(const Specialization& specialization) {
	const auto constant = [](const auto& entries) {
		for (size_t index = 0; index < entries.size(); ++index) {
			const auto& entry = entries[index];
			if (entry.indirect_root == index && !entry.indirect_mapping_header) return true;
		}
		return false;
	};
	for (size_t index = 0; index < specialization.images.size(); ++index) {
		const auto& image = specialization.images[index];
		if (image.indirect_root == index && !image.bindless && !image.indirect_mapping_header)
			return true;
	}
	return constant(specialization.buffers) || constant(specialization.samplers);
}

} // namespace

ShaderComputeInputInfo ComputeProgramRecord::StaticInput(const ShaderComputeInputInfo& input) {
	ShaderComputeInputInfo out {};
	std::ranges::copy(input.threads_num, out.threads_num);
	out.lds_size_dwords     = input.lds_size_dwords;
	out.scratch_size_dwords = input.scratch_size_dwords;
	out.wave_size           = input.wave_size;
	out.async_compute       = input.async_compute;
	out.float_mode          = input.float_mode;
	std::ranges::copy(input.group_id, out.group_id);
	out.dispatch_thread_dimensions = input.dispatch_thread_dimensions;
	out.thread_ids_num             = input.thread_ids_num;
	out.workgroup_register         = input.workgroup_register;
	out.tg_size_en                 = input.tg_size_en;
	return out;
}

uint64_t ComputeProgramRecord::Key() const {
	const auto body = ProgramManifest::EncodeComputeBody(*this);
	return XXH3_64bits(body.data(), body.size());
}

std::vector<uint8_t> ProgramManifest::EncodeComputeBody(const ComputeProgramRecord& record) {
	std::vector<uint8_t> out;
	out.reserve(128 + record.specialization.buffers.size() * 32u +
	            record.specialization.images.size() * 40u);
	Writer w(out);
	w.Put(KindComputeProgram);
	w.Put(record.shader_hash);
	w.Put(static_cast<uint32_t>(record.code != nullptr ? record.code->size() : 0u));
	w.Put(record.user_data_count);
	PutInput(w, record.input);
	w.Put(record.push_data_cursor);
	ManifestFormat::PutSpecialization(w, record.specialization);
	return out;
}

bool ProgramManifest::DecodeComputeBody(std::span<const uint8_t> body, ComputeProgramRecord& record,
                                        uint32_t& code_words) {
	Reader  r(body);
	uint8_t kind = 0;
	record       = {};
	return r.Get(kind) && kind == KindComputeProgram && r.Get(record.shader_hash) &&
	       r.Get(code_words) && r.Get(record.user_data_count) &&
	       record.user_data_count <= MaxUserData && GetInput(r, record.input) &&
	       r.Get(record.push_data_cursor) &&
	       ManifestFormat::GetSpecialization(r, record.specialization) && r.AtEnd();
}

std::vector<uint8_t> ProgramManifest::EncodeCode(uint64_t hash, std::span<const uint32_t> code) {
	std::vector<uint8_t> out;
	out.reserve(16 + code.size_bytes());
	Writer w(out);
	w.Put(KindCode);
	w.Put(hash);
	w.Put(static_cast<uint32_t>(code.size()));
	w.PutWords(code);
	return out;
}

std::filesystem::path ProgramManifest::PathFor(const std::filesystem::path& directory,
                                               const std::string&           title_id) {
	// The format is part of the name: builds of another format share the directory.
	return directory / (title_id + ".v4.programs");
}

std::filesystem::path ProgramManifest::PreviousPathFor(const std::filesystem::path& directory,
                                                       const std::string&           title_id) {
	return directory / (title_id + ".v3.programs");
}

std::filesystem::path ProgramManifest::CreatedPathFor(const std::filesystem::path& directory,
                                                      const std::string&           title_id) {
	return directory / (title_id + ".v4.created");
}

ProgramManifest::ProgramManifest(std::filesystem::path path, std::filesystem::path previous)
    : m_path(std::move(path)), m_previous(std::move(previous)) {
	if (!m_path.empty()) {
		Load();
	}
}

ProgramManifest::~ProgramManifest() = default;

bool ProgramManifest::ParseEntry(std::span<const uint8_t> payload, uint32_t version) {
	Reader  r(payload);
	uint8_t kind = 0;
	if (!r.Get(kind)) {
		return false;
	}
	if (kind == KindCode) {
		uint64_t hash  = 0;
		uint32_t words = 0;
		auto     code  = std::make_shared<std::vector<uint32_t>>();
		if (!r.Get(hash) || !r.Get(words) || !r.GetWords(*code, words) || !r.AtEnd()) {
			return false;
		}
		m_code.try_emplace({hash, words}, std::move(code));
		return true;
	}
	if (kind != KindComputeProgram) {
		return false;
	}
	auto     record     = std::make_shared<ComputeProgramRecord>();
	uint32_t code_words = 0;
	if (!r.Get(record->shader_hash) || !r.Get(code_words) || !r.Get(record->user_data_count) ||
	    record->user_data_count > MaxUserData || !GetInput(r, record->input) ||
	    !r.Get(record->push_data_cursor) ||
	    !ManifestFormat::GetSpecialization(r, record->specialization,
	                                       {.mapping_header = version >= FirstHeaderVersion,
	                                        .image_heap     = version >= FirstHeapVersion}) ||
	    !r.Get(record->spirv_hash) || !r.AtEnd()) {
		return false;
	}
	// An older record keeps its program only while the module it names can still be needed.
	if (version < FirstHeaderVersion && HasConstantKeyMap(record->specialization)) {
		return true;
	}
	// A program whose code entry was lost cannot be translated; skip it, the entry is intact.
	const auto code = m_code.find({record->shader_hash, code_words});
	if (code == m_code.end()) {
		return true;
	}
	record->code = code->second;
	if (m_keys.insert(record->Key()).second) {
		m_shaders.insert(record->shader_hash);
		m_records.push_back(std::move(record));
	}
	return true;
}

void ProgramManifest::Load() {
	std::vector<uint8_t> data;
	const auto           read = [&data](const std::filesystem::path& path) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (file) {
			const auto size = static_cast<std::streamoff>(file.tellg());
			if (size > 0) {
				data.resize(static_cast<size_t>(size));
				file.seekg(0);
				if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
					data.clear();
				}
			}
		}
	};
	read(m_path);
	std::error_code error;
	if (data.empty() && !m_previous.empty() && !std::filesystem::exists(m_path, error)) {
		read(m_previous);
	}
	const auto header  = FileHeader();
	size_t     offset  = 0;
	uint32_t   version = ProgramManifest::FormatVersion;
	bool       valid   = false;
	// Older versions' records are still readable; the file is rewritten in this one.
	for (uint32_t candidate = FirstHeaderVersion - 1u; candidate <= ProgramManifest::FormatVersion;
	     candidate++) {
		const auto expected = FileHeader(candidate);
		if (data.size() >= expected.size() &&
		    std::equal(expected.begin(), expected.end(), data.begin())) {
			valid   = true;
			version = candidate;
		}
	}
	if (valid) {
		offset = header.size();
		while (offset < data.size()) {
			uint32_t size = 0;
			uint64_t hash = 0;
			if (data.size() - offset < EntryHeaderSize) {
				break;
			}
			std::memcpy(&size, data.data() + offset, sizeof(size));
			std::memcpy(&hash, data.data() + offset + sizeof(size), sizeof(hash));
			if (size == 0 || size > MaxPayloadSize ||
			    data.size() - offset - EntryHeaderSize < size) {
				break;
			}
			const auto payload = std::span(data).subspan(offset + EntryHeaderSize, size);
			if (XXH3_64bits(payload.data(), payload.size()) != hash ||
			    !ParseEntry(payload, version)) {
				break;
			}
			offset += EntryHeaderSize + size;
		}
	}
	m_loaded.records      = m_records.size();
	m_loaded.code_entries = m_code.size();
	if (!valid || offset != data.size() || version != ProgramManifest::FormatVersion) {
		m_loaded.dropped_bytes = valid ? data.size() - offset : data.size();
		m_loaded.rewritten     = !data.empty();
		// A file that cannot be rewritten is not appended to: new entries would follow the
		// unreadable ones. The manifest then lives in memory for this session.
		if (!Rewrite()) {
			return;
		}
	}
	m_file.open(m_path, std::ios::binary | std::ios::app);
}

bool ProgramManifest::Rewrite() {
	// Only the entries that were read are written back: every record's code, then the records.
	std::vector<uint8_t> out    = FileHeader();
	auto                 append = [&out](std::span<const uint8_t> payload) {
		Writer w(out);
		w.Put(static_cast<uint32_t>(payload.size()));
		w.Put(static_cast<uint64_t>(XXH3_64bits(payload.data(), payload.size())));
		out.insert(out.end(), payload.begin(), payload.end());
	};
	std::unordered_set<const std::vector<uint32_t>*> written;
	for (const auto& record: m_records) {
		if (written.insert(record->code.get()).second) {
			append(EncodeCode(record->shader_hash, *record->code));
		}
		auto body = EncodeComputeBody(*record);
		Writer(body).Put(record->spirv_hash);
		append(body);
	}
	auto temp = m_path;
	temp += ".tmp";
	std::error_code error;
	if (!m_path.parent_path().empty()) {
		std::filesystem::create_directories(m_path.parent_path(), error);
	}
	{
		std::ofstream file(temp, std::ios::binary | std::ios::trunc);
		file.write(reinterpret_cast<const char*>(out.data()),
		           static_cast<std::streamsize>(out.size()));
		if (!file.flush()) {
			return false;
		}
	}
	std::filesystem::rename(temp, m_path, error);
	if (error) {
		return false;
	}
	m_code.clear();
	for (const auto& record: m_records) {
		m_code.try_emplace({record->shader_hash, static_cast<uint32_t>(record->code->size())},
		                   record->code);
	}
	return true;
}

void ProgramManifest::AppendEntry(std::span<const uint8_t> payload) {
	if (!m_file.is_open()) {
		return;
	}
	std::vector<uint8_t> header;
	Writer               w(header);
	w.Put(static_cast<uint32_t>(payload.size()));
	w.Put(static_cast<uint64_t>(XXH3_64bits(payload.data(), payload.size())));
	m_file.write(reinterpret_cast<const char*>(header.data()),
	             static_cast<std::streamsize>(header.size()));
	m_file.write(reinterpret_cast<const char*>(payload.data()),
	             static_cast<std::streamsize>(payload.size()));
	// Flushed per entry: a session that ends in a crash keeps what it recorded.
	m_file.flush();
}

bool ProgramManifest::Add(const ComputeProgramRecord& record) {
	if (record.code == nullptr || record.code->empty() || record.user_data_count > MaxUserData) {
		return false;
	}
	auto            body = EncodeComputeBody(record);
	const auto      key  = XXH3_64bits(body.data(), body.size());
	std::lock_guard lock(m_mutex);
	if (!m_keys.insert(key).second) {
		return false;
	}
	auto       stored   = std::make_shared<ComputeProgramRecord>(record);
	const auto code_key = CodeKey {record.shader_hash, static_cast<uint32_t>(record.code->size())};
	if (const auto code = m_code.find(code_key); code != m_code.end()) {
		stored->code = code->second;
	} else {
		m_code.emplace(code_key, record.code);
		AppendEntry(EncodeCode(record.shader_hash, *record.code));
	}
	Writer(body).Put(record.spirv_hash);
	AppendEntry(body);
	m_shaders.insert(record.shader_hash);
	m_records.push_back(std::move(stored));
	return true;
}

bool ProgramManifest::HasShader(uint64_t shader_hash) const {
	std::lock_guard lock(m_mutex);
	return m_shaders.contains(shader_hash);
}

std::vector<std::shared_ptr<const ComputeProgramRecord>> ProgramManifest::Records() const {
	std::lock_guard lock(m_mutex);
	return m_records;
}

size_t ProgramManifest::Size() const {
	std::lock_guard lock(m_mutex);
	return m_records.size();
}

PrecompileAttempts::PrecompileAttempts(std::filesystem::path path, std::filesystem::path previous)
    : m_path(std::move(path)) {
	if (m_path.empty()) {
		return;
	}
	std::error_code error;
	const auto      source =
	    !previous.empty() && !std::filesystem::exists(m_path, error) ? previous : m_path;
	{
		std::ifstream file(source);
		std::string   line;
		while (std::getline(file, line)) {
			// A torn last line is ignored.
			if (line.size() < 3 || line[1] != ' ' ||
			    (line[0] != 'B' && line[0] != 'E' && line[0] != 'T')) {
				continue;
			}
			char*      end = nullptr;
			const auto key = std::strtoull(line.c_str() + 2, &end, 16);
			if (end == line.c_str() + 2 || *end != '\0') {
				continue;
			}
			switch (line[0]) {
				// One unfinished begin per process: a process builds a key once.
				case 'B': m_loaded[key].unfinished++; break;
				case 'T': m_loaded[key].timed_out = true; break;
				default: m_loaded.erase(key); break;
			}
		}
	}
	m_file.open(m_path, std::ios::trunc);
	for (const auto& [key, state]: m_loaded) {
		if (state.timed_out) {
			m_file << "T " << std::hex << key << '\n';
		}
		for (uint32_t i = 0; i < std::min(state.unfinished, 2u); i++) {
			m_file << "B " << std::hex << key << '\n';
		}
	}
	m_file.flush();
}

PrecompileAttempts::~PrecompileAttempts() = default;

std::filesystem::path PrecompileAttempts::PathFor(const std::filesystem::path& directory,
                                                  const std::string&           title_id) {
	return directory / (title_id + ".v5.attempts");
}

std::filesystem::path PrecompileAttempts::PreviousPathFor(const std::filesystem::path& directory,
                                                          const std::string&           title_id) {
	return directory / (title_id + ".v4.attempts");
}

bool PrecompileAttempts::Failed(uint64_t key) const {
	std::lock_guard lock(m_mutex);
	const auto      state = m_loaded.find(key);
	return state != m_loaded.end() && (state->second.timed_out || state->second.unfinished >= 2u);
}

bool PrecompileAttempts::TimedOut(uint64_t key) const {
	std::lock_guard lock(m_mutex);
	const auto      state = m_loaded.find(key);
	return state != m_loaded.end() && state->second.timed_out;
}

bool PrecompileAttempts::Retried(uint64_t key) const {
	std::lock_guard lock(m_mutex);
	const auto      state = m_loaded.find(key);
	return state != m_loaded.end() && !state->second.timed_out && state->second.unfinished == 1u;
}

size_t PrecompileAttempts::FailedCount() const {
	std::lock_guard lock(m_mutex);
	return static_cast<size_t>(std::ranges::count_if(m_loaded, [](const auto& entry) {
		return entry.second.timed_out || entry.second.unfinished >= 2u;
	}));
}

size_t PrecompileAttempts::RetriedCount() const {
	std::lock_guard lock(m_mutex);
	return static_cast<size_t>(std::ranges::count_if(m_loaded, [](const auto& entry) {
		return !entry.second.timed_out && entry.second.unfinished == 1u;
	}));
}

void PrecompileAttempts::Begin(uint64_t key) {
	Append('B', key);
}

void PrecompileAttempts::End(uint64_t key) {
	Append('E', key);
}

void PrecompileAttempts::TimeOut(uint64_t key) {
	Append('T', key);
}

void PrecompileAttempts::Append(char event, uint64_t key) {
	std::lock_guard lock(m_mutex);
	if (!m_file.is_open()) {
		return;
	}
	// Flushed per event: the process may end in the build that follows.
	m_file << event << ' ' << std::hex << key << '\n';
	m_file.flush();
}

} // namespace Libs::Graphics
