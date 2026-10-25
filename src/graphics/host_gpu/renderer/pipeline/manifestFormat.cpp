#include "graphics/host_gpu/renderer/pipeline/manifestFormat.h"

#include <system_error>
#include <xxhash.h>

namespace Libs::Graphics::ManifestFormat {

using Specialization = ShaderRecompiler::IR::ResourceSpecialization;

// Serialized field by field; these trip when a stored struct changes so that the formats are
// revisited (a new field must be stored, or the versions bumped).
static_assert(sizeof(Specialization::Buffer) == 32);
static_assert(sizeof(Specialization::Image) == 40);
static_assert(sizeof(Specialization::Sampler) == 16);

namespace {
constexpr size_t EntryHeaderSize = sizeof(uint32_t) + sizeof(uint64_t);
} // namespace

void PutSpecialization(Writer& w, const Specialization& specialization) {
	w.Put(static_cast<uint32_t>(specialization.buffers.size()));
	for (const auto& buffer: specialization.buffers) {
		w.Put(buffer.packed_stride);
		w.PutEnum(buffer.descriptor_format);
		w.Put(buffer.descriptor_swizzle);
		w.PutBool(buffer.zero_stride_oob);
		w.PutBool(buffer.scalar_empty);
		w.Put(buffer.indirect_root);
		w.Put(buffer.indirect_mapping_offset);
		w.Put(buffer.indirect_search_iterations);
		w.PutBool(buffer.unbound);
		w.PutBool(buffer.indirect_mapping_header);
	}
	w.Put(static_cast<uint32_t>(specialization.images.size()));
	for (const auto& image: specialization.images) {
		w.PutEnum(image.numeric_class);
		w.PutEnum(image.dimension);
		w.Put(image.mip_count);
		w.PutEnum(image.conversion_format);
		w.Put(image.shader_swizzle);
		w.Put(image.indirect_root);
		w.Put(image.indirect_mapping_offset);
		w.Put(image.indirect_search_iterations);
		w.PutBool(image.cube);
		w.PutBool(image.fmask);
		w.PutBool(image.bindless);
		w.PutBool(image.bindless_byte_keys);
		w.PutBool(image.indirect_mapping_header);
		w.PutBool(image.heap);
	}
	w.Put(static_cast<uint32_t>(specialization.samplers.size()));
	for (const auto& sampler: specialization.samplers) {
		w.Put(sampler.indirect_root);
		w.Put(sampler.indirect_mapping_offset);
		w.Put(sampler.indirect_search_iterations);
		w.PutBool(sampler.indirect_mapping_header);
	}
}

bool GetSpecialization(Reader& r, Specialization& specialization, SpecializationFields fields) {
	const auto get_header = [&](bool& value) {
		value = false;
		return !fields.mapping_header || r.GetBool(value);
	};
	const auto get_heap = [&](bool& value) {
		value = false;
		return !fields.image_heap || r.GetBool(value);
	};
	uint32_t count = 0;
	if (!r.Get(count) || count > MaxResources) {
		return false;
	}
	specialization.buffers.resize(count);
	for (auto& buffer: specialization.buffers) {
		if (!r.Get(buffer.packed_stride) || !r.GetEnum(buffer.descriptor_format) ||
		    !r.Get(buffer.descriptor_swizzle) || !r.GetBool(buffer.zero_stride_oob) ||
		    !r.GetBool(buffer.scalar_empty) || !r.Get(buffer.indirect_root) ||
		    !r.Get(buffer.indirect_mapping_offset) || !r.Get(buffer.indirect_search_iterations) ||
		    !r.GetBool(buffer.unbound) || !get_header(buffer.indirect_mapping_header)) {
			return false;
		}
	}
	if (!r.Get(count) || count > MaxResources) {
		return false;
	}
	specialization.images.resize(count);
	for (auto& image: specialization.images) {
		if (!r.GetEnum(image.numeric_class) || !r.GetEnum(image.dimension) ||
		    !r.Get(image.mip_count) || !r.GetEnum(image.conversion_format) ||
		    !r.Get(image.shader_swizzle) || !r.Get(image.indirect_root) ||
		    !r.Get(image.indirect_mapping_offset) || !r.Get(image.indirect_search_iterations) ||
		    !r.GetBool(image.cube) || !r.GetBool(image.fmask) || !r.GetBool(image.bindless) ||
		    !r.GetBool(image.bindless_byte_keys) || !get_header(image.indirect_mapping_header) ||
		    !get_heap(image.heap)) {
			return false;
		}
	}
	if (!r.Get(count) || count > MaxResources) {
		return false;
	}
	specialization.samplers.resize(count);
	for (auto& sampler: specialization.samplers) {
		if (!r.Get(sampler.indirect_root) || !r.Get(sampler.indirect_mapping_offset) ||
		    !r.Get(sampler.indirect_search_iterations) ||
		    !get_header(sampler.indirect_mapping_header)) {
			return false;
		}
	}
	return true;
}

std::vector<uint8_t> EntryFileHeader(const Magic& magic, uint32_t version) {
	std::vector<uint8_t> header(magic.begin(), magic.end());
	Writer(header).Put(version);
	return header;
}

void FrameEntry(std::vector<uint8_t>& out, std::span<const uint8_t> payload) {
	Writer w(out);
	w.Put(static_cast<uint32_t>(payload.size()));
	w.Put(static_cast<uint64_t>(XXH3_64bits(payload.data(), payload.size())));
	out.insert(out.end(), payload.begin(), payload.end());
}

EntryFileRead ReadEntryFile(const std::filesystem::path& path, const Magic& magic,
                            const std::function<bool(std::span<const uint8_t>, uint32_t)>& parse) {
	EntryFileRead        result;
	std::vector<uint8_t> data;
	{
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file) {
			return result;
		}
		const auto size = static_cast<std::streamoff>(file.tellg());
		if (size > 0) {
			data.resize(static_cast<size_t>(size));
			file.seekg(0);
			if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
				data.clear();
			}
		}
	}
	result.size            = data.size();
	constexpr size_t start = sizeof(Magic) + sizeof(uint32_t);
	if (data.size() < start || std::memcmp(data.data(), magic.data(), magic.size()) != 0) {
		return result;
	}
	result.valid = true;
	std::memcpy(&result.version, data.data() + magic.size(), sizeof(uint32_t));
	size_t offset = start;
	while (data.size() - offset >= EntryHeaderSize) {
		uint32_t size = 0;
		uint64_t hash = 0;
		std::memcpy(&size, data.data() + offset, sizeof(size));
		std::memcpy(&hash, data.data() + offset + sizeof(size), sizeof(hash));
		if (size == 0 || size > MaxPayloadSize || data.size() - offset - EntryHeaderSize < size) {
			break;
		}
		const auto payload = std::span(data).subspan(offset + EntryHeaderSize, size);
		if (XXH3_64bits(payload.data(), payload.size()) != hash ||
		    !parse(payload, result.version)) {
			break;
		}
		offset += EntryHeaderSize + size;
	}
	result.intact = offset;
	return result;
}

bool WriteEntryFile(const std::filesystem::path& path, const Magic& magic, uint32_t version,
                    std::span<const std::vector<uint8_t>> payloads) {
	auto out = EntryFileHeader(magic, version);
	for (const auto& payload: payloads) {
		FrameEntry(out, payload);
	}
	std::error_code error;
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path(), error);
	}
	auto temp = path;
	temp += ".tmp";
	{
		std::ofstream file(temp, std::ios::binary | std::ios::trunc);
		file.write(reinterpret_cast<const char*>(out.data()),
		           static_cast<std::streamsize>(out.size()));
		if (!file.flush()) {
			return false;
		}
	}
	std::filesystem::rename(temp, path, error);
	return !error;
}

void EntryAppender::Append(std::span<const uint8_t> payload) {
	if (!m_file.is_open()) {
		return;
	}
	m_buffer.clear();
	FrameEntry(m_buffer, payload);
	m_file.write(reinterpret_cast<const char*>(m_buffer.data()),
	             static_cast<std::streamsize>(m_buffer.size()));
	m_file.flush();
}

} // namespace Libs::Graphics::ManifestFormat
