#include "libs/libVideoDec2.cpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

namespace {
std::vector<Loader::SymbolResolve> registrations;
}
namespace Loader {
void SymbolDatabase::Add(const SymbolResolve& symbol, uint64_t, const std::string&) {
	registrations.push_back(symbol);
}
} // namespace Loader

namespace {
using namespace Libs::VideoDec2;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "VideoDec2Tests:%d: %s\n", __LINE__, #condition);                 \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

constexpr uint8_t GUARD = 0xa7;
constexpr size_t  GUARD_SIZE = 4096;

uint32_t AlignUp(uint32_t value, uint32_t alignment) {
	return (value + alignment - 1u) / alignment * alignment;
}

// Guest memory block: aligned like the game's allocators and followed by guard bytes that must
// stay untouched, so a write past the size the API reported is caught.
struct GuestMemory {
	std::vector<uint8_t> storage;
	uint8_t*             data;
	size_t               size;
	GuestMemory(size_t bytes, size_t alignment): storage(bytes + GUARD_SIZE + alignment), size(bytes) {
		const auto misalignment = reinterpret_cast<uintptr_t>(storage.data()) % alignment;
		data = storage.data() + (alignment - misalignment) % alignment;
		std::memset(data, GUARD, bytes + GUARD_SIZE);
	}
	void CheckGuard() const {
		for (size_t i = 0; i < GUARD_SIZE; ++i) CHECK(data[size + i] == GUARD);
	}
};

// --- A generated H.264 stream: High profile IDR pictures made only of I_PCM macroblocks, so every
// decoded sample is known exactly without an encoder.

class BitWriter {
public:
	void Bit(uint32_t bit) {
		m_current = static_cast<uint8_t>((m_current << 1u) | (bit & 1u));
		if (++m_count == 8) {
			bytes.push_back(m_current);
			m_current = 0;
			m_count   = 0;
		}
	}
	void Bits(uint32_t value, int count) {
		for (int i = count - 1; i >= 0; --i) Bit(value >> static_cast<uint32_t>(i));
	}
	void Ue(uint32_t value) {
		const uint64_t code   = static_cast<uint64_t>(value) + 1u;
		int            length = 0;
		while ((code >> (length + 1)) != 0) ++length;
		Bits(0, length);
		for (int i = length; i >= 0; --i) Bit(static_cast<uint32_t>(code >> i));
	}
	void Se(int32_t value) { Ue(value <= 0 ? static_cast<uint32_t>(-2 * value) : static_cast<uint32_t>(2 * value - 1)); }
	void AlignZero() {
		while (m_count != 0) Bit(0);
	}
	void Trailing() {
		Bit(1);
		AlignZero();
	}
	void Byte(uint8_t value) {
		CHECK(m_count == 0);
		bytes.push_back(value);
	}
	std::vector<uint8_t> bytes;

private:
	uint8_t m_current = 0;
	int     m_count   = 0;
};

void AppendNal(std::vector<uint8_t>* stream, uint8_t header, const std::vector<uint8_t>& payload) {
	stream->insert(stream->end(), {0, 0, 0, 1, header});
	int zeros = 0;
	for (const uint8_t byte: payload) {
		if (zeros >= 2 && byte <= 3) {
			stream->push_back(3); // emulation prevention
			zeros = 0;
		}
		stream->push_back(byte);
		zeros = byte == 0 ? zeros + 1 : 0;
	}
}

uint8_t PatternY(uint32_t x, uint32_t y, uint32_t frame) {
	return static_cast<uint8_t>(16u + (x * 7u + y * 3u + frame * 29u) % 220u);
}
uint8_t PatternU(uint32_t x, uint32_t y, uint32_t frame) {
	return static_cast<uint8_t>(32u + (x * 5u + y * 11u + frame * 17u) % 190u);
}
uint8_t PatternV(uint32_t x, uint32_t y, uint32_t frame) {
	return static_cast<uint8_t>(40u + (x * 3u + y * 13u + frame * 7u) % 180u);
}

struct GeneratedStream {
	uint32_t                          width;
	uint32_t                          height;
	std::vector<std::vector<uint8_t>> access_units;
};

GeneratedStream GeneratePcmStream(uint32_t width, uint32_t height, uint32_t frames) {
	CHECK(width % 16 == 0 && height % 2 == 0);
	const uint32_t mbs_w = width / 16;
	const uint32_t mbs_h = (height + 15) / 16;
	GeneratedStream result {width, height, {}};
	std::vector<uint8_t> headers;
	{
		BitWriter sps;
		sps.Bits(100, 8); // profile_idc: High
		sps.Bits(0, 8);   // constraint flags
		sps.Bits(42, 8);  // level_idc
		sps.Ue(0);        // seq_parameter_set_id
		sps.Ue(1);        // chroma_format_idc: 4:2:0
		sps.Ue(0);        // bit_depth_luma_minus8
		sps.Ue(0);        // bit_depth_chroma_minus8
		sps.Bit(0);       // qpprime_y_zero_transform_bypass_flag
		sps.Bit(0);       // seq_scaling_matrix_present_flag
		sps.Ue(0);        // log2_max_frame_num_minus4
		sps.Ue(2);        // pic_order_cnt_type: output order is decoding order
		sps.Ue(1);        // max_num_ref_frames
		sps.Bit(0);       // gaps_in_frame_num_value_allowed_flag
		sps.Ue(mbs_w - 1);
		sps.Ue(mbs_h - 1);
		sps.Bit(1); // frame_mbs_only_flag
		sps.Bit(1); // direct_8x8_inference_flag
		const uint32_t crop_bottom = mbs_h * 16 - height;
		sps.Bit(crop_bottom != 0 ? 1 : 0);
		if (crop_bottom != 0) {
			sps.Ue(0);
			sps.Ue(0);
			sps.Ue(0);
			sps.Ue(crop_bottom / 2); // in chroma rows
		}
		sps.Bit(0); // vui_parameters_present_flag
		sps.Trailing();
		AppendNal(&headers, 0x67, sps.bytes);

		BitWriter pps;
		pps.Ue(0);  // pic_parameter_set_id
		pps.Ue(0);  // seq_parameter_set_id
		pps.Bit(0); // entropy_coding_mode_flag: CAVLC
		pps.Bit(0); // bottom_field_pic_order_in_frame_present_flag
		pps.Ue(0);  // num_slice_groups_minus1
		pps.Ue(0);  // num_ref_idx_l0_default_active_minus1
		pps.Ue(0);  // num_ref_idx_l1_default_active_minus1
		pps.Bit(0); // weighted_pred_flag
		pps.Bits(0, 2);
		pps.Se(0);  // pic_init_qp_minus26
		pps.Se(0);  // pic_init_qs_minus26
		pps.Se(0);  // chroma_qp_index_offset
		pps.Bit(1); // deblocking_filter_control_present_flag
		pps.Bit(0); // constrained_intra_pred_flag
		pps.Bit(0); // redundant_pic_cnt_present_flag
		pps.Trailing();
		AppendNal(&headers, 0x68, pps.bytes);
	}
	for (uint32_t frame = 0; frame < frames; ++frame) {
		BitWriter slice;
		slice.Ue(0);         // first_mb_in_slice
		slice.Ue(7);         // slice_type: I, all slices
		slice.Ue(0);         // pic_parameter_set_id
		slice.Bits(0, 4);    // frame_num
		slice.Ue(frame % 2); // idr_pic_id differs between consecutive IDR pictures
		slice.Bit(0);        // no_output_of_prior_pics_flag
		slice.Bit(0);        // long_term_reference_flag
		slice.Se(0);         // slice_qp_delta
		slice.Ue(1);         // disable_deblocking_filter_idc
		for (uint32_t my = 0; my < mbs_h; ++my) {
			for (uint32_t mx = 0; mx < mbs_w; ++mx) {
				slice.Ue(25); // mb_type: I_PCM
				slice.AlignZero();
				for (uint32_t y = 0; y < 16; ++y)
					for (uint32_t x = 0; x < 16; ++x) slice.Byte(PatternY(mx * 16 + x, my * 16 + y, frame));
				for (uint32_t y = 0; y < 8; ++y)
					for (uint32_t x = 0; x < 8; ++x) slice.Byte(PatternU(mx * 8 + x, my * 8 + y, frame));
				for (uint32_t y = 0; y < 8; ++y)
					for (uint32_t x = 0; x < 8; ++x) slice.Byte(PatternV(mx * 8 + x, my * 8 + y, frame));
			}
		}
		slice.Trailing();
		std::vector<uint8_t> au = frame == 0 ? headers : std::vector<uint8_t> {};
		AppendNal(&au, 0x65, slice.bytes);
		result.access_units.push_back(std::move(au));
	}
	return result;
}

// --- Annex B access-unit splitting for real elementary streams.

std::vector<std::vector<uint8_t>> SplitAccessUnits(const std::vector<uint8_t>& stream) {
	std::vector<size_t> starts;
	for (size_t i = 0; i + 3 < stream.size(); ++i) {
		if (stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1) {
			starts.push_back(i + 3);
			i += 2;
		}
	}
	std::vector<std::vector<uint8_t>> units;
	std::vector<uint8_t>              current;
	bool                              has_picture = false;
	for (size_t n = 0; n + 1 < starts.size(); ++n) {
		const size_t begin = starts[n];
		size_t       end   = starts[n + 1] - 3;
		while (end > begin && stream[end - 1] == 0) --end;
		const uint8_t type     = stream[begin] & 0x1fu;
		const bool    vcl      = type == 1 || type == 5;
		const bool    first_mb = vcl && begin + 1 < end && (stream[begin + 1] & 0x80u) != 0;
		if (has_picture && ((type >= 6 && type <= 9) || first_mb)) {
			units.push_back(std::move(current));
			current.clear();
			has_picture = false;
		}
		current.insert(current.end(), {0, 0, 0, 1});
		current.insert(current.end(), stream.begin() + static_cast<ptrdiff_t>(begin),
		               stream.begin() + static_cast<ptrdiff_t>(end));
		has_picture = has_picture || vcl;
	}
	// The last unit may be cut by the read limit, so it is dropped.
	return units;
}

// --- Reference: the same access units decoded directly by FFmpeg.

struct ReferenceFrame {
	uint32_t             width  = 0;
	uint32_t             height = 0;
	std::vector<uint8_t> y, u, v; // tightly packed planes
};

std::vector<ReferenceFrame> DecodeReference(const std::vector<std::vector<uint8_t>>& units) {
	const AVCodec*  codec   = avcodec_find_decoder(AV_CODEC_ID_H264);
	AVCodecContext* context = avcodec_alloc_context3(codec);
	CHECK(context != nullptr && avcodec_open2(context, codec, nullptr) == 0);
	AVPacket*                   packet = av_packet_alloc();
	AVFrame*                    frame  = av_frame_alloc();
	std::vector<ReferenceFrame> frames;
	auto receive = [&]() {
		while (avcodec_receive_frame(context, frame) == 0) {
			CHECK(frame->format == AV_PIX_FMT_YUV420P || frame->format == AV_PIX_FMT_YUVJ420P);
			ReferenceFrame copy;
			copy.width  = static_cast<uint32_t>(frame->width);
			copy.height = static_cast<uint32_t>(frame->height);
			for (uint32_t y = 0; y < copy.height; ++y) {
				const uint8_t* row = frame->data[0] + static_cast<ptrdiff_t>(y) * frame->linesize[0];
				copy.y.insert(copy.y.end(), row, row + copy.width);
			}
			for (uint32_t y = 0; y < (copy.height + 1) / 2; ++y) {
				const uint8_t* u = frame->data[1] + static_cast<ptrdiff_t>(y) * frame->linesize[1];
				const uint8_t* v = frame->data[2] + static_cast<ptrdiff_t>(y) * frame->linesize[2];
				copy.u.insert(copy.u.end(), u, u + (copy.width + 1) / 2);
				copy.v.insert(copy.v.end(), v, v + (copy.width + 1) / 2);
			}
			frames.push_back(std::move(copy));
			av_frame_unref(frame);
		}
	};
	for (const auto& unit: units) {
		CHECK(av_new_packet(packet, static_cast<int>(unit.size())) == 0);
		std::memcpy(packet->data, unit.data(), unit.size());
		CHECK(avcodec_send_packet(context, packet) == 0);
		av_packet_unref(packet);
		receive();
	}
	CHECK(avcodec_send_packet(context, nullptr) == 0);
	receive();
	av_frame_free(&frame);
	av_packet_free(&packet);
	avcodec_free_context(&context);
	return frames;
}

// --- Videodec2 driven the way Ghost of Yotei (PPSA26344) drives it.

Videodec2DecoderConfigInfo GameConfig(int32_t max_width, int32_t max_height) {
	Videodec2DecoderConfigInfo config {};
	config.this_size                  = sizeof(config);
	config.resource_type              = VIDEODEC2_RESOURCE_TYPE_COMPUTE;
	config.codec_type                 = 1;   // AVC
	config.profile                    = 100; // High
	config.max_level                  = 42;
	config.max_frame_width            = max_width;
	config.max_frame_height           = max_height;
	config.max_dpb_frame_count        = -1;
	config.decode_input_queue_depth   = 4;
	config.cpu_affinity_mask          = 0;
	config.cpu_thread_priority        = -1;
	config.optimize_progressive_video = true;
	config.check_memory_type          = true;
	return config;
}

struct Session {
	Videodec2DecoderConfigInfo config;
	Videodec2ComputeMemoryInfo compute_memory {sizeof(Videodec2ComputeMemoryInfo)};
	Videodec2DecoderMemoryInfo memory {sizeof(Videodec2DecoderMemoryInfo)};
	std::vector<GuestMemory>   blocks;
	std::vector<GuestMemory>   frames;
	Videodec2ComputeQueue      queue   = nullptr;
	Videodec2Decoder           decoder = nullptr;

	Session(int32_t max_width, int32_t max_height): config(GameConfig(max_width, max_height)) {
		CHECK(QueryComputeMemoryInfo(&compute_memory) == OK);
		CHECK(compute_memory.cpu_gpu_memory_size != 0);
		blocks.reserve(4);
		blocks.emplace_back(compute_memory.cpu_gpu_memory_size, 0x100);
		compute_memory.cpu_gpu_memory = blocks.back().data;
		Videodec2ComputeConfigInfo compute_config {sizeof(compute_config), 1, 0, false, 0, 0};
		CHECK(AllocateComputeQueue(&compute_config, &compute_memory, &queue) == OK);
		CHECK(queue != nullptr);
		config.compute_queue = queue;

		CHECK(QueryDecoderMemoryInfo(&config, &memory) == OK);
		CHECK(memory.frame_buffer_alignment == 0x100);
		blocks.emplace_back(memory.cpu_memory_size, 0x20);
		memory.cpu_memory = blocks.back().data;
		blocks.emplace_back(memory.gpu_memory_size, 0x100);
		memory.gpu_memory = blocks.back().data;
		blocks.emplace_back(memory.cpu_gpu_memory_size, 0x100);
		memory.cpu_gpu_memory = blocks.back().data;
		CHECK(CreateDecoder(&config, &memory, &decoder) == OK);
		CHECK(decoder != nullptr);

		// The game carves one frame buffer of max_frame_buffer_size per queue entry.
		frames.reserve(config.decode_input_queue_depth);
		for (uint32_t i = 0; i < config.decode_input_queue_depth; ++i)
			frames.emplace_back(memory.max_frame_buffer_size, memory.frame_buffer_alignment);
	}
	~Session() {
		CHECK(DeleteDecoder(decoder) == OK);
		CHECK(DeleteDecoder(decoder) == VIDEODEC2_ERROR_DECODER_INSTANCE);
		CHECK(ReleaseComputeQueue(queue) == OK);
		for (const auto& block: blocks) block.CheckGuard();
	}
	Session(const Session&)            = delete;
	Session& operator=(const Session&) = delete;
};

// Checks one output picture: the reported geometry, the buffer the API says it wrote, and the
// linear NV12 layout (luma rows at pitch, interleaved CbCr rows at pitch right after the luma
// plane, zeroed row padding) against the expected planes.
void CheckOutput(const Videodec2OutputInfo& output, const GuestMemory& buffer, const ReferenceFrame& expected) {
	CHECK(output.is_valid && output.picture_count == 1 && !output.is_error_frame);
	CHECK(output.codec_type == 1);
	CHECK(output.frame_width == expected.width && output.frame_height == expected.height);
	const uint32_t pitch = AlignUp(expected.width, 256);
	CHECK(output.frame_pitch == pitch && output.frame_pitch_in_bytes == pitch);
	CHECK(output.frame_format == VIDEODEC2_FRAME_FORMAT_DEFAULT);
	CHECK(output.frame_buffer == buffer.data && output.frame_buffer_size == buffer.size);
	const uint32_t chroma_rows = (expected.height + 1) / 2;
	CHECK(static_cast<size_t>(pitch) * (expected.height + chroma_rows) <= buffer.size);
	const uint32_t chroma_width = (expected.width + 1) / 2;
	for (uint32_t y = 0; y < expected.height; ++y) {
		const uint8_t* row = buffer.data + static_cast<size_t>(y) * pitch;
		CHECK(std::memcmp(row, expected.y.data() + static_cast<size_t>(y) * expected.width, expected.width) == 0);
		for (uint32_t x = expected.width; x < pitch; ++x) CHECK(row[x] == 0);
	}
	const uint8_t* chroma = buffer.data + static_cast<size_t>(pitch) * expected.height;
	for (uint32_t y = 0; y < chroma_rows; ++y) {
		const uint8_t* row = chroma + static_cast<size_t>(y) * pitch;
		for (uint32_t x = 0; x < chroma_width; ++x) {
			CHECK(row[2 * x] == expected.u[static_cast<size_t>(y) * chroma_width + x]);
			CHECK(row[2 * x + 1] == expected.v[static_cast<size_t>(y) * chroma_width + x]);
		}
		for (uint32_t x = 2 * chroma_width; x < pitch; ++x) CHECK(row[x] == 0);
	}
	buffer.CheckGuard();
}

// Feeds the access units like the game's MovieDecoder thread: the same free frame buffer is
// offered until a picture lands in it, then the next one of the ring is used. Returns the number
// of pictures checked against `expected`.
uint32_t DecodeAndCheck(Session* session, const std::vector<std::vector<uint8_t>>& units,
					const std::vector<ReferenceFrame>& expected) {
	uint32_t pictures = 0;
	uint32_t slot     = 0;
	auto handle = [&](const Videodec2OutputInfo& output) {
		if (!output.is_valid) return;
		CHECK(pictures < expected.size());
		CheckOutput(output, session->frames[slot], expected[pictures]);
		Videodec2AvcPictureInfo picture {sizeof(picture)};
		CHECK(GetPictureInfo(&output, &picture, nullptr) == OK);
		CHECK(picture.is_valid && picture.profile_idc == 100);
		++pictures;
		slot = (slot + 1) % static_cast<uint32_t>(session->frames.size());
	};
	for (size_t i = 0; i < units.size(); ++i) {
		Videodec2InputData input {sizeof(input), const_cast<uint8_t*>(units[i].data()), units[i].size(),
		                          i * 1000u, i * 1000u, 0x5000u + i};
		auto&                frame = session->frames[slot];
		Videodec2FrameBuffer buffer {sizeof(buffer), frame.data, frame.size, false};
		Videodec2OutputInfo  output {sizeof(output)};
		CHECK(Decode(session->decoder, &input, &buffer, &output) == OK);
		CHECK(buffer.is_accepted == output.is_valid);
		handle(output);
	}
	for (;;) {
		auto&                frame = session->frames[slot];
		Videodec2FrameBuffer buffer {sizeof(buffer), frame.data, frame.size, false};
		Videodec2OutputInfo  output {sizeof(output)};
		CHECK(Flush(session->decoder, &buffer, &output) == OK);
		if (!output.is_valid) break;
		handle(output);
	}
	return pictures;
}

// The memory the API asks for must fit the budgets Ghost of Yotei carves it from (its movie memory
// slots: CPU pool, GPU pool for the compute queue, the CPU-GPU block and the frame buffers, and a
// second GPU pool for the GPU block), with at least four frame buffers, as the game requires.
void TestGameMemoryBudget() {
	struct Budget {
		int32_t max_width, max_height;
		size_t  cpu_pool, frame_pool, gpu_pool;
	};
	constexpr size_t MiB = 1024u * 1024u;
	for (const Budget b: {Budget {1920, 1088, 0x1d99999, 72 * MiB, 34 * MiB},
	                      Budget {1280, 720, 0x1d99999, 72 * MiB, 34 * MiB},
	                      Budget {544, 304, 19 * MiB, 13 * MiB, 9 * MiB}}) {
		Videodec2ComputeMemoryInfo compute {sizeof(compute)};
		CHECK(QueryComputeMemoryInfo(&compute) == OK);
		auto                       config = GameConfig(b.max_width, b.max_height);
		Videodec2DecoderMemoryInfo memory {sizeof(memory)};
		CHECK(QueryDecoderMemoryInfo(&config, &memory) == OK);
		const size_t pitch = AlignUp(static_cast<uint32_t>(b.max_width), 256);
		CHECK(memory.max_frame_buffer_size == pitch * b.max_height * 3 / 2);
		CHECK(memory.cpu_memory_size <= b.cpu_pool && memory.gpu_memory_size <= b.gpu_pool);
		size_t used = 0;
		for (const size_t block: {compute.cpu_gpu_memory_size, memory.cpu_gpu_memory_size})
			used = (used + 0xff) / 0x100 * 0x100 + block;
		used = (used + 0xff) / 0x100 * 0x100;
		const size_t frame = (memory.max_frame_buffer_size + 0xff) / 0x100 * 0x100;
		CHECK(used < b.frame_pool && (b.frame_pool - used) / frame >= 4);
	}
	// Without a size bound the frame buffer still holds a 4K picture.
	auto                       config = GameConfig(-1, -1);
	Videodec2DecoderMemoryInfo memory {sizeof(memory)};
	CHECK(QueryDecoderMemoryInfo(&config, &memory) == OK);
	CHECK(memory.max_frame_buffer_size >= size_t {4096} * 2160 * 3 / 2);
	// CreateDecoder still insists on the sizes it reported.
	Session session(1920, 1088);
	Videodec2Decoder second = nullptr;
	auto             small  = session.memory;
	small.max_frame_buffer_size--;
	CHECK(CreateDecoder(&session.config, &small, &second) == VIDEODEC2_ERROR_MEMORY_SIZE);
	small = session.memory;
	small.cpu_memory_size--;
	CHECK(CreateDecoder(&session.config, &small, &second) == VIDEODEC2_ERROR_MEMORY_SIZE);
	CHECK(second == nullptr);
}

void TestRegistration() {
	Loader::SymbolDatabase symbols;
	InitVideoDec2_1(&symbols);
	Libs::g_print_name = false;
	constexpr std::array nids {"RnDibcGCPKw", "eD+X2SmxUt4", "UvtA3FAiF4Y", "qqMCwlULR+E",
	                           "CNNRoRYd8XI", "jwImxXRGSKA", "852F5+q6+iM", "l1hXwscLuCY",
	                           "wJXikG6QFN8", "NtXRa3dRzU0", "kjrLbcyhEiw"};
	CHECK(registrations.size() == nids.size());
	for (size_t i = 0; i < nids.size(); ++i) {
		CHECK(registrations[i].name == nids[i]);
		CHECK(registrations[i].library == "Videodec2" && registrations[i].module == "Videodec2");
	}
}

// Generated I_PCM pictures at the three sizes the game configures (max 1920x1088, 1280x720 and
// 544x304); 1920 and 544 need row padding, 1280 does not.
void TestGeneratedStreams() {
	struct Case {
		uint32_t width, height;
		int32_t  max_width, max_height;
	};
	for (const Case c: {Case {1920, 1080, 1920, 1088}, Case {1280, 720, 1280, 720}, Case {544, 304, 544, 304}}) {
		constexpr uint32_t frames = 3;
		const auto         stream = GeneratePcmStream(c.width, c.height, frames);
		std::vector<ReferenceFrame> expected(frames);
		for (uint32_t f = 0; f < frames; ++f) {
			auto& e  = expected[f];
			e.width  = c.width;
			e.height = c.height;
			for (uint32_t y = 0; y < c.height; ++y)
				for (uint32_t x = 0; x < c.width; ++x) e.y.push_back(PatternY(x, y, f));
			for (uint32_t y = 0; y < c.height / 2; ++y)
				for (uint32_t x = 0; x < c.width / 2; ++x) {
					e.u.push_back(PatternU(x, y, f));
					e.v.push_back(PatternV(x, y, f));
				}
		}
		// The generator itself is checked against FFmpeg first.
		const auto reference = DecodeReference(stream.access_units);
		CHECK(reference.size() == frames);
		for (uint32_t f = 0; f < frames; ++f) {
			CHECK(reference[f].width == c.width && reference[f].height == c.height);
			CHECK(reference[f].y == expected[f].y && reference[f].u == expected[f].u && reference[f].v == expected[f].v);
		}
		Session session(c.max_width, c.max_height);
		CHECK(DecodeAndCheck(&session, stream.access_units, expected) == frames);
	}
}

// Real game movies, given as a ';'-separated list in KYTY_VIDEODEC2_TEST_STREAMS (game files are
// not part of the repository). Only the first few megabytes of each are decoded.
void TestRealStreams() {
	const char* list = std::getenv("KYTY_VIDEODEC2_TEST_STREAMS");
	if (list == nullptr || *list == '\0') {
		std::puts("Videodec2 real-stream test skipped (KYTY_VIDEODEC2_TEST_STREAMS is not set)");
		return;
	}
	std::string paths = list;
	size_t      begin = 0;
	while (begin <= paths.size()) {
		size_t end = paths.find(';', begin);
		if (end == std::string::npos) end = paths.size();
		const std::string path = paths.substr(begin, end - begin);
		begin                  = end + 1;
		if (path.empty()) continue;

		std::ifstream file(path, std::ios::binary);
		CHECK(file.good());
		std::vector<uint8_t> data(4u * 1024u * 1024u);
		file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
		data.resize(static_cast<size_t>(file.gcount()));
		auto units = SplitAccessUnits(data);
		if (units.size() > 48) units.resize(48);
		CHECK(units.size() >= 8);
		const auto reference = DecodeReference(units);
		CHECK(reference.size() >= 4);
		const auto max_width  = static_cast<int32_t>(AlignUp(reference[0].width, 32));
		const auto max_height = static_cast<int32_t>(AlignUp(reference[0].height, 16));
		Session    session(max_width, max_height);
		const auto pictures = DecodeAndCheck(&session, units, reference);
		CHECK(pictures >= 4);
		std::printf("Videodec2 real stream %s: %ux%u, %zu access units, %u pictures match FFmpeg\n", path.c_str(),
		            reference[0].width, reference[0].height, units.size(), pictures);
	}
}
} // namespace

int main() {
	TestRegistration();
	TestGameMemoryBudget();
	TestGeneratedStreams();
	TestRealStreams();
	std::puts("Videodec2 tests passed");
}
