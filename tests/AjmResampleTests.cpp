#include "libs/ajm.cpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <opus.h>
#include <opus_multistream.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

using namespace Libs::Audio::Ajm;

namespace {

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AjmResampleTests: %s\n", message);
		std::abort();
	}
}

struct DecodeResult {
	AjmSidebandResult result;
	AjmSidebandStream stream;
	AjmSidebandMFrame frames;
};

struct Fixture {
	uint32_t                  instance = 0;
	uint32_t                  channels;
	uint32_t                  mapping_family = 0;
	std::array<uint8_t, 4096> batch_data {};
	AjmBatchInfo              batch {};

	explicit Fixture(uint32_t channels = 1, AjmSampleEncoding encoding = AjmSampleEncoding::S16,
	                 uint64_t flags = 0x1000)
	    : channels(channels) {
		Check(AjmInstanceCreate(1, 24, flags | channels | (uint64_t(encoding) << 7), &instance) ==
		          OK,
		      "create Opus instance");
		Initialize();
	}
	~Fixture() { AjmInstanceDestroy(1, instance); }

	AjmBatchInfo* Batch() {
		batch = {batch_data.data(), 0, batch_data.size(), nullptr, nullptr};
		return &batch;
	}

	void Initialize(int32_t expected = OK) {
		AjmDecOpusInitializeParameters params {channels, 48000, mapping_family};
		AjmSidebandResult              result {};
		AjmBatchJobInitialize(Batch(), instance, &params, sizeof(params), &result);
		Check(result.result == expected, "initialize Opus instance");
	}

	void Parameters(float ratio, uint32_t flags, float change = 0, bool extended = false) {
		AjmSidebandResult result {};
		if (extended) {
			AjmBatchJobSetResampleParametersEx(Batch(), instance, ratio, change, flags, &result);
		} else {
			AjmBatchJobSetResampleParameters(Batch(), instance, ratio, flags, &result);
		}
		Check(result.result == OK, "set resampling parameters");
	}

	AjmSidebandResampleInfo Info() {
		struct {
			AjmSidebandResult       result;
			AjmSidebandResampleInfo info;
		} result {};
		AjmBatchJobGetResampleInfo(Batch(), instance, &result);
		Check(result.result.result == OK, "get resampling info");
		return result.info;
	}

	void Gapless(uint32_t samples, uint16_t skip) {
		AjmSidebandGaplessDecode params {samples, skip, 0};
		AjmSidebandResult        result {};
		AjmBatchJobSetGaplessDecode(Batch(), instance, &params, 1, &result);
		Check(result.result == OK, "set gapless parameters");
	}

	DecodeResult Decode(const void* input, size_t size, void* output, size_t capacity) {
		DecodeResult result {};
		AjmBatchJobDecode(Batch(), instance, input, size, output, capacity, &result);
		Check((result.result.result & ~(AJM_RESULT_NOT_ENOUGH_ROOM | AJM_RESULT_PARTIAL_INPUT)) ==
		          0,
		      "decode returned unexpected error");
		return result;
	}
};

std::vector<uint8_t> Encode(uint32_t channels, int packets) {
	int   error   = 0;
	auto* encoder = opus_encoder_create(48000, channels, OPUS_APPLICATION_AUDIO, &error);
	Check(encoder != nullptr && error == OPUS_OK, "create test encoder");
	std::vector<uint8_t>      stream;
	std::vector<int16_t>      pcm(960 * channels);
	std::array<uint8_t, 4096> packet;
	for (int p = 0; p < packets; ++p) {
		for (size_t i = 0; i < pcm.size(); ++i) {
			pcm[i] = static_cast<int16_t>(
			    12000 * std::sin((i / channels + p * 960) * 0.041 + (i % channels)));
		}
		const int size = opus_encode(encoder, pcm.data(), 960, packet.data(), packet.size());
		Check(size > 0, "encode test packet");
		stream.push_back(static_cast<uint8_t>(size));
		stream.push_back(static_cast<uint8_t>(size >> 8));
		stream.insert(stream.end(), packet.begin(), packet.begin() + size);
	}
	opus_encoder_destroy(encoder);
	return stream;
}

void SmallOutputs(const std::vector<uint8_t>& packet, uint16_t skip = 0) {
	Fixture f;
	f.Gapless(960 - skip, skip);
	f.Parameters(1, 3);
	std::array<uint8_t, 1030> output;
	output.fill(0x5a);
	auto first = f.Decode(packet.data(), packet.size(), output.data(), 1024);
	Check(first.stream.size_consumed == static_cast<int>(packet.size()),
	      "small output must consume packet");
	Check(first.stream.size_produced == 1024, "small output must produce 1024 bytes");
	Check(first.frames.num_frames == 1 && first.stream.total_decoded_samples == 960 - skip,
	      "decoded frame counted once before resampling");
	Check(f.Info().num_samples == 448 - skip, "report retained source samples");
	Check(std::all_of(output.begin() + 1024, output.end(), [](auto v) { return v == 0x5a; }),
	      "do not write past output capacity");
	const auto last = f.Decode(nullptr, 0, output.data(), 1024);
	Check(last.result.result == OK && last.stream.size_consumed == 0 &&
	          last.stream.size_produced == 896 - skip * 2 && last.frames.num_frames == 0 &&
	          last.stream.total_decoded_samples == first.stream.total_decoded_samples,
	      "END drains retained PCM without decoding or reporting partial input");
	Check(f.Decode(nullptr, 0, output.data(), 1024).stream.size_produced == 0,
	      "END drains only once");
}

void UnityMatchesWholeDecode(uint32_t channels, AjmSampleEncoding encoding) {
	const auto stream       = Encode(channels, 6);
	const auto sample_bytes = channels * AjmBytesPerSample(encoding);
	Fixture    reference(channels, encoding, 0);
	reference.Gapless(5328, 312);
	std::vector<uint8_t> expected(5760 * sample_bytes);
	const auto           full =
	    reference.Decode(stream.data(), stream.size(), expected.data(), expected.size());
	expected.resize(full.stream.size_produced);
	Check(expected.size() == 5328 * sample_bytes, "reference gapless length");

	Fixture f(channels, encoding);
	f.Gapless(5328, 312);
	f.Parameters(1, 3);
	std::vector<uint8_t>      actual;
	std::array<uint8_t, 1025> output;
	size_t                    offset = 0;
	uint32_t                  frames = 0;
	for (int i = 0; i < 100; ++i) {
		const auto decoded = f.Decode(offset < stream.size() ? stream.data() + offset : nullptr,
		                              stream.size() - offset, output.data(), output.size());
		offset += decoded.stream.size_consumed;
		frames += decoded.frames.num_frames;
		actual.insert(actual.end(), output.begin(), output.begin() + decoded.stream.size_produced);
		if (decoded.stream.size_produced == 0) {
			break;
		}
	}
	Check(actual == expected && frames == 6 && offset == stream.size(),
	      "chunked PCM must exactly match whole decode, without loss or duplication");
}

void ChunkedRates(const std::vector<uint8_t>& stream, uint32_t samples, uint16_t skip) {
	for (float ratio: {0.125f, 0.749153f, 1.334840f}) {
		Fixture reference;
		reference.Gapless(samples, skip);
		reference.Parameters(ratio, 3);
		std::vector<uint8_t> expected(
		    (static_cast<size_t>(std::ceil(samples / double(ratio))) + 1) * 2);
		const auto full =
		    reference.Decode(stream.data(), stream.size(), expected.data(), expected.size());
		expected.resize(full.stream.size_produced);
		Check(expected.size() == static_cast<size_t>(std::ceil(samples / double(ratio))) * 2,
		      "fractional resampling rounds up final output sample count");

		Fixture f;
		f.Gapless(samples, skip);
		f.Parameters(ratio, 1);
		std::vector<uint8_t>      actual;
		std::array<uint8_t, 1024> output;
		size_t                    offset = 0;
		for (int i = 0; i < 1000; ++i) {
			if (offset == stream.size()) {
				// The caller can signal END only after submitting the last packet.
				f.Parameters(0, 2 | 8);
			}
			const auto decoded = f.Decode(offset < stream.size() ? stream.data() + offset : nullptr,
			                              stream.size() - offset, output.data(), output.size());
			offset += decoded.stream.size_consumed;
			actual.insert(actual.end(), output.begin(),
			              output.begin() + decoded.stream.size_produced);
			if (decoded.stream.size_produced == 0) {
				break;
			}
		}
		Check(actual == expected && offset == stream.size(),
		      "non-unity PCM is independent of output chunk boundaries and deferred END");
	}
}

void LoopBoundaries() {
	const auto loop   = Encode(1, 2);
	auto       stream = loop;
	stream.insert(stream.end(), loop.begin(), loop.end());
	for (uint64_t flags: {0x1400u, 0x3400u}) {
		Fixture f(1, AjmSampleEncoding::S16, flags);
		f.Gapless(1500, 100);
		f.Parameters(1, 1);
		std::array<uint8_t, 8192> pcm;
		const auto                first = f.Decode(stream.data(), stream.size(), pcm.data(), 1024);
		Check(first.stream.size_produced == 1024, "loop first partial output");
		const auto second =
		    f.Decode(stream.data() + first.stream.size_consumed,
			         stream.size() - first.stream.size_consumed, pcm.data(), pcm.size());
		const bool seamless = flags == 0x3400;
		Check(second.stream.size_produced == (seamless ? 6000 - 1024 : 3000 - 1024),
		      "non-seamless loop stops job at stream boundary");
		const auto consumed = first.stream.size_consumed + second.stream.size_consumed;
		Check(consumed == static_cast<int>(seamless ? stream.size() : loop.size()),
		      "loop boundary input consumption");
		if (!seamless) {
			Check(
			    f.Decode(stream.data() + consumed, stream.size() - consumed, pcm.data(), pcm.size())
			            .stream.size_produced == 3000,
			    "next job resumes gapless loop");
		}
	}
}

void RatesAndControls(const std::vector<uint8_t>& packet) {
	for (float ratio: {0.125f, 0.5f, 1.0f, 1.25f, 2.0f, 4.0f}) {
		Fixture f;
		f.Parameters(ratio, 3);
		std::vector<int16_t> pcm(8000);
		const auto result = f.Decode(packet.data(), packet.size(), pcm.data(), pcm.size() * 2);
		Check(result.stream.size_produced == static_cast<int>(std::ceil(960.0 / ratio)) * 2,
		      "resampling ratio controls output duration");
		Check(result.stream.total_decoded_samples == 960,
		      "ratio does not change decoded sample count");
	}
	Fixture f;
	f.Parameters(1, 1);
	std::array<int16_t, 512> pcm;
	f.Decode(packet.data(), packet.size(), pcm.data(), sizeof(pcm));
	f.Initialize();
	Check(f.Info().num_samples == 448, "codec initialize must preserve resampler PCM");
	f.Parameters(0, 0);
	const auto frozen = f.Decode(nullptr, 0, pcm.data(), sizeof(pcm));
	Check(
	    frozen.stream.size_produced == sizeof(pcm) && f.Info().num_samples == 448 &&
	        std::all_of(pcm.begin(), pcm.end(), [&](auto sample) { return sample == pcm.front(); }),
	    "zero ratio repeats one sample without advancing input");
	f.Parameters(1, 0, 0.25f, true);
	f.Parameters(std::numeric_limits<float>::quiet_NaN(), 8);
	f.Decode(nullptr, 0, pcm.data(), 4 * sizeof(int16_t));
	Check(f.Info().ratio == 2.0f, "ignore preserves ratio ramp");
	f.Parameters(1, 1);
	Check(f.Info().num_samples == 0, "resample RESET discards retained PCM");
	Check(f.Decode(nullptr, 0, pcm.data(), sizeof(pcm)).stream.size_produced == 0,
	      "RESET leaves no stale audio");
}

void UnbufferedAndSplit(const std::vector<uint8_t>& packet) {
	Fixture                   raw(1, AjmSampleEncoding::S16, 0);
	std::array<uint8_t, 1024> pcm;
	const auto too_small = raw.Decode(packet.data(), packet.size(), pcm.data(), pcm.size());
	Check(too_small.result.result == AJM_RESULT_NOT_ENOUGH_ROOM &&
	          too_small.stream.size_consumed == 0 && too_small.stream.size_produced == 0,
	      "ordinary decoding still requires room for a whole packet");
	Fixture f;
	f.Parameters(1, 3);
	AjmBuffer    input[] {{const_cast<uint8_t*>(packet.data()), 1},
	                      {const_cast<uint8_t*>(packet.data()) + 1, packet.size() - 1}};
	AjmBuffer    output[] {{pcm.data(), 257}, {pcm.data() + 257, 767}};
	DecodeResult result {};
	AjmBatchJobDecodeSplit(f.Batch(), f.instance, input, 2, output, 2, &result);
	Check(result.stream.size_produced == 1024 &&
	          result.stream.size_consumed == static_cast<int>(packet.size()),
	      "split buffers retain PCM across jobs");
}

void FormatChangeRequiresReset() {
	Fixture f(2);
	f.Parameters(1, 1);
	const auto                stereo = Encode(2, 1);
	std::array<uint8_t, 4096> pcm;
	f.Decode(stereo.data(), stereo.size(), pcm.data(), 1024);
	f.channels = 1;
	f.Initialize();
	DecodeResult result {};
	AjmBatchJobDecode(f.Batch(), f.instance, nullptr, 0, pcm.data(), pcm.size(), &result);
	Check(result.result.result == AJM_RESULT_INVALID_PARAMETER && result.stream.size_produced == 0,
	      "format change cannot reinterpret retained PCM with a different channel stride");
	f.Parameters(1, 3);
	const auto mono = Encode(1, 1);
	Check(f.Decode(mono.data(), mono.size(), pcm.data(), pcm.size()).stream.size_produced == 1920,
	      "resampler RESET permits a new stream format");
}

void EndPersistsUntilReset(const std::vector<uint8_t>& packet) {
	Fixture f;
	f.Parameters(0.5f, 1);
	std::array<int16_t, 64> pcm;
	auto   decoded  = f.Decode(packet.data(), packet.size(), pcm.data(), sizeof(pcm));
	size_t produced = decoded.stream.size_produced;
	f.Parameters(0.5f, 2);
	for (int i = 0; i < 40; ++i) {
		f.Parameters(0.5f, 0);
		decoded = f.Decode(nullptr, 0, pcm.data(), sizeof(pcm));
		produced += decoded.stream.size_produced;
		Check((decoded.result.result & AJM_RESULT_PARTIAL_INPUT) == 0,
		      "END remains active across subsequent ratio controls");
		if (f.Info().num_samples < 0) {
			break;
		}
	}
	Check(produced == 3840 && f.Info().num_samples < 0,
	      "fully drained END reports a negative retained sample count");
	f.Parameters(1, 1);
	Check(f.Info().num_samples == 0, "RESET clears end-of-stream state");
}

void MappingFamilies() {
	// Family 1 codes stereo as family 0 does.
	const auto           stereo = Encode(2, 1);
	std::vector<uint8_t> expected(3840), actual(3840);
	Fixture              f(2, AjmSampleEncoding::S16, 0);
	f.Decode(stereo.data(), stereo.size(), expected.data(), expected.size());
	f.mapping_family = 1;
	f.Initialize();
	f.Decode(stereo.data(), stereo.size(), actual.data(), actual.size());
	Check(actual == expected, "family 1 decodes stereo as family 0");

	// Family 255: two uncoupled mono streams, a loud tone in the first, a quiet one in the second.
	int                 error = 0;
	const unsigned char mapping[] {0, 1};
	auto* encoder = opus_multistream_encoder_create(48000, 2, 2, 0, mapping, OPUS_APPLICATION_AUDIO,
	                                                &error);
	Check(encoder != nullptr && error == OPUS_OK, "create multistream test encoder");
	std::vector<int16_t> pcm(960 * 2);
	for (size_t i = 0; i < 960; ++i) {
		pcm[i * 2]     = static_cast<int16_t>(12000 * std::sin(i * 0.041));
		pcm[i * 2 + 1] = static_cast<int16_t>(3000 * std::sin(i * 0.11));
	}
	std::vector<uint8_t> packet(4096);
	const int size = opus_multistream_encode(encoder, pcm.data(), 960, packet.data() + 2, 4094);
	opus_multistream_encoder_destroy(encoder);
	Check(size > 0, "encode multistream test packet");
	packet.resize(size + 2);
	packet[0] = static_cast<uint8_t>(size);
	packet[1] = static_cast<uint8_t>(size >> 8);

	f.mapping_family = 255;
	f.Initialize();
	std::vector<int16_t> out(960 * 2);
	const auto decoded = f.Decode(packet.data(), packet.size(), out.data(), out.size() * 2);
	Check(decoded.stream.size_consumed == static_cast<int>(packet.size()) &&
	          decoded.stream.size_produced == 960 * 2 * 2,
	      "family 255 decodes the whole multistream packet");
	double left = 0, right = 0;
	for (size_t i = 0; i < 960; ++i) {
		left += std::abs(out[i * 2]);
		right += std::abs(out[i * 2 + 1]);
	}
	Check(2 * right < left && left < 8 * right, "family 255 keeps each stream in its own channel");

	f.mapping_family = 2;
	f.Initialize(AJM_RESULT_INVALID_PARAMETER | AJM_RESULT_FATAL);
}

void Mp3ParseFrame() {
	// MPEG-1 Layer III, 128 kbps, 44.1 kHz, joint stereo, no CRC.
	std::array<uint8_t, 4>  frame = {0xff, 0xfb, 0x90, 0x64};
	AjmDecMp3ParseFrameInfo info;
	std::memset(&info, 0xff, sizeof(info));
	Check(AjmDecMp3ParseFrame(frame.data(), frame.size(), 1, &info) == OK, "parse MP3 frame");
	Check(info.frame_size == 417 && info.num_channels == 2 && info.samples_per_channel == 1152 &&
	          info.bitrate == 128000 && info.sample_rate == 44100 && info.encoder_delay == 0 &&
	          info.num_frames == 0 && info.total_samples == 0 && info.ofl_type == 0,
	      "MP3 frame info");
	Check(AjmDecMp3ParseFrame(frame.data(), frame.size(), 0, nullptr) ==
	          AJM_ERROR_INVALID_PARAMETER,
	      "MP3 parse without output");
	frame[1] = 0xfd;
	Check(AjmDecMp3ParseFrame(frame.data(), frame.size(), 0, &info) == AJM_ERROR_INVALID_PARAMETER,
	      "MP3 parse of a Layer II frame");
}

// Demon's Souls' haptic banks hold 48 kHz dual-mono ATRAC9 streams (192-byte superframes of one
// frame) with a few frames LibAtrac9 cannot unpack; PS5 keeps decoding them. An all-zero frame
// fails to unpack the same way (scale factor out of range).
void UndecodableAt9FrameIsSilent() {
	AjmAt9Decoder decoder(2, 48000, AjmSampleEncoding::S16, 0);
	const AjmDecAt9InitializeParameters parameters {{0xfe, 0x72, 0x17, 0xe0}, 0};
	Check(decoder.Initialize(&parameters, sizeof(parameters)).result == OK, "initialize ATRAC9");
	const std::vector<uint8_t> input(2 * 192, 0);
	std::vector<uint8_t>       output(2 * 256 * 2 * sizeof(int16_t), 0xcd);
	const auto result = decoder.Decode(input.data(), input.size(), output.data(), output.size(),
	                                   true, nullptr);
	Check(result.result == OK, "an unpackable ATRAC9 frame does not fail the job");
	Check(result.frames == 2 && result.input_consumed == input.size() &&
	          result.output_written == output.size() && result.total_decoded_samples == 512,
	      "an unpackable ATRAC9 frame takes its superframe and plays its samples");
	Check(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0; }),
	      "an unpackable ATRAC9 frame plays as silence");
}

// Input that ends right before an inaccessible page, as an exact-size heap buffer can.
class GuardedInput {
public:
	explicit GuardedInput(size_t size) {
#if defined(_WIN32)
		m_base = static_cast<uint8_t*>(
		    VirtualAlloc(nullptr, 2 * PageSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
		DWORD old_protect = 0;
		Check(m_base != nullptr &&
		          VirtualProtect(m_base + PageSize, PageSize, PAGE_NOACCESS, &old_protect) != 0,
		      "allocate a guarded input page");
#else
		void* base = mmap(nullptr, 2 * PageSize, PROT_READ | PROT_WRITE,
		                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		Check(base != MAP_FAILED && mprotect(static_cast<uint8_t*>(base) + PageSize, PageSize,
		                                     PROT_NONE) == 0,
		      "allocate a guarded input page");
		m_base = static_cast<uint8_t*>(base);
#endif
		Check(size <= PageSize, "guarded input fits a page");
		m_data = m_base + PageSize - size;
	}
	~GuardedInput() {
#if defined(_WIN32)
		VirtualFree(m_base, 0, MEM_RELEASE);
#else
		munmap(m_base, 2 * PageSize);
#endif
	}
	GuardedInput(const GuardedInput&)            = delete;
	GuardedInput& operator=(const GuardedInput&) = delete;

	[[nodiscard]] uint8_t* Data() const { return m_data; }

private:
	static constexpr size_t PageSize = 4096;
	uint8_t*                m_base   = nullptr;
	uint8_t*                m_data   = nullptr;
};

// LibAtrac9 takes no input size and reads past the bits a frame uses, far past them for a frame it
// cannot unpack. A split decode job copies its input into an exact-size buffer; Saros' 320-byte
// job once ended at an unmapped page and the decoder faulted reading its 321st byte. Frames of
// noise in 48 kHz mono 64-byte superframes end at such a page here (the fourth read past it;
// later ones reach a LibAtrac9 band-extension overrun of noise, unrelated to the input size).
void At9FramesReadOnlyTheirInput() {
	AjmAt9Decoder decoder(1, 48000, AjmSampleEncoding::S16, 0);
	const AjmDecAt9InitializeParameters parameters {{0xfe, 0x70, 0x07, 0xe0}, 0};
	Check(decoder.Initialize(&parameters, sizeof(parameters)).result == OK,
	      "initialize mono ATRAC9");
	GuardedInput         input(64);
	std::vector<uint8_t> output(256 * sizeof(int16_t));
	uint32_t             seed = 0x12345678u;
	for (int frame = 0; frame < 8; frame++) {
		for (size_t i = 0; i < 64; i++) {
			seed           = seed * 1664525u + 1013904223u;
			input.Data()[i] = static_cast<uint8_t>(seed >> 24u);
		}
		const auto result =
		    decoder.Decode(input.Data(), 64, output.data(), output.size(), true, nullptr);
		Check(result.frames <= 1 && result.input_consumed <= 64,
		      "a noise ATRAC9 frame takes at most its superframe");
	}
}

// LibAtrac9's band extension tables cover blocks of 13 to 20 coded quantization units. A frame
// that enables band extension with other band counts indexed past them, and the noise spectrum
// it then filled ran past the channel's 256 coefficients (Saros 01.005.002 died in
// ApplyBandExtension). Such a frame fails to unpack instead.
void At9BandExtensionNeedsItsBandRange() {
	// 48 kHz mono: minimum band count 3, extension bands up to 18.
	unsigned char config[4] = {0xfe, 0x70, 0x07, 0xe0};
	// First block of a superframe, new band parameters, band count, band extension on,
	// extension band: 4 bands (12 quantization units) extended to 5, and 11 bands (21 units)
	// extended to 12.
	for (const auto& [frame_bits, bands]: {std::pair {std::array<uint8_t, 2> {0x06, 0x40}, 4},
	                                        std::pair {std::array<uint8_t, 2> {0x23, 0x20}, 11}}) {
		void* handle = Atrac9GetHandle();
		Check(handle != nullptr && Atrac9InitDecoder(handle, config) == 0,
		      "initialize a mono ATRAC9 handle");
		std::vector<uint8_t> frame(16 * 1024, 0);
		std::copy(frame_bits.begin(), frame_bits.end(), frame.begin());
		std::vector<int16_t> pcm(256, 0);
		int                  used   = 0;
		const int            status = Atrac9Decode(handle, frame.data(), pcm.data(), &used, 0);
		Atrac9ReleaseHandle(handle);
		Check(status == ERR_UNPACK_BAND_PARAMS_INVALID,
		      bands == 4 ? "band extension of 12 quantization units is rejected"
		                 : "band extension of 21 quantization units is rejected");
	}
}

// Instance ids have 14 slot bits. Saros creates and destroys ATRAC9 instances all the time; once
// ids wrapped, a new instance took the id of a live one, replaced its decoder and fed it another
// stream's frames (LibAtrac9 then read a null Huffman table). A live instance keeps its id.
void InstanceIdsNeverAliasLiveInstances() {
	std::array<uint8_t, 4096> batch_data {};
	AjmBatchInfo              batch {};
	const auto                Batch = [&]() {
		batch = {batch_data.data(), 0, batch_data.size(), nullptr, nullptr};
		return &batch;
	};
	const auto CodecInfo = [&](uint32_t instance) {
		struct {
			AjmSidebandResult          result;
			AjmSidebandDecAt9CodecInfo info;
		} result {};
		AjmBatchJobGetCodecInfo(Batch(), instance, &result, sizeof(result));
		return result.info;
	};
	const uint32_t at9 = static_cast<uint32_t>(AjmCodec::DecAt9);
	uint32_t       live = 0;
	Check(AjmInstanceCreate(1, at9, 0x1001, &live) == OK, "create a live ATRAC9 instance");
	const AjmDecAt9InitializeParameters mono {{0xfe, 0x70, 0x07, 0xe0}, 0};
	AjmSidebandResult                   init_result {};
	AjmBatchJobInitialize(Batch(), live, &mono, sizeof(mono), &init_result);
	Check(init_result.result == OK && CodecInfo(live).super_frame_size == 64,
	      "initialize the live instance");

	for (uint32_t i = 0; i < 2 * AJM_INSTANCE_SLOT_COUNT; i++) {
		uint32_t other = 0;
		Check(AjmInstanceCreate(1, at9, 0x1001, &other) == OK, "create a short-lived instance");
		Check(other != live, "a new instance never takes a live instance's id");
		Check(other % AJM_INSTANCE_SLOT_COUNT != 0, "slot 0 is not handed out");
		Check(AjmInstanceDestroy(1, other) == OK, "destroy the short-lived instance");
	}
	Check(CodecInfo(live).super_frame_size == 64, "the live instance keeps its decoder");

	// Every slot taken: creation fails until one is freed, then that slot is handed out.
	std::vector<uint32_t> all;
	for (;;) {
		uint32_t   other  = 0;
		const auto result = AjmInstanceCreate(1, at9, 0x1001, &other);
		if (result != OK) {
			Check(result == AJM_ERROR_OUT_OF_RESOURCES, "creation past the last slot fails");
			break;
		}
		all.push_back(other);
	}
	Check(all.size() == AJM_INSTANCE_SLOT_COUNT - 2, "every slot but 0 and the live one is used");
	const auto freed = all[all.size() / 2];
	Check(AjmInstanceDestroy(1, freed) == OK, "free one slot");
	uint32_t again = 0;
	Check(AjmInstanceCreate(1, at9, 0x1001, &again) == OK && again == freed,
	      "the freed slot is handed out again");
	all[all.size() / 2] = again;
	for (const auto instance: all) {
		AjmInstanceDestroy(1, instance);
	}
	AjmInstanceDestroy(1, live);
}

} // namespace

int main(int argc, char** argv) {
	InstanceIdsNeverAliasLiveInstances();
	Mp3ParseFrame();
	UndecodableAt9FrameIsSilent();
	At9FramesReadOnlyTheirInput();
	At9BandExtensionNeedsItsBandRange();
	const auto packet = Encode(1, 1);
	SmallOutputs(packet);
	SmallOutputs(packet, 312);
	for (uint32_t channels: {1u, 2u}) {
		for (auto encoding:
		     {AjmSampleEncoding::S16, AjmSampleEncoding::S32, AjmSampleEncoding::Float}) {
			UnityMatchesWholeDecode(channels, encoding);
		}
	}
	RatesAndControls(packet);
	UnbufferedAndSplit(packet);
	ChunkedRates(Encode(1, 6), 5328, 312);
	LoopBoundaries();
	FormatChangeRequiresReset();
	EndPersistsUntilReset(packet);
	MappingFamilies();
	if (argc >= 2) {
		std::ifstream        file(argv[1], std::ios::binary);
		std::vector<uint8_t> real_packet((std::istreambuf_iterator<char>(file)), {});
		Check(real_packet.size() == 209, "expected real 209-byte packet fixture");
		SmallOutputs(real_packet, 312);
	}
	if (argc >= 3) {
		std::ifstream        file(argv[2], std::ios::binary);
		std::vector<uint8_t> real_stream((std::istreambuf_iterator<char>(file)), {});
		Check(real_stream.size() == 4048, "expected complete real stream fixture");
		ChunkedRates(real_stream, 25920, 312);
	}
	std::puts("AjmResampleTests: passed");
}
