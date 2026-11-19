#include "graphics/host_gpu/renderer/pipeline/selectionFeedback.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cinttypes>
#include <xxhash.h>

namespace Libs::Graphics {

namespace IR = ShaderRecompiler::IR;

namespace {

constexpr uint64_t Mix(uint64_t hash, uint64_t value) {
	hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
	return hash;
}

} // namespace

SelectionFeedback::SelectionFeedback(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_report_time(std::chrono::steady_clock::now()) {
	m_current.frame = m_frame;
	m_current.stamp = Stamp(m_frame);
}

SelectionFeedback::~SelectionFeedback() = default;

bool SelectionFeedback::Create() {
	if (m_ring != nullptr) return true;
	if (m_failed || m_graphics.allocator == nullptr) return false;
	m_failed         = true;
	const auto bytes = uint64_t {RingWords} * sizeof(uint32_t);
	m_ring     = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	                                      vk::BufferUsageFlagBits::eStorageBuffer |
	                                          vk::BufferUsageFlagBits::eTransferSrc |
	                                          vk::BufferUsageFlagBits::eShaderDeviceAddress,
	                                      bytes);
	m_readback = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                      vk::BufferUsageFlagBits::eTransferDst, bytes);
	if (m_readback->Mapped().empty()) {
		m_ring.reset();
		m_readback.reset();
		return false;
	}
	// Words never written hold whatever the memory held; one equal to a frame's stamp only
	// reports a selection that did not happen, which keeps a candidate bound.
	m_failed = false;
	return true;
}

// The ring holds the regions of the frames whose feedback is still to be read, oldest first,
// then the current frame's; a frame's regions may wrap around the ring end once.
uint32_t SelectionFeedback::Allocate(uint32_t words) {
	if (words == 0 || words > RingWords / 4u) return IR::NoSelectionRegion;
	const bool used = !m_pending.empty() || m_head != m_current.start || m_current.wrap_end != 0;
	if (!used) {
		m_head          = 0;
		m_current.start = 0;
	}
	const uint32_t tail = m_pending.empty() ? m_current.start : m_pending.front().start;
	if (m_head < tail) {
		// Wrapped: the free words end at the oldest region.
		if (m_head + words >= tail) return IR::NoSelectionRegion;
	} else if (m_head + words > RingWords) {
		if (words >= tail) return IR::NoSelectionRegion;
		if (m_head == m_current.start && m_current.wrap_end == 0) {
			m_current.start = 0;
		} else {
			m_current.wrap_end = m_head;
		}
		m_head = 0;
	}
	const auto base = m_head;
	m_head += words;
	return base;
}

void SelectionFeedback::PrepareStage(const IR::CompiledShaderInfo& program,
                                     const IR::ResourceSnapshot&   snapshot,
                                     std::vector<uint32_t>& words, std::vector<uint8_t>& demoted) {
	words.clear();
	demoted.clear();
	const auto& bindings = program.bindings;
	if (bindings.selection_roots == 0) return;
	words.assign(IR::SelectionFeedbackHeaderDwords + bindings.selection_roots,
	             IR::NoSelectionRegion);
	words[0] = words[1] = words[2] = 0;
	if (!Config::SelectionFeedbackEnabled() || !Create()) return;
	const auto address = m_ring->BufferDeviceAddress();
	words[0]           = static_cast<uint32_t>(address);
	words[1]           = static_cast<uint32_t>(address >> 32u);
	words[2]           = m_current.stamp;
	const auto& images = program.info.images;
	demoted.assign(images.size(), 0);
	uint32_t ordinal = 0;
	for (uint32_t root = 0; root < images.size(); root++) {
		const auto& resource = images[root];
		if (!IR::IsSelectionFeedbackRoot(resource, root)) continue;
		const auto count = static_cast<uint32_t>(resource.indirect_resources.size());
		auto& table = m_selection.FindTable(Mix(Mix(program.shader_hash, root), count), m_frame);
		const bool writes  = resource.written || resource.atomic;
		uint64_t   content = Mix(reinterpret_cast<uint64_t>(&table), count);
		m_scratch.assign(count, {});
		for (uint32_t k = 1; k < count; k++) {
			const auto index = resource.indirect_resources[k];
			if (index >= snapshot.images.size()) continue;
			const auto& value = snapshot.images[index].dwords;
			if (std::ranges::all_of(value, [](uint32_t word) { return word == 0; })) {
				continue; // A spare.
			}
			auto& candidate = CandidateSelection::FindCandidate(
			    table, XXH3_64bits(value.data(), sizeof(value)), m_frame);
			const bool demote = CandidateSelection::Demotes(table, candidate, writes);
			m_scratch[k]      = {&candidate, demote};
			content = Mix(content, reinterpret_cast<uint64_t>(&candidate) | uint64_t {demote});
		}
		uint32_t   base  = IR::NoSelectionRegion;
		const auto found = m_regions.find(content);
		if (found != m_regions.end()) {
			base = found->second;
		} else {
			base = Allocate(count);
			if (base == IR::NoSelectionRegion) {
				// Without feedback nothing is demoted; reading the frame counts every
				// candidate as selected.
				m_report_full++;
				for (auto& entry: m_scratch)
					entry.demoted = false;
			} else {
				m_regions.emplace(content, base);
			}
			m_current.records.push_back({&table, base,
			                             static_cast<uint32_t>(m_current.entries.size()), count,
			                             program.shader_hash, root});
			m_current.entries.insert(m_current.entries.end(), m_scratch.begin(), m_scratch.end());
		}
		words[IR::SelectionFeedbackHeaderDwords + ordinal++] = base;
		m_report_binds++;
		for (uint32_t k = 1; k < count; k++) {
			if (m_scratch[k].candidate == nullptr) continue;
			m_report_bound++;
			if (m_scratch[k].demoted) {
				demoted[resource.indirect_resources[k]] = 1;
				m_report_demoted++;
			}
		}
	}
}

void SelectionFeedback::EndFrame() {
	if (m_head != m_current.start || m_current.wrap_end != 0) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto        handle = command.Handle();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		handle.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
		                       nullptr);
		const auto Region = [](uint32_t begin, uint32_t end) {
			return vk::BufferCopy {uint64_t {begin} * 4u, uint64_t {begin} * 4u,
			                       uint64_t {end - begin} * 4u};
		};
		std::vector<vk::BufferCopy> regions;
		if (m_current.wrap_end != 0) {
			if (m_current.wrap_end > m_current.start) {
				regions.push_back(Region(m_current.start, m_current.wrap_end));
			}
			if (m_head != 0) regions.push_back(Region(0, m_head));
		} else {
			regions.push_back(Region(m_current.start, m_head));
		}
		handle.copyBuffer(m_ring->Handle(), m_readback->Handle(),
		                  static_cast<uint32_t>(regions.size()), regions.data());
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eHostRead;
		handle.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eHost, {}, 1, &after, 0, nullptr, 0,
		                       nullptr);
		m_current.end  = m_head;
		m_current.tick = m_scheduler.CurrentTick();
		m_pending.push_back(std::move(m_current));
	} else if (!m_current.records.empty()) {
		// Only bindings without feedback: nothing to copy, the frame is read at once.
		Decode(m_current);
	}
	m_current = {};
	m_regions.clear();
	m_frame++;
	m_current.frame = m_frame;
	m_current.stamp = Stamp(m_frame);
	m_current.start = m_head;
	while (!m_pending.empty() && m_scheduler.IsFree(m_pending.front().tick)) {
		Decode(m_pending.front());
		m_pending.pop_front();
	}
	if ((m_frame & 255u) == 0) {
		m_selection.Forget(m_pending.empty() ? m_frame : m_pending.front().frame);
	}
	m_report_frames++;
	Report();
}

void SelectionFeedback::Decode(PendingFrame& frame) {
	const uint32_t* words = nullptr;
	if (frame.end != frame.start || frame.wrap_end != 0) {
		m_readback->Invalidate(0, m_readback->Size());
		words = reinterpret_cast<const uint32_t*>(m_readback->Mapped().data());
	}
	for (const auto& record: frame.records) {
		auto* entries = frame.entries.data() + record.first;
		for (uint32_t k = 0; k < record.count; k++) {
			if (record.base != IR::NoSelectionRegion && words[record.base + k] != frame.stamp) {
				continue;
			}
			auto& entry = entries[k];
			CandidateSelection::Selected(*record.table, entry.candidate, frame.frame);
			if (entry.demoted && m_selection.Missed(*record.table)) {
				LOGF("Selection feedback: shader %016" PRIx64
				     " selected a candidate of its table %u "
				     "(%u candidates) it had left unused at frame %" PRIu64
				     "; every candidate of the table stays bound\n",
				     record.shader, record.root, record.count, frame.frame);
			}
		}
	}
	for (const auto& record: frame.records) {
		const auto* entries = frame.entries.data() + record.first;
		for (uint32_t k = 0; k < record.count; k++) {
			CandidateSelection::Observed(*record.table, entries[k].candidate, frame.frame);
		}
	}
	m_decoded = std::max(m_decoded, frame.frame);
}

// One line every 30 s while candidates are demoted or a table turned eager.
void SelectionFeedback::Report() {
	const auto now = std::chrono::steady_clock::now();
	if (now - m_report_time < std::chrono::seconds(30)) return;
	if (m_report_demoted != 0 || m_selection.EagerTables() != 0) {
		const auto frames = static_cast<double>(std::max<uint64_t>(m_report_frames, 1));
		LOGF("Selection feedback: %.0f table bindings/frame, %.0f of %.0f candidates/frame bound "
		     "null, %zu tables (%zu eager), %" PRIu64 " misses, %" PRIu64
		     " bindings without feedback, frame lag %" PRIu64 "\n",
		     m_report_binds / frames, m_report_demoted / frames, m_report_bound / frames,
		     m_selection.TableCount(), m_selection.EagerTables(), m_selection.Misses(),
		     m_report_full, m_frame - m_decoded);
	}
	m_report_time   = now;
	m_report_frames = m_report_binds = m_report_demoted = m_report_bound = m_report_full = 0;
}

} // namespace Libs::Graphics
