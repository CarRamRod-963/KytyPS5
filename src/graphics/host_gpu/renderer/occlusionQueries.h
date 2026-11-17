#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// The guest's query results (16-byte aligned): a begin/end pair of running z-pass counters per DB,
// 16 bytes apart; bit 63 marks a written counter. A query's samples are the sum of end - begin.
constexpr uint32_t OCCLUSION_QUERY_DBS   = 16;
constexpr uint64_t OCCLUSION_QUERY_READY = 1ull << 63u;

// A dump of the running count, all of it in DB 0. A begin also clears its end counters.
inline void WriteZPassCounters(volatile uint64_t* results, uint64_t samples, bool begin) {
	for (uint32_t db = 0; db < OCCLUSION_QUERY_DBS; db++) {
		results[db * 2] = OCCLUSION_QUERY_READY | (db == 0 ? samples : 0);
		if (begin) {
			results[db * 2 + 1] = 0;
		}
	}
}

// The guest side of the z-pass dumps, without Vulkan: whether a guest query is open, and the
// samples credited to draws that took no host query. A dump's counter is the samples of the host
// queries before it plus those credits.
class ZPassLedger {
public:
	struct Dump {
		uint64_t address;
		// Host queries begun before the dump.
		uint64_t queries;
		// Samples credited before the dump.
		uint64_t uncounted;
	};

	// A guest query is open: draws count their samples.
	[[nodiscard]] bool Counting() const { return m_open != 0; }
	// One sample for a draw that took no host query (the pool had no free query).
	void Credit() { m_uncounted++; }
	// A draw the host does not issue while a guest query is open, because it cannot run it or
	// replaces it with other work, counts one sample. Its object is then visible: a hidden result
	// (no samples) would let the guest cull geometry its draw shows, while a visible one costs
	// at most the work the guest would have done anyway.
	void CountUnissuedDraw() {
		if (m_open != 0) {
			m_uncounted++;
		}
	}
	// A PIXEL_PIPE_STAT_DUMP to address after `queries` host queries: begins dump to the
	// results, ends 8 bytes into them.
	Dump Record(uint64_t address, uint64_t queries) {
		if ((address & 8u) == 0) {
			m_open++;
		} else if (m_open != 0) {
			m_open--;
		}
		return {address, queries, m_uncounted};
	}

private:
	uint32_t m_open      = 0;
	uint64_t m_uncounted = 0;
};

// The guest's z-pass counters from host occlusion queries: a draw counts its samples while a guest
// query is open, and the dumps' counters are written before the guest can see the GPU ran them.
class OcclusionQueries {
public:
	OcclusionQueries(GraphicContext& graphics, CommandScheduler& scheduler);
	~OcclusionQueries();
	KYTY_CLASS_NO_COPY(OcclusionQueries);

	[[nodiscard]] bool Enabled() const { return m_pool != nullptr; }

	// In the draw's rendering: the query that counts its samples, or -1.
	[[nodiscard]] int32_t Begin(vk::CommandBuffer command) {
		return m_ledger.Counting() ? BeginQuery(command) : -1;
	}
	// Whether a draw recorded now counts its samples for a guest query.
	[[nodiscard]] bool Counting() const { return m_ledger.Counting(); }
	// See ZPassLedger::CountUnissuedDraw.
	void CountUnissuedDraw() { m_ledger.CountUnissuedDraw(); }
	void End(vk::CommandBuffer command, int32_t query) const {
		if (query >= 0) {
			command.endQuery(m_pool, static_cast<uint32_t>(query));
		}
	}

	// A PIXEL_PIPE_STAT_DUMP to address, after the draws recorded so far.
	void Dump(uint64_t address);
	// Waits for the GPU to run the dumps and writes their counters.
	void Publish() {
		if (!m_dumps.empty()) {
			PublishDumps();
		}
	}

private:
	struct PendingDump {
		ZPassLedger::Dump dump;
		uint64_t          tick;
	};

	int32_t BeginQuery(vk::CommandBuffer command);
	void    PublishDumps();

	CommandScheduler&        m_scheduler;
	vk::Device               m_device;
	vk::QueryPool            m_pool;
	vk::QueryControlFlags    m_control;
	std::vector<PendingDump> m_dumps;
	std::vector<uint64_t>    m_results;
	ZPassLedger              m_ledger;
	uint64_t                 m_begun   = 0;
	uint64_t                 m_read    = 0;
	uint64_t                 m_samples = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
