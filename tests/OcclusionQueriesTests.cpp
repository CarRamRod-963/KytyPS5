#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {
using namespace Libs::Graphics;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "OcclusionQueriesTests:%d: %s\n", __LINE__, #condition);          \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

using Results = std::array<uint64_t, OCCLUSION_QUERY_DBS * 2>;

// The samples a query passed, as ZPass predication sums them; -1 before every counter is written.
int64_t Passed(const Results& results) {
	int64_t samples = 0;
	for (uint32_t db = 0; db < OCCLUSION_QUERY_DBS; db++) {
		const auto begin = results[db * 2];
		const auto end   = results[db * 2 + 1];
		if ((begin & end & OCCLUSION_QUERY_READY) == 0) {
			return -1;
		}
		samples += static_cast<int64_t>(end - begin);
	}
	return samples;
}

void TestCounters() {
	Results results {};
	WriteZPassCounters(results.data(), 100, true);
	CHECK(Passed(results) == -1);
	WriteZPassCounters(results.data() + 1, 100, false);
	CHECK(Passed(results) == 0);
	WriteZPassCounters(results.data() + 1, 142, false);
	CHECK(Passed(results) == 42);
	CHECK(results[0] == (OCCLUSION_QUERY_READY | 100) && results[30] == OCCLUSION_QUERY_READY);

	// A begin clears the end a previous query wrote late, so the query is not ready until its own.
	WriteZPassCounters(results.data(), 142, true);
	CHECK(Passed(results) == -1);
	WriteZPassCounters(results.data() + 1, 150, false);
	CHECK(Passed(results) == 8);
}
// The counters of a ledger's dumps when every host query passed `per_query` samples.
void Publish(uint64_t* results, const ZPassLedger::Dump& dump, uint64_t per_query) {
	WriteZPassCounters(results + (dump.address & 8u) / sizeof(uint64_t),
	                   dump.queries * per_query + dump.uncounted, (dump.address & 8u) == 0);
}

void TestUnissuedDraws() {
	alignas(16) Results results {};
	const auto      base = reinterpret_cast<uint64_t>(results.data());
	ZPassLedger     ledger;

	// Outside a guest query nothing counts.
	CHECK(!ledger.Counting());
	ledger.CountUnissuedDraw();
	auto begin = ledger.Record(base, 0);
	CHECK(ledger.Counting() && begin.uncounted == 0);
	Publish(results.data(), begin, 0);

	// A query whose only draw the host left out reads visible, not hidden.
	ledger.CountUnissuedDraw();
	auto end = ledger.Record(base + 8u, 0);
	CHECK(!ledger.Counting());
	Publish(results.data(), end, 0);
	CHECK(Passed(results) == 1);

	// Queried draws count their samples; an unissued one adds one, a closed query none.
	begin = ledger.Record(base, 3);
	Publish(results.data(), begin, 10);
	ledger.CountUnissuedDraw();
	end = ledger.Record(base + 8u, 5);
	Publish(results.data(), end, 10);
	CHECK(Passed(results) == 21);
	ledger.CountUnissuedDraw();
	begin = ledger.Record(base, 5);
	Publish(results.data(), begin, 10);
	end = ledger.Record(base + 8u, 5);
	Publish(results.data(), end, 10);
	CHECK(Passed(results) == 0);

	// A draw past the pool's queries is credited one sample.
	begin = ledger.Record(base, 5);
	Publish(results.data(), begin, 10);
	ledger.Credit();
	end = ledger.Record(base + 8u, 5);
	Publish(results.data(), end, 10);
	CHECK(Passed(results) == 1);
}
} // namespace

int main() {
	TestCounters();
	TestUnissuedDraws();
	std::puts("Occlusion query tests passed");
}
