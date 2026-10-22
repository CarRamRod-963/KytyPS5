#include "graphics/host_gpu/renderer/pipeline/resourceLookahead.h"

#include "graphics/host_gpu/renderer/pipeline/scalarReadCache.h"

#include <algorithm>
#include <cstring>
#include <span>

namespace Libs::Graphics {

// The helper's backend for the scalar read cache: plain backing reads (a helper cannot see
// GPU-dirty bytes; the owner checks them), each one recorded with its bytes for the owner's
// comparison.
class ResourceLookahead::HelperReader {
public:
	HelperReader(const LookaheadMemory& memory, Job& job): m_memory(memory), m_job(job) {}

	bool Read(uint64_t address, void* data, uint64_t size) {
		const bool mapped = m_memory.read(address, data, size);
		Record(address, size, mapped, data);
		return mapped;
	}
	static bool     HasGpuDirtyBytes(uint64_t /*address*/, uint64_t /*size*/) { return false; }
	static void     ReadGpuDirtyBytes(std::span<const ScalarReadRange> /*ranges*/) {}
	static uint64_t GpuWriteSequence() { return 0; }

	// The specialization reader: the same bytes the owner's buffer-backing reader returns when
	// the range holds no GPU-dirty bytes, which the owner checks.
	static bool ReadSpecialization(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto&      self = static_cast<Context*>(userdata)->reader;
		const bool mapped =
		    !values.empty() && self.m_memory.read(address, values.data(), values.size_bytes());
		if (!values.empty()) self.Record(address, values.size_bytes(), mapped, values.data());
		return mapped;
	}
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
		static_cast<Context*>(userdata)->cache.Read(address, values);
		return true;
	}

	struct Context {
		HelperReader&                  reader;
		ScalarReadCache<HelperReader>& cache;
	};

private:
	void Record(uint64_t address, uint64_t size, bool mapped, const void* data) {
		auto& job = m_job;
		// A mapped read that continues the previous one extends it: the owner then checks the
		// union with one read, which is mapped exactly when both pieces are.
		if (mapped && !job.reads.empty()) {
			auto& last = job.reads.back();
			if (last.mapped && last.address + last.size == address &&
			    uint64_t {last.size} + size <= UINT32_MAX) {
				last.size += static_cast<uint32_t>(size);
				const auto* bytes = static_cast<const uint8_t*>(data);
				job.bytes.insert(job.bytes.end(), bytes, bytes + size);
				return;
			}
		}
		job.reads.push_back({.address = address,
		                     .size    = static_cast<uint32_t>(size),
		                     .offset  = static_cast<uint32_t>(job.bytes.size()),
		                     .mapped  = mapped});
		if (mapped) {
			const auto* bytes = static_cast<const uint8_t*>(data);
			job.bytes.insert(job.bytes.end(), bytes, bytes + size);
		}
	}

	const LookaheadMemory& m_memory;
	Job&                   m_job;
};

ResourceLookahead::ResourceLookahead(LookaheadMemory memory, uint32_t helpers): m_memory(memory) {
	m_helpers.reserve(helpers);
	for (uint32_t i = 0; i < helpers; ++i) {
		m_helpers.push_back(std::make_unique<Helper>());
	}
	for (auto& helper: m_helpers) {
		helper->thread = std::thread([this, h = helper.get()] { RunHelper(*h); });
	}
}

ResourceLookahead::~ResourceLookahead() {
	for (auto& helper: m_helpers) {
		{
			std::scoped_lock lock(helper->mutex);
			helper->stop = true;
		}
		helper->wake.notify_one();
	}
	for (auto& helper: m_helpers) {
		helper->thread.join();
	}
}

ResourceLookahead::Stream& ResourceLookahead::FindStream(const void* key) {
	for (auto& stream: m_streams) {
		if (stream.key == key) return stream;
	}
	return m_streams.emplace_back(Stream {.key = key});
}

ResourceLookahead::Job* ResourceLookahead::NewJob() {
	if (m_free.empty()) Reclaim();
	Job* job = nullptr;
	if (!m_free.empty()) {
		job = m_free.back();
		m_free.pop_back();
	} else {
		job = m_storage.emplace_back(std::make_unique<Job>()).get();
	}
	job->state.store(State::Queued, std::memory_order_relaxed);
	job->cancelled.store(false, std::memory_order_relaxed);
	job->reads.clear();
	job->bytes.clear();
	return job;
}

void ResourceLookahead::Retire(Job* job) {
	job->cancelled.store(true, std::memory_order_relaxed);
	m_retired.push_back(job);
}

// A retired job returns to the pool once no helper can touch it any more.
void ResourceLookahead::Reclaim() {
	std::erase_if(m_retired, [this](Job* job) {
		const auto state = job->state.load(std::memory_order_acquire);
		if (state == State::Queued || state == State::Running) return false;
		job->plan.reset();
		m_free.push_back(job);
		return true;
	});
}

void ResourceLookahead::BeginRun(const void* stream_key) {
	auto& stream = FindStream(stream_key);
	for (size_t i = stream.next; i < stream.jobs.size(); ++i) {
		Retire(stream.jobs[i]);
	}
	stream.jobs.clear();
	stream.next = 0;
}

void ResourceLookahead::Plan(const void* stream_key, const void* command,
                             std::shared_ptr<LookaheadPlan> plan, const LookaheadInputs& inputs) {
	if (m_helpers.empty() || plan == nullptr) return;
	auto* job         = NewJob();
	job->command      = command;
	job->inputs       = inputs;
	const auto helper = plan->Helper() % m_helpers.size();
	job->plan         = std::move(plan);
	FindStream(stream_key).jobs.push_back(job);
	m_helpers[helper]->staged.push_back(job);
	m_stats.planned++;
}

void ResourceLookahead::Kick() {
	for (auto& helper: m_helpers) {
		if (helper->staged.empty()) continue;
		{
			std::scoped_lock lock(helper->mutex);
			helper->inbox.insert(helper->inbox.end(), helper->staged.begin(), helper->staged.end());
		}
		helper->staged.clear();
		helper->wake.notify_one();
	}
}

void ResourceLookahead::WaitIdle() {
	for (auto& helper: m_helpers) {
		std::unique_lock lock(helper->mutex);
		helper->idle.wait(lock, [&] { return helper->inbox.empty() && !helper->busy; });
	}
}

void ResourceLookahead::RunHelper(Helper& helper) {
	std::vector<Job*> batch;
	for (;;) {
		{
			std::unique_lock lock(helper.mutex);
			helper.busy = false;
			helper.idle.notify_all();
			helper.wake.wait(lock, [&] { return helper.stop || !helper.inbox.empty(); });
			if (helper.stop) return;
			batch.swap(helper.inbox);
			helper.busy = true;
		}
		for (auto* job: batch) {
			if (job->cancelled.load(std::memory_order_relaxed)) {
				job->state.store(State::Skipped, std::memory_order_release);
				continue;
			}
			job->state.store(State::Running, std::memory_order_relaxed);
			RunJob(*job);
		}
		batch.clear();
	}
}

void ResourceLookahead::RunJob(Job& job) {
	const auto* plan = job.plan->Get();
	if (plan == nullptr) {
		job.state.store(State::Failed, std::memory_order_release);
		return;
	}
	HelperReader                  reader(m_memory, job);
	ScalarReadCache<HelperReader> cache(reader);
	HelperReader::Context         context {.reader = reader, .cache = cache};
	static_assert(ScalarReadCache<HelperReader>::ChunkBytes ==
	              ShaderRecompiler::IR::SrtReadCacheView::ChunkBytes);
	const ShaderRecompiler::IR::SrtReadCacheView view {
	    .tags       = cache.Tags(),
	    .data       = cache.Data(),
	    .stride     = ScalarReadCache<HelperReader>::Stride(),
	    .index_mask = static_cast<uint32_t>(ScalarReadCache<HelperReader>::Entries - 1u),
	};
	const ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data =
	        std::span<const uint32_t>(job.inputs.user_data).first(job.inputs.user_data_count),
	    .shader_base                = job.inputs.shader_base,
	    .read_memory                = HelperReader::ReadOrdinary,
	    .userdata                   = &context,
	    .read_specialization_memory = HelperReader::ReadSpecialization,
	    .workgroup_counts           = job.inputs.workgroup_counts,
	    .indirect_args_address      = job.inputs.indirect_args_address,
	    .read_cache                 = &view,
	    // Earlier commands may still write the data it reads; Outcome::Failed counts it.
	    .speculative = true,
	};
	const bool done = ShaderRecompiler::IR::MaterializeResources(*plan, runtime, job.snapshot,
	                                                             job.specialization);
	job.state.store(done ? State::Done : State::Failed, std::memory_order_release);
}

// Every range the helper read: no GPU-dirty bytes, and the backing still reads the same.
bool ResourceLookahead::Unchanged(const Job& job, Outcome& outcome) {
	for (const auto& read: job.reads) {
		if (m_memory.gpu_dirty(read.address, read.size)) {
			outcome = Outcome::Dirty;
			return false;
		}
	}
	for (const auto& read: job.reads) {
		bool same = false;
		if (m_memory.same != nullptr) {
			same =
			    m_memory.same(read.address, job.bytes.data() + read.offset, read.size, read.mapped);
		} else {
			if (m_check.size() < read.size) m_check.resize(read.size);
			const bool mapped = m_memory.read(read.address, m_check.data(), read.size);
			same = mapped == read.mapped &&
			       (!mapped ||
			        std::memcmp(m_check.data(), job.bytes.data() + read.offset, read.size) == 0);
		}
		if (!same) {
			outcome = Outcome::Moved;
			return false;
		}
	}
	return true;
}

ResourceLookahead::Outcome
ResourceLookahead::TryAdopt(const void* stream_key, const void* command, const LookaheadPlan* plan,
                            const LookaheadInputs&                        inputs,
                            ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
                            ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	const auto outcome = [&] {
		auto&  stream = FindStream(stream_key);
		size_t found  = stream.next;
		// A command plans one job per program it walks; one of another program of the command
		// stays for that program.
		while (found < stream.jobs.size() &&
		       (stream.jobs[found]->command != command || stream.jobs[found]->plan.get() != plan)) {
			++found;
		}
		if (found == stream.jobs.size()) return Outcome::NotPlanned;
		// Commands are reached in the order they were planned: the jobs of earlier commands are
		// not needed.
		for (; stream.next < found && stream.jobs[stream.next]->command != command; ++stream.next)
			Retire(stream.jobs[stream.next]);
		Job* job = stream.jobs[found];
		std::swap(stream.jobs[found], stream.jobs[stream.next]);
		stream.next++;
		Retire(job);
		if (!(job->inputs == inputs)) return Outcome::Changed;
		const auto state = job->state.load(std::memory_order_acquire);
		if (state == State::Queued || state == State::Running) return Outcome::Late;
		if (state != State::Done) return Outcome::Failed;
		Outcome result = Outcome::Adopted;
		if (!Unchanged(*job, result)) return result;
		std::swap(snapshot, job->snapshot);
		std::swap(specialization, job->specialization);
		return Outcome::Adopted;
	}();
	m_stats.outcomes[static_cast<size_t>(outcome)]++;
	return outcome;
}

ResourceLookahead::Stats ResourceLookahead::TakeStats() {
	auto stats = m_stats;
	m_stats    = {};
	return stats;
}

} // namespace Libs::Graphics
