#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RESOURCELOOKAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RESOURCELOOKAHEAD_H_

#include "graphics/host_gpu/renderer/pipeline/srtWalkerCompileQueue.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Everything a resource walk reads besides guest memory. A walk is a pure function of its plan,
// these inputs and the guest bytes it reads, so a walk evaluated earlier on another thread gives
// exactly the result the GPU thread would compute when all three are unchanged.
struct LookaheadInputs {
	uint64_t                 shader_base           = 0;
	uint64_t                 indirect_args_address = 0;
	std::array<uint32_t, 3>  workgroup_counts {};
	uint32_t                 user_data_count = 0;
	std::array<uint32_t, 40> user_data {};

	[[nodiscard]] bool operator==(const LookaheadInputs& other) const {
		return shader_base == other.shader_base &&
		       indirect_args_address == other.indirect_args_address &&
		       workgroup_counts == other.workgroup_counts &&
		       user_data_count == other.user_data_count &&
		       std::equal(user_data.begin(), user_data.begin() + user_data_count,
		                  other.user_data.begin());
	}
};

// A program's plan for helper threads: its own extraction (a plan carries the scratch of its
// walks, so it is never shared between threads), whose walker reads through the helper's reader.
// Only the helper the plan is assigned to walks it.
class LookaheadPlan {
public:
	LookaheadPlan(std::shared_ptr<SrtWalkerCompileQueue::Job> job, uint32_t helper)
	    : m_job(std::move(job)), m_helper(helper) {}

	// Helper thread: the plan once its walker is compiled (or found incomplete), else null.
	[[nodiscard]] const ShaderRecompiler::IR::ResourcePlan* Get() const { return m_job->Plan(); }
	[[nodiscard]] uint32_t                                  Helper() const { return m_helper; }

private:
	std::shared_ptr<SrtWalkerCompileQueue::Job> m_job;
	uint32_t                                    m_helper;
};

// Guest memory as the walks see it.
struct LookaheadMemory {
	// Any thread: the backing bytes; false when the range is not mapped.
	bool (*read)(uint64_t address, void* data, uint64_t size) = nullptr;
	// Owner thread only: whether the range holds GPU writes not yet in the backing.
	bool (*gpu_dirty)(uint64_t address, uint64_t size) = nullptr;
	// Owner thread: whether `read` would return exactly `data` (true) or fail exactly when
	// `mapped` is false; null to compare through `read`.
	bool (*same)(uint64_t address, const void* data, uint64_t size, bool mapped) = nullptr;
	// Any thread: SrtRuntime::mapped_range_size (null: every candidate counts as mapped).
	uint64_t (*mapped_size)(uint64_t address, uint64_t size) = nullptr;
};

// Materializes the resources of commands the GPU thread has not reached yet on helper threads.
//
// The owner (the GPU thread) plans a job for each command of a command stream whose inputs it
// can tell before it gets there, hands the jobs to the helpers in one batch, and at the command
// adopts the helper's snapshot instead of walking itself, but only when the job was planned
// for the same plan and inputs, finished, and every byte it read still reads the same and holds
// no GPU write that is not in the backing. Anything else falls back to the owner's own walk, so
// the result is always the one the owner would compute.
class ResourceLookahead {
public:
	enum class Outcome : uint32_t {
		Adopted,
		NotPlanned, // no job for the command
		Changed,    // planned with another plan or other inputs
		Late,       // not finished
		Failed,     // the helper's walk failed
		Dirty,      // read bytes with GPU writes not in the backing
		Moved,      // read bytes changed
		Heap,       // a resident image heap gave the command another view
		Count
	};

	// With a resident image heap, walks take the views of its tables that the owner acquired
	// last (ImageHeapSource::Peek), and the owner acquires them again at the command.
	ResourceLookahead(LookaheadMemory memory, uint32_t helpers,
	                  ShaderRecompiler::IR::ImageHeapSource* heap = nullptr);
	~ResourceLookahead();
	ResourceLookahead(const ResourceLookahead&)            = delete;
	ResourceLookahead& operator=(const ResourceLookahead&) = delete;

	[[nodiscard]] uint32_t HelperCount() const { return static_cast<uint32_t>(m_helpers.size()); }

	// Owner thread. A stream starts a run of commands it plans ahead; jobs it planned before are
	// dropped. Streams are told apart by an opaque key (one per command processor).
	void BeginRun(const void* stream);
	void Plan(const void* stream, const void* command, std::shared_ptr<LookaheadPlan> plan,
	          const LookaheadInputs& inputs);
	// Hands the jobs planned since the last call to the helpers.
	void Kick();
	// Waits until the helpers finished every job handed to them.
	void WaitIdle();
	// At the command: moves the job's result into snapshot and specialization when it may be
	// used. Jobs of the stream planned before this command are dropped.
	Outcome TryAdopt(const void* stream, const void* command, const LookaheadPlan* plan,
	                 const LookaheadInputs&                        inputs,
	                 ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
	                 ShaderRecompiler::IR::ResourceSpecialization& specialization);

	// Owner thread: outcomes and planned jobs since the last call.
	struct Stats {
		std::array<uint64_t, static_cast<size_t>(Outcome::Count)> outcomes {};
		uint64_t                                                  planned = 0;
	};
	Stats TakeStats();

private:
	struct Read {
		uint64_t address = 0;
		uint32_t size    = 0;
		uint32_t offset  = 0;
		bool     mapped  = false;
	};
	enum class State : uint32_t { Queued, Running, Done, Failed, Skipped };
	// What a walk took the resident image heap to do with a table.
	struct HeapUse {
		uint64_t                                           base  = 0;
		uint64_t                                           size  = 0;
		uint64_t                                           step  = 0;
		uint64_t                                           first = 0;
		uint64_t                                           count = 0;
		ShaderRecompiler::IR::ImageResource                root;
		uint64_t                                           slots      = 0;
		uint32_t                                           slot_count = 0;
		std::vector<ShaderRecompiler::IR::DescriptorValue> kinds;
		bool                                               resident = false;
	};
	struct Job {
		std::atomic<State>                           state {State::Queued};
		std::atomic<bool>                            cancelled {false};
		const void*                                  command = nullptr;
		std::shared_ptr<LookaheadPlan>               plan;
		LookaheadInputs                              inputs;
		ShaderRecompiler::IR::ResourceSnapshot       snapshot;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Read>                            reads;
		std::vector<uint8_t>                         bytes;
		std::vector<HeapUse>                         heap_uses;
	};
	struct Stream {
		const void*       key = nullptr;
		std::vector<Job*> jobs;
		size_t            next = 0;
	};
	struct Helper {
		std::mutex              mutex;
		std::condition_variable wake;
		std::vector<Job*>       inbox;
		bool                    stop = false;
		bool                    busy = false;
		std::condition_variable idle;
		std::vector<Job*>       staged; // owner side, handed over by Kick
		std::thread             thread;
	};

	class HelperReader;
	class HelperHeap;

	void    RunHelper(Helper& helper);
	void    RunJob(Job& job);
	bool    Unchanged(const Job& job, Outcome& outcome);
	bool    SameHeapViews(const Job& job);
	Job*    NewJob();
	void    Retire(Job* job);
	void    Reclaim();
	Stream& FindStream(const void* key);

	LookaheadMemory                      m_memory;
	ShaderRecompiler::IR::ImageHeapSource* m_heap = nullptr;
	std::vector<std::unique_ptr<Helper>> m_helpers;
	std::vector<std::unique_ptr<Job>>    m_storage;
	std::vector<Job*>                    m_free;
	std::vector<Job*>                    m_retired;
	std::vector<Stream>                  m_streams;
	std::vector<uint8_t>                 m_check;
	Stats                                m_stats;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RESOURCELOOKAHEAD_H_ */
