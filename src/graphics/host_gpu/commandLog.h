#ifndef KYTY_GRAPHICS_HOST_GPU_COMMAND_LOG_H_
#define KYTY_GRAPHICS_HOST_GPU_COMMAND_LOG_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

// The draws and dispatches recorded into a scheduler's recent submissions, kept so that a lost
// device can name the work of the submission the GPU did not finish. Vulkan-free.
//
// Noting a command costs one atomic increment and a few relaxed stores into a fixed ring: no
// allocation, no lock, no hashing. Readers (the thread that reports a lost device) may run while
// commands are noted: every slot carries the sequence number of the command it holds, written
// last, and a reader keeps a slot only when that number is the same before and after it copied
// the fields.
namespace Libs::Graphics::CommandLog {

enum class Kind : uint8_t {
	None,
	Dispatch,         // x, y, z: workgroup counts
	DispatchIndirect, // arguments read on the GPU
	Draw,             // x: vertex count, y: instance count
	DrawIndexed,      // x: index count, y: instance count
	DrawIndirect,     // x: draw count (the maximum with a count buffer)
	DrawMesh,         // x, y, z: mesh workgroup counts
	DrawMeshIndirect, // x: draw count
	Internal,         // an emulator pass: `name`
	Scatter,          // an upload scatter dispatch (see Command)
	UploadCopies,     // the copy commands that stood in for a scatter dispatch
};

inline constexpr uint8_t FlagAsyncCompute = 1u << 0u; // from a guest compute queue
// Binds the bindless image table; `bindless_*` hold the table's slot counters at recording.
inline constexpr uint8_t FlagBindless = 1u << 1u;

// The slot counters of the bindless image table (BindlessTable): descriptor writes, the highest
// slot index written, and reuses: writes of an image into a slot that held a descriptor before
// (a slot freed and handed out again).
struct BindlessCounters {
	uint64_t writes  = 0;
	uint32_t highest = 0;
	uint64_t reuses  = 0;
};

// A command keeps the counters in one word: the writes and reuses modulo these bounds (a pending
// command never trails the table by that many), the slot below BindlessSlotBound.
inline constexpr uint64_t BindlessWritesBound = uint64_t {1} << 32u;
inline constexpr uint32_t BindlessSlotBound   = 1u << 12u;
inline constexpr uint64_t BindlessReusesBound = uint64_t {1} << 20u;
static_assert(std::countr_zero(BindlessWritesBound) + std::countr_zero(BindlessSlotBound) +
                  std::countr_zero(BindlessReusesBound) ==
              64);

inline uint64_t PackBindless(const BindlessCounters& counters) noexcept {
	return (counters.writes % BindlessWritesBound) |
	       (uint64_t {counters.highest % BindlessSlotBound} << 32u) |
	       ((counters.reuses % BindlessReusesBound) << 44u);
}

inline BindlessCounters UnpackBindless(uint64_t word) noexcept {
	return {word % BindlessWritesBound, static_cast<uint32_t>((word >> 32u) % BindlessSlotBound),
	        word >> 44u};
}

// Counts the bindless table's slot descriptor writes. One writer (the thread that writes the
// descriptors); any thread may read. O(1) per write.
class BindlessSlotTally {
public:
	// A descriptor write of `slot` (below BindlessSlotBound); `image` is false for a null one.
	void Note(uint32_t slot, bool image) noexcept {
		Increment(m_writes);
		if (!m_written[slot]) {
			m_written[slot] = true;
			Increment(m_slots_written);
		} else if (image) {
			Increment(m_reuses);
		}
		if (slot > m_highest.load(std::memory_order_relaxed)) {
			m_highest.store(slot, std::memory_order_relaxed);
		}
	}

	[[nodiscard]] BindlessCounters Counters() const noexcept {
		return {m_writes.load(std::memory_order_relaxed), m_highest.load(std::memory_order_relaxed),
		        m_reuses.load(std::memory_order_relaxed)};
	}
	[[nodiscard]] uint64_t Writes() const noexcept {
		return m_writes.load(std::memory_order_relaxed);
	}
	// Slots below the highest one written that never got a descriptor.
	[[nodiscard]] uint32_t NeverWritten() const noexcept {
		const uint64_t written = m_slots_written.load(std::memory_order_relaxed);
		const uint64_t slots   = uint64_t {m_highest.load(std::memory_order_relaxed)} + 1u;
		return written == 0 ? 0 : static_cast<uint32_t>(slots - std::min(written, slots));
	}

private:
	template <typename T>
	static void Increment(std::atomic<T>& counter) noexcept {
		counter.store(counter.load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
	}

	std::atomic<uint64_t>          m_writes {0};
	std::atomic<uint64_t>          m_slots_written {0};
	std::atomic<uint64_t>          m_reuses {0};
	std::atomic<uint32_t>          m_highest {0};
	std::bitset<BindlessSlotBound> m_written;
};

struct Command {
	Kind    kind  = Kind::None;
	uint8_t flags = 0;
	// Shader hashes: the compute shader, or the last vertex-pipeline stage (VS, ES/GS or mesh)
	// and the pixel shader; 0 for none.
	uint64_t shader0 = 0;
	uint64_t shader1 = 0;
	// Kind::Scatter, Kind::UploadCopies: x, y = the first and the end source offset copied;
	// Kind::Scatter: z = its pieces.
	// Kind::DispatchIndirect: x, y = low and high word of the guest address of the arguments.
	uint32_t x       = 0;
	uint32_t y       = 0;
	uint32_t z       = 0;
	// For Kind::Internal: a string with static storage duration.
	const char* name = nullptr;
	// Kind::DispatchIndirect: the device address of the arguments. Kind::Scatter and
	// Kind::UploadCopies: the device address of the source buffer the offsets are in.
	uint64_t address = 0;
	// Kind::Scatter: the device address of its first piece table (header, then the pieces).
	uint64_t table = 0;
	// With FlagBindless: the bindless table's slot counters when the command was recorded
	// (modulo the BindlessCounters bounds).
	uint64_t bindless_writes  = 0;
	uint32_t bindless_highest = 0;
	uint64_t bindless_reuses  = 0;
};

struct Entry {
	uint64_t sequence = 0; // 1 for the first command the log noted
	uint64_t tick     = 0; // the submission the command was recorded into
	Command  command;
};

// One checkpoint the GPU passed on its way through a pipeline stage
// (VK_NV_device_diagnostic_checkpoints), with the marker MakeMarker gave the command.
struct Checkpoint {
	std::string_view stage;
	uint64_t         marker = 0;
};

// A checkpoint marker names the log (several schedulers share a queue) and the sequence.
inline constexpr uint32_t MarkerLogShift = 56;
inline constexpr uint64_t MakeMarker(uint32_t log_id, uint64_t sequence) {
	return (uint64_t {log_id} << MarkerLogShift) |
	       (sequence & ((uint64_t {1} << MarkerLogShift) - 1u));
}

class Ring {
public:
	// `capacity` is rounded up to a power of two.
	explicit Ring(uint32_t log_id, uint32_t capacity)
	    : m_log_id(log_id), m_mask(std::bit_ceil(std::max(capacity, 2u)) - 1u),
	      m_slots(std::make_unique<Slot[]>(size_t {m_mask} + 1u)) {}

	[[nodiscard]] uint32_t LogId() const noexcept { return m_log_id; }
	[[nodiscard]] uint32_t Capacity() const noexcept { return m_mask + 1u; }
	// Commands noted so far: the sequence number of the newest one.
	[[nodiscard]] uint64_t Count() const noexcept { return m_next.load(std::memory_order_acquire); }

	// Notes a command recorded into submission `tick`; returns its sequence number. Commands of
	// one writer are noted in recording order, so their ticks never decrease.
	uint64_t Note(uint64_t tick, const Command& command) noexcept {
		const uint64_t sequence = m_next.fetch_add(1, std::memory_order_relaxed) + 1u;
		Slot&          slot     = m_slots[sequence & m_mask];
		slot.sequence.store(0, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		slot.tick.store(tick, std::memory_order_relaxed);
		slot.shader0.store(command.shader0, std::memory_order_relaxed);
		slot.shader1.store(command.shader1, std::memory_order_relaxed);
		slot.xy.store(uint64_t {command.x} | (uint64_t {command.y} << 32u),
		              std::memory_order_relaxed);
		slot.z_kind.store(uint64_t {command.z} |
		                      (uint64_t {static_cast<uint8_t>(command.kind)} << 32u) |
		                      (uint64_t {command.flags} << 40u),
		                  std::memory_order_relaxed);
		slot.word.store(PackWord(command), std::memory_order_relaxed);
		slot.address.store(command.address, std::memory_order_relaxed);
		slot.sequence.store(sequence, std::memory_order_release);
		return sequence;
	}

	// The command with this sequence number, or false when it was overwritten, not yet noted or
	// being rewritten.
	bool Read(uint64_t sequence, Entry& out) const noexcept {
		if (sequence == 0) {
			return false;
		}
		const Slot& slot = m_slots[sequence & m_mask];
		if (slot.sequence.load(std::memory_order_acquire) != sequence) {
			return false;
		}
		Entry entry;
		entry.sequence        = sequence;
		entry.tick            = slot.tick.load(std::memory_order_relaxed);
		entry.command.shader0 = slot.shader0.load(std::memory_order_relaxed);
		entry.command.shader1 = slot.shader1.load(std::memory_order_relaxed);
		const uint64_t xy     = slot.xy.load(std::memory_order_relaxed);
		const uint64_t z_kind = slot.z_kind.load(std::memory_order_relaxed);
		const uint64_t word   = slot.word.load(std::memory_order_relaxed);
		entry.command.address = slot.address.load(std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_acquire);
		if (slot.sequence.load(std::memory_order_relaxed) != sequence) {
			return false;
		}
		entry.command.x     = static_cast<uint32_t>(xy);
		entry.command.y     = static_cast<uint32_t>(xy >> 32u);
		entry.command.z     = static_cast<uint32_t>(z_kind);
		entry.command.kind  = static_cast<Kind>(static_cast<uint8_t>(z_kind >> 32u));
		entry.command.flags = static_cast<uint8_t>(z_kind >> 40u);
		UnpackWord(word, entry.command);
		out = entry;
		return true;
	}

private:
	// One slot word holds the field that matters for the kind: a pass name, a scatter's table
	// address or the bindless slot counters. That keeps a slot in one cache line.
	static uint64_t PackWord(const Command& command) noexcept {
		switch (command.kind) {
			case Kind::Internal: return reinterpret_cast<uintptr_t>(command.name);
			case Kind::Scatter: return command.table;
			default:
				return PackBindless(
				    {command.bindless_writes, command.bindless_highest, command.bindless_reuses});
		}
	}
	static void UnpackWord(uint64_t word, Command& command) noexcept {
		switch (command.kind) {
			case Kind::Internal:
				command.name = reinterpret_cast<const char*>(static_cast<uintptr_t>(word));
				break;
			case Kind::Scatter: command.table = word; break;
			default: {
				const auto counters      = UnpackBindless(word);
				command.bindless_writes  = counters.writes;
				command.bindless_highest = counters.highest;
				command.bindless_reuses  = counters.reuses;
				break;
			}
		}
	}

	struct alignas(64) Slot {
		std::atomic<uint64_t>    sequence {0};
		std::atomic<uint64_t>    tick {0};
		std::atomic<uint64_t>    shader0 {0};
		std::atomic<uint64_t>    shader1 {0};
		std::atomic<uint64_t>    xy {0};
		std::atomic<uint64_t>    z_kind {0};
		std::atomic<uint64_t>    word {0};
		std::atomic<uint64_t>    address {0};
	};
	static_assert(sizeof(Slot) == 64);

	uint32_t                m_log_id;
	uint32_t                m_mask;
	std::unique_ptr<Slot[]> m_slots;
	std::atomic<uint64_t>   m_next {0};
};

// At most this many commands of the lost submission are written from its start, and as many
// from its end; at most this many distinct shaders are counted.
inline constexpr size_t ShownCommands = 64;
inline constexpr size_t ShownShaders  = 32;

inline const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::Dispatch: return "dispatch";
		case Kind::DispatchIndirect: return "dispatch indirect";
		case Kind::Draw: return "draw";
		case Kind::DrawIndexed: return "draw indexed";
		case Kind::DrawIndirect: return "draw indirect";
		case Kind::DrawMesh: return "draw mesh";
		case Kind::DrawMeshIndirect: return "draw mesh indirect";
		case Kind::Internal: return "internal";
		case Kind::Scatter: return "upload scatter";
		case Kind::UploadCopies: return "upload copies";
		default: return "unknown";
	}
}

inline bool IsDispatch(Kind kind) {
	return kind == Kind::Dispatch || kind == Kind::DispatchIndirect;
}

// One command as the end of a line: what it is, its shaders and counts. `bindless_now`, when
// known, are the bindless table's counters at the report: a bindless command also says how many
// slots were written and reused while it was pending.
inline std::string Describe(const Command&                         c,
                            const std::optional<BindlessCounters>& bindless_now = std::nullopt) {
	char line[200];
	switch (c.kind) {
		case Kind::Dispatch:
			std::snprintf(line, sizeof(line), "dispatch CS %016" PRIx64 " groups %ux%ux%u",
			              c.shader0, c.x, c.y, c.z);
			break;
		case Kind::DispatchIndirect:
			std::snprintf(line, sizeof(line),
			              "dispatch indirect CS %016" PRIx64 " args 0x%" PRIx64 " (guest 0x%" PRIx64
			              ")",
			              c.shader0, c.address, uint64_t {c.x} | (uint64_t {c.y} << 32u));
			break;
		case Kind::Draw:
		case Kind::DrawIndexed:
			std::snprintf(line, sizeof(line),
			              "%s VS %016" PRIx64 " PS %016" PRIx64 " count %u instances %u",
			              KindName(c.kind), c.shader0, c.shader1, c.x, c.y);
			break;
		case Kind::DrawIndirect:
		case Kind::DrawMeshIndirect:
			std::snprintf(line, sizeof(line), "%s %s %016" PRIx64 " PS %016" PRIx64 " draws %u",
			              KindName(c.kind), c.kind == Kind::DrawMeshIndirect ? "MS" : "VS",
			              c.shader0, c.shader1, c.x);
			break;
		case Kind::DrawMesh:
			std::snprintf(line, sizeof(line),
			              "draw mesh MS %016" PRIx64 " PS %016" PRIx64 " groups %ux%ux%u",
			              c.shader0, c.shader1, c.x, c.y, c.z);
			break;
		case Kind::Internal:
			std::snprintf(line, sizeof(line), "internal %s", c.name != nullptr ? c.name : "pass");
			break;
		case Kind::Scatter:
			std::snprintf(line, sizeof(line),
			              "upload scatter source 0x%" PRIx64
			              " offsets 0x%x-0x%x pieces %u table 0x%" PRIx64,
			              c.address, c.x, c.y, c.z, c.table);
			break;
		case Kind::UploadCopies:
			std::snprintf(line, sizeof(line),
			              "upload copies source 0x%" PRIx64 " offsets 0x%x-0x%x", c.address, c.x,
			              c.y);
			break;
		default: std::snprintf(line, sizeof(line), "unknown command"); break;
	}
	std::string out = line;
	if ((c.flags & FlagBindless) != 0) {
		std::snprintf(line, sizeof(line),
		              " bindless (slot writes %" PRIu64 ", highest slot %u, reuses %" PRIu64,
		              c.bindless_writes, c.bindless_highest, c.bindless_reuses);
		out += line;
		if (bindless_now) {
			const auto writes = (bindless_now->writes - c.bindless_writes) % BindlessWritesBound;
			const auto reuses = (bindless_now->reuses - c.bindless_reuses) % BindlessReusesBound;
			std::snprintf(line, sizeof(line),
			              "; since: %" PRIu64 " slot writes, %" PRIu64 " reuses, highest slot %u",
			              writes, reuses, bindless_now->highest);
			out += line;
		}
		out += ')';
	}
	if ((c.flags & FlagAsyncCompute) != 0) {
		out += " (compute queue)";
	}
	return out;
}

// The last completed tick a report can rely on. A timeline counter read after a lost device may
// be any value (some drivers return UINT64_MAX): only a tick that was submitted counts, otherwise
// the last value the scheduler read before the loss (`known`).
inline uint64_t CompletedTick(bool read, uint64_t counter, uint64_t known, uint64_t submitted) {
	if (read && counter <= submitted) {
		return counter;
	}
	return std::min(known, submitted);
}

// A line about a counter read that CompletedTick did not use, or an empty string.
inline std::string FormatCounterNote(std::string_view log_name, bool read, uint64_t counter,
                                     uint64_t known, uint64_t submitted) {
	if (read && counter <= submitted) {
		return {};
	}
	char      line[200];
	const int name_size = static_cast<int>(std::min<size_t>(log_name.size(), 32));
	if (read) {
		std::snprintf(line, sizeof(line),
		              "Lost submission (%.*s): GPU counter after the loss %" PRIu64
		              " is not a submitted tick; using the last one read before, %" PRIu64 "\n",
		              name_size, log_name.data(), counter, std::min(known, submitted));
	} else {
		std::snprintf(line, sizeof(line),
		              "Lost submission (%.*s): GPU counter unreadable after the loss; using the "
		              "last one read before, %" PRIu64 "\n",
		              name_size, log_name.data(), std::min(known, submitted));
	}
	return line;
}

// The report about one log: the commands noted for the first submission the GPU did not finish
// (`completed` + 1), which holds a command that never finished, in recording order. When none of
// its commands is held any more, or `completed` is not a submitted tick (unknown), the commands
// of the oldest submission still held are listed instead: the first unfinished one is that one or
// an older one. `log_name` names the scheduler. `checkpoints` (may be empty) are the queue's
// checkpoints; those of other logs are left out. `bindless_now`: see Describe. Lines end in a
// newline.
inline std::string Format(const Ring& ring, std::string_view log_name, uint64_t completed,
                          uint64_t submitted, std::span<const Checkpoint> checkpoints = {},
                          const std::optional<BindlessCounters>& bindless_now = std::nullopt) {
	std::string    out;
	char           line[256];
	const bool     known  = completed <= submitted;
	uint64_t       lost   = known ? completed + 1u : 0;
	const uint64_t newest = ring.Count();
	const auto     prefix = "Lost submission (" + std::string(log_name) + ")";
	if (known && submitted < lost) {
		std::snprintf(line, sizeof(line), "%s: none, GPU at tick %" PRIu64 "\n", prefix.c_str(),
		              completed);
		return out + line;
	}
	// The commands of the lost tick are contiguous: find them from the newest command back.
	const uint64_t oldest = newest > ring.Capacity() ? newest - ring.Capacity() + 1u : 1u;
	uint64_t       first  = 0;
	uint64_t       last   = 0;
	uint64_t       later  = 0; // commands of later submissions
	bool           gap    = false;
	bool           older  = false;      // reached a command of an earlier submission
	uint64_t       lowest = UINT64_MAX; // the lowest tick of the commands read
	Entry          entry;
	const auto     scan = [&](uint64_t target) {
		first = 0;
		last  = 0;
		later = 0;
		gap   = false;
		older = false;
		for (uint64_t sequence = newest; sequence >= oldest && sequence != 0; sequence--) {
			if (!ring.Read(sequence, entry)) {
				gap = true;
				break;
			}
			lowest = std::min(lowest, entry.tick);
			if (entry.tick > target) {
				later++;
				continue;
			}
			if (entry.tick < target) {
				older = true;
				break;
			}
			if (last == 0) {
				last = sequence;
			}
			first = sequence;
		}
	};
	scan(lost);
	// Nothing of the lost tick is held and the held commands do not reach back before it.
	const bool fallback = first == 0 && !older && later != 0;
	if (fallback) {
		lost = lowest;
		scan(lost);
	}
	// When the oldest command still held belongs to the lost submission, earlier ones may have
	// been overwritten.
	const bool     truncated = first != 0 && (first == oldest || gap) && first > 1u;
	const uint64_t count     = first != 0 ? last - first + 1u : 0;
	char           gpu[32];
	if (known) {
		std::snprintf(gpu, sizeof(gpu), "%" PRIu64, completed);
	} else {
		std::snprintf(gpu, sizeof(gpu), "unknown");
	}
	std::snprintf(line, sizeof(line),
	              "%s: tick %" PRIu64 "%s, GPU at %s, submitted up to %" PRIu64 ", %" PRIu64
	              " command(s) noted%s\n",
	              prefix.c_str(), lost,
	              fallback ? " (the oldest held: the first unfinished one is not held)" : "", gpu,
	              submitted, count, truncated ? " (earlier ones overwritten)" : "");
	out += line;

	struct ShaderCount {
		Kind     kind    = Kind::None;
		uint64_t shader0 = 0;
		uint64_t shader1 = 0;
		uint64_t count   = 0;
	};
	std::array<ShaderCount, ShownShaders> shaders {};
	size_t                                distinct = 0;
	uint64_t                              others   = 0;
	for (uint64_t sequence = first; count != 0 && sequence <= last; sequence++) {
		const uint64_t index = sequence - first;
		const bool     shown = index < ShownCommands || last - sequence < ShownCommands;
		if (!ring.Read(sequence, entry)) {
			if (shown) {
				std::snprintf(line, sizeof(line), "%s command %" PRIu64 ": overwritten\n",
				              prefix.c_str(), index);
				out += line;
			}
			continue;
		}
		if (shown) {
			std::snprintf(line, sizeof(line), "%s command %" PRIu64 ": ", prefix.c_str(), index);
			out += line;
			out += Describe(entry.command, bindless_now);
			out += '\n';
		} else if (index == ShownCommands) {
			std::snprintf(line, sizeof(line), "%s commands not shown: %" PRIu64 "\n",
			              prefix.c_str(), count - 2u * ShownCommands);
			out += line;
		}
		const auto& c = entry.command;
		if (c.kind == Kind::Internal || c.kind == Kind::Scatter || c.kind == Kind::UploadCopies) {
			continue;
		}
		const Kind group = IsDispatch(c.kind) ? Kind::Dispatch : Kind::Draw;
		auto       found =
		    std::find_if(shaders.begin(), shaders.begin() + distinct, [&](const ShaderCount& s) {
			    return s.kind == group && s.shader0 == c.shader0 && s.shader1 == c.shader1;
		    });
		if (found != shaders.begin() + distinct) {
			found->count++;
		} else if (distinct < shaders.size()) {
			shaders[distinct++] = {group, c.shader0, c.shader1, 1};
		} else {
			others++;
		}
	}
	for (size_t i = 0; i < distinct; i++) {
		const auto& s = shaders[i];
		if (s.kind == Kind::Dispatch) {
			std::snprintf(line, sizeof(line), "%s shader: CS %016" PRIx64 " x %" PRIu64 "\n",
			              prefix.c_str(), s.shader0, s.count);
		} else {
			std::snprintf(line, sizeof(line),
			              "%s shader: VS %016" PRIx64 " PS %016" PRIx64 " x %" PRIu64 "\n",
			              prefix.c_str(), s.shader0, s.shader1, s.count);
		}
		out += line;
	}
	if (others != 0) {
		std::snprintf(line, sizeof(line), "%s shader: others x %" PRIu64 "\n", prefix.c_str(),
		              others);
		out += line;
	}
	if (later != 0) {
		std::snprintf(line, sizeof(line), "%s later submissions: %" PRIu64 " command(s) noted\n",
		              prefix.c_str(), later);
		out += line;
	}
	// A checkpoint marker precedes each noted command: the last one a stage passed names the
	// newest command that reached it.
	for (const auto& checkpoint: checkpoints) {
		if ((checkpoint.marker >> MarkerLogShift) != ring.LogId()) {
			continue;
		}
		const uint64_t sequence = checkpoint.marker & ((uint64_t {1} << MarkerLogShift) - 1u);
		std::snprintf(line, sizeof(line), "%s checkpoint %.*s: ", prefix.c_str(),
		              static_cast<int>(std::min<size_t>(checkpoint.stage.size(), 64)),
		              checkpoint.stage.data());
		out += line;
		if (!ring.Read(sequence, entry)) {
			std::snprintf(line, sizeof(line), "command #%" PRIu64 " (overwritten)\n", sequence);
			out += line;
			continue;
		}
		if (entry.tick == lost && first != 0) {
			std::snprintf(line, sizeof(line), "command %" PRIu64 " of tick %" PRIu64 ", ",
			              sequence - first, entry.tick);
		} else {
			std::snprintf(line, sizeof(line), "tick %" PRIu64 ", ", entry.tick);
		}
		out += line;
		out += Describe(entry.command);
		out += '\n';
	}
	return out;
}

} // namespace Libs::Graphics::CommandLog

#endif /* KYTY_GRAPHICS_HOST_GPU_COMMAND_LOG_H_ */
