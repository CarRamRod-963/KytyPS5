#include "graphics/host_gpu/vulkanRecording.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/recordingStream.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

namespace Libs::Graphics::VulkanRecording {
namespace {

using Dispatcher = vk::detail::DispatchLoaderDynamic;
using Recording::Stream;
using Recording::Writer;

// The driver's entries: the default dispatcher's entries before Install replaced them.
Dispatcher g_driver;
// The driver's entries of the previous Install (a dispatcher initialized again keeps the thunks
// in the entries its initialization does not load).
Dispatcher        g_previous;
std::atomic<bool> g_installed {false};

// The calling thread's stream, while it is a producer.
thread_local Stream* t_stream = nullptr;
// The command buffer the producer queued its last call for (owned until its end or a Drain).
thread_local VkCommandBuffer t_owned = VK_NULL_HANDLE;

// Command buffers a producer has begun and not ended or handed over (Drain). Only the producer's
// recording thread may record into them: a call from another thread would race with it and lose
// its order.
constexpr size_t OpenSlots = 64;

struct OpenBuffer {
	std::atomic<VkCommandBuffer> buffer {VK_NULL_HANDLE};
	std::atomic<const Stream*>   owner {nullptr};
};

std::array<OpenBuffer, OpenSlots> g_open {};

void NoteOpen(VkCommandBuffer buffer, const Stream* owner) {
	for (auto& slot: g_open) {
		if (slot.buffer.load(std::memory_order_relaxed) == buffer) {
			slot.owner.store(owner, std::memory_order_relaxed);
			return;
		}
	}
	for (auto& slot: g_open) {
		VkCommandBuffer empty = VK_NULL_HANDLE;
		if (slot.buffer.compare_exchange_strong(empty, buffer, std::memory_order_relaxed)) {
			slot.owner.store(owner, std::memory_order_relaxed);
			return;
		}
	}
	// More open buffers than slots: the check below misses the rest, nothing else depends on it.
}

void NoteClosed(VkCommandBuffer buffer) {
	for (auto& slot: g_open) {
		VkCommandBuffer open = buffer;
		if (slot.buffer.compare_exchange_strong(open, VK_NULL_HANDLE, std::memory_order_relaxed)) {
			return;
		}
	}
}

// The producer made every queued call: its open buffers may be used by other threads.
void ReleaseOpen(const Stream* owner) {
	t_owned = VK_NULL_HANDLE;
	for (auto& slot: g_open) {
		if (slot.owner.load(std::memory_order_relaxed) == owner) {
			slot.owner.store(nullptr, std::memory_order_relaxed);
			slot.buffer.store(VK_NULL_HANDLE, std::memory_order_relaxed);
		}
	}
}

void CheckNotOpen(VkCommandBuffer buffer, const char* name) {
	for (const auto& slot: g_open) {
		if (slot.buffer.load(std::memory_order_relaxed) == buffer) {
			EXIT("Vulkan recording: %s on another thread into a command buffer the recording "
			     "thread fills\n",
			     name);
		}
	}
}

void CheckReplayed(VkResult result, const char* name) {
	if (result != VK_SUCCESS) {
		EXIT("Vulkan recording: %s failed on the recording thread: %s\n", name,
		     vk::to_string(static_cast<vk::Result>(result)).c_str());
	}
}

// A copy of `count` structures with an empty extension chain (others cannot be copied: the call
// then goes to the driver directly).
template <typename T>
T* CopyStructs(Writer& writer, const T* source, size_t count) {
	auto* copy = writer.Copy(source, count);
	if constexpr (requires { copy->pNext; }) {
		for (size_t index = 0; copy != nullptr && index < count; ++index) {
			if (copy[index].pNext != nullptr) {
				writer.Fail();
				return nullptr;
			}
		}
	}
	return copy;
}

const VkWriteDescriptorSet* CopyWrites(Writer& writer, const VkWriteDescriptorSet* writes,
                                       uint32_t count) {
	auto* copy = CopyStructs(writer, writes, count);
	for (uint32_t index = 0; copy != nullptr && index < count; ++index) {
		auto&       write  = copy[index];
		const auto& source = writes[index];
		// Only the payload the descriptor type selects is read; the other fields need not point to
		// readable memory.
		write.pImageInfo       = nullptr;
		write.pBufferInfo      = nullptr;
		write.pTexelBufferView = nullptr;
		switch (write.descriptorType) {
			case VK_DESCRIPTOR_TYPE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
				write.pImageInfo = writer.Copy(source.pImageInfo, write.descriptorCount);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
				write.pBufferInfo = writer.Copy(source.pBufferInfo, write.descriptorCount);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
				write.pTexelBufferView =
				    writer.Copy(source.pTexelBufferView, write.descriptorCount);
				break;
			default: writer.Fail(); return nullptr;
		}
	}
	return copy;
}

const VkDependencyInfo* CopyDependency(Writer& writer, const VkDependencyInfo* info) {
	auto* copy = CopyStructs(writer, info, 1);
	if (copy != nullptr) {
		copy->pMemoryBarriers =
		    CopyStructs(writer, copy->pMemoryBarriers, copy->memoryBarrierCount);
		copy->pBufferMemoryBarriers =
		    CopyStructs(writer, copy->pBufferMemoryBarriers, copy->bufferMemoryBarrierCount);
		copy->pImageMemoryBarriers =
		    CopyStructs(writer, copy->pImageMemoryBarriers, copy->imageMemoryBarrierCount);
	}
	return copy;
}

const VkRenderingInfo* CopyRendering(Writer& writer, const VkRenderingInfo* info) {
	auto* copy = CopyStructs(writer, info, 1);
	if (copy != nullptr) {
		copy->pColorAttachments =
		    CopyStructs(writer, copy->pColorAttachments, copy->colorAttachmentCount);
		copy->pDepthAttachment   = CopyStructs(writer, copy->pDepthAttachment, 1);
		copy->pStencilAttachment = CopyStructs(writer, copy->pStencilAttachment, 1);
	}
	return copy;
}

// An argument a recorded call can keep by value: anything but a pointer to data (handles are
// pointers to non-const opaque types).
template <typename T>
constexpr bool ByValue = !(std::is_pointer_v<T> && std::is_const_v<std::remove_pointer_t<T>>);

template <typename T>
bool NullOrValue(T value) {
	if constexpr (ByValue<T>) {
		return true;
	} else {
		return value == nullptr;
	}
}

template <typename First, typename... Rest>
void OwnFirst(const Stream* stream, First first, Rest...) {
	if constexpr (std::is_same_v<First, VkCommandBuffer>) {
		if (first != t_owned) {
			NoteOpen(first, stream);
			t_owned = first;
		}
	}
}

template <typename First, typename... Rest>
void CheckFirstNotOpen(const char* name, First first, Rest...) {
	if constexpr (std::is_same_v<First, VkCommandBuffer>) {
		CheckNotOpen(first, name);
	}
}

template <auto Member, typename Pfn = std::remove_cvref_t<decltype(g_driver.*Member)>>
struct Entry;

template <auto Member, typename R, typename... A>
struct Entry<Member, R(VKAPI_PTR*)(A...)> {
	static constexpr bool by_value = (ByValue<A> && ...);

	static inline const char*       name = "";
	static inline std::atomic<bool> reported {false};

	static R Driver(A... args) { return (g_driver.*Member)(args...); }

	// The producer could not queue the call: it waits for the recording thread and calls the
	// driver itself. Reported once per entry (the calls stay correct, only slower).
	static void Direct(Stream& stream) {
		if (!reported.exchange(true, std::memory_order_relaxed)) {
			LOGF("Vulkan recording: %s with arguments the stream does not copy: called "
			     "directly\n",
			     name);
		}
		stream.Drain();
	}

	// A command buffer command with data pointers no thunk copies: waits for the queued calls like
	// Ordered. Reported once per entry, since every call costs a wait for the recording thread.
	static VKAPI_ATTR R VKAPI_CALL OrderedCommand(A... args) {
		if (t_stream != nullptr && !reported.exchange(true, std::memory_order_relaxed)) {
			LOGF("Vulkan recording: %s has arguments the stream does not copy: waits for the "
			     "queued calls\n",
			     name);
		}
		return Ordered(args...);
	}

	// Waits for the queued calls, then calls the driver.
	static VKAPI_ATTR R VKAPI_CALL Ordered(A... args) {
		if (auto* stream = t_stream) {
			stream->Drain();
		} else if constexpr (sizeof...(A) != 0) {
			CheckFirstNotOpen(name, args...);
		}
		return Driver(args...);
	}

	// Queues the call when every data pointer among its arguments is null (an allocator, an
	// optional array): its arguments are then values.
	static VKAPI_ATTR void VKAPI_CALL Queued(A... args) {
		static_assert(std::is_void_v<R>);
		if (auto* stream = t_stream) {
			OwnFirst(stream, args...);
			if ((NullOrValue(args) && ...) &&
			    stream->Enqueue([&](Writer& writer) { writer.Call([=] { Driver(args...); }); })) {
				return;
			}
			Direct(*stream);
		} else if constexpr (sizeof...(A) != 0) {
			CheckFirstNotOpen(name, args...);
		}
		Driver(args...);
	}
};

// A call whose arguments `capture` copies: queued on a producer, made directly otherwise.
template <auto Member, typename Capture, typename... A>
void Record(const Capture& capture, A... args) {
	if (auto* stream = t_stream) {
		OwnFirst(stream, args...);
		if (stream->Enqueue(capture)) {
			return;
		}
		Entry<Member>::Direct(*stream);
	} else {
		CheckFirstNotOpen(Entry<Member>::name, args...);
	}
	(g_driver.*Member)(args...);
}

// The thunks of commands with data pointers, one per argument shape; each serves the entry
// `Member` (a command and its aliases share a shape).

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer b, VkPipelineStageFlags src,
                                              VkPipelineStageFlags dst, VkDependencyFlags flags,
                                              uint32_t mc, const VkMemoryBarrier* m, uint32_t bc,
                                              const VkBufferMemoryBarrier* bb, uint32_t ic,
                                              const VkImageMemoryBarrier* ib) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* m_copy  = CopyStructs(w, m, mc);
		    auto* bb_copy = CopyStructs(w, bb, bc);
		    auto* ib_copy = CopyStructs(w, ib, ic);
		    w.Call([=] {
			    (g_driver.*Member)(b, src, dst, flags, mc, m_copy, bc, bb_copy, ic, ib_copy);
		    });
	    },
	    b, src, dst, flags, mc, m, bc, bb, ic, ib);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdDependency(VkCommandBuffer b, const VkDependencyInfo* info) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyDependency(w, info);
		    w.Call([=] { (g_driver.*Member)(b, copy); });
	    },
	    b, info);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdBeginRendering(VkCommandBuffer b, const VkRenderingInfo* info) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyRendering(w, info);
		    w.Call([=] { (g_driver.*Member)(b, copy); });
	    },
	    b, info);
}

// (b, x, y, count, regions): copies, blits, buffer fills of regions.
template <auto Member, typename X, typename Y, typename Region>
VKAPI_ATTR void VKAPI_CALL CmdRegions2(VkCommandBuffer b, X x, Y y, uint32_t count,
                                       const Region* regions) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, regions, count);
		    w.Call([=] { (g_driver.*Member)(b, x, y, count, copy); });
	    },
	    b, x, y, count, regions);
}

// (b, src, src layout, dst, dst layout, count, regions).
template <auto Member, typename Region>
VKAPI_ATTR void VKAPI_CALL CmdImageRegions(VkCommandBuffer b, VkImage src, VkImageLayout sl,
                                           VkImage dst, VkImageLayout dl, uint32_t count,
                                           const Region* regions) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, regions, count);
		    w.Call([=] { (g_driver.*Member)(b, src, sl, dst, dl, count, copy); });
	    },
	    b, src, sl, dst, dl, count, regions);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer b, VkImage src, VkImageLayout sl,
                                        VkImage dst, VkImageLayout dl, uint32_t count,
                                        const VkImageBlit* regions, VkFilter filter) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, regions, count);
		    w.Call([=] { (g_driver.*Member)(b, src, sl, dst, dl, count, copy, filter); });
	    },
	    b, src, sl, dst, dl, count, regions, filter);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer b, VkBuffer src, VkImage dst,
                                                VkImageLayout layout, uint32_t count,
                                                const VkBufferImageCopy* regions) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, regions, count);
		    w.Call([=] { (g_driver.*Member)(b, src, dst, layout, count, copy); });
	    },
	    b, src, dst, layout, count, regions);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer b, VkImage src,
                                                VkImageLayout layout, VkBuffer dst, uint32_t count,
                                                const VkBufferImageCopy* regions) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, regions, count);
		    w.Call([=] { (g_driver.*Member)(b, src, layout, dst, count, copy); });
	    },
	    b, src, layout, dst, count, regions);
}

// (b, image, layout, value, count, ranges): image clears.
template <auto Member, typename Value>
VKAPI_ATTR void VKAPI_CALL CmdClearImage(VkCommandBuffer b, VkImage image, VkImageLayout layout,
                                         const Value* value, uint32_t count,
                                         const VkImageSubresourceRange* ranges) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* value_copy  = CopyStructs(w, value, 1);
		    auto* ranges_copy = CopyStructs(w, ranges, count);
		    w.Call([=] { (g_driver.*Member)(b, image, layout, value_copy, count, ranges_copy); });
	    },
	    b, image, layout, value, count, ranges);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(VkCommandBuffer b, uint32_t ac,
                                               const VkClearAttachment* attachments, uint32_t rc,
                                               const VkClearRect* rects) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* a_copy = CopyStructs(w, attachments, ac);
		    auto* r_copy = CopyStructs(w, rects, rc);
		    w.Call([=] { (g_driver.*Member)(b, ac, a_copy, rc, r_copy); });
	    },
	    b, ac, attachments, rc, rects);
}

// (b, first, count, items): viewports, scissors.
template <auto Member, typename Item>
VKAPI_ATTR void VKAPI_CALL CmdFirstCount(VkCommandBuffer b, uint32_t first, uint32_t count,
                                         const Item* items) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, items, count);
		    w.Call([=] { (g_driver.*Member)(b, first, count, copy); });
	    },
	    b, first, count, items);
}

// (b, count, items): viewports and scissors with count.
template <auto Member, typename Item>
VKAPI_ATTR void VKAPI_CALL CmdCount(VkCommandBuffer b, uint32_t count, const Item* items) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, items, count);
		    w.Call([=] { (g_driver.*Member)(b, count, copy); });
	    },
	    b, count, items);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer b, VkPipelineBindPoint point,
                                                 VkPipelineLayout layout, uint32_t first,
                                                 uint32_t count, const VkDescriptorSet* sets,
                                                 uint32_t oc, const uint32_t* offsets) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* s_copy = w.Copy(sets, count);
		    auto* o_copy = w.Copy(offsets, oc);
		    w.Call([=] { (g_driver.*Member)(b, point, layout, first, count, s_copy, oc, o_copy); });
	    },
	    b, point, layout, first, count, sets, oc, offsets);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(VkCommandBuffer b, uint32_t first, uint32_t count,
                                                const VkBuffer*     buffers,
                                                const VkDeviceSize* offsets) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* b_copy = w.Copy(buffers, count);
		    auto* o_copy = w.Copy(offsets, count);
		    w.Call([=] { (g_driver.*Member)(b, first, count, b_copy, o_copy); });
	    },
	    b, first, count, buffers, offsets);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers2(VkCommandBuffer b, uint32_t first, uint32_t count,
                                                 const VkBuffer*     buffers,
                                                 const VkDeviceSize* offsets,
                                                 const VkDeviceSize* sizes,
                                                 const VkDeviceSize* strides) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* b_copy  = w.Copy(buffers, count);
		    auto* o_copy  = w.Copy(offsets, count);
		    auto* sz_copy = w.Copy(sizes, count);
		    auto* st_copy = w.Copy(strides, count);
		    w.Call([=] { (g_driver.*Member)(b, first, count, b_copy, o_copy, sz_copy, st_copy); });
	    },
	    b, first, count, buffers, offsets, sizes, strides);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer b, VkPipelineLayout layout,
                                            VkShaderStageFlags stages, uint32_t offset,
                                            uint32_t size, const void* values) {
	Record<Member>(
	    [&](Writer& w) {
		    const void* copy = w.CopyBytes(values, size);
		    w.Call([=] { (g_driver.*Member)(b, layout, stages, offset, size, copy); });
	    },
	    b, layout, stages, offset, size, values);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer b, VkBuffer buffer, VkDeviceSize offset,
                                           VkDeviceSize size, const void* data) {
	Record<Member>(
	    [&](Writer& w) {
		    const void* copy = w.CopyBytes(data, static_cast<size_t>(size));
		    w.Call([=] { (g_driver.*Member)(b, buffer, offset, size, copy); });
	    },
	    b, buffer, offset, size, data);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdSetBlendConstants(VkCommandBuffer b, const float constants[4]) {
	Record<Member>(
	    [&](Writer& w) {
		    const float* copy = w.Copy(constants, 4);
		    w.Call([=] { (g_driver.*Member)(b, copy); });
	    },
	    b, constants);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL
CmdBeginConditionalRendering(VkCommandBuffer b, const VkConditionalRenderingBeginInfoEXT* info) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyStructs(w, info, 1);
		    w.Call([=] { (g_driver.*Member)(b, copy); });
	    },
	    b, info);
}

template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSet(VkCommandBuffer b, VkPipelineBindPoint point,
                                                VkPipelineLayout layout, uint32_t set,
                                                uint32_t                    count,
                                                const VkWriteDescriptorSet* writes) {
	Record<Member>(
	    [&](Writer& w) {
		    auto* copy = CopyWrites(w, writes, count);
		    w.Call([=] { (g_driver.*Member)(b, point, layout, set, count, copy); });
	    },
	    b, point, layout, set, count, writes);
}

// The marker is a value the driver hands back, not data it reads.
template <auto Member>
VKAPI_ATTR void VKAPI_CALL CmdSetCheckpoint(VkCommandBuffer b, const void* marker) {
	Record<Member>([&](Writer& w) { w.Call([=] { (g_driver.*Member)(b, marker); }); }, b, marker);
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer                 b,
                                                  const VkCommandBufferBeginInfo* info) {
	using Self = Entry<&Dispatcher::vkBeginCommandBuffer>;
	if (auto* stream = t_stream) {
		NoteOpen(b, stream);
		t_owned           = b;
		const bool queued = stream->Enqueue([&](Writer& w) {
			auto* copy = CopyStructs(w, info, 1);
			if (copy != nullptr && copy->pInheritanceInfo != nullptr) {
				w.Fail();
			}
			w.Call([=] { CheckReplayed(g_driver.vkBeginCommandBuffer(b, copy), Self::name); });
		});
		if (queued) {
			return VK_SUCCESS;
		}
		Self::Direct(*stream);
	} else {
		CheckNotOpen(b, Self::name);
	}
	return g_driver.vkBeginCommandBuffer(b, info);
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer b) {
	using Self = Entry<&Dispatcher::vkEndCommandBuffer>;
	if (auto* stream = t_stream) {
		NoteClosed(b);
		t_owned = VK_NULL_HANDLE;
		if (stream->Enqueue([&](Writer& w) {
			    w.Call([=] { CheckReplayed(g_driver.vkEndCommandBuffer(b), Self::name); });
		    })) {
			return VK_SUCCESS;
		}
		Self::Direct(*stream);
	} else {
		CheckNotOpen(b, Self::name);
	}
	return g_driver.vkEndCommandBuffer(b);
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t wc,
                                                const VkWriteDescriptorSet* writes, uint32_t cc,
                                                const VkCopyDescriptorSet* copies) {
	Record<&Dispatcher::vkUpdateDescriptorSets>(
	    [&](Writer& w) {
		    auto* w_copy = CopyWrites(w, writes, wc);
		    auto* c_copy = CopyStructs(w, copies, cc);
		    w.Call([=] { g_driver.vkUpdateDescriptorSets(device, wc, w_copy, cc, c_copy); });
	    },
	    device, wc, writes, cc, copies);
}

// Sets `slot` to `thunk` when the driver provides the entry.
template <auto Member, typename Thunk>
void Replace(Dispatcher& dispatcher, Thunk thunk, const char* name) {
	auto& slot = dispatcher.*Member;
	if (slot == thunk) {
		g_driver.*Member = g_previous.*Member;
	}
	if (g_driver.*Member == nullptr) {
		return;
	}
	Entry<Member>::name = name;
	slot                = thunk;
}

// Every command buffer command: queued when its arguments are values, made after a drain
// otherwise (the thunks with copies below replace the latter where the renderer uses them).
template <auto Member>
void ReplaceCommand(Dispatcher& dispatcher, const char* name) {
	using Command = Entry<Member>;
	if constexpr (Command::by_value) {
		Replace<Member>(dispatcher, &Command::Queued, name);
	} else {
		Replace<Member>(dispatcher, &Command::OrderedCommand, name);
	}
}

} // namespace

void Install() {
	EXIT_IF(t_stream != nullptr);
	// Again for every device the dispatcher is initialized for (test harnesses create several);
	// never while a producer records.
	auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	g_previous       = g_driver;
	g_driver         = dispatcher;

#define KYTY_VULKAN_COMMAND(name) ReplaceCommand<&Dispatcher::name>(dispatcher, #name);
#include "graphics/host_gpu/vulkanCommandList.inc"
#undef KYTY_VULKAN_COMMAND

#define KYTY_REPLACE(name, ...) Replace<&Dispatcher::name>(dispatcher, __VA_ARGS__, #name)
	KYTY_REPLACE(vkCmdPipelineBarrier, &CmdPipelineBarrier<&Dispatcher::vkCmdPipelineBarrier>);
	KYTY_REPLACE(vkCmdPipelineBarrier2, &CmdDependency<&Dispatcher::vkCmdPipelineBarrier2>);
	KYTY_REPLACE(vkCmdPipelineBarrier2KHR, &CmdDependency<&Dispatcher::vkCmdPipelineBarrier2KHR>);
	KYTY_REPLACE(vkCmdBeginRendering, &CmdBeginRendering<&Dispatcher::vkCmdBeginRendering>);
	KYTY_REPLACE(vkCmdBeginRenderingKHR, &CmdBeginRendering<&Dispatcher::vkCmdBeginRenderingKHR>);
	KYTY_REPLACE(vkCmdCopyBuffer,
	             &CmdRegions2<&Dispatcher::vkCmdCopyBuffer, VkBuffer, VkBuffer, VkBufferCopy>);
	KYTY_REPLACE(vkCmdCopyImage, &CmdImageRegions<&Dispatcher::vkCmdCopyImage, VkImageCopy>);
	KYTY_REPLACE(vkCmdResolveImage,
	             &CmdImageRegions<&Dispatcher::vkCmdResolveImage, VkImageResolve>);
	KYTY_REPLACE(vkCmdBlitImage, &CmdBlitImage<&Dispatcher::vkCmdBlitImage>);
	KYTY_REPLACE(vkCmdCopyBufferToImage,
	             &CmdCopyBufferToImage<&Dispatcher::vkCmdCopyBufferToImage>);
	KYTY_REPLACE(vkCmdCopyImageToBuffer,
	             &CmdCopyImageToBuffer<&Dispatcher::vkCmdCopyImageToBuffer>);
	KYTY_REPLACE(vkCmdClearColorImage,
	             &CmdClearImage<&Dispatcher::vkCmdClearColorImage, VkClearColorValue>);
	KYTY_REPLACE(
	    vkCmdClearDepthStencilImage,
	    &CmdClearImage<&Dispatcher::vkCmdClearDepthStencilImage, VkClearDepthStencilValue>);
	KYTY_REPLACE(vkCmdClearAttachments, &CmdClearAttachments<&Dispatcher::vkCmdClearAttachments>);
	KYTY_REPLACE(vkCmdSetViewport, &CmdFirstCount<&Dispatcher::vkCmdSetViewport, VkViewport>);
	KYTY_REPLACE(vkCmdSetScissor, &CmdFirstCount<&Dispatcher::vkCmdSetScissor, VkRect2D>);
	KYTY_REPLACE(vkCmdSetViewportWithCount,
	             &CmdCount<&Dispatcher::vkCmdSetViewportWithCount, VkViewport>);
	KYTY_REPLACE(vkCmdSetViewportWithCountEXT,
	             &CmdCount<&Dispatcher::vkCmdSetViewportWithCountEXT, VkViewport>);
	KYTY_REPLACE(vkCmdSetScissorWithCount,
	             &CmdCount<&Dispatcher::vkCmdSetScissorWithCount, VkRect2D>);
	KYTY_REPLACE(vkCmdSetScissorWithCountEXT,
	             &CmdCount<&Dispatcher::vkCmdSetScissorWithCountEXT, VkRect2D>);
	KYTY_REPLACE(vkCmdBindDescriptorSets,
	             &CmdBindDescriptorSets<&Dispatcher::vkCmdBindDescriptorSets>);
	KYTY_REPLACE(vkCmdBindVertexBuffers,
	             &CmdBindVertexBuffers<&Dispatcher::vkCmdBindVertexBuffers>);
	KYTY_REPLACE(vkCmdBindVertexBuffers2,
	             &CmdBindVertexBuffers2<&Dispatcher::vkCmdBindVertexBuffers2>);
	KYTY_REPLACE(vkCmdBindVertexBuffers2EXT,
	             &CmdBindVertexBuffers2<&Dispatcher::vkCmdBindVertexBuffers2EXT>);
	KYTY_REPLACE(vkCmdPushConstants, &CmdPushConstants<&Dispatcher::vkCmdPushConstants>);
	KYTY_REPLACE(vkCmdUpdateBuffer, &CmdUpdateBuffer<&Dispatcher::vkCmdUpdateBuffer>);
	KYTY_REPLACE(vkCmdSetBlendConstants,
	             &CmdSetBlendConstants<&Dispatcher::vkCmdSetBlendConstants>);
	KYTY_REPLACE(vkCmdPushDescriptorSet,
	             &CmdPushDescriptorSet<&Dispatcher::vkCmdPushDescriptorSet>);
	KYTY_REPLACE(vkCmdPushDescriptorSetKHR,
	             &CmdPushDescriptorSet<&Dispatcher::vkCmdPushDescriptorSetKHR>);
	KYTY_REPLACE(vkCmdSetCheckpointNV, &CmdSetCheckpoint<&Dispatcher::vkCmdSetCheckpointNV>);
	KYTY_REPLACE(vkCmdBeginConditionalRenderingEXT,
	             &CmdBeginConditionalRendering<&Dispatcher::vkCmdBeginConditionalRenderingEXT>);

	KYTY_REPLACE(vkBeginCommandBuffer, &BeginCommandBuffer);
	KYTY_REPLACE(vkEndCommandBuffer, &EndCommandBuffer);
	KYTY_REPLACE(vkUpdateDescriptorSets, &UpdateDescriptorSets);

	// Host calls that keep their place among the queued calls: they name objects queued calls
	// use, and no answer is needed.
#define KYTY_QUEUED(name) KYTY_REPLACE(name, &Entry<&Dispatcher::name>::Queued)
	KYTY_QUEUED(vkDestroyBuffer);
	KYTY_QUEUED(vkDestroyBufferView);
	KYTY_QUEUED(vkDestroyImage);
	KYTY_QUEUED(vkDestroyImageView);
	KYTY_QUEUED(vkDestroySampler);
	KYTY_QUEUED(vkFreeMemory);
	KYTY_QUEUED(vkDestroyPipeline);
	KYTY_QUEUED(vkDestroyPipelineLayout);
	KYTY_QUEUED(vkDestroyDescriptorSetLayout);
	KYTY_QUEUED(vkDestroyShaderModule);
	KYTY_QUEUED(vkDestroyQueryPool);
	KYTY_QUEUED(vkDestroySemaphore);
	KYTY_QUEUED(vkDestroyFence);
	KYTY_QUEUED(vkDestroyEvent);
	KYTY_QUEUED(vkDestroyAccelerationStructureKHR);
	KYTY_QUEUED(vkResetQueryPool);
	KYTY_QUEUED(vkResetQueryPoolEXT);
#undef KYTY_QUEUED

	// Host calls that use what the queued calls use (a command pool, a descriptor pool, the
	// queue, a fence of a submission) or wait for their results: after a drain.
#define KYTY_ORDERED(name) KYTY_REPLACE(name, &Entry<&Dispatcher::name>::Ordered)
	KYTY_ORDERED(vkQueueSubmit);
	KYTY_ORDERED(vkQueueSubmit2);
	KYTY_ORDERED(vkQueueSubmit2KHR);
	KYTY_ORDERED(vkQueueBindSparse);
	KYTY_ORDERED(vkQueueWaitIdle);
	KYTY_ORDERED(vkDeviceWaitIdle);
	KYTY_ORDERED(vkQueuePresentKHR);
	KYTY_ORDERED(vkAllocateCommandBuffers);
	KYTY_ORDERED(vkFreeCommandBuffers);
	KYTY_ORDERED(vkResetCommandPool);
	KYTY_ORDERED(vkResetCommandBuffer);
	KYTY_ORDERED(vkTrimCommandPool);
	KYTY_ORDERED(vkDestroyCommandPool);
	KYTY_ORDERED(vkResetDescriptorPool);
	KYTY_ORDERED(vkFreeDescriptorSets);
	KYTY_ORDERED(vkDestroyDescriptorPool);
	KYTY_ORDERED(vkUpdateDescriptorSetWithTemplate);
	KYTY_ORDERED(vkResetFences);
	KYTY_ORDERED(vkWaitForFences);
	KYTY_ORDERED(vkGetFenceStatus);
	KYTY_ORDERED(vkSetEvent);
	KYTY_ORDERED(vkResetEvent);
	KYTY_ORDERED(vkGetEventStatus);
	KYTY_ORDERED(vkDestroyDevice);
#undef KYTY_ORDERED
#undef KYTY_REPLACE

	g_installed.store(true, std::memory_order_release);
}

bool Installed() {
	return g_installed.load(std::memory_order_acquire);
}

ProducerScope::ProducerScope(bool enabled) {
	if (!enabled || !Installed()) {
		return;
	}
	EXIT_IF(t_stream != nullptr);
	t_stream = new Stream([] {
		KYTY_PROFILER_THREAD("Thread_GpuRecord");
		// It runs at the producer's priority: its latency is in every wait of the producer for the
		// GPU.
		if (Config::GpuThreadPriorityEnabled()) {
			Common::Thread::RaiseCurrentThreadPriority();
		}
	});
	m_stream = t_stream;
}

ProducerScope::~ProducerScope() {
	auto* stream = static_cast<Stream*>(m_stream);
	if (stream == nullptr) {
		return;
	}
	stream->Drain();
	ReleaseOpen(stream);
	if (t_stream == stream) {
		t_stream = nullptr;
	}
	delete stream;
}

bool Recording() {
	return t_stream != nullptr;
}

namespace {
// Distinct per thread while it runs: the scope's identity check.
thread_local char t_thread_tag = 0;
} // namespace

PauseScope::PauseScope() {
	auto* stream = t_stream;
	if (stream == nullptr) {
		return;
	}
	stream->Drain();
	ReleaseOpen(stream);
	t_stream = nullptr;
	m_stream = stream;
	m_thread = &t_thread_tag;
}

PauseScope::~PauseScope() {
	if (m_stream != nullptr && m_thread == &t_thread_tag && t_stream == nullptr) {
		t_stream = static_cast<Stream*>(m_stream);
	}
}

void Drain() {
	if (auto* stream = t_stream) {
		stream->Drain();
		ReleaseOpen(stream);
	}
}

bool EnqueueOrdered(void (*function)(const void*), const void* data, size_t size) {
	auto* stream = t_stream;
	if (stream == nullptr || function == nullptr) {
		return false;
	}
	const bool queued = stream->Enqueue([&](Writer& writer) {
		const void* copy = writer.CopyBytes(data, size);
		writer.Call([function, copy] { function(copy); });
	});
	if (queued) {
		stream->Publish();
		return true;
	}
	stream->Drain();
	return false;
}

} // namespace Libs::Graphics::VulkanRecording
