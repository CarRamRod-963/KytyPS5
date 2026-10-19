#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages();
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::DestroyRecycledImages() {
	std::scoped_lock lock(recycled_images_mutex);
	for (const auto& recycled: recycled_images) {
		vmaDestroyImage(allocator, recycled.image, recycled.allocation);
	}
	recycled_images.clear();
	recycled_image_bytes = 0;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

GraphicContext::MemoryHeaps GraphicContext::SampleMemoryHeaps() const {
	MemoryHeaps heaps;
	const bool  discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	if (!CanReportMemoryUsage() || allocator == nullptr || !discrete) {
		return heaps;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		auto& state = device_local ? heaps.device : heaps.host;
		state.budget += budgets[heap].budget;
		state.usage += budgets[heap].usage;
		state.blocks += budgets[heap].statistics.blockBytes;
	}
	return heaps;
}

void GraphicContext::MonitorMemoryBudget() {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return;
	}
	vmaSetCurrentFrameIndex(allocator, ++memory_budget_frame);
	const auto heaps = SampleMemoryHeaps();
	if (heaps.device.budget == 0) {
		return;
	}
	const auto now_ms =
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                              std::chrono::steady_clock::now().time_since_epoch())
	                              .count());
	const auto report = memory_budget_monitor.Sample(heaps.device, heaps.host, now_ms);
	if (report == MemoryBudget::Monitor::Report::None) {
		return;
	}
	using MemoryBudget::MiB;
	const auto& device = heaps.device;
	LOGF("%sGPU memory: device %" PRIu64 " of %" PRIu64 " MiB budget, driver %" PRIu64
	     " MiB, caches %" PRIu64 " MiB (limit %" PRIu64
	     " MiB); driver memory in the host heap %" PRIu64 " MiB\n",
	     report == MemoryBudget::Monitor::Report::Demoted
	         ? "Warning: driver memory grew in the host heap (demoted from device memory?). "
	         : "",
	     device.usage / MiB, device.budget / MiB, MemoryBudget::DriverUsage(device) / MiB,
	     device.blocks / MiB, MemoryBudget::CacheLimit(device) / MiB,
	     MemoryBudget::DriverUsage(heaps.host) / MiB);
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	{
		std::scoped_lock lock(recycled_images_mutex);
		const auto       recycled =
		    std::find_if(recycled_images.begin(), recycled_images.end(),
		                 [&](const RecycledImage& candidate) { return candidate.info == image_info; });
		if (recycled != recycled_images.end()) {
			image.image      = recycled->image;
			image.allocation = recycled->allocation;
			recycled_image_bytes -= recycled->bytes;
			recycled_images.erase(recycled);
		}
	}
	if (image.image == nullptr) {
		const auto create = [&](VkMemoryPropertyFlags required) {
			VmaAllocationCreateInfo alloc_info {};
			alloc_info.requiredFlags  = required;
			alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

			vk::Image::CType native_image = VK_NULL_HANDLE;
			const auto       result       = static_cast<vk::Result>(vmaCreateImage(
			    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
			    &alloc_info, &native_image, &image.allocation, nullptr));
			image.image                   = native_image;
			return result == vk::Result::eSuccess;
		};
		bool created = create(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (!created) {
			// Device memory is exhausted. The recycled backings are the only memory freed
			// right away; next, any memory type the image supports (system memory over the bus)
			// keeps the title running while the caches release what they no longer use.
			DestroyRecycledImages();
			created = create(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			if (!created) {
				created          = create(0);
				const auto count = created ? ++image_fallback_count : uint64_t {0};
				if (count % 256 == 1) {
					LOGF("Image placed outside device-local memory (%" PRIu64 " so far)\n", count);
					LogMemoryBudget();
				}
			}
		}
		if (!created) {
			LogMemoryBudget();
			return false;
		}
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.create_info = image_info;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	VmaAllocationInfo allocation_info {};
	vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
	const uint64_t bytes = allocation_info.size;
	// Only a chain-free create info can be compared by value.
	if (image.create_info.pNext == nullptr && bytes <= RecycledImageBytesLimit) {
		std::scoped_lock lock(recycled_images_mutex);
		while (!recycled_images.empty() && (recycled_images.size() >= RecycledImageLimit ||
		                                    recycled_image_bytes + bytes > RecycledImageBytesLimit)) {
			const auto& oldest = recycled_images.front();
			vmaDestroyImage(allocator, oldest.image, oldest.allocation);
			recycled_image_bytes -= oldest.bytes;
			recycled_images.erase(recycled_images.begin());
		}
		recycled_images.push_back({.info       = image.create_info,
		                           .image      = image.image,
		                           .allocation = image.allocation,
		                           .bytes      = bytes});
		recycled_image_bytes += bytes;
	} else {
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
