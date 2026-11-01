#include "hostInfo.h"

#include <QCoreApplication>
#include <QFileInfo>
#include <QVersionNumber>
#include <QtGui/qtguiglobal.h> // QT_CONFIG(vulkan)

#if QT_CONFIG(vulkan)
#include <QVulkanFunctions>
#include <QVulkanInstance>
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

namespace HostInfo {

namespace {

#if QT_CONFIG(vulkan)
constexpr uint32_t VENDOR_NVIDIA = 0x10DE;
constexpr uint32_t VENDOR_INTEL  = 0x8086;

QString DriverVersion(const VkPhysicalDeviceProperties& properties) {
	const uint32_t v = properties.driverVersion;
	if (properties.vendorID == VENDOR_NVIDIA) {
		return QStringLiteral("%1.%2").arg((v >> 22U) & 0x3FFU).arg((v >> 14U) & 0xFFU, 2, 10, QLatin1Char('0'));
	}
#if defined(_WIN32)
	if (properties.vendorID == VENDOR_INTEL) {
		return QStringLiteral("%1.%2").arg(v >> 14U).arg(v & 0x3FFFU);
	}
#endif
	return QStringLiteral("%1.%2.%3").arg(VK_VERSION_MAJOR(v)).arg(VK_VERSION_MINOR(v)).arg(VK_VERSION_PATCH(v));
}
#endif

QList<Gpu> ListGpus() {
	QList<Gpu> gpus;
#if QT_CONFIG(vulkan)
#if defined(__APPLE__)
	if (!qEnvironmentVariableIsSet("QT_VULKAN_LIB")) {
		const auto base    = QCoreApplication::applicationDirPath();
		auto       library = base + "/libMoltenVK.dylib";
		if (!QFileInfo::exists(library)) {
			library = base + "/../Frameworks/libMoltenVK.dylib";
		}
		if (QFileInfo::exists(library)) {
			qputenv("QT_VULKAN_LIB", library.toUtf8());
		}
	}
#endif
	// The same instance as the game settings dialog's device list (configurationEditDialog.cpp).
	QVulkanInstance instance;
	instance.setApiVersion(QVersionNumber(1, 3, 0));
#if !defined(__APPLE__) && QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
	instance.setFlags(QVulkanInstance::NoPortabilityDrivers);
#endif
	if (!instance.create()) {
		return gpus;
	}
	auto*    f     = instance.functions();
	uint32_t count = 0;
	if (f->vkEnumeratePhysicalDevices(instance.vkInstance(), &count, nullptr) != VK_SUCCESS || count == 0) {
		return gpus;
	}
	std::vector<VkPhysicalDevice> devices(count);
	if (f->vkEnumeratePhysicalDevices(instance.vkInstance(), &count, devices.data()) < 0) {
		return gpus;
	}
	devices.resize(count);
	for (auto* device: devices) {
		VkPhysicalDeviceProperties properties {};
		f->vkGetPhysicalDeviceProperties(device, &properties);
		VkPhysicalDeviceMemoryProperties memory {};
		f->vkGetPhysicalDeviceMemoryProperties(device, &memory);
		Gpu gpu;
		gpu.name     = QString::fromUtf8(properties.deviceName);
		gpu.discrete = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
		gpu.driver   = DriverVersion(properties);
		for (uint32_t heap = 0; heap < memory.memoryHeapCount; heap++) {
			if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
				gpu.vram_bytes = std::max<quint64>(gpu.vram_bytes, memory.memoryHeaps[heap].size);
			}
		}
		gpus.append(gpu);
	}
#endif
	return gpus;
}

} // namespace

const QList<Gpu>& Gpus() {
	static const QList<Gpu> gpus = ListGpus();
	return gpus;
}

int ResolveGpuIndex(int gpu_index) {
	const auto& gpus = Gpus();
	if (gpu_index >= 0 && gpu_index < gpus.size()) {
		return gpu_index;
	}
	int best = gpus.isEmpty() ? -1 : 0;
	for (int i = 0; i < gpus.size(); i++) {
		if (gpus.at(i).discrete) {
			best = i;
		}
	}
	return best;
}

quint64 TotalRamBytes() {
#if defined(_WIN32)
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	return GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 0;
#elif defined(__APPLE__)
	uint64_t size   = 0;
	size_t   length = sizeof(size);
	return sysctlbyname("hw.memsize", &size, &length, nullptr, 0) == 0 ? size : 0;
#else
	const long pages = sysconf(_SC_PHYS_PAGES);
	const long page  = sysconf(_SC_PAGE_SIZE);
	return pages > 0 && page > 0 ? static_cast<quint64>(pages) * static_cast<quint64>(page) : 0;
#endif
}

QString Gigabytes(quint64 bytes) {
	constexpr double GIB = 1024.0 * 1024.0 * 1024.0;
	return QStringLiteral("%1 GB").arg(std::llround(static_cast<double>(bytes) / GIB));
}

QString GpuText(const Gpu& gpu) {
	return gpu.vram_bytes == 0 ? gpu.name : gpu.name + QStringLiteral(" · ") + Gigabytes(gpu.vram_bytes);
}

} // namespace HostInfo
