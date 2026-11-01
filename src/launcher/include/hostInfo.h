#ifndef LAUNCHER_INCLUDE_HOST_INFO_H_
#define LAUNCHER_INCLUDE_HOST_INFO_H_

#include <QList>
#include <QString>
#include <QtGlobal>

// What the launcher shows about this PC (Play checks, First start, Graphics card, bug report).
namespace HostInfo {

struct Gpu {
	QString name;
	quint64 vram_bytes = 0; // the largest device-local memory heap
	bool    discrete   = false;
	QString driver;         // driver version, written the way the vendor numbers it
};

// The Vulkan devices in vkEnumeratePhysicalDevices order, which is the order of the emulator's --gpu index. Listed
// once per launcher run; empty when Qt has no Vulkan or no device answers.
const QList<Gpu>& Gpus();

// The device the emulator starts on for a gpu_index setting: that device, or for -1 (and an index that is gone) the
// last discrete GPU, else the first device (VulkanFindPhysicalDevice keeps the last discrete one that qualifies).
// -1 when there is no device.
int ResolveGpuIndex(int gpu_index);

[[nodiscard]] quint64 TotalRamBytes();

// "16 GB" (binary gigabytes, rounded).
[[nodiscard]] QString Gigabytes(quint64 bytes);

// "NVIDIA GeForce RTX 5070 Ti · 16 GB".
[[nodiscard]] QString GpuText(const Gpu& gpu);

} // namespace HostInfo

#endif // LAUNCHER_INCLUDE_HOST_INFO_H_
