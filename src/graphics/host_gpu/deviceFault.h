#ifndef KYTY_GRAPHICS_HOST_GPU_DEVICE_FAULT_H_
#define KYTY_GRAPHICS_HOST_GPU_DEVICE_FAULT_H_

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

// The record VK_EXT_device_fault keeps about a lost device, as log lines. Vulkan-free: the
// caller copies the driver's structures into these.
namespace Libs::Graphics::DeviceFault {

// At most this many address and vendor records are queried and written.
inline constexpr uint32_t MaxAddressInfos = 16;
inline constexpr uint32_t MaxVendorInfos  = 8;
// Descriptions are cut after this many characters (VK_MAX_DESCRIPTION_SIZE).
inline constexpr size_t MaxDescription = 256;

struct AddressInfo {
	uint32_t type      = 0; // VkDeviceFaultAddressTypeEXT
	uint64_t address   = 0;
	uint64_t precision = 0;
};

struct VendorInfo {
	std::string_view description;
	uint64_t         code = 0;
	uint64_t         data = 0;
};

struct Record {
	std::string_view description;
	// The counts the driver reported; more than the spans hold when they were clamped.
	uint32_t                     address_count = 0;
	uint32_t                     vendor_count  = 0;
	std::span<const AddressInfo> addresses;
	std::span<const VendorInfo>  vendors;
};

// VkDeviceFaultAddressTypeEXT values.
inline const char* AddressTypeName(uint32_t type) {
	switch (type) {
		case 0: return "None";
		case 1: return "ReadInvalid";
		case 2: return "WriteInvalid";
		case 3: return "ExecuteInvalid";
		case 4: return "InstructionPointerUnknown";
		case 5: return "InstructionPointerInvalid";
		case 6: return "InstructionPointerFault";
		default: return "Unknown";
	}
}

// A driver string as one quoted line: up to the first NUL, at most MaxDescription characters,
// control characters and quotes replaced.
inline std::string Quote(std::string_view text) {
	text            = text.substr(0, std::min(text.find('\0'), MaxDescription));
	std::string out = "\"";
	for (const char c: text) {
		const auto u = static_cast<unsigned char>(c);
		out += (u < 0x20 || u == 0x7f || c == '"') ? '?' : c;
	}
	out += '"';
	return out;
}

// The record as lines, each ending in a newline.
inline std::string Format(const Record& record) {
	std::string out;
	char        line[160];
	const auto  addresses = record.addresses.first(
	    std::min<size_t>({record.addresses.size(), record.address_count, MaxAddressInfos}));
	const auto vendors = record.vendors.first(
	    std::min<size_t>({record.vendors.size(), record.vendor_count, MaxVendorInfos}));
	out += "Device fault: " + Quote(record.description);
	std::snprintf(line, sizeof(line), ", %u address record(s), %u vendor record(s)\n",
	              record.address_count, record.vendor_count);
	out += line;
	for (size_t i = 0; i < addresses.size(); i++) {
		const auto& a = addresses[i];
		// The faulting access lies in the precision-sized, precision-aligned range (a power of
		// two) around the reported address.
		const bool     ranged = a.precision > 1 && (a.precision & (a.precision - 1)) == 0;
		const uint64_t begin  = ranged ? a.address & ~(a.precision - 1) : a.address;
		if (ranged) {
			std::snprintf(line, sizeof(line),
			              "Device fault address %zu: %s at 0x%016" PRIx64 " (range 0x%016" PRIx64
			              "-0x%016" PRIx64 ")\n",
			              i, AddressTypeName(a.type), a.address, begin, begin + a.precision);
		} else {
			std::snprintf(line, sizeof(line),
			              "Device fault address %zu: %s at 0x%016" PRIx64 " (precision %" PRIu64
			              ")\n",
			              i, AddressTypeName(a.type), a.address, a.precision);
		}
		out += line;
	}
	if (record.address_count > addresses.size()) {
		std::snprintf(line, sizeof(line), "Device fault address records not shown: %zu\n",
		              record.address_count - addresses.size());
		out += line;
	}
	for (size_t i = 0; i < vendors.size(); i++) {
		const auto& v = vendors[i];
		std::snprintf(line, sizeof(line), "Device fault vendor %zu: ", i);
		out += line;
		out += Quote(v.description);
		std::snprintf(line, sizeof(line), ", code 0x%" PRIx64 ", data 0x%" PRIx64 "\n", v.code,
		              v.data);
		out += line;
	}
	if (record.vendor_count > vendors.size()) {
		std::snprintf(line, sizeof(line), "Device fault vendor records not shown: %zu\n",
		              record.vendor_count - vendors.size());
		out += line;
	}
	return out;
}

} // namespace Libs::Graphics::DeviceFault

#endif /* KYTY_GRAPHICS_HOST_GPU_DEVICE_FAULT_H_ */
