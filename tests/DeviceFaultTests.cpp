#include "graphics/host_gpu/deviceFault.h"

#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Libs::Graphics::DeviceFault::AddressInfo;
using Libs::Graphics::DeviceFault::AddressTypeName;
using Libs::Graphics::DeviceFault::Format;
using Libs::Graphics::DeviceFault::MaxAddressInfos;
using Libs::Graphics::DeviceFault::MaxVendorInfos;
using Libs::Graphics::DeviceFault::Record;
using Libs::Graphics::DeviceFault::VendorInfo;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "DeviceFaultTests: failed: %s\n", text);
    std::abort();
  }
}

void CheckText(const std::string &actual, const std::string &expected,
               const char *text) {
  if (actual != expected) {
    std::fprintf(stderr, "DeviceFaultTests: got:\n%s\nexpected:\n%s\n",
                 actual.c_str(), expected.c_str());
  }
  Check(actual == expected, text);
}

size_t Count(const std::string &text, std::string_view part) {
  size_t count = 0;
  for (size_t at = text.find(part); at != std::string::npos;
       at = text.find(part, at + 1)) {
    count++;
  }
  return count;
}

void TestTypeNames() {
  Check(std::string_view(AddressTypeName(1)) == "ReadInvalid",
        "type 1 is not ReadInvalid");
  Check(std::string_view(AddressTypeName(2)) == "WriteInvalid",
        "type 2 is not WriteInvalid");
  Check(std::string_view(AddressTypeName(3)) == "ExecuteInvalid",
        "type 3 is not ExecuteInvalid");
  Check(std::string_view(AddressTypeName(4)) == "InstructionPointerUnknown",
        "type 4 is not InstructionPointerUnknown");
  Check(std::string_view(AddressTypeName(5)) == "InstructionPointerInvalid",
        "type 5 is not InstructionPointerInvalid");
  Check(std::string_view(AddressTypeName(6)) == "InstructionPointerFault",
        "type 6 is not InstructionPointerFault");
  Check(std::string_view(AddressTypeName(7)) == "Unknown",
        "an unknown type has a name");
}

void TestRecord() {
  // A driver description fills its array; the text ends at the first NUL.
  char description[256] = "Page fault\n in \"shader\"";
  const char vendor_text[256] = "MMU fault";
  const std::vector<AddressInfo> addresses = {
      {.type = 1, .address = 0x43f55c123, .precision = 0x1000},
      {.type = 6, .address = 0x7fff0010, .precision = 0},
      {.type = 2, .address = 0x6304c0000, .precision = 3},
  };
  const std::vector<VendorInfo> vendors = {
      {.description = {vendor_text, sizeof(vendor_text)},
       .code = 0x1f,
       .data = 0xdeadbeef},
  };
  const auto text = Format({.description = {description, sizeof(description)},
                            .address_count = 3,
                            .vendor_count = 1,
                            .addresses = addresses,
                            .vendors = vendors});
  CheckText(
      text,
      "Device fault: \"Page fault? in ?shader?\", 3 address record(s), 1 "
      "vendor record(s)\n"
      "Device fault address 0: ReadInvalid at 0x000000043f55c123 (range "
      "0x000000043f55c000-0x000000043f55d000)\n"
      "Device fault address 1: InstructionPointerFault at "
      "0x000000007fff0010 (precision 0)\n"
      "Device fault address 2: WriteInvalid at 0x00000006304c0000 "
      "(precision 3)\n"
      "Device fault vendor 0: \"MMU fault\", code 0x1f, data 0xdeadbeef\n",
      "a record is not written as expected");
}

void TestEmptyRecord() {
  const auto text = Format({.description = "Device lost"});
  CheckText(text,
            "Device fault: \"Device lost\", 0 address record(s), 0 vendor "
            "record(s)\n",
            "an empty record is not one line");
}

void TestClamped() {
  // The driver reported more records than were queried: the rest are counted.
  std::vector<AddressInfo> addresses(MaxAddressInfos + 4,
                                     {.type = 1, .address = 0x1000});
  std::vector<VendorInfo> vendors(MaxVendorInfos + 2,
                                  {.description = "v", .code = 1});
  auto text = Format({.description = "many",
                      .address_count = 100,
                      .vendor_count = 50,
                      .addresses = addresses,
                      .vendors = vendors});
  Check(Count(text, ": ReadInvalid at ") == MaxAddressInfos,
        "more address records than the maximum are written");
  Check(Count(text, "\"v\", code") == MaxVendorInfos,
        "more vendor records than the maximum are written");
  Check(text.find("Device fault address records not shown: 84\n") !=
            std::string::npos,
        "the address records not shown are not counted");
  Check(text.find("Device fault vendor records not shown: 42\n") !=
            std::string::npos,
        "the vendor records not shown are not counted");
  Check(text.find(", 100 address record(s), 50 vendor record(s)") !=
            std::string::npos,
        "the reported counts are not written");

  // Fewer records filled than reported (VK_INCOMPLETE): only those are read.
  text = Format({.description = "short",
                 .address_count = 3,
                 .vendor_count = 0,
                 .addresses = std::span(addresses).first(1)});
  Check(Count(text, ": ReadInvalid at ") == 1,
        "records past the filled ones are read");
  Check(text.find("Device fault address records not shown: 2\n") !=
            std::string::npos,
        "records reported but not filled are not counted");

  // A count below the filled records limits them.
  text = Format({.description = "less",
                 .address_count = 2,
                 .vendor_count = 1,
                 .addresses = addresses,
                 .vendors = vendors});
  Check(Count(text, ": ReadInvalid at ") == 2 &&
            Count(text, "\"v\", code") == 1,
        "the reported counts do not limit the records written");
  Check(text.find("not shown") == std::string::npos,
        "records are reported missing although all were written");
}

void TestLongDescription() {
  const std::string long_text(1000, 'x');
  const auto text = Format({.description = long_text});
  Check(Count(text, "x") == 256, "a description is not cut at 256 characters");
}

} // namespace

int main() {
  TestTypeNames();
  TestRecord();
  TestEmptyRecord();
  TestClamped();
  TestLongDescription();
  std::printf("DeviceFaultTests: ok\n");
  return 0;
}
