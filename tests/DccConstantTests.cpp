#include "graphics/host_gpu/renderer/cache/dccConstant.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace DccConstant = Libs::Graphics::DccConstant;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "DccConstantTests: failed: %s\n", message);
    std::abort();
  }
}

void TestSliceClassification() {
  std::vector<uint8_t> keys(0x1000, DccConstant::Key);
  Check(DccConstant::IsConstantSlice(keys),
        "a slice of single-value keys was rejected");
  Check(!DccConstant::IsConstantSlice({}), "an empty slice was decoded");
  keys[0x7ff] = 0xff;
  Check(!DccConstant::IsConstantSlice(keys),
        "a slice with an uncompressed block was decoded");
  for (const uint8_t clear : {0x00, 0x20, 0x40, 0x80, 0xc0, 0xff}) {
    const std::vector<uint8_t> other(0x1000, clear);
    Check(!DccConstant::IsConstantSlice(other),
          "a clear or uncompressed code was taken for single-value blocks");
  }
}

void TestElementSizes() {
  for (const uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
    Check(DccConstant::SupportsElementBytes(bytes),
          "a power-of-two element was rejected");
  }
  for (const uint32_t bytes : {0u, 3u, 6u, 12u, 32u}) {
    Check(!DccConstant::SupportsElementBytes(bytes),
          "an unsupported element was accepted");
  }
}

// Every element of each 256-byte block becomes the block's first element.
void TestExpansion() {
  for (const uint32_t element : {1u, 2u, 4u, 8u, 16u}) {
    std::vector<uint8_t> data(3 * DccConstant::BlockBytes);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    const auto original = data;
    Check(DccConstant::ExpandBlocks(data, element),
          "a valid expansion was refused");
    for (size_t block = 0; block < data.size();
         block += DccConstant::BlockBytes) {
      for (size_t offset = 0; offset < DccConstant::BlockBytes; offset++) {
        Check(data[block + offset] == original[block + offset % element],
              "an expanded byte does not repeat the block's first element");
      }
    }
  }
  // The motion-vector reset of Ghost of Yotei: RG16F NaN in the first dword of
  // each block.
  std::vector<uint8_t> motion(2 * DccConstant::BlockBytes, 0);
  const uint32_t nan = 0x7e007e00u;
  std::memcpy(motion.data(), &nan, sizeof(nan));
  std::memcpy(motion.data() + DccConstant::BlockBytes, &nan, sizeof(nan));
  Check(DccConstant::ExpandBlocks(motion, 4), "the RG16F reset was refused");
  for (size_t offset = 0; offset < motion.size(); offset += sizeof(nan)) {
    uint32_t value = 0;
    std::memcpy(&value, motion.data() + offset, sizeof(value));
    Check(value == nan, "an RG16F pixel kept its stale bytes");
  }
  std::vector<uint8_t> partial(DccConstant::BlockBytes + 4);
  Check(!DccConstant::ExpandBlocks(partial, 4), "a partial block was expanded");
  std::vector<uint8_t> block(DccConstant::BlockBytes);
  Check(!DccConstant::ExpandBlocks(block, 12),
        "a non-power-of-two element was expanded");
}

} // namespace

int main() {
  TestSliceClassification();
  TestElementSizes();
  TestExpansion();
  std::printf("DccConstantTests: ok\n");
  return 0;
}
