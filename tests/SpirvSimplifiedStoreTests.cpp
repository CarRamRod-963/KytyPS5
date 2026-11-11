#include "graphics/shader/recompiler/backend/spirv/SpirvSimplifiedStore.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::ShaderRecompiler::Spirv::SimplifiedStore;
using namespace std::chrono_literals;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "SpirvSimplifiedStoreTests: failed: %s\n", text);
    std::abort();
  }
}

std::filesystem::path TempPath(const char *name) {
  auto dir = std::filesystem::temp_directory_path() /
             "kyty_spirv_simplified_store_tests";
  std::filesystem::remove_all(dir);
  return dir / "cache" / name;
}

std::vector<uint32_t> Words(uint32_t count, uint32_t seed) {
  std::vector<uint32_t> words(count);
  for (uint32_t i = 0; i < count; i++) {
    words[i] = i * 2654435761u + seed;
  }
  return words;
}

void TestKeys() {
  const auto a = Words(100, 1);
  auto b = a;
  b[99] ^= 1u;
  Check(SimplifiedStore::MakeKey(a, 0) == SimplifiedStore::MakeKey(a, 0),
        "equal modules have equal keys");
  Check(!(SimplifiedStore::MakeKey(a, 0) == SimplifiedStore::MakeKey(b, 0)),
        "a changed word changes the key");
  Check(!(SimplifiedStore::MakeKey(a, 0) == SimplifiedStore::MakeKey(a, 1)),
        "the simplifier options are part of the key");
}

void TestAcrossSessions() {
  const auto path = TempPath("title.v1.simplified");
  const auto key_a = SimplifiedStore::MakeKey(Words(50, 1), 0);
  const auto key_b = SimplifiedStore::MakeKey(Words(60, 2), 0);
  {
    SimplifiedStore store(path, "build-1", 1ms);
    Check(store.Loaded() == 0, "a new file holds nothing");
    std::vector<uint32_t> words;
    Check(!store.Find(key_a, words), "nothing is found in a new store");
    store.Insert(key_a, Words(30, 3));
    store.Insert(key_b, {});
    Check(store.Find(key_a, words) && words == Words(30, 3),
          "an inserted module is found in the same session");
  }
  {
    SimplifiedStore store(path, "build-1", 1ms);
    std::vector<uint32_t> words;
    Check(store.Loaded() == 2 && store.Find(key_a, words) &&
              words == Words(30, 3) && store.Find(key_b, words) &&
              words.empty(),
          "the next session of the same build finds the modules");
    store.Insert(key_a, Words(5, 5));
    Check(store.Find(key_a, words) && words == Words(30, 3),
          "a stored module is not replaced");
  }
  {
    SimplifiedStore store(path, "build-2", 1ms);
    std::vector<uint32_t> words;
    Check(store.Loaded() == 0 && !store.Find(key_a, words),
          "another build's file is not used");
    store.Insert(key_b, Words(7, 7));
  }
  {
    SimplifiedStore store(path, "build-2", 1ms);
    std::vector<uint32_t> words;
    Check(store.Loaded() == 1 && store.Find(key_b, words) &&
              words == Words(7, 7),
          "the replaced file holds the new build's modules");
  }
}

void TestWriterAppends() {
  const auto path = TempPath("title.v1.simplified");
  const auto key = SimplifiedStore::MakeKey(Words(10, 9), 0);
  SimplifiedStore store(path, "build", 1ms);
  store.Insert(key, Words(20, 9));
  // The writer thread appends the record while the session goes on, so a
  // killed session keeps it.
  bool written = false;
  for (int i = 0; i < 500 && !written; i++) {
    std::this_thread::sleep_for(10ms);
    SimplifiedStore other(path, "build", 1ms);
    written = other.Loaded() == 1;
  }
  Check(written, "the writer thread did not append the record");
}

void TestTornRecord() {
  const auto path = TempPath("title.v1.simplified");
  const auto key_a = SimplifiedStore::MakeKey(Words(10, 1), 0);
  const auto key_b = SimplifiedStore::MakeKey(Words(10, 2), 0);
  {
    SimplifiedStore store(path, "build", 1ms);
    store.Insert(key_a, Words(40, 1));
    store.Flush();
    store.Insert(key_b, Words(40, 2));
  }
  // An append cut short (the process ended in the middle of it).
  const auto size = std::filesystem::file_size(path);
  std::filesystem::resize_file(path, size - 6);
  {
    SimplifiedStore store(path, "build", 1ms);
    std::vector<uint32_t> words;
    Check(store.Loaded() == 1 && store.Find(key_a, words) &&
              !store.Find(key_b, words),
          "a torn last record is dropped, the records before it are kept");
    store.Insert(key_b, Words(40, 2));
  }
  {
    SimplifiedStore store(path, "build", 1ms);
    std::vector<uint32_t> words;
    Check(store.Loaded() == 2 && store.Find(key_b, words) &&
              words == Words(40, 2),
          "records appended after a dropped torn record are read");
  }
  // A damaged record (its words do not match the checksum).
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(-4, std::ios::end);
    const uint32_t junk = 0xdeadbeefu;
    file.write(reinterpret_cast<const char *>(&junk), sizeof(junk));
  }
  {
    SimplifiedStore store(path, "build", 1ms);
    Check(store.Loaded() == 1, "a damaged record is dropped");
  }
}

} // namespace

int main() {
  TestKeys();
  TestAcrossSessions();
  TestWriterAppends();
  TestTornRecord();
  std::filesystem::remove_all(std::filesystem::temp_directory_path() /
                              "kyty_spirv_simplified_store_tests");
  std::printf("SpirvSimplifiedStoreTests: all tests passed\n");
  return 0;
}
