#include "graphics/host_gpu/renderer/pipeline/srtWalkerCompileQueue.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include <xbyak/xbyak.h>

#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <thread>

#if defined(__linux__)
#include <csignal>
#include <sys/mman.h>
#include <ucontext.h>
#endif

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::Program SrtProgram(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return program;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(&dword)));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestNativeNestedSrtRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::array<uint32_t, 3> first{0u, 0x12345678u, 0xabcdef01u};
  std::array<uint32_t, 3> second{0u, 0x87654321u, 0x10fedcbau};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address) | 3u,
                                  static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
  };
  auto table = pointer_words(first.data());
  const auto user_data = pointer_words(table.data());
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 2;
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto low = emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))});
  const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto read = [&](Value address, uint32_t offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {address, Value(offset), Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto pointer_low = read(root, 0u);
  const auto pointer_high = read(root, 4u);
  auto *leaf = program.srt_reads[0].value.Instruction();
  leaf->Arg(0).Instruction()->SetArg(0, pointer_low);
  leaf->Arg(0).Instruction()->SetArg(1, pointer_high);
  leaf->SetArg(1, Value(7u));
  read(leaf->Arg(0), 11u);
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_code != nullptr, "nested user-data tree did not compile an x86 walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const SrtRuntime runtime{.user_data = user_data};
  {
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(snapshot.flattened_srt),
          "native SRT refresh failed");
    const auto loaded = first[1];
    first[1] ^= 0xffffffffu;
    uint32_t value = 0;
    Check(walker.Evaluate(plan.srt_reads[0].value, value) && value == loaded,
          "descriptor evaluation reloaded a value already copied by the native walker");
    first[1] = loaded;
  }
  for (const auto *data : {&first, &second}) {
    table = pointer_words(data->data());
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt ==
                  std::vector<uint32_t>{(*data)[1], table[0], table[1], (*data)[2]},
          "native SRT refresh lost pointer masking, alignment, or current values");
  }
#if defined(__linux__)
  struct sigaction action {}, previous {};
  action.sa_sigaction = +[](int, siginfo_t *, void *context) {
    auto &registers = static_cast<ucontext_t *>(context)->uc_mcontext.gregs;
    const auto length = SrtWalkerLoadLength(static_cast<uint64_t>(registers[REG_RIP]));
    if (length == 0) std::abort();
    registers[REG_RAX] = 0;
    registers[REG_RIP] += length;
  };
  action.sa_flags = SA_SIGINFO;
  auto *page = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Check(page != MAP_FAILED && sigaction(SIGSEGV, &action, &previous) == 0 &&
            MaterializeResources(plan, {.user_data = pointer_words(page)}, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>(4, 0u),
        "faulting walker loads of a table and a null pointer did not read zero");
  sigaction(SIGSEGV, &previous, nullptr);
  munmap(page, 4096);
#endif
  Check(!MaterializeResources(plan, {.user_data = std::span(user_data).first(1)},
                              snapshot, specialization),
        "native SRT walk accepted truncated user data");
  const std::array<uint32_t, 2> fake_user_data{0x1003u, 0xabcd0000u};
  uint32_t reads = 0;
  const SrtRuntime callback_runtime{
      .user_data = fake_user_data,
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++*static_cast<uint32_t *>(data);
        switch (address) {
          case 0x1000u: words[0] = 0x2003u; break;
          case 0x1004u: words[0] = 0xabcd0000u; break;
          case 0x2004u: words[0] = 0x1234u; break;
          case 0x2008u: words[0] = 0x5678u; break;
          default: return false;
        }
        return true;
      },
      .userdata = &reads};
  Check(MaterializeResources(plan, callback_runtime, snapshot, specialization) && reads == 4 &&
            snapshot.flattened_srt ==
                std::vector<uint32_t>{0x1234u, 0x2003u, 0xabcd0000u, 0x5678u},
        "native-eligible SRT bypassed its callback or repeated nested pointer reads");
  leaf->SetArg(0, root);
  leaf->SetArg(1, Value(0u));
  program.srt_reads.resize(1);
  for (const int32_t offset : {-1, 7}) {
    program.memory_info[0].offset = static_cast<uint32_t>(offset);
    auto bounded = ExtractResourcePlan(program);
    auto base = pointer_words(second.data() + 1);
    Check(bounded.srt_code != nullptr &&
              MaterializeResources(bounded, {.user_data = base}, snapshot, specialization) &&
              snapshot.flattened_srt[0] == second[offset < 0 ? 0 : 2],
          "native SRT applied a signed immediate offset incorrectly");
    base = offset < 0 ? std::array<uint32_t, 2>{0u, 0u}
                      : std::array<uint32_t, 2>{0xfffffffcu, 0xffffu};
    Check(!MaterializeResources(bounded, {.user_data = base}, snapshot, specialization),
          "native SRT accepted address underflow or 48-bit overflow");
  }
  high.Instruction()->SetArg(0, Value(ScalarReg(6)));
  Check(ExtractResourcePlan(program).srt_code == nullptr,
        "native SRT accepted nonadjacent user-data pointer words");
  high.Instruction()->SetArg(0, Value(ScalarReg(5)));
  program.memory_info[0].offset = 0;
  program.srt_reads.push_back({pointer_low, 1});
  program.srt_reads.push_back({pointer_high, 2});
  leaf->SetArg(0, emit(ValueOpcode::GetAddressResource, {pointer_low, pointer_high}));
  pointer_high.Instruction()->SetArg(1, Value(8u));
  Check(ExtractResourcePlan(program).srt_code == nullptr,
        "native SRT accepted nonadjacent nested pointer words");
}

void TestGeneralWalkerScopedMemo() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // A chain of selects whose arms both use the previous link: without a memo
  // inside conditional regions the compiled code doubles at every link.
  std::array<uint32_t, 8> table{0x11u, 0x12u, 0x13u, 0x14u,
                                0x15u, 0x16u, 0x17u, 0x18u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 3> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 3;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto control = emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))});
  constexpr uint32_t Links = 24;
  auto link = control;
  for (uint32_t index = 0; index < Links; ++index) {
    const auto chosen = emit(ValueOpcode::IEqual32, {control, Value(index)});
    link = emit(ValueOpcode::SelectU32,
                {chosen, emit(ValueOpcode::IAdd32, {link, Value(1u)}),
                 emit(ValueOpcode::IAdd32, {link, Value(2u)})});
  }
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto offset =
      emit(ValueOpcode::ShiftLeftLogical32,
           {emit(ValueOpcode::BitwiseAnd32, {link, Value(7u)}), Value(2u)});
  const auto value =
      emit(ValueOpcode::LoadAddressU32, {root, offset, Value(0u), Value(true)});
  value.Instruction()->SetFlags(MemoryFlags{.index = 0});
  program.srt_reads.push_back({value, 0u});
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_general_code != nullptr,
        "select chain did not compile the general walker");
  Check(plan.srt_general_code->getSize() < (64u << 10u),
        "compiled select chain re-emits shared values inside its arms");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const uint32_t selector : {0u, 5u, 23u, 99u}) {
    user_data[2] = selector;
    const SrtRuntime runtime{.user_data = user_data};
    // Every link adds 2 except the selected one, which adds 1.
    const uint32_t expected =
        selector + 2u * Links - (selector < Links ? 1u : 0u);
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt ==
                  std::vector<uint32_t>{table[expected & 7u]},
          "compiled select chain returned the wrong value");
  }
}

void TestGeneralWalkerAssumesExecFactor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // A lane read under an EXEC factor the walker cannot evaluate: selects on
  // that factor take their true arm, and a conjunction with it is false when
  // its right operand is false under the same assumption.
  std::array<uint32_t, 8> table{0x11u, 0x12u, 0x13u, 0x14u,
                                0x15u, 0x16u, 0x17u, 0x18u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 4> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u,
                                    0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 4;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto lane =
      emit(ValueOpcode::GetBuiltin,
           {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
            Value(0u)});
  const auto active = emit(ValueOpcode::ULessThan32, {lane, Value(32u)});
  const auto index = emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))});
  const auto masked = emit(ValueOpcode::SelectU32, {active, index, Value(7u)});
  const auto uniform = emit(ValueOpcode::ReadFirstLane, {masked, active});
  // The conjunction is false whenever the selector word is not 3, even though
  // its left operand is unknown.
  const auto undef = emit(ValueOpcode::UndefU1, {});
  const auto gate = emit(
      ValueOpcode::LogicalAnd,
      {undef, emit(ValueOpcode::IEqual32,
                   {emit(ValueOpcode::SelectU32, {undef, index, Value(3u)}),
                    Value(3u)})});
  const auto chosen = emit(ValueOpcode::SelectU32, {gate, Value(5u), uniform});
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto value =
      emit(ValueOpcode::LoadAddressU32,
           {root, emit(ValueOpcode::ShiftLeftLogical32, {chosen, Value(2u)}),
            Value(0u), Value(true)});
  value.Instruction()->SetFlags(MemoryFlags{.index = 0});
  program.srt_reads.push_back({value, 0u});
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_general_code != nullptr,
        "lane read under an EXEC factor did not compile the general walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const uint32_t selector : {0u, 2u, 6u}) {
    user_data[2] = selector;
    const SrtRuntime runtime{.user_data = user_data};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt == std::vector<uint32_t>{table[selector]},
          "compiled walker did not assume the EXEC factor");
    auto code = std::move(plan.srt_general_code);
    ResourceSnapshot interpreted;
    Check(MaterializeResources(plan, runtime, interpreted, specialization) &&
              interpreted.flattened_srt == snapshot.flattened_srt,
          "interpreter and compiled walker disagree under an assumption");
    plan.srt_general_code = std::move(code);
  }
  // Selector 3 leaves the conjunction unknown: neither evaluator can
  // materialize.
  user_data[2] = 3u;
  Check(!MaterializeResources(plan, {.user_data = user_data}, snapshot,
                              specialization),
        "compiled walker decided an unknown conjunction");
}

void TestGeneralWalkerZeroUnsignedBound() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // No U32 is below zero: a zero upper bound decides the compare without its
  // unknown left operand, in the compiled walker as in the interpreter.
  constexpr uint64_t table_address = 0x10000u;
  struct Reads {
    std::array<uint32_t, 4> table{};
    uint32_t count = 0;
  } reads;
  std::array<uint32_t, 2> user_data{
      static_cast<uint32_t>(table_address),
      static_cast<uint32_t>(table_address >> 32u)};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 2;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto read = [&](Value byte_offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {root, byte_offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto bound = read(Value(0u));
  const auto below =
      emit(ValueOpcode::ULessThan32, {emit(ValueOpcode::UndefU32, {}), bound});
  read(emit(ValueOpcode::SelectU32, {below, Value(8u), Value(4u)}));
  auto plan = ExtractResourcePlan(program);
  Check(
      plan.srt_general_code != nullptr,
      "compare with a zero unsigned bound did not compile the general walker");
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory =
          +[](void *data, uint64_t address, std::span<uint32_t> words) {
            auto &reads = *static_cast<Reads *>(data);
            if (address < table_address || address >= table_address + 16u ||
                address % 4u || words.size() != 1)
              return false;
            ++reads.count;
            words[0] = reads.table[(address - table_address) / 4u];
            return true;
          },
      .userdata = &reads};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  reads.table = {0u, 0x11u, 0x12u, 0x13u};
  // The compiled walker reads the bound and the selected word once each; a
  // failed native run would read them again in the interpreter.
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>{0u, 0x11u} &&
            reads.count == 2u,
        "compiled walker did not decide a compare with a zero unsigned bound");
  auto code = std::move(plan.srt_general_code);
  ResourceSnapshot interpreted;
  Check(MaterializeResources(plan, runtime, interpreted, specialization) &&
            interpreted.flattened_srt == snapshot.flattened_srt,
        "interpreter and compiled walker disagree on a zero unsigned bound");
  plan.srt_general_code = std::move(code);
  // A nonzero bound leaves the compare unknown: neither evaluator decides it.
  reads.table[0] = 3u;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization),
        "compiled walker decided a compare with an unknown operand");
  plan.srt_general_code.reset();
  Check(!MaterializeResources(plan, runtime, interpreted, specialization),
        "interpreter decided a compare with an unknown operand");
}

void TestGeneralWalkerCompares64() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // A 64-bit pointer tested against null or another address compiles in the
  // general walker and compares all 64 bits, as the interpreter does: a
  // pointer with a zero low half is not null.
  std::array<uint32_t, 4> table{0x11u, 0x12u, 0x13u, 0x14u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 4> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u,
                                    0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 4;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto pointer =
      emit(ValueOpcode::CompositeConstructU64,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(7))})});
  const auto read = [&](Value byte_offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {root, byte_offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
  };
  constexpr uint64_t other = uint64_t{1} << 32u;
  read(emit(ValueOpcode::SelectU32,
            {emit(ValueOpcode::IEqual64, {pointer, Value(uint64_t{0})}),
             Value(0u), Value(4u)}));
  read(emit(ValueOpcode::SelectU32,
            {emit(ValueOpcode::INotEqual64, {pointer, Value(other)}), Value(8u),
             Value(12u)}));
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_general_code != nullptr,
        "64-bit compares did not compile the general walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const uint64_t value :
       {uint64_t{0}, other, uint64_t{1}, uint64_t{0xffffffff00000000u}}) {
    user_data[2] = static_cast<uint32_t>(value);
    user_data[3] = static_cast<uint32_t>(value >> 32u);
    const std::vector<uint32_t> expected{value == 0u ? 0x11u : 0x12u,
                                         value != other ? 0x13u : 0x14u};
    Check(MaterializeResources(plan, {.user_data = user_data}, snapshot,
                               specialization) &&
              snapshot.flattened_srt == expected,
          "compiled walker compared a 64-bit pointer by its low half");
    auto code = std::move(plan.srt_general_code);
    ResourceSnapshot interpreted;
    Check(MaterializeResources(plan, {.user_data = user_data}, interpreted,
                               specialization) &&
              interpreted.flattened_srt == expected,
          "interpreter and compiled walker disagree on a 64-bit compare");
    plan.srt_general_code = std::move(code);
  }
}

void TestGeneralWalkerLeavesFloatToInterpreter() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // The compiled walker does not model floating point: a plan whose read
  // offset goes through float arithmetic stays with the interpreter.
  std::array<uint32_t, 4> table{0x11u, 0x12u, 0x13u, 0x14u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 3> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 3;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto index =
      emit(ValueOpcode::ConvertU32F32,
           {emit(ValueOpcode::FPMul32,
                 {emit(ValueOpcode::ConvertF32U32,
                       {emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))})}),
                  Value::F32(1.f)})});
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto value =
      emit(ValueOpcode::LoadAddressU32,
           {root, emit(ValueOpcode::ShiftLeftLogical32, {index, Value(2u)}),
            Value(0u), Value(true)});
  value.Instruction()->SetFlags(MemoryFlags{.index = 0});
  program.srt_reads.push_back({value, 0u});
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_plan_complete && plan.srt_general_code == nullptr,
        "the general walker compiled float arithmetic");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const uint32_t selector : {0u, 3u}) {
    user_data[2] = selector;
    Check(MaterializeResources(plan, {.user_data = user_data}, snapshot,
                               specialization) &&
              snapshot.flattened_srt == std::vector<uint32_t>{table[selector]},
          "the interpreter did not walk a float-computed offset");
  }
}

void TestGeneralWalkerSharedValuesCompileOnce() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Many reads at offsets selected by user data, every arm using one long
  // chain over a guest read: the chain is compiled once, not once per arm.
  std::array<uint32_t, 8> table{3u,    0x12u, 0x13u, 0x14u,
                                0x15u, 0x16u, 0x17u, 0x18u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 3> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 3;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto read = [&](Value byte_offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {root, byte_offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto base = read(Value(0u));
  auto chain = base;
  constexpr uint32_t Links = 256;
  for (uint32_t link = 0; link < Links; ++link)
    chain = emit(ValueOpcode::IAdd32, {chain, base});
  const auto control = emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))});
  constexpr uint32_t Reads = 64;
  for (uint32_t index = 0; index < Reads; ++index) {
    const auto chosen = emit(ValueOpcode::IEqual32, {control, Value(index)});
    const auto element =
        emit(ValueOpcode::SelectU32,
             {chosen, chain, emit(ValueOpcode::IAdd32, {chain, Value(1u)})});
    read(emit(
        ValueOpcode::ShiftLeftLogical32,
        {emit(ValueOpcode::BitwiseAnd32, {element, Value(7u)}), Value(2u)}));
  }
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_general_code != nullptr &&
            plan.srt_general_code->getSize() < (256u << 10u),
        "shared values were compiled once per use");
  ResourceSnapshot snapshot;
  ResourceSnapshot interpreted;
  ResourceSpecialization specialization;
  for (const uint32_t selector : {0u, 17u, 99u}) {
    user_data[2] = selector;
    const SrtRuntime runtime{.user_data = user_data};
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "compiled walker failed on shared values");
    // chain = base * (Links + 1); the selected read adds nothing, the others 1.
    for (uint32_t index = 0; index < Reads; ++index) {
      const auto element = 3u * (Links + 1u) + (index == selector ? 0u : 1u);
      Check(snapshot.flattened_srt[1u + index] == table[element & 7u],
            "compiled walker returned the wrong shared value");
    }
    auto code = std::move(plan.srt_general_code);
    Check(MaterializeResources(plan, runtime, interpreted, specialization) &&
              interpreted.flattened_srt == snapshot.flattened_srt,
          "interpreter and compiled walker disagree on shared values");
    plan.srt_general_code = std::move(code);
  }
}

void TestGeneralWalkerRemembersUnknownValues() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // A disjunction whose operands are one unknown value evaluates its right
  // operand on both paths. Each link of the chain doubles the paths, so a
  // walker that evaluates an unknown value again on every path never ends;
  // the interpreter remembers it, and so must the compiled walker.
  std::array<uint32_t, 4> table{0x11u, 0x12u, 0x13u, 0x14u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 3> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 3;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  auto unknown = emit(ValueOpcode::UndefU1, {});
  for (uint32_t link = 0; link < 64; ++link)
    unknown = emit(ValueOpcode::LogicalOr, {unknown, unknown});
  // The selector word decides the select before the unknown chain is needed.
  const auto known =
      emit(ValueOpcode::IEqual32,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))}), Value(1u)});
  const auto gate = emit(ValueOpcode::LogicalOr, {known, unknown});
  const auto root =
      emit(ValueOpcode::GetAddressResource,
           {emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto value =
      emit(ValueOpcode::LoadAddressU32,
           {root, emit(ValueOpcode::SelectU32, {gate, Value(4u), Value(8u)}),
            Value(0u), Value(true)});
  value.Instruction()->SetFlags(MemoryFlags{.index = 0});
  program.srt_reads.push_back({value, 0u});
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_general_code != nullptr,
        "an unknown chain did not compile the general walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  user_data[2] = 1u;
  Check(MaterializeResources(plan, {.user_data = user_data}, snapshot,
                             specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>{table[1]},
        "compiled walker did not decide a disjunction with a true operand");
  user_data[2] = 0u;
  Check(!MaterializeResources(plan, {.user_data = user_data}, snapshot,
                              specialization),
        "compiled walker decided an unknown disjunction");
}

void TestRangeOverlapIndex() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // The sorted index answers exactly as a scan of every range does, including
  // empty ranges, nested ranges and ranges that only touch the query.
  std::mt19937_64 random(7u);
  for (uint32_t round = 0; round < 200; ++round) {
    std::vector<std::pair<uint64_t, uint64_t>> ranges(random() % 40u);
    for (auto &[address, bytes] : ranges) {
      address = 0x1000u + random() % 4096u;
      bytes = random() % 4u == 0u ? 0u : random() % 512u;
    }
    std::vector<std::pair<uint64_t, uint64_t>> storage;
    RangeOverlapIndex index;
    for (uint32_t query = 0; query < 64; ++query) {
      const uint64_t base = 0x1000u + random() % 4800u;
      const uint64_t size = random() % 8u == 0u ? 0u : 1u + random() % 300u;
      bool expected = false;
      for (const auto [address, bytes] : ranges)
        expected |= bytes != 0u && size != 0u && address < base + size &&
                    base < address + bytes;
      Check(index.Overlaps(ranges, storage, base, size) == expected,
            "range overlap index disagrees with a scan");
    }
  }
}

void TestGeneralNativeSrtWalker() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Two tables; user data holds a pointer to each and a selector/index word.
  std::array<uint32_t, 8> first{0x11u, 0x12u, 0x13u, 0x14u,
                                0x15u, 0x16u, 0x17u, 0x18u};
  std::array<uint32_t, 8> second{0x21u, 0x22u, 0x23u, 0x24u,
                                 0x25u, 0x26u, 0x27u, 0x28u};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address),
                                   static_cast<uint32_t>(address >> 32u)};
  };
  const auto first_words = pointer_words(first.data());
  const auto second_words = pointer_words(second.data());
  std::array<uint32_t, 5> user_data{first_words[0], first_words[1],
                                    second_words[0], second_words[1], 0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 5;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto control = emit(ValueOpcode::GetUserData, {Value(ScalarReg(8))});
  // A pointer chosen by a user-data test, read at an offset computed from user
  // data.
  const auto chosen = emit(ValueOpcode::IEqual32, {control, Value(1u)});
  const auto low =
      emit(ValueOpcode::SelectU32,
           {chosen, emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))})});
  const auto high =
      emit(ValueOpcode::SelectU32,
           {chosen, emit(ValueOpcode::GetUserData, {Value(ScalarReg(7))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto offset =
      emit(ValueOpcode::ShiftLeftLogical32,
           {emit(ValueOpcode::BitwiseAnd32, {control, Value(6u)}), Value(2u)});
  const auto read = [&](Value address, Value byte_offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {address, byte_offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  read(root, offset);
  read(root, Value(4u));
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_code == nullptr && plan.srt_general_code != nullptr,
        "selected pointer with a computed offset did not compile the general "
        "walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const uint32_t selector : {0u, 1u, 3u, 7u}) {
    user_data[4] = selector;
    const SrtRuntime runtime{.user_data = user_data};
    const auto &table = selector == 1u ? second : first;
    const auto index = selector & 6u;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt ==
                  std::vector<uint32_t>{table[index], table[1]},
          "general native walker returned the wrong selected table or offset");
    uint32_t value = 0;
    SrtWalker walker(plan, runtime);
    Check(walker.Evaluate(plan.srt_reads[0].value, value) &&
              value == table[index],
          "general native walker did not seed the descriptor memo");
    // The interpreter agrees with the compiled walker.
    auto code = std::move(plan.srt_general_code);
    ResourceSnapshot interpreted;
    Check(MaterializeResources(plan, runtime, interpreted, specialization) &&
              interpreted.flattened_srt == snapshot.flattened_srt &&
              interpreted.images == snapshot.images,
          "interpreter and general native walker disagree");
    plan.srt_general_code = std::move(code);
  }
  // A failed native run (user data too short for the selector register) falls
  // back to the interpreter, which rejects the plan the same way.
  Check(!MaterializeResources(plan,
                              {.user_data = std::span(user_data).first(4)},
                              snapshot, specialization),
        "general native walker accepted truncated user data");
}

// The pipeline cache interprets a new shader's plan right away and compiles its
// general walker on a second extraction in the background; the compiled plan
// then replaces the interpreted one and must refresh identically.
void TestBackgroundGeneralWalker() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::array<uint32_t, 8> first{0x11u, 0x12u, 0x13u, 0x14u,
                                0x15u, 0x16u, 0x17u, 0x18u};
  std::array<uint32_t, 8> second{0x21u, 0x22u, 0x23u, 0x24u,
                                 0x25u, 0x26u, 0x27u, 0x28u};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address),
                                   static_cast<uint32_t>(address >> 32u)};
  };
  const auto first_words = pointer_words(first.data());
  const auto second_words = pointer_words(second.data());
  std::array<uint32_t, 5> user_data{first_words[0], first_words[1],
                                    second_words[0], second_words[1], 0u};
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 5;
  program.srt_reads.clear();
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto control = emit(ValueOpcode::GetUserData, {Value(ScalarReg(8))});
  const auto chosen = emit(ValueOpcode::IEqual32, {control, Value(1u)});
  const auto low =
      emit(ValueOpcode::SelectU32,
           {chosen, emit(ValueOpcode::GetUserData, {Value(ScalarReg(6))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))})});
  const auto high =
      emit(ValueOpcode::SelectU32,
           {chosen, emit(ValueOpcode::GetUserData, {Value(ScalarReg(7))}),
            emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))})});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto offset =
      emit(ValueOpcode::ShiftLeftLogical32,
           {emit(ValueOpcode::BitwiseAnd32, {control, Value(6u)}), Value(2u)});
  for (const auto byte_offset : {offset, Value(4u)}) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {root, byte_offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
  }

  auto interpreted =
      ExtractResourcePlan(program, SrtWalkerCompilation::SimpleOnly);
  Check(interpreted.srt_general_code == nullptr,
        "a plan extracted for the background compile already had a general "
        "walker");
  std::unique_ptr<ResourcePlan> compiled;
  {
    Libs::Graphics::SrtWalkerCompileQueue queue;
    // A job its owner dropped is skipped without harm.
    (void)queue.Submit(std::make_unique<ResourcePlan>(
        ExtractResourcePlan(program, SrtWalkerCompilation::SimpleOnly)));
    auto job = queue.Submit(std::make_unique<ResourcePlan>(
        ExtractResourcePlan(program, SrtWalkerCompilation::SimpleOnly)));
    for (uint32_t wait = 0; wait < 10000u && !job->Ready(); ++wait)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(job->Ready(), "the background walker compile did not finish");
    compiled = job->Take();
  }
  Check(compiled != nullptr && compiled->srt_general_code != nullptr,
        "the background compile did not produce a general walker");
  for (const uint32_t selector : {0u, 1u, 3u, 7u}) {
    user_data[4] = selector;
    const SrtRuntime runtime{.user_data = user_data};
    const auto &table = selector == 1u ? second : first;
    ResourceSnapshot from_interpreter, from_compiled;
    ResourceSpecialization interpreter_specialization, compiled_specialization;
    Check(MaterializeResources(interpreted, runtime, from_interpreter,
                               interpreter_specialization) &&
              MaterializeResources(*compiled, runtime, from_compiled,
                                   compiled_specialization) &&
              from_interpreter.flattened_srt ==
                  std::vector<uint32_t>{table[selector & 6u], table[1]} &&
              from_compiled.flattened_srt == from_interpreter.flattened_srt &&
              compiled_specialization == interpreter_specialization,
          "the plan compiled in the background refreshed differently");
  }
}

void TestGeneralWalkerLoopFixpoint() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  // entry -> header (enter) or skip; header -> body (loop flag and an opaque
  // loop-carried test) or loop exit; body reads and returns to the header;
  // skip and loop exit join on a control phi that picks the use block, which
  // reads again and holds a buffer descriptor. The phi is unknown while every
  // block counts as visited, so reaching the right blocks takes the
  // interpreter's fixpoint, which the compiled walker must reproduce.
  std::array<uint32_t, 8> table{0x11u, 0x12u, 0x13u, 0x14u,
                                0x15u, 0x16u, 0x17u, 0x18u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  std::array<uint32_t, 5> user_data{static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u), 0u,
                                    0u, 3u};
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 5;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 8; ++i)
    AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &header = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &loop_exit = *program.blocks[3];
  auto &join = *program.blocks[4];
  auto &use = *program.blocks[5];
  auto &exit = *program.blocks[6];
  auto &skip = *program.blocks[7];
  const auto emit = [](Block &block, ValueOpcode opcode,
                       std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto branch = [](Block &block, Block &target, Value condition = {},
                         Block *other = nullptr) {
    block.terminator.kind =
        other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other)
      block.AddBranch(other);
  };
  const auto user = [&](uint32_t reg) {
    return emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(reg))});
  };
  const auto low = user(0);
  const auto high = user(1);
  const auto enter = user(2);
  const auto loop_flag = user(3);
  const auto index = user(4);
  program.memory_info.push_back(
      {.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto root = emit(entry, ValueOpcode::GetAddressResource, {low, high});
  const auto read = [&](Value byte_offset) {
    const auto value = emit(entry, ValueOpcode::LoadAddressU32,
                            {root, byte_offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
  };
  read(emit(entry, ValueOpcode::ShiftLeftLogical32, {index, Value(2u)}));
  read(Value(28u));
  const auto srt = emit(entry, ValueOpcode::GetSrtResource, {});
  emit(body, ValueOpcode::ReadConst, {srt, Value(0u)});
  const auto flat = emit(use, ValueOpcode::ReadConst, {srt, Value(1u)});
  branch(entry, header, emit(entry, ValueOpcode::IEqual32, {enter, Value(1u)}),
         &skip);

  auto &carried = header.AppendNewInst(ValueOpcode::Phi);
  carried.SetFlags(Type::U32);
  const auto next =
      emit(body, ValueOpcode::IAdd32, {Value(&carried), Value(1u)});
  carried.AddPhiOperand(&entry, index);
  carried.AddPhiOperand(&body, next);
  const auto again = emit(
      header, ValueOpcode::LogicalAnd,
      {emit(header, ValueOpcode::IEqual32, {loop_flag, Value(1u)}),
       emit(header, ValueOpcode::INotEqual32, {Value(&carried), Value(100u)})});
  branch(header, body, again, &loop_exit);
  header.terminator.loop_header = true;
  header.terminator.continue_block = &body;
  header.terminator.merge_block = &loop_exit;
  branch(body, header);
  branch(loop_exit, join);
  branch(skip, join);
  auto &selected = join.AppendNewInst(ValueOpcode::Phi);
  selected.SetFlags(Type::U32);
  selected.AddPhiOperand(&skip, Value(5u));
  selected.AddPhiOperand(&loop_exit, Value(6u));
  branch(join, use,
         emit(join, ValueOpcode::IEqual32, {Value(&selected), Value(5u)}),
         &exit);
  branch(use, exit);
  exit.terminator.kind = TerminatorKind::Return;

  DescriptorSource source;
  source.dwords = {flat, Value(1u << 16u), Value(16u), Value(3u << 28u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = emit(
      use, ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  emit(use, ValueOpcode::LoadBufferU32,
       {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .Instruction()
      ->SetFlags(MemoryFlags{.index = 1});

  auto plan = ExtractResourcePlan(program);
  Check(plan.control_flow.size() == 8 && plan.srt_general_code != nullptr &&
            plan.jit_loops,
        "loop-shaped plan did not compile the general walker");
  struct Case {
    uint32_t enter;
    uint32_t loop;
    std::vector<uint8_t> visited;
    std::vector<uint32_t> flat;
  };
  const std::array cases{
      // The body may run: both header successors; the join phi settles on the
      // loop exit's incoming in the second iteration and skips the use block.
      Case{1u, 1u, {1, 1, 1, 1, 1, 0, 1, 0}, {table[3], 0u}},
      // The loop flag skips the body.
      Case{1u, 0u, {1, 1, 0, 1, 1, 0, 1, 0}, {0u, 0u}},
      // The loop is not reached: the phi takes the skip incoming.
      Case{0u, 1u, {1, 0, 0, 0, 1, 1, 1, 1}, {0u, table[7]}},
  };
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto &expected : cases) {
    user_data[2] = expected.enter;
    user_data[3] = expected.loop;
    const SrtRuntime runtime{.user_data = user_data};
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "compiled loop walker failed to materialize");
    const auto visited = plan.visited_blocks;
    const auto active = plan.active_sources;
    Check(visited == expected.visited,
          "compiled loop walker reached the wrong blocks");
    Check(snapshot.flattened_srt == expected.flat,
          "compiled loop walker refreshed the wrong reads");
    Check(active == std::vector<uint8_t>{expected.visited[5]} &&
              snapshot.buffers[0].dwords[0] == expected.flat[1],
          "compiled loop walker activated the wrong descriptors");
    // The buffer's descriptor is evaluated by the compiled walker in its final
    // pass when the use block is reached, and left to the interpreter when not.
    Check(plan.jit_descriptor_sources == std::vector<uint32_t>{0u} &&
              plan.jit_descriptor_known ==
                  std::vector<uint8_t>{expected.visited[5]} &&
              (!expected.visited[5] ||
               plan.jit_descriptors[0] == expected.flat[1]),
          "compiled loop walker did not evaluate the active descriptor");
    auto code = std::move(plan.srt_general_code);
    ResourceSnapshot interpreted;
    Check(MaterializeResources(plan, runtime, interpreted, specialization),
          "interpreter failed on the loop-shaped plan");
    Check(plan.visited_blocks == visited && plan.active_sources == active &&
              interpreted.flattened_srt == snapshot.flattened_srt &&
              interpreted.buffers == snapshot.buffers,
          "interpreter and compiled loop walker disagree");
    plan.srt_general_code = std::move(code);
  }
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtProgram(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestSrtAliasesRetainReadPolicy() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = SrtProgram(0x1000u);
  auto &block = *program.blocks[0];
  const auto append_read = [&](uint32_t address) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                       {Value(address), Value(0u)});
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(0u), Value(0u), Value(true)});
    // Different reads deliberately share MemoryInfo but have different guest PCs.
    read.SetFlags(MemoryFlags{.index = 0, .pc = address});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&read), slot});
    auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto ordinary = append_read(0x2000u);
  const auto pointer = append_read(0x3000u);
  program.srt_reads[0].value.Instruction()->Arg(0).Instruction()->SetArg(0, pointer);
  auto &mask = block.AppendNewInst(ValueOpcode::INotEqual32, {ordinary, Value(0u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                    {Value(9u), Value(&mask)});
  program.descriptor_sources.push_back({.dwords = {Value(&first), Value(0u)},
                                       .dword_count = 2});
  DescriptorSource indirect;
  indirect.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1};
  program.descriptor_sources.push_back(indirect);
  auto plan = ExtractResourcePlan(program);
  for (const auto &inst : plan.value_storage) {
    Check(inst.GetOpcode() != ValueOpcode::ReadConst &&
              inst.GetOpcode() != ValueOpcode::GetSrtResource,
          "retained plan still contains flattened-read aliases");
  }
  Check(plan.descriptor_sources[0].dwords[0] == plan.srt_reads[0].value &&
            plan.srt_reads[0].value.Instruction()->Flags<SrtReadFlags>().clean == 1u &&
            plan.srt_reads[1].value.Instruction()->Flags<SrtReadFlags>().clean == 0u &&
            plan.srt_reads[2].value.Instruction()->Flags<SrtReadFlags>().clean == 1u,
        "alias normalization lost read identity or retained discarded EXEC provenance");
  Check(plan.srt_code == nullptr,
        "captured strict reads compiled an unused x86 walker");
  struct Reads { uint32_t strict = 0; uint32_t ordinary = 0; bool dirty = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        if (address == 0x2000u) words[0] = 0x2222u;
        else if (address == 0x3000u) words[0] = 0x8000u;
        else if (address == 0x8000u) words[0] = 0xeeeeu;
        else return false;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (reads.dirty) return false;
        if (address == 0x3000u) words[0] = 0x1000u;
        else if (address == 0x1000u) words[0] = 0x1111u;
        else return false;
        return true;
      }};
  std::vector<uint32_t> flat;
  DescriptorValue descriptor;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, &clean);
    Check(walker.RefreshFlatBuffer(flat) &&
              flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
              walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == 0x1111u &&
              reads.strict == 2 && reads.ordinary == 1,
          "normalized aliases changed nested strict reads or repeated a shared read");
  }
  Check(!SrtWalker(plan, runtime).RefreshFlatBuffer(flat),
        "strict flat read accepted a missing clean evaluator");
  auto no_reader = runtime;
  no_reader.read_specialization_memory = nullptr;
  {
    SrtWalker clean(plan, CleanRuntime(no_reader));
    Check(!SrtWalker(plan, no_reader, &clean).RefreshFlatBuffer(flat),
          "strict flat read accepted a missing strict reader");
  }
  Check(SrtWalker(plan, runtime).EvaluateDescriptor(0, descriptor) &&
            descriptor.dwords[0] == 0xeeeeu && reads.strict == 2 && reads.ordinary == 3,
        "direct descriptor evaluation without a clean evaluator changed read domains");
  reads.dirty = true;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    Check(!SrtWalker(plan, runtime, &clean).RefreshFlatBuffer(flat) &&
              reads.strict == 3 && reads.ordinary == 3,
          "dirty strict pointer fell back to an ordinary read");
  }
}

void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t strict = 0; uint32_t ordinary = 0; bool clean = true; } reads;
  const SrtRuntime runtime{
      .read_memory = [](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 999;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = ExtractResourcePlan(program);
    Check(plan.control_flow.empty() && plan.requires_specialization_memory &&
              plan.capture_specialization_reads,
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.strict == 1 && reads.ordinary == 0 &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "vector descriptor input was not read and captured exactly once");
    reads.clean = false;
    reads.strict = 0;
    Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.strict == 1 && reads.ordinary == 0,
          "dirty vector descriptor input fell back to an ordinary memory read");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  program.descriptor_sources.push_back({.dwords = {Value(&inactive)}, .dword_count = 1});
  reads.strict = 0;
  uint32_t result = 99;
  {
    const auto plan = ExtractResourcePlan(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "literal false EXEC read vector memory");
  }
  count.SetArg(4, Value(true));
  handle.SetArg(3, Value(0x204u));
  program.descriptor_sources.back().dwords[0] = Value(&count);
  {
    const auto plan = ExtractResourcePlan(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "invalid vector buffer format read memory");
  }
  handle.SetArg(3, Value(0x16204u));
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(SrtWalker(program, {}).Evaluate(Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!SrtWalker(program, {}).Evaluate(Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorPredicateReads(bool memory_condition) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // Only memory-derived host decisions need clean, disjoint descriptor reads.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
      {memory_condition ? Value(&read) : Value(&offset),
       Value(memory_condition ? 0x8000u : 4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.blocks[0]->condition = Value(&condition);
  program.blocks[0]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.blocks[0]->terminator.true_block = program.blocks[1];
  program.blocks[0]->terminator.false_block = program.blocks[2];
  program.blocks[1]->id = 1;
  program.blocks[1]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.blocks[2]->id = 2;
  program.blocks[2]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  Check(plan.capture_specialization_reads == memory_condition && !plan.control_flow.empty(),
        "conditional writable descriptor used the wrong memory dependency policy");
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; bool clean = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 0x8000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool clean : {false, true}) {
    reads = {.clean = clean};
    // A failed predicate explores both edges; its dirty read also rejects the writable descriptor.
    const bool expected = !memory_condition || clean;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) == expected &&
              reads.ordinary == (memory_condition ? 0u : 1u) &&
              reads.strict == (memory_condition ? 1u : 0u),
          "writable descriptor changed reader policy or bypassed a dirty memory predicate");
    if (!expected) continue;
    Check(snapshot.buffers[0].dwords[0] == 0x8000u &&
              snapshot.specialization_reads == (memory_condition
                  ? std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}}
                  : std::vector<std::pair<uint64_t, uint64_t>>{}),
          "writable descriptor lost its address or captured an immutable predicate");
  }
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestSharedControlPredicates() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 8;
  for (const auto operation : {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr}) {
    Program program;
    program.user_data_count = 1;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    auto &block = AddValueBlock(program);
    const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
      return Value(&block.AppendNewInst(opcode, args));
    };
    const auto user = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
    const auto uniform = emit(ValueOpcode::INotEqual32, {user, Value(0u)});
    const auto lane = emit(ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    auto varying = emit(ValueOpcode::INotEqual32, {lane, Value(0u)});
    // Both failed and partially uniform subgraphs share their operands exponentially.
    for (uint32_t i = 0; i < depth; ++i)
      varying = emit(ValueOpcode::LogicalOr, {varying, varying});
    auto condition = emit(operation, {varying, uniform});
    for (uint32_t i = 0; i < depth; ++i)
      condition = emit(ValueOpcode::LogicalOr, {condition, condition});
    condition = emit(ValueOpcode::LogicalNot, {condition});
    condition = emit(ValueOpcode::ConditionRef, {condition});
    for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
    for (uint32_t i = 0; i < 2; ++i) {
      program.blocks[i]->condition = condition;
      program.blocks[i]->terminator.kind = TerminatorKind::ConditionalBranch;
      program.blocks[i]->terminator.true_block = program.blocks[i == 0 ? 1 : 3];
      program.blocks[i]->terminator.false_block = program.blocks[2];
    }
    auto plan = ExtractResourcePlan(program);
    Check(plan.value_storage.size() <= 2u * depth + 12u &&
              plan.control_flow.size() == 4 &&
              plan.control_flow[0].condition == plan.control_flow[1].condition &&
              !plan.capture_specialization_reads,
          "shared predicates expanded into trees or retained varying dependencies");
    for (const uint32_t flag : {0u, 1u}) {
      const std::array<uint32_t, 1> user_data{flag};
      SrtWalker walker(plan, {.user_data = user_data});
      uint32_t result = 99;
      const bool conjunction = operation == ValueOpcode::LogicalAnd;
      const bool known = conjunction ? flag == 0 : flag != 0;
      Check(walker.Evaluate(plan.control_flow[0].condition, result) == known &&
                (!known || result == uint32_t{conjunction}),
            "shared predicate changed its uniform factor or cached runtime values");
      std::vector<uint32_t> flat;
      Check(walker.RefreshFlatBuffer(flat) &&
                plan.visited_blocks == (known
                    ? (conjunction ? std::vector<uint8_t>{1, 1, 0, 1}
                                   : std::vector<uint8_t>{1, 0, 1, 0})
                    : std::vector<uint8_t>{1, 1, 1, 1}),
            "shared predicate skipped an unknown branch or followed a rejected branch");
    }
  }
}

void TestNestedExecPredicatesReuseReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 12;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &use = *program.blocks[1];
  auto &exit = *program.blocks[2];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&entry.AppendNewInst(opcode, args));
  };
  const auto lane = emit(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  auto condition = emit(ValueOpcode::LogicalNot,
      {emit(ValueOpcode::ULessThanEqual32, {count, lane})});
  // A compact shared DAG must not become a tree of fresh EXEC evaluations.
  for (uint32_t i = 0; i < depth; ++i) {
    const auto mask = emit(ValueOpcode::INotEqual32, {lane, Value(i)});
    condition = emit(ValueOpcode::LogicalAnd,
        {mask, emit(ValueOpcode::LogicalOr, {condition, condition})});
  }
  entry.condition = condition;
  entry.terminator.kind = TerminatorKind::ConditionalBranch;
  entry.terminator.true_block = &use;
  entry.terminator.false_block = &exit;
  entry.AddBranch(&use);
  entry.AddBranch(&exit);
  use.terminator.kind = TerminatorKind::Branch;
  use.terminator.true_block = &exit;
  use.AddBranch(&exit);
  DescriptorSource source;
  source.dwords = {Value(0x8000u), Value(1u << 16u), Value(16u), Value(3u << 28u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = Value(&use.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
  use.AppendNewInst(ValueOpcode::LoadBufferU32,
      {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  struct Reads { uint32_t value = 0; bool clean = true; uint32_t strict = 0, ordinary = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        Check(reads.strict <= depth + 1,
              "shared EXEC predicates repeated scalar reads beyond their distinct masks");
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{64u, true}, Reads{0u, true}, Reads{0u, false}, Reads{64u, true}}) {
    reads = state;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "nested EXEC predicates rejected a valid resource snapshot");
    const bool inactive = reads.clean && reads.value == 0;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "nested EXEC predicates lost an unknown branch or retained stale runtime facts");
    Check(reads.strict > 0 && reads.ordinary == 0 &&
              snapshot.specialization_reads.size() <= depth + 1,
          "shared EXEC predicates repeated scalar reads beyond their distinct masks");
  }
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = ExtractResourcePlan(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
    uint32_t ordinary = 0;
  } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    const auto offset = specialization.images[0].indirect_mapping_offset;
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated scalar reads or bypassed clean provenance");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
}

void TestEmptyTraversalSkipsUnmappedDescriptor(bool independent_mask, bool counted_loop = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 6; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &loop = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &merge = *program.blocks[3];
  auto &use = *program.blocks[4];
  auto &exit = *program.blocks[5];
  const auto emit = [](Block &block, ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto branch = [](Block &block, Block &target, Value condition = {}, Block *other = nullptr) {
    block.terminator.kind = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other) block.AddBranch(other);
  };
  Value lane, active, initial;
  if (!counted_loop) {
    lane = emit(entry, ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    active = emit(entry, ValueOpcode::INotEqual32, {lane, Value(0u)});
    const auto initial_mask = independent_mask
        ? emit(entry, ValueOpcode::IEqual32, {lane, Value(0u)}) : active;
    initial = emit(entry, ValueOpcode::SelectU32,
        {initial_mask, Value(UINT32_MAX), Value(0u)});
  }
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(entry, ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(entry, ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  if (counted_loop) {
    branch(entry, loop);
  } else {
    const auto rejected = emit(entry, ValueOpcode::ULessThanEqual32, {count, lane});
    const auto accepted = emit(entry, ValueOpcode::LogicalNot, {rejected});
    branch(entry, loop, emit(entry, ValueOpcode::LogicalAnd, {active, accepted}), &merge);
  }

  // A nonempty traversal contains a varying loop-carried value. Its cycle must
  // remain unknown; only the proven empty traversal can select the bypass arm.
  auto &carried = loop.AppendNewInst(ValueOpcode::Phi);
  carried.SetFlags(Type::U32);
  const auto next = emit(body, ValueOpcode::IAdd32, {Value(&carried), Value(1u)});
  carried.AddPhiOperand(&entry, counted_loop ? Value(0u) : lane);
  carried.AddPhiOperand(&body, next);
  const auto condition = counted_loop
      ? emit(loop, ValueOpcode::ULessThan32, {Value(&carried), count})
      : emit(loop, ValueOpcode::INotEqual32, {Value(&carried), Value(UINT32_MAX)});
  branch(loop, counted_loop ? use : body, condition, &merge);
  loop.terminator.loop_header = true;
  loop.terminator.continue_block = &body;
  loop.terminator.merge_block = &merge;
  branch(body, loop);
  if (counted_loop) {
    branch(merge, exit);
    branch(use, body);
  } else {
    auto &selected = merge.AppendNewInst(ValueOpcode::Phi);
    selected.SetFlags(Type::U32);
    selected.AddPhiOperand(&entry, initial);
    selected.AddPhiOperand(&loop, Value(&carried));
    const auto valid = emit(merge, ValueOpcode::INotEqual32, {Value(&selected), Value(UINT32_MAX)});
    branch(merge, use, emit(merge, ValueOpcode::LogicalAnd, {active, valid}), &exit);
    branch(use, exit);
  }
  DescriptorSource source;
  const auto descriptor_address = emit(use, ValueOpcode::GetAddressResource,
      {Value(0x2000u), Value(0u)});
  for (uint32_t i = 0; i < 4; ++i) {
    const auto index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
    source.dwords[i] = emit(use, ValueOpcode::LoadAddressU32,
        {descriptor_address, Value(i * 4u), Value(0u), Value(true)});
    source.dwords[i].Instruction()->SetFlags(MemoryFlags{.index = index});
  }
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  const auto buffer_index = static_cast<uint32_t>(program.memory_info.size());
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = emit(use, ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  emit(use, ValueOpcode::LoadBufferU32, {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .Instruction()->SetFlags(MemoryFlags{.index = buffer_index});
  auto plan = ExtractResourcePlan(program);
  struct Reads { uint32_t count = 0; bool clean = true; uint32_t descriptors = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        constexpr std::array<uint32_t, 4> descriptor{0x8000u, 1u << 16u, 16u,
                                                   (3u << 28u) | 0x4dfacu};
        if (address < 0x2000u || address >= 0x2010u || address % 4u || words.size() != 1)
          return false;
        ++static_cast<Reads *>(data)->descriptors;
        words[0] = descriptor[(address - 0x2000u) / 4u];
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        const auto &reads = *static_cast<Reads *>(data);
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.count;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{0u, true}, Reads{4u, true}, Reads{0u, false}, Reads{0u, true}}) {
    reads = state;
    if (!counted_loop) {
      uint32_t condition = 99;
      Check(!SrtWalker(plan, runtime).Evaluate(plan.control_flow[3].condition, condition),
            "direct merge evaluation used missing or previous runtime reachability");
    }
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "partial traversal proof rejected a valid resource snapshot");
    const bool inactive = !independent_mask && reads.clean && reads.count == 0u;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "empty traversal kept its descriptor or an uncertain traversal lost it");
    Check(specialization.buffers[0].zero_stride_oob == inactive,
          "inactive traversal did not expose dead vector reads to specialization");
    Check(reads.descriptors == (inactive ? 0u : 4u),
          "inactive traversal read its descriptor or an uncertain traversal skipped it");
    if (inactive) {
      Check(!snapshot.specialization_reads.empty(), "empty traversal lost its clean scalar read");
      for (const auto &read : snapshot.specialization_reads)
        Check(read == std::pair<uint64_t, uint64_t>{0x1000u, 4u},
              "empty traversal proof captured an unrelated memory range");
    }
  }
}

void TestCyclicControlPredicatesRemainUnknown() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 4; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &left = *program.blocks[1];
  auto &right = *program.blocks[2];
  auto &exit = *program.blocks[3];
  const auto lane = Value(&entry.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)}));
  const auto u = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(0u)}));
  const auto v = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(1u)}));
  auto &p = left.AppendNewInst(ValueOpcode::Phi);
  auto &q = right.AppendNewInst(ValueOpcode::Phi);
  p.SetFlags(Type::U1);
  q.SetFlags(Type::U1);
  const auto a = Value(&left.AppendNewInst(ValueOpcode::LogicalAnd, {u, Value(&p)}));
  const auto b = Value(&right.AppendNewInst(ValueOpcode::LogicalAnd, {v, Value(&q)}));
  p.AddPhiOperand(&entry, Value(true));
  p.AddPhiOperand(&right, b);
  q.AddPhiOperand(&entry, Value(true));
  q.AddPhiOperand(&left, a);
  const auto branch = [](Block &block, Value condition, Block &yes, Block &no) {
    block.condition = condition;
    block.terminator.kind = TerminatorKind::ConditionalBranch;
    block.terminator.true_block = &yes;
    block.terminator.false_block = &no;
    block.AddBranch(&yes);
    block.AddBranch(&no);
  };
  // Two loop entries have no single header whose Phi can be discarded up front.
  branch(entry, u, left, right);
  branch(left, a, right, exit);
  branch(right, b, left, exit);
  auto plan = ExtractResourcePlan(program);
  SrtWalker walker(plan, {});
  uint32_t value = 99;
  Check(!walker.Evaluate(plan.control_flow[1].condition, value) &&
            !walker.Evaluate(plan.control_flow[2].condition, value),
        "cyclic varying control acquired an unsupported uniform value");
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) && plan.visited_blocks == std::vector<uint8_t>({1, 1, 1, 1}),
        "cyclic control lost a possible branch during reachability");
}

void TestEmptyScalarBufferSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct Fixture {
    Program program;
    std::vector<std::pair<Inst *, Type>> uses;
    Inst *nonnull = nullptr;
  };
  const auto make_fixture = [] {
    Fixture fixture;
    auto &program = fixture.program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    program.user_data_count = 4u;
    auto &block = AddValueBlock(program);
    DescriptorSource source;
    source.dword_count = 4u;
    for (uint32_t word = 0; word < 4u; ++word)
      source.dwords[word] = Value(&block.AppendNewInst(ValueOpcode::GetUserData,
          {Value(static_cast<ScalarReg>(word))}));
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 0u, .read = true, .scalar = true});
    const auto table = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
    std::array<Value, 4> words;
    for (uint32_t word = 0; word < 4u; ++word) {
      program.memory_info.push_back({.kind = ResourceKind::ScalarBuffer, .resource = 0u,
                                     .offset = word * 4u});
      auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {table, Value(0u)});
      read.SetFlags(MemoryFlags{.index = word});
      words[word] = Value(&read);
    }
    const auto handle = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], words[3]}));
    for (const auto opcode : {ValueOpcode::ReadConstBuffer, ValueOpcode::LoadBufferU8,
                              ValueOpcode::LoadBufferU16, ValueOpcode::LoadBufferU32,
                              ValueOpcode::LoadBufferU32x2, ValueOpcode::LoadBufferU32x3,
                              ValueOpcode::LoadBufferU32x4}) {
      const auto index = static_cast<uint32_t>(program.memory_info.size());
      program.memory_info.push_back({.kind = ResourceKind::IndirectBuffer, .resource = UINT32_MAX,
          .data_dwords = BufferComponentCount(opcode),
          .data_bits = opcode == ValueOpcode::LoadBufferU8 ? 8u : opcode == ValueOpcode::LoadBufferU16 ? 16u : 32u,
          .formatted = opcode == ValueOpcode::LoadBufferU32});
      auto &read = opcode == ValueOpcode::ReadConstBuffer
          ? block.AppendNewInst(opcode, {handle, Value(0u)})
          : block.AppendNewInst(opcode, {handle, Value(0u), Value(0u), Value(0u), Value(true)});
      read.SetFlags(MemoryFlags{.index = index});
      fixture.uses.emplace_back(&block.AppendNewInst(ValueOpcode::Identity, {Value(&read)}), read.GetType());
    }
    const auto nonnull = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], Value(1u)}));
    auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {nonnull, Value(0u)});
    read.SetFlags(MemoryFlags{.index = 4u});
    fixture.nonnull = &block.AppendNewInst(ValueOpcode::Identity, {Value(&read)});
    return fixture;
  };
  auto fixture = make_fixture();
  auto plan = ExtractResourcePlan(fixture.program);
  std::array<uint32_t, 4> descriptor{0x1000u, 8u << 16u, 0u, 3u << 28u};
  uint32_t reads = 0u;
  const SrtRuntime runtime{.user_data = descriptor, .read_memory = RejectSpecializationRead,
                           .userdata = &reads, .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, empty, populated;
  for (const uint32_t records : {0u, 1u, 0x20000000u, 0u}) {
    descriptor[2] = records;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0u &&
              specialization.buffers[0].scalar_empty == (records == 0u),
          "scalar emptiness read table contents or truncated its 64-bit byte size");
    if (records == 0u) {
      if (empty.buffers.empty()) empty = specialization;
      else Check(specialization == empty, "returning to an empty scalar table changed its specialization");
    }
    else if (records == 1u) populated = specialization;
    else Check(specialization == populated, "nonempty scalar size created a shader permutation");
    auto current = make_fixture();
    ApplyResourceSpecialization(current.program, specialization);
    for (const auto &[use, type] : current.uses) {
      const auto value = use->Arg(0).Resolve();
      Check(value.GetType() == type, "empty scalar-table folding changed the load result type");
      if (records != 0u) {
        Check(value.TryInstruction() != nullptr && BufferAccessOf(value.Instruction()->GetOpcode()) == BufferAccess::Read,
              "nonempty scalar table lost a runtime buffer load");
      } else if (value.IsImmediate()) {
        Check(value == Value(uint8_t{0}) || value == Value(uint16_t{0}) || value == Value(0u),
              "empty scalar table did not produce a typed zero");
      } else {
        for (size_t component = 0; component < value.Instruction()->NumArgs(); ++component)
          Check(value.Instruction()->Arg(component) == Value(0u),
                "empty scalar table retained a nonzero vector component");
      }
    }
    Check(current.nonnull->Arg(0).Resolve().TryInstruction() != nullptr,
          "a partly nonzero indirect descriptor was treated as unbound");
  }
  Check(empty != populated, "empty-to-populated scalar transition reused the folded specialization");
  plan.info.buffers[0].scalar = false;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            !specialization.buffers[0].scalar_empty,
        "vector-only resource acquired scalar-empty specialization");
  empty.buffers[0].indirect_root = 0u;
  empty.buffers.push_back(empty.buffers[0]);
  empty.buffers[1].scalar_empty = false;
  ApplyResourceSpecialization(fixture.program, empty);
  for (const auto &[use, type] : fixture.uses)
    Check(use->Arg(0).GetType() == type && use->Arg(0).Resolve().TryInstruction() != nullptr &&
              BufferAccessOf(use->Arg(0).ResolveInstruction()->GetOpcode()) == BufferAccess::Read,
          "empty root folded a scalar read with nonempty candidates");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

void TestSignedStorageSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using F = Libs::Graphics::Prospero::BufferFormat;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 1u;
  auto &block = AddValueBlock(program);
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  constexpr auto swizzle = Libs::Graphics::DstSel(5, 4, 1, 0);
  source.dwords[3] = Value(swizzle |
      (uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
  program.descriptor_sources.push_back(source);
  program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D, .written = true});
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 1> data{};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, signed16;
  for (const auto format : {F::k16SInt, F::k32SInt, F::k16SInt}) {
    data[0] = uint32_t(format) << 20u;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "write-only signed storage image failed materialization");
    const auto conversion = format == F::k16SInt ? format : F::kInvalid;
    const auto view = format == F::k16SInt ? F::k16UInt : F::k32UInt;
    Check(specialization.images[0].numeric_class == N::Uint &&
              specialization.images[0].conversion_format == conversion &&
              specialization.images[0].shader_swizzle == swizzle &&
              snapshot.images[0].dwords[1] == data[0] &&
              RawSignedStorageFormat(plan.info.images[0], format) == view,
          "signed storage specialization lost guest format, conversion, swizzle or raw view width");
    if (format == F::k16SInt) {
      if (signed16.images.empty()) signed16 = specialization;
      else Check(specialization == signed16, "signed storage format refresh retained a stale specialization");
    } else {
      Check(specialization != signed16, "signed 16/32-bit stores shared incompatible conversion state");
    }
    ApplyResourceSpecialization(program, specialization);
    const auto binding = DescriptorBindingForImage(program.info.images[0]);
    Check(binding && ImageBindingResourceClass(*binding) == ImageResourceClass::Storage,
          "signed storage conversion did not reuse the unsigned storage binding");
  }
  for (const auto format : {F::k8SInt, F::k16SInt, F::k32SInt}) {
    data[0] = uint32_t(format) << 20u;
    auto &image = plan.info.images[0];
    image.read = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed raw storage view accepted a non-atomic read/write image");
    image.atomic = true;
    Check((RawSignedStorageFormat(image, format) != F::kInvalid) == (format == F::k32SInt) &&
              MaterializeResources(plan, runtime, snapshot, specialization) == (format == F::k32SInt),
          "signed storage conversion changed atomic format eligibility");
    image.atomic64 = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed 16/32-bit image acquired a 64-bit atomic raw view");
    image.resource_class = ImageResourceClass::Sampled;
    image.written = image.atomic = image.atomic64 = false;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == N::Sint &&
              specialization.images[0].conversion_format == F::kInvalid &&
              Libs::Graphics::Prospero::NumBytesPerElement(format) ==
                  (format == F::k8SInt ? 1u : format == F::k16SInt ? 2u : 4u) &&
              RawSignedStorageFormat(image, format) == F::kInvalid,
          "storage conversion changed ordinary signed image reads");
    image.resource_class = ImageResourceClass::Storage;
    image.written = true;
    image.read = false;
  }
}

void TestIndirectImageSamplerClasses(bool mixed) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::TextureNumericClass;
  auto program = MixedSamplerProgram();
  auto &first = program.descriptor_sources[0];
  first.dwords[0] = Value(0x100u);
  first.dwords[1] = Value(static_cast<uint32_t>(BufferFormat::k8UNorm) << 20u);
  first.dwords[3] = program.descriptor_sources[1].dwords[3];
  program.descriptor_sources[1].dwords[1] = Value(static_cast<uint32_t>(
      mixed ? BufferFormat::k8UInt : BufferFormat::k8UNorm) << 20u);
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 0};
  program.info.images.resize(1);
  program.info.images[0].source = static_cast<uint32_t>(program.descriptor_sources.size());
  program.descriptor_sources.push_back(root);
  program.info.samplers.resize(1);
  program.info.sampled_pairs.resize(1);
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 1> user_data{0x11111111u};
  uint32_t reads = 0;
  const SrtRuntime runtime{.user_data = user_data, .userdata = &reads,
                           .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0 &&
            snapshot.images.size() == 2 && snapshot.images[0].dwords[0] == 0x100u &&
            snapshot.images[1].dwords[0] == 1u &&
            specialization.images[0].numeric_class == TextureNumericClass::Float &&
            specialization.images[1].numeric_class ==
                (mixed ? TextureNumericClass::Uint : TextureNumericClass::Float),
        "indirect image materialization rejected or changed a typed candidate");
  const auto offset = specialization.images[0].indirect_mapping_offset;
  Check(snapshot.flattened_srt[offset] == 3u &&
            snapshot.flattened_srt[offset + 2u] == 0u &&
            snapshot.flattened_srt[offset + 4u] == 1u &&
            snapshot.flattened_srt[offset + 6u] == 0u,
        "mixed image classes changed selector ordinals or duplicated candidates");
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.images[0].indirect_resources == std::vector<uint32_t>({0, 1}) &&
            snapshot.samplers.size() == 1 && program.info.samplers.size() == (mixed ? 2u : 1u),
        "indirect images lost candidates or created unnecessary sampler variants");
  user_data[0] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed image sampler refresh failed");
  for (uint32_t index = 0; index < program.info.samplers.size(); ++index) {
    const auto &sampler = program.info.samplers[index];
    Check(sampler.snapshot_index == 0u && sampler.source == plan.info.samplers[0].source &&
              sampler.force_point_filtering == (index != 0u) &&
              sampler.integer_border == (index != 0u) &&
              snapshot.samplers[sampler.snapshot_index].dwords[0] == user_data[0],
          "typed sampler variants lost filtering, border class, or shared live descriptor");
  }
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  TestMappedSrtUsesDirectReaderByDefault();
  TestNativeNestedSrtRefresh();
  TestGeneralNativeSrtWalker();
  TestBackgroundGeneralWalker();
  TestGeneralWalkerScopedMemo();
  TestGeneralWalkerAssumesExecFactor();
  TestGeneralWalkerZeroUnsignedBound();
  TestGeneralWalkerCompares64();
  TestGeneralWalkerLeavesFloatToInterpreter();
  TestGeneralWalkerSharedValuesCompileOnce();
  TestGeneralWalkerRemembersUnknownValues();
  TestRangeOverlapIndex();
  TestGeneralWalkerLoopFixpoint();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestSrtAliasesRetainReadPolicy();
  TestUniformVectorDescriptorRead();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  for (const bool memory_condition : {false, true})
    TestWrittenDescriptorPredicateReads(memory_condition);
  TestFailedMaterializationRejectsStage();
  TestSharedControlPredicates();
  TestNestedExecPredicatesReuseReads();
  TestFiniteImageRefreshReusesScalarReads();
  for (const bool independent_mask : {false, true})
    TestEmptyTraversalSkipsUnmappedDescriptor(independent_mask);
  TestEmptyTraversalSkipsUnmappedDescriptor(false, true);
  TestCyclicControlPredicatesRemainUnknown();
  TestEmptyScalarBufferSpecialization();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestSignedStorageSpecialization();
  for (const bool mixed : {false, true}) TestIndirectImageSamplerClasses(mixed);
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
