#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		// V# stride layout; the stride field is 1 for any nonzero stride (the stride itself is
		// read at run time, see FillBufferStrides).
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		// A direct V# the renderer binds as the null buffer (no base or no records), where every
		// access is out of bounds. It never changes the module; SpecializationServes uses it.
		bool                   unbound                         = false;
		// indirect_mapping_offset is a plan-fixed flattened-SRT word holding the key map's
		// offset, so the module does not depend on where the scene's key maps end up.
		bool                   indirect_mapping_header         = false;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          heap                       = false;
		// Resolved at run time through the bindless translation table (no candidates).
		bool                          bindless                   = false;
		// The bindless key is the byte offset of the T# in its heap's records (an image heap the
		// shader indexes at a scalar offset); the translation index is the key divided by four.
		bool                          bindless_byte_keys             = false;
		// See Buffer::indirect_mapping_header.
		bool                          indirect_mapping_header        = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		// See Buffer::indirect_mapping_header.
		bool indirect_mapping_header          = false;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG. Extraction is deterministic: two
// plans extracted from one program are interchangeable.
enum class SrtWalkerCompilation {
	All,        // the simple and the general compiled walker
	SimpleOnly, // the general walker is left to CompileSrtWalkers (it may take long)
};
ResourcePlan ExtractResourcePlan(const Program&       program,
                                 SrtWalkerCompilation walkers = SrtWalkerCompilation::All);

// The user-data slots (SGPR minus the plan's user_data_base) the plan's values read, one bit each
// for slots below 64; a slot at or above 64 sets every bit.
[[nodiscard]] uint64_t PlanUserDataReads(const ResourcePlan& plan);

// Refreshes cached resources and specialization in place. A failed refresh must not be used;
// snapshot.unavailable tells a failure caused by the draw's guest data from an unsupported shader.
// Answers whether a range overlaps any of a list of (address, bytes) ranges, in logarithmic
// time per query after one sort: materialization checks every GPU-selected store candidate (up
// to thousands) against every read a table walk recorded (up to tens of thousands). The index
// is built into `index` on the first query; later queries must pass the same list.
class RangeOverlapIndex {
public:
	[[nodiscard]] bool Overlaps(std::span<const std::pair<uint64_t, uint64_t>> ranges,
	                            std::vector<std::pair<uint64_t, uint64_t>>& index, uint64_t base,
	                            uint64_t size);

private:
	bool m_built = false;
};

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// How many specialization failures MaterializeResources has printed: one line per shader per
// session, none for a speculative walk (SrtRuntime::speculative).
uint64_t SpecializationFailureReports();
// The failure line this thread's last reporting MaterializeResources call printed, empty when
// none printed one since the last take.
std::string TakeSpecializationFailureReport();

// Whether a module compiled for `compiled` behaves exactly as one compiled for `current`
// would. Equal layouts do; so does any layout of an unbound direct buffer when both layouts read
// zero out of bounds, since an unbound buffer is bound so that every access is out of bounds
// (loads read the out-of-bounds value, stores and atomics are dropped).
bool SpecializationServes(const ResourcePlan& program, const ResourceSpecialization& compiled,
                          const ResourceSpecialization& current);

// Gives the unbound buffers of `target` the bound layouts of `donor` that serve them, so a
// module compiled for `target` also serves those bound cases.
void AdoptServingBuffers(const ResourcePlan& program, ResourceSpecialization& target,
                         const ResourceSpecialization& donor);

// Where an indirect root's key map starts in the snapshot's flattened SRT: its mapping offset, or
// the word its header holds (SIZE_MAX when the header is out of range).
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Buffer& root);
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Image& root);
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Sampler& root);

// Whether a module compiled for `compiled` serves the draw `snapshot` and `current` describe once
// its GPU-selected image candidates are moved into the module's candidate slots: every candidate
// a key selects needs a slot of its own layout (a null one any free slot), a key search needs at
// most the module's depth, and a root the draw does not map is served by a module that maps it
// (the draw's key map header names the empty map). Rewrites the snapshot's images and key map
// ordinals and `current` to the module's layout when it does; leaves both unchanged otherwise.
bool AdaptToSpecialization(const ResourcePlan& program, const ResourceSpecialization& compiled,
                           ResourceSnapshot& snapshot, ResourceSpecialization& current);

// Widens `current` to a layout that also serves the draws `prior` serves - per image root, the
// prior candidate slots plus the ones the draw has beyond them, the deeper key searches - and
// moves the draw into it. A shader whose tables change with the scene thereby compiles modules
// whose layouts only grow. Leaves both unchanged when the layouts do not combine.
void WidenSpecialization(const ResourcePlan& program, const ResourceSpecialization& prior,
                         ResourceSnapshot& snapshot, ResourceSpecialization& current);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);
// Whether ApplyResourceSpecialization accepts `specialization` for a translated program: its
// resource counts and indirect roots fit the program's resources. A specialization recorded for
// another recompiler version may not.
bool SpecializationApplies(const Program& program, const ResourceSpecialization& specialization);

// Writes the V# stride of every buffer binding with a runtime stride into the shader data.
void FillBufferStrides(const ShaderInfo& info, const Bindings& bindings,
                       const ResourceSnapshot& snapshot, std::span<uint32_t> shader_data);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
