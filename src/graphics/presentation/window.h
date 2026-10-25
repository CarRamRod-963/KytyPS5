#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

#include <filesystem>

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
// --precompile-only: builds the recorded pipelines into the driver cache and saves it. Returns
// the process exit code: 0 done, 1 the driver cache was not written, 2 the window was closed.
[[nodiscard]] int WindowPrecompilePipelines();
// --export-pipeline-list: whether the list was written.
[[nodiscard]] bool       WindowExportPipelineList(const std::filesystem::path& path);
void                     WindowShutdown();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
