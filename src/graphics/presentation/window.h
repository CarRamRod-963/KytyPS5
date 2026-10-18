#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
// --precompile-only: builds the recorded pipelines into the driver cache and saves it.
void                     WindowPrecompilePipelines();
void                     WindowShutdown();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
