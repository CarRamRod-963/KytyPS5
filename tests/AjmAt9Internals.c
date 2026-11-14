// LibAtrac9's handle layout for ajm_resample_tests (its headers are C only).
#include "decoder.h"

void AjmTestAt9BandUnits(const void* handle, int* quantization_units, int* stereo_units) {
	const Block* block = &((const Atrac9Handle*)handle)->Frame.Blocks[0];
	*quantization_units = block->QuantizationUnitCount;
	*stereo_units       = block->StereoQuantizationUnit;
}
