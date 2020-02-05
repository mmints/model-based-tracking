#ifndef MT_KERNEL_FUNCTIONS_H
#define MT_KERNEL_FUNCTIONS_H

/**
 * This is the collection of all kernel functions
 * that are in use in the ModelTracker.
 */
#include <device_launch_parameters.h>

#include "ParticleGrid.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

#include <cuda_runtime.h>
#include <sl/Camera.hpp>

namespace mt
{
    void calculateWeight(const sl::Mat &in_zed, float *dev_weight_memory, cudaArray *particle_grid_tex_array, mt::ParticleGrid &particleGrid);
}

#endif //MT_KERNEL_FUNCTIONS_H
