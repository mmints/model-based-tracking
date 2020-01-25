#ifndef MT_KERNEL_FUNCTIONS_H
#define MT_KERNEL_FUNCTIONS_H

/**
 * This is the collection of all kernel functions
 * that are in use in the ModelTracker.
 */

#include <cuda_runtime.h>
#include <sl/Camera.hpp>

namespace mt
{
    // testing kernel that was used in the deprecated version of the tracker
    void testCallKernel(sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, int width, int height, cudaArray *particle_grid_tex_array, float *dev_global_weight_memory);

    /**
     * Compare pixel pairwise from the particle_grid_texture and corresponding zed_in.
     * Return the weights of the particles in dev_global_weight_memory
     */
    void compareRedPixel(sl::uchar4 *zed_in,  size_t step,
                         int particle_scale, int particle_grid_dimension,
                         int particle_width, int particle_height,
                         cudaArray *particle_grid_tex_array,
                         float *dev_global_weight_memory,
                         sl::uchar4 *debug_img_out,
                         sl::uchar4 *debug_clean_in); // DEBUG IMG
}

#endif //MT_KERNEL_FUNCTIONS_H
