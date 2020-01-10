#include <cuda_runtime.h>
#include <sl/Camera.hpp>

#ifndef MT_KERNEL_H
#define MT_KERNEL_H

void callKernel(sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, int width, int height, cudaArray *particle_grid_tex_array, float *dev_global_weight_memory);

#endif //MT_KERNEL_H
