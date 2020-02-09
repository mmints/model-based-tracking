#include <cuda_runtime.h>
#include <sl/Camera.hpp>

#ifndef CVK_2_KERNEL_H
#define CVK_2_KERNEL_H

void callKernel(sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, int width, int height, cudaArray *tex_array);
void callKernel2(int width, int height, cudaArray *tex_array);

#endif //CVK_2_KERNEL_H
