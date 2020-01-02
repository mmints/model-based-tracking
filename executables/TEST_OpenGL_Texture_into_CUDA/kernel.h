#include <cuda_runtime.h>
#include <sl/Camera.hpp>

#ifndef CVK_2_KERNEL_H
#define CVK_2_KERNEL_H

void callKernel(int width, int height, cudaArray *tex_array, sl::uchar1 *d_in, sl::uchar1 *d_out, size_t step);

#endif //CVK_2_KERNEL_H
