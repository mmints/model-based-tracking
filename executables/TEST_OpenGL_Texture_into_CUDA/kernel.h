#include <cuda_runtime.h>

#ifndef CVK_2_KERNEL_H
#define CVK_2_KERNEL_H

void callKernel(int width, int height, cudaArray *gl_texture);

#endif //CVK_2_KERNEL_H
