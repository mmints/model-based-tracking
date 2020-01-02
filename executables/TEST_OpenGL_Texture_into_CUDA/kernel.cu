#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>


texture<float4 , 2, cudaReadModeElementType> tex_ref;

__global__ void kernel(sl::uchar1 *d_in, sl::uchar1 *d_out, size_t step) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // if texture-pixel-color != 0,0,0
        // write d_in-pixel into d_out-pixel
    // else:
        // write texture-pixel into d_out-pixel

}

void callKernel(int width, int height, cudaArray *tex_array, sl::uchar1 *d_in, sl::uchar1 *d_out, size_t step)
{
    cudaChannelFormatDesc desc;
    HANDLE_CUDA_ERROR(cudaGetChannelDesc(&desc, tex_array));

    printf("CUDA Array channel descriptor, bits per component:\n");
    printf("X %d Y %d Z %d W %d, kind %d\n",
           desc.x,desc.y,desc.z,desc.w,desc.f);

    HANDLE_CUDA_ERROR(cudaBindTextureToArray(tex_ref, tex_array));

    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    kernel<<<dimGrid, dimBlock>>>(d_in, d_out, step);
}


