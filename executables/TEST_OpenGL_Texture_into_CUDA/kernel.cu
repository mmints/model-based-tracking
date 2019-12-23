#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

texture<float4 , 2, cudaReadModeElementType> tex_ref;

// turns all pixels red
__global__ void kernel() {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;


}

void callKernel(int width, int height, cudaArray *tex_array)
{
    cudaChannelFormatDesc desc;
    HANDLE_CUDA_ERROR(cudaGetChannelDesc(&desc, tex_array));

    printf("CUDA Array channel descriptor, bits per component:\n");
    printf("X %d Y %d Z %d W %d, kind %d\n",
           desc.x,desc.y,desc.z,desc.w,desc.f);

    HANDLE_CUDA_ERROR(cudaBindTextureToArray(tex_ref, tex_array));

    dim3 grids(width/16, height/16);
    dim3 threads(16, 16);

}


