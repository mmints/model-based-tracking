#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>


texture<uchar4, 2, cudaReadModeElementType> tex_ref;

__global__ void kernel(sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = x + y * step;

    uchar4 texel_value = tex2D(tex_ref, x, y);

    if (texel_value.x == 0 && texel_value.y == 0 && texel_value.z == 0) {
        zed_out[offset].x = zed_in[offset].z;
        zed_out[offset].y = zed_in[offset].y;
        zed_out[offset].z = zed_in[offset].x;
        return;
    }
    zed_out[offset].x = texel_value.x;
    zed_out[offset].y = texel_value.y;
    zed_out[offset].z = texel_value.z;
}

void callKernel(sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, int width, int height, cudaArray *tex_array)
{
/*
    cudaChannelFormatDesc desc;
    HANDLE_CUDA_ERROR(cudaGetChannelDesc(&desc, tex_array));

    printf("++++ Size of texture: %i \n", sizeof(uchar4));

    printf("CUDA Array channel descriptor, bits per component:\n");
    printf("X %d Y %d Z %d W %d, kind %d\n",
           desc.x,desc.y,desc.z,desc.w,desc.f);
*/

    HANDLE_CUDA_ERROR(cudaBindTextureToArray(tex_ref, tex_array));

    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    kernel<<<dimGrid, dimBlock>>>(zed_in, zed_out, step);
    HANDLE_CUDA_ERROR(cudaUnbindTexture(tex_ref));

}

__global__ void kernel2() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    uchar4 texel_value = tex2D(tex_ref, x, y);

    if (texel_value.x != 0 || texel_value.y != 0 || texel_value.z != 0) {
        printf("(%i, %i) = r: %i, g: %i, b %i \n",x, y, texel_value.x, texel_value.y, texel_value.z);
    }
}

void callKernel2(int width, int height, cudaArray *tex_array)
{
    cudaChannelFormatDesc desc;
    HANDLE_CUDA_ERROR(cudaGetChannelDesc(&desc, tex_array));

    printf("++++ Size of texture: %i \n", sizeof(uchar4));

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

    kernel2<<<dimGrid, dimBlock>>>();
    HANDLE_CUDA_ERROR(cudaUnbindTexture(tex_ref));

}

