#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>


texture<uchar4, 2, cudaReadModeElementType> particle_grid_texture_ref;

__global__ void kernel(int width, int height, sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, float *global_weight_memory) {

    // Get the texel value from particleGrid.texture (parts as particle_grid_texture_ref)
    uint32_t gird_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t grid_y = threadIdx.y + blockIdx.y * blockDim.y;
    uchar4 texel_value = tex2D(particle_grid_texture_ref, gird_x, grid_y);

    // Transfer texel coordinate to ZED pixel coordinates
    uint32_t zed_x = gird_x % width;
    uint32_t zed_y = grid_y % height;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

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

void callKernel(sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, int width, int height, cudaArray *particle_grid_tex_array, float *dev_global_weight_memory)
{
    HANDLE_CUDA_ERROR(cudaBindTextureToArray(particle_grid_texture_ref, particle_grid_tex_array));

    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    kernel<<<dimGrid, dimBlock>>>(width,height, zed_in, zed_out, step, dev_global_weight_memory);
    HANDLE_CUDA_ERROR(cudaUnbindTexture(particle_grid_texture_ref));
}