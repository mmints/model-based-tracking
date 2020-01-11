#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

#define PARTICLE 4

texture<uchar4, 2, cudaReadModeElementType> particle_grid_texture_ref;

__global__ void kernel(int width, int height, sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, float *global_weight_memory) {

    // Get the texel value from particleGrid.texture (parts as particle_grid_texture_ref)
    // use unsigned integer because the numbers can become very large
    uint32_t particle_grid_texture_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t particle_grid_texture_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Transfer texel coordinate to ZED pixel coordinates
    uint32_t zed_x = particle_grid_texture_x % width;
    uint32_t zed_y = particle_grid_texture_y % height;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

    uchar4 texel_value = tex2D(particle_grid_texture_ref, zed_x, zed_y);
    uchar4 texel_value_2 = tex2D(particle_grid_texture_ref, zed_x + width, zed_y); // switch 1 particle to right

    // Calculate the index of the current corresponding particle to the given texel
    int particle_index = (int)(particle_grid_texture_x / width) + (int)(particle_grid_texture_y / height) * PARTICLE;

    // Write dummy data into global_weight_memory for testing
    if (global_weight_memory[particle_index] == 0.f)
    {
        global_weight_memory[particle_index] = (float) particle_index;
    }

    bool cube_1 = (texel_value.x == 0 && texel_value.y == 0 && texel_value.z == 0);

    if (cube_1) {
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

    // Given Code
    const size_t BLOCKSIZE_X = 32; // Max Threads in X
    const size_t BLOCKSIZE_Y = 8;  // Max Threads in Y

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y}; // Threads
    dim3 dimGrid; // Blocks

    // Launch enough blocks for the requestet amount of threads
    dimGrid.x = (width * PARTICLE + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height * PARTICLE + dimBlock.y - 1) / dimBlock.y;

    kernel<<<dimGrid, dimBlock>>>(width,height, zed_in, zed_out, step, dev_global_weight_memory);
    HANDLE_CUDA_ERROR(cudaUnbindTexture(particle_grid_texture_ref));
}