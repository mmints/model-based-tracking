#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

#define PARTICLE 32

texture<uchar4, 2, cudaReadModeElementType> particle_grid_texture_ref;

// A super simple likelihood function that compares two pixels and sets a weight
// calculate the difference between every channel and create on this way a weight
__device__ void likelihood(float &weight, const uchar4 &particle_pixel, const sl::uchar4 &zed_pixel) {
    if (particle_pixel.x < zed_pixel.z || particle_pixel.y < zed_pixel.y || particle_pixel.z < zed_pixel.x) {
        weight = 1.f;
    }
    else {
        weight = 0.f;
    }
}

__global__ void kernel(int width, int height, sl::uchar4 *zed_in, sl::uchar4 *zed_out,  size_t step, float *global_weight_memory) {

    // Get the texel value from particleGrid.texture (parts as particle_grid_texture_ref)
    // use unsigned integer because the numbers can become very large
    uint32_t particle_grid_texture_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t particle_grid_texture_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Transfer texel coordinate to ZED pixel coordinates
    uint32_t zed_x = (particle_grid_texture_x % width) * 2;//(zed_width/width);
    uint32_t zed_y = (particle_grid_texture_y % height) * 2;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

    uchar4 particle_grid_texel_value = tex2D(particle_grid_texture_ref, zed_x, zed_y);
    uchar4 particle_grid_texel_value_2 = tex2D(particle_grid_texture_ref, zed_x + width, zed_y);

    // Calculate the index of the current corresponding particle to the given texel
    int particle_index = (int)(particle_grid_texture_x / width) + (int)(particle_grid_texture_y / height) * PARTICLE;

    // Calculate the weight of the current pixel
    float weight = 0.f;
    likelihood(weight, particle_grid_texel_value, zed_in[offset]);
    atomicAdd(&global_weight_memory[particle_index],weight);

    // Write dummy data into global_weight_memory for testing
    if (global_weight_memory[particle_index] == 0.f)
    {
        global_weight_memory[particle_index] = (float) particle_index;
    }

    // VISUALISATION
    bool cube = (particle_grid_texel_value.x == 0 && particle_grid_texel_value.y == 0 && particle_grid_texel_value.z == 0);
    bool cube2 = (particle_grid_texel_value_2.x == 0 && particle_grid_texel_value_2.y == 0 && particle_grid_texel_value_2.z == 0);

    if (cube && cube2) {
        zed_out[offset].x = zed_in[offset].z;
        zed_out[offset].y = zed_in[offset].y;
        zed_out[offset].z = zed_in[offset].x;
        return;
    }

    if (cube) {
        zed_out[offset].x = particle_grid_texel_value_2.x;
        zed_out[offset].y = particle_grid_texel_value_2.y;
        zed_out[offset].z = particle_grid_texel_value_2.z;
        return;
    }

    if (cube2) {
        zed_out[offset].x = particle_grid_texel_value.x;
        zed_out[offset].y = particle_grid_texel_value.y;
        zed_out[offset].z = particle_grid_texel_value.z;
        return;
    }
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