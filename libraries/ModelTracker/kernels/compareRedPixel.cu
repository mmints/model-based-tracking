#include "ModelTracker/kernel_functions.h"
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>


texture<uchar4, 2, cudaReadModeElementType> particle_grid_texture_ref;

// Compare the pixel values of the particle and the zed
__device__ void compare(float &weight, const uchar4 &particle_pixel, const sl::uchar4 &zed_pixel)
{
    if (particle_pixel.x == zed_pixel.x) //&& particle_pixel.y == zed_pixel.y && particle_pixel.z == zed_pixel.z)
        weight = 1.f;
    else
        weight = 0.f;
}

__global__ void compareRedPixelKernel(sl::uchar4 *zed_in, size_t step,
                                int particle_scale, int particle_grid_dimension,
                                int particle_width, int particle_height,
                                float *global_weight_memory,
                                sl::uchar4 *debug_img_out,
                                sl::uchar4 *debug_clean_in)
{
    // Get the texel value from particleGrid.texture (parts as particle_grid_texture_ref)
    // use unsigned integer because the numbers can become very large
    uint32_t particle_grid_texture_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t particle_grid_texture_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Transfer particle grid pixel coordinate to ZED pixel coordinate
    uint32_t zed_x = (particle_grid_texture_x % particle_width) * particle_scale;
    uint32_t zed_y = (particle_grid_texture_y % particle_height) * particle_scale;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

    float weight = 0.f;

    // Calculate the index of the current corresponding particle to the given texel
    int particle_index = (int)(particle_grid_texture_x / particle_width) + (int)(particle_grid_texture_y / particle_height) * particle_grid_dimension;

    // If the RED channel of the RedMapMat is
    if (zed_in[offset].x != 0)
    {
        uchar4 particle_grid_pixel_value = tex2D(particle_grid_texture_ref, particle_grid_texture_x, particle_grid_texture_y);

        debug_img_out[offset] = sl::uchar4(0, 0, 0, 0);

        compare(weight, particle_grid_pixel_value, zed_in[offset]);

        //Debug: Have a look how the particle are spread on the input img
        if (weight > 0.f) {
            debug_img_out[offset] = sl::uchar4(0, 255, 0, 0);
        }
        else {
/*            debug_img_out[offset].x = debug_clean_in[offset].z;
            debug_img_out[offset].y = debug_clean_in[offset].y;
            debug_img_out[offset].z = debug_clean_in[offset].x;*/
            debug_img_out[offset] = sl::uchar4(0, 0, 0, 0);
        }
    }

    atomicAdd(&global_weight_memory[particle_index], weight);
}

void mt::compareRedPixel(sl::uchar4 *zed_in,  size_t step,
                        int particle_scale,
                        int particle_grid_dimension,
                        int particle_width, int particle_height,
                        cudaArray *particle_grid_tex_array,
                        float *dev_global_weight_memory,
                         sl::uchar4 *debug_img_out,
                         sl::uchar4 *debug_clean_in)
{
    HANDLE_CUDA_ERROR(cudaBindTextureToArray(particle_grid_texture_ref, particle_grid_tex_array));

    // Given Code
    const size_t BLOCKSIZE_X = 32; // Max Threads in X
    const size_t BLOCKSIZE_Y = 8;  // Max Threads in Y

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y}; // Threads
    dim3 dimGrid; // Blocks

    // Launch enough blocks for the requestet amount of threads
    dimGrid.x = (particle_width * particle_grid_dimension + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (particle_height * particle_grid_dimension + dimBlock.y - 1) / dimBlock.y;

    compareRedPixelKernel<<<dimGrid, dimBlock>>>(zed_in,  step, particle_scale,
                                                 particle_grid_dimension,
                                                 particle_width, particle_height,
                                                 dev_global_weight_memory,
                                                 debug_img_out,
                                                 debug_clean_in);

    HANDLE_CUDA_ERROR(cudaUnbindTexture(particle_grid_texture_ref));
}
