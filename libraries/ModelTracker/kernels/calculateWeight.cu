#include "ModelTracker/kernel_functions.h"

texture<uchar4, 2, cudaReadModeElementType> particle_grid_texture_ref;

__global__ void calculateWeightKernel(sl::uchar4 *zed_in, size_t step, int particle_scale,
                                      int particle_grid_dimension, int particle_width, int particle_height,
                                      float *weight_memory)
{
    // Get the pixel value from particleGrid.texture (parts as particle_grid_texture_ref)
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

    // Just for testing
    weight = (float) particle_index;
    weight_memory[particle_index] = weight;

    //atomicAdd(&weight_memory[particle_index], weight);
}

void mt::calculateWeight(const sl::Mat &in_zed, float *dev_weight_memory, cudaArray *particle_grid_tex_array, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaBindTextureToArray(particle_grid_texture_ref, particle_grid_tex_array));

    size_t width = in_zed.getWidth();
    size_t height = in_zed.getHeight();
    size_t step = in_zed.getStep(sl::MEM_GPU);

    int particle_width = particleGrid.getParticleWidth();
    int particle_height = particleGrid.getParticleHeight();
    int particle_gird_dimension = particleGrid.getParticleGridDimension();

    int particle_scale = width / particle_width;
    sl::uchar4 *in_zed_ptr = in_zed.getPtr<sl::uchar4>(sl::MEM_GPU);

    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (particle_width * particle_gird_dimension + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (particle_height * particle_gird_dimension + dimBlock.y - 1) / dimBlock.y;

    calculateWeightKernel<<<dimGrid, dimBlock>>>(in_zed_ptr, step, particle_scale,
            particle_gird_dimension, particle_width, particle_height,
            dev_weight_memory);

    HANDLE_CUDA_ERROR(cudaUnbindTexture(particle_grid_texture_ref));
}
