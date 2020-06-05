#include "ModelTracker/kernel_functions.h"
#include <cstdlib>

texture<uchar4, 2, cudaReadModeElementType> particle_grid_texture_ref;

__device__ float colorWeight(const uchar4 &particle_pixel, const sl::uchar4 &zed_pixel)
{
    // calculate difference between pixel values
    int diff_x = std::abs(particle_pixel.x - zed_pixel.x);
    int diff_y = std::abs(particle_pixel.y - zed_pixel.y);
    int diff_z = std::abs(particle_pixel.z - zed_pixel.z);

    // Normalize total difference
    float diff_total = diff_x + diff_y + diff_z;
    diff_total /= (255.f*3.f);

    float weight = 1.f -diff_total;
    weight*=weight;

    return weight;
}

__device__ float depthWeight(const uchar4 &particle_pixel, const sl::uchar4 &zed_pixel)
{
    // Use same function as for color. Depth data is provided in gray scale BUT in a uchar4 image.
    // All channels have the same value, so only one is needed for calculation weight.

    // calculate difference between pixel values
    int diff = std::abs(particle_pixel.x - zed_pixel.x);

    // Normalize difference
    diff /= 255.f;

    float weight = 1.f -diff;
    weight*=weight;

    return weight;
}

__device__ float normalsWeight(const uchar4 &particle_pixel, const sl::uchar4 &zed_pixel)
{
    // Normalize ZED Normals
    float n_x = zed_pixel.x / 255.f;
    float n_y = zed_pixel.y / 255.f;
    float n_z = zed_pixel.z / 255.f;

    float n_accum = 0.f;
    n_accum += n_x * n_x;
    n_accum += n_y * n_y;
    n_accum += n_z * n_z;

    float n_norm = sqrt(n_accum);

    // Normalize Particle Normals
    float p_x = particle_pixel.x / 255.f;
    float p_y = particle_pixel.y / 255.f;
    float p_z = particle_pixel.z / 255.f;

    float p_accum = 0.f;
    p_accum += p_x * p_x;
    p_accum += p_y * p_y;
    p_accum += p_z * p_z;

    float p_norm = sqrt(p_accum);

    float cos_theta = (n_x * p_x + n_y * p_y + n_z * p_z) / (n_norm * p_norm); // cos theta = dot(n, p)/(n_norm * p_norm);
    cos_theta *= cos_theta;

    // cos_theta = weight -> if cos_theta == 0 -> angle btw. n and p is 90deg | If cos_theta == 1 -> n = p
    return cos_theta;
}

__global__ void calculateWeightKernel(sl::uchar4 *zed_in, size_t step, int particle_scale,
                                      int particle_grid_dimension, int particle_width, int particle_height,
                                      float *weight_memory, LIKELIHOOD type)
{
    // Get the pixel value from particleGrid.texture (parts as particle_grid_texture_ref)
    // use unsigned integer because the numbers can become very large
    uint32_t particle_grid_texture_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t particle_grid_texture_y = threadIdx.y + blockIdx.y * blockDim.y;
    uchar4 particle_grid_pixel_value = tex2D(particle_grid_texture_ref, particle_grid_texture_x, particle_grid_texture_y);

    if (particle_grid_pixel_value.x <= 0 && particle_grid_pixel_value.y <= 0 && particle_grid_pixel_value.z <= 0)
        return;

        // Transfer particle grid pixel coordinate to ZED pixel coordinate
    uint32_t zed_x = (particle_grid_texture_x % particle_width) * particle_scale;
    uint32_t zed_y = (particle_grid_texture_y % particle_height) * particle_scale;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

    // Calculate the index of the current corresponding particle to the given texel
    int particle_index = (int)(particle_grid_texture_x / particle_width) + (int)(particle_grid_texture_y / particle_height) * particle_grid_dimension;

    // TODO: Calculate weight for all measurement types
    float weight = 0.f;

    switch(type) {
        case COLOR: weight = colorWeight(particle_grid_pixel_value, zed_in[offset]); break;
        case DEPTH: weight = depthWeight(particle_grid_pixel_value, zed_in[offset]); break;
        case NORMAL: weight = normalsWeight(particle_grid_pixel_value, zed_in[offset]); break;
    }
    atomicAdd(&weight_memory[particle_index], weight);
}

void mt::calculateWeight(const sl::Mat &in_zed, float *dev_weight_memory, cudaArray *particle_grid_tex_array, mt::ParticleGrid &particleGrid, LIKELIHOOD type)
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
            dev_weight_memory, type);

    HANDLE_CUDA_ERROR(cudaUnbindTexture(particle_grid_texture_ref));
}
