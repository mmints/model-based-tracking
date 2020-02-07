#include "ParticleFilter.h"

mt::ParticleFilter::ParticleFilter(mt::ParticleGrid &particleGrid)
{
    m_particle_count = particleGrid.getParticleCount();

    // Allocate weight memory on host
    m_color_weight_memory   = new float[m_particle_count];
    m_depth_weight_memory   = new float[m_particle_count];
    m_normals_weight_memory = new float[m_particle_count];

    // Allocate weight memory on device
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_color_weight_memory, m_particle_count * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_depth_weight_memory, m_particle_count * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_normals_weight_memory, m_particle_count * sizeof(float)));

    // Register and map texture to CudaArray
    mapGLTextureToCudaArray(particleGrid.getColorTexture(), m_color_texture_array);
    mapGLTextureToCudaArray(particleGrid.getDepthTexture(), m_depth_texture_array);
    mapGLTextureToCudaArray(particleGrid.getNormalTexture(), m_normals_texture_array);
}

void mt::ParticleFilter::mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array)
{
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_texture_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &m_texture_resource));

    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_array, m_texture_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_texture_resource));
}

void mt::ParticleFilter::calculateWeightColor(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_color_weight_memory, m_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_color_weight_memory, m_color_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_color_weight_memory, dev_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
}

void mt::ParticleFilter::calculateWeightDepth(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_depth_weight_memory, m_depth_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_depth_weight_memory, m_depth_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_depth_weight_memory, dev_depth_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
}

void mt::ParticleFilter::calculateWeightNormals(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_normals_weight_memory, m_normals_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_normals_weight_memory, m_normals_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_normals_weight_memory, dev_normals_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
}
