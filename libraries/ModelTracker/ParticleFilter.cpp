#include "ParticleFilter.h"

mt::ParticleFilter::ParticleFilter(mt::ParticleGrid &particleGrid)
{
    m_particle_count = particleGrid.getParticleCount();
    m_color_weight_memory = new float[m_particle_count];

    // Allocate and register memory
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_color_weight_memory, m_particle_count * sizeof(float)));
    mapGLTextureToCudaArray(particleGrid.getColorTexture(), m_color_texture_array);
}


void mt::ParticleFilter::mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array)
{
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_texture_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &m_texture_resource));

    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_array, m_texture_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_texture_resource));
}

void mt::ParticleFilter::convertBGRtoRGB(sl::Mat in, sl::Mat out)
{
    filter::convertBGRtoRGB(in, out);
}

void mt::ParticleFilter::calculateWeightColor(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_color_weight_memory, m_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_color_weight_memory, m_color_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_color_weight_memory, dev_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));

    // TODO: Temporary, because we want to set the particle weight from all likelihoods
    for (int i = 0; i < m_particle_count; i++) {
        particleGrid.m_particles[i].setWeight(m_color_weight_memory[i]);
        m_color_weight_memory[i] = 0.f;

        if (particleGrid.m_particles[i].getWeight() > 5.f){
            printf("FIT: Particle %i - W: %f \n", i, particleGrid.m_particles[i].getWeight());
        }

    }
}
