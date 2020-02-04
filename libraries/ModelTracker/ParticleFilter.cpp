#include "ParticleFilter.h"

void mt::ParticleFilter::mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array)
{
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_texture_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &m_texture_resource));

    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_array, m_texture_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_texture_resource));
}