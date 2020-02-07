#ifndef MT_PARTICLEFILTER_H
#define MT_PARTICLEFILTER_H

#include "kernel_functions.h"
#include <CVK_2/CVK_Framework.h>
#include "ParticleGrid.h"

#include <iostream>
#include <cuda_gl_interop.h>
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

#include <sl/Camera.hpp>

namespace mt
{

class ParticleFilter
{

private:
    // Parameter
    int m_particle_count;

    // Weight array memory space
    float *m_color_weight_memory;
    float *m_depth_weight_memory;
    float *m_normals_weight_memory;

    // Cuda Memory and Resources
    cudaGraphicsResource* m_texture_resource;
    cudaArray_t m_color_texture_array;
    cudaArray_t m_depth_texture_array;
    cudaArray_t m_normals_texture_array;

    float *dev_color_weight_memory;
    float *dev_depth_weight_memory;
    float *dev_normals_weight_memory;


    // Private Functions

public:
    ParticleFilter(mt::ParticleGrid &particleGrid);
    void mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array);

    // weight Calculation
    void calculateWeightColor(sl::Mat in, mt::ParticleGrid &particleGrid);
    void calculateWeightDepth(sl::Mat in, mt::ParticleGrid &particleGrid);
    void calculateWeightNormals(sl::Mat in, mt::ParticleGrid &particleGrid);
};

}

#endif //MT_PARTICLEFILTER_H
