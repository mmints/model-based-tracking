#ifndef MT_PARTICLEFILTER_H
#define MT_PARTICLEFILTER_H

#include <CVK_2/CVK_Framework.h>
#include "ParticleGrid.h"

#include <iostream>
#include <cuda_gl_interop.h>
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

#include <sl/Camera.hpp>

#include <ImageFilter/ImageFilter.h>
#include "kernel_functions.h"

namespace mt
{

class ParticleFilter
{

private:
    // Parameter
    int m_particle_count;

    // Weight array memory space
    float *m_color_weight_memory;

    // Cuda Memory and Resources
    cudaGraphicsResource* m_texture_resource;
    cudaArray_t m_color_texture_array;


    float *dev_color_weight_memory; // Pointer to Cuda Memory Space


    // Private Functions

public:
    ParticleFilter(mt::ParticleGrid &particleGrid);
    void mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array);

    // Filter
    void convertBGRtoRGB(sl::Mat in, sl::Mat out);

    // weight Calculation
    void calculateWeightColor(sl::Mat in, mt::ParticleGrid &particleGrid);
};

}

#endif //MT_PARTICLEFILTER_H
