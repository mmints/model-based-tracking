#ifndef MT_PARTICLEFILTER_H
#define MT_PARTICLEFILTER_H

#include <iostream>
#include <cuda_gl_interop.h>
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

#include <sl/Camera.hpp>

#include <ImageFilter/ImageFilter.h>

namespace mt
{

class ParticleFilter
{

private:
    cudaGraphicsResource* m_texture_resource;

public:
    void mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array);
    void convertBGRtoRGB(sl::Mat in, sl::Mat out);
};

}

#endif //MT_PARTICLEFILTER_H
