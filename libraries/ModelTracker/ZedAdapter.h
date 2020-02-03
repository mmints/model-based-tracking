#ifndef MODEL_BASED_TRACKER_ZED_HELPER_H
#define MODEL_BASED_TRACKER_ZED_HELPER_H

#include <iostream>
#include <CVK_2/CVK_Framework.h>

#include <sl/Camera.hpp>
#include <cuda_gl_interop.h>

#include "helper_functions.h"
#include "ErrorHandling/HANDLE_CUDA_ERROR.h"

using namespace sl;

namespace mt // model based tracking
{

class ZedAdapter
{

private:
    // Parameters
    int m_width;
    int m_height;

    // Cuda resources for CUDA-OpenGL interoperability
    cudaGraphicsResource* m_texture_resource;

    // GL Texture and Shader for Rendering
    const char *m_texture_shader_paths[2] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag"};
    CVK::ShaderSimpleTexture *m_texture_shader;
    cudaArray_t m_texture_array;
    GLuint m_display_texture;

public:
    ZedAdapter(int width, int height);

    void initZedCamera(sl::Camera &zed, const char* path_to_file);
    void initBasicZedCameraHD720(sl::Camera &zed);
    void initSVOZedCamera(sl::Camera &zed, const char* path_to_file);

    void imageToGlTexture(Mat &zed_img);
    void renderImage();

    void displayImage(Mat &zed_img);
};
}

#endif //MODEL_BASED_TRACKER_ZED_HELPER_H