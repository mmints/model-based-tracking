#ifndef MODEL_BASED_TRACKER_ZED_HELPER_H
#define MODEL_BASED_TRACKER_ZED_HELPER_H

#include <iostream>
#include <CVK_2/CVK_Framework.h>

#include <sl/Camera.hpp>
#include <cuda_gl_interop.h>

#include "helper_functions.h"
#include "ErrorHandling/HANDLE_CUDA_ERROR.h"
#include "ErrorHandling/HANDLE_ZED_ERROR.h"

using namespace sl;

namespace mt // model based tracking
{

    class ZedAdapter
    {

    private:
        // Parameters
        int m_width;
        int m_height;

        // GL Texture and Shader for Rendering and Cuda resource for CUDA-OpenGL interoperability
        const char *m_texture_shader_paths[2] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag"};
        CVK::ShaderSimpleTexture *m_texture_shader;
        cudaGraphicsResource* m_texture_resource;
        cudaArray_t m_texture_array;
        GLuint m_display_texture;

        // Private Functions
        /**
         * Set the resolution of the ZED Camera and of the GL texture
         * @param res Resolution type from sl::RESOLUTION
         */
        void setResolution(RESOLUTION res);

        /**
         * Initialize shader, generate GL texture und register CUDA resource
         */
        void initTextureInteroperation();

        /**
         * Initiate Hardware ZED Camera with a basic parameter preset.
         * @param zed Reference to sl::Camera
         * @param res Resolution type from sl::RESOLUTION
         */
        void initHardwareCamera(sl::Camera &zed, RESOLUTION res);

        /**
         * Initiate ZED Camera from SVO file with a basic parameter preset.
         * @param zed Reference to sl::Camera
         * @param res Resolution type from sl::RESOLUTION
         */
        void initSVOZedCamera(sl::Camera &zed, const char* path_to_file);

    public:
        /**
         * Constructor for Hardware Camera Adapter
         * @param zed Reference to sl::Camera
         * @param res Resolution type from sl::RESOLUTION
         */
        ZedAdapter(Camera &zed, RESOLUTION resolution);

        /**
         * Constructor for SVO file Camera Adapter
         * @param zed Reference to sl::Camera
         * @param res Resolution type from sl::RESOLUTION
         * @param path_to_svo_file path to file location
         */
        ZedAdapter(Camera &zed, RESOLUTION resolution, const char* path_to_svo_file);

        /**
         * Transfer a sl::Mat saved on GPU into a GL texture
         * @param zed_img sl::Mat image
         */
        void imageToGlTexture(Mat &zed_img);

        /**
         * Render the last generated sl::Mat
         */
        void renderImage();
    };
}

#endif //MODEL_BASED_TRACKER_ZED_HELPER_H
