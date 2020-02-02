#include "ZedAdapter.h"

mt::ZedAdapter::ZedAdapter(int width, int height)
{
    m_width = width;
    m_height = height;

    m_texture_shader = new CVK::ShaderSimpleTexture( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_texture_shader_paths);
    generateGlTexture(m_display_texture, width, height);
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_texture_resource, m_display_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

void mt::ZedAdapter::imageToGlTexture(Mat &zed_img)
{
    cudaArray_t texture_array;
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &m_texture_resource, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_array, m_texture_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaMemcpy2DToArray(texture_array, 0, 0, zed_img.getPtr<sl::uchar1>(MEM_GPU), zed_img.getStepBytes(MEM_GPU), zed_img.getWidth() * sizeof(sl::uchar4), zed_img.getHeight(), cudaMemcpyDeviceToDevice));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_texture_resource, 0));
}

void mt::ZedAdapter::renderImage()
{
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_texture_shader->setTextureInput(0, m_display_texture);
    m_texture_shader->useProgram();
    m_texture_shader->update();
    m_texture_shader->renderZED();
}

// *** Deprecated *** //

void mt::initBasicZedCameraHD720(sl::Camera &zed)
{
    // Init ZED Camera
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION_HD720;
    init_parameters.camera_fps = 30.f;
    zed.open(init_parameters);
}

void mt::initSVOZedCamera(sl::Camera &zed, const char* path_to_file)
{
    // Init ZED Camera from SVO files
    sl::InitParameters initParameters;
    initParameters.svo_input_filename.set(path_to_file);
    initParameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
    sl::ERROR_CODE err = zed.open(initParameters);

    // ERRCODE display
    if (err != sl::SUCCESS) {
        zed.close();
    }
}

void mt::initZedCamera(sl::Camera &zed, const char *path_to_file) {

    if (path_to_file) {
        printf("##### Load from file \n");
        mt::initSVOZedCamera(zed, path_to_file);
    }
    else {
        printf("###### Load Camera \n");
        mt::initBasicZedCameraHD720(zed);
    }

}