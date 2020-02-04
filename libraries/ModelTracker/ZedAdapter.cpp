#include "ZedAdapter.h"

mt::ZedAdapter::ZedAdapter(Camera &zed, RESOLUTION resolution)
{
    setResolution(resolution);
    printf("Initialize ZED with resolution: %i x %i \n", m_width, m_height);

    initHardwareCamera(zed, resolution);
    initTextureInteroperation();
}

mt::ZedAdapter::ZedAdapter(Camera &zed, RESOLUTION resolution, const char* path_to_svo_file)
{
    setResolution(resolution);
    printf("Initialize ZED with resolution: %i x %i \n", m_width, m_height);

    initSVOZedCamera(zed, path_to_svo_file);
    initTextureInteroperation();
}


void mt::ZedAdapter::imageToGlTexture(Mat &zed_img)
{
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &m_texture_resource, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&m_texture_array, m_texture_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaMemcpy2DToArray(m_texture_array, 0, 0, zed_img.getPtr<sl::uchar1>(MEM_GPU), zed_img.getStepBytes(MEM_GPU), zed_img.getWidth() * sizeof(sl::uchar4), zed_img.getHeight(), cudaMemcpyDeviceToDevice));
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

// *** Private Functions *** //

void mt::ZedAdapter::initTextureInteroperation()
{
    m_texture_shader = new CVK::ShaderSimpleTexture( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_texture_shader_paths);
    generateGlTexture(m_display_texture, m_width, m_height);
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_texture_resource, m_display_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

void mt::ZedAdapter::setResolution(RESOLUTION res)
{
    switch(res)
    {
        case RESOLUTION_HD720:  m_width = 1280; m_height = 720; break;
        case RESOLUTION_HD1080: m_width = 1920; m_height = 1080; break;
        case RESOLUTION_HD2K:   m_width = 2208; m_height = 1242; break;
        case RESOLUTION_VGA:    m_width = 672; m_height = 376; break;
    }
}

void mt::ZedAdapter::initHardwareCamera(sl::Camera &zed, RESOLUTION res)
{
    // Init ZED Camera
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = res;
    init_parameters.camera_fps = 60.f;
    init_parameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
    zed.open(init_parameters);
}

void mt::ZedAdapter::initSVOZedCamera(sl::Camera &zed, const char* path_to_file)
{
    // Init ZED Camera from SVO files
    sl::InitParameters init_parameters;
    init_parameters.svo_input_filename.set(path_to_file);
    init_parameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
    sl::ERROR_CODE err = zed.open(init_parameters);

    // ERRCODE display
    if (err != sl::SUCCESS) {
        zed.close();
    }
}