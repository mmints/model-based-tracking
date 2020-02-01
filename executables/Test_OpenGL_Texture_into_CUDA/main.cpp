#include <CVK_2/CVK_Framework.h>

#include <ErrorHandling/HANDLE_CUDA_ERROR.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>

#include <cuda_gl_interop.h>
#include <sl/Camera.hpp>

#include "kernel.h"

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 1

#define PARTICLE_WIDTH WIDTH
#define PARTICLE_HEIGHT HEIGHT

using namespace sl;

GLFWwindow* window;

void initGL()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH, HEIGHT, "[TEST] OpenGL Texture into CUDA", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glewInit();
    glEnable(GL_DEPTH_TEST);

    CVK::State::getInstance()->setBackgroundColor(BLACK);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);
}

// Renders texture onto a screen filling quad.
void renderTextureToScreen(GLuint textureID, CVK::ShaderSimpleTexture &simpleTextureShader)
{
    glViewport(0, 0, WIDTH, HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simpleTextureShader.setTextureInput(0, textureID);
    simpleTextureShader.useProgram();
    simpleTextureShader.update();
    simpleTextureShader.renderZED();
}

int main()
{
    initGL();

    Camera zed;
    mt::initSVOZedCamera(zed, "~/Documents/ZED/HD720_SN11351_13-04-52.svo");

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture);

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);

    // Creating Color Particle Grid
    mt::ParticleGrid particleGrid(RESOURCES_PATH "/simple_cube/simple_cube.obj", PARTICLE_WIDTH, PARTICLE_HEIGHT, PARTICLE_COUNT);
    particleGrid.renderColorTexture();

    // CUDA interopertion part
    struct cudaGraphicsResource *tex_resource;
    //HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&tex_resource, fbo.getColorTexture(0), GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&tex_resource, particleGrid.getColorTexture(), GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &tex_resource));

    cudaArray *tex_array;
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&tex_array, tex_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &tex_resource));

    // zed interopertion
    Mat zed_in_img  =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat zed_out_img =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    GLuint zed_tex;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &zed_tex);
    glBindTexture(GL_TEXTURE_2D, zed_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsResource* zed_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&zed_resource, zed_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaArray *zed_tex_array;

    while(!glfwWindowShouldClose( window))
    {
        particleGrid.update(0.2f, 0.8f);
        particleGrid.renderColorTexture();

        zed.grab();
        if (zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU) == SUCCESS) {
            callKernel(zed_in_img.getPtr<sl::uchar4>(MEM_GPU), zed_out_img.getPtr<sl::uchar4>(MEM_GPU), zed_in_img.getStep(MEM_GPU), WIDTH, HEIGHT, tex_array);
            HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &zed_resource, 0));
            HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&zed_tex_array, zed_resource, 0, 0));
            HANDLE_CUDA_ERROR(cudaMemcpy2DToArray(
                    zed_tex_array, 0, 0,
                    zed_out_img.getPtr<sl::uchar1>(MEM_GPU),
                    zed_out_img.getStepBytes(MEM_GPU),
                    zed_out_img.getWidth() * sizeof(sl::uchar4),
                    zed_out_img.getHeight(),
                    cudaMemcpyDeviceToDevice
                    ));

            HANDLE_CUDA_ERROR( cudaGraphicsUnmapResources(1, &zed_resource, 0));
        }

        renderTextureToScreen(zed_tex, simpleTextureShader);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}