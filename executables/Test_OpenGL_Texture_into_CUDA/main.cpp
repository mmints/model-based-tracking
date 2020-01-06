#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>
#include <cuda_gl_interop.h>
#include <sl/Camera.hpp>
//#include <cuda.h>
//#include <cuda_runtime.h>

#include "kernel.h"

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 400

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

// Set up all necessary shader and use the Particle Generator to create a geometry, that will be
// rendered into tha Color Buffer of ang given FBO
void renderParticlesIntoFBO(CVK::FBO &fbo, ShaderSimple &shaderSimple, mt::ParticleGenerator &particleGenerator, std::vector<mt::Particle> &particles)
{
    CVK::State::getInstance()->setShader( &shaderSimple);

    // set up the location of matrices in shader program
    GLuint viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    GLuint projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WIDTH/HEIGHT, 1.0f, 100.0f);

    fbo.bind();
    shaderSimple.useProgram();
    // pipe uniform variables to shader
    glUniformMatrix4fv(viewMatrixHandle, 1, GL_FALSE, value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionMatrixHandle, 1, GL_FALSE, value_ptr(projectionMatrix));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    particleGenerator.renderParticleTextureGrid(particles);
    glFinish(); // Wait until everything is done.
    fbo.unbind();
}

// Renders texture onto a screen filling quad.
void renderTextureToScreen(GLuint textureID, CVK::ShaderSimpleTexture &simpleTextureShader)
{
    glViewport(0, 0, WIDTH, HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simpleTextureShader.setTextureInput(0, textureID);
    simpleTextureShader.useProgram();
    simpleTextureShader.update();
    simpleTextureShader.render();
}

int main()
{
    initGL();

    Camera zed;
    mt::initSVOZedCamera(zed, "~/Documents/ZED/HD720_SN11351_13-04-52.svo"); // This breaks the Kernal call for some reason.. err: peer access has not been enabled in .....

    //mt::initBasicZedCameraHD720(zed);
//    mt::initZedCamera(zed, "~/Documents/ZED/HD720_SN11351_13-04-52.svo"); // This breaks the Kernal call for some reason.. err: peer access has not been enabled in .....

    // Because of the errors I try the default way for initialization, to figure out, where the error is
   // initParameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
    // initParameters.sdk_cuda_ctx = 0;
    //cuCtxGetCurrent(&initParameters.sdk_cuda_ctx);

    //sl::ERROR_CODE err = zed.open(initParameters);// This breaks the Kernel call for some reason.. err: peer access has not been enabled in .....
    //zed.close();
    //HANDLE_CUDA_ERROR(cudaDeviceEnablePeerAccess(0,0));

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture);

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);

    mt::ParticleGenerator particleGenerator(RESOURCES_PATH "/rubiks_cube/rubiks_cube.obj", PARTICLE_COUNT, WIDTH, HEIGHT);
    std::vector<mt::Particle> particles;
    particleGenerator.initializeParticles(particles, 1.8f);

    CVK::FBO fbo( WIDTH, HEIGHT, 1, true);
    renderParticlesIntoFBO(fbo, shaderSimple, particleGenerator, particles);



    // CUDA interopertion part
    struct cudaGraphicsResource *tex_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&tex_resource, fbo.getColorTexture(0), GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &tex_resource));

    cudaArray *tex_array;
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&tex_array, tex_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &tex_resource));

    Mat zed_in_img;
    Mat zed_out_img =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    GLuint zed_tex;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &zed_tex);
    glBindTexture(GL_TEXTURE_2D, zed_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsResource* zed_resource; // Cuda resources for CUDA-OpenGL interoperability
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&zed_resource, zed_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaArray_t zed_tex_array;

    while(!glfwWindowShouldClose( window))
    {
        zed.grab();
        if (zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU) == SUCCESS) {
            // callKernel(zed_in_img.getPtr<sl::uchar4>(MEM_GPU), zed_out_img.getPtr<sl::uchar4>(MEM_GPU), zed_in_img.getStepBytes(MEM_GPU), WIDTH, HEIGHT, tex_array);
            HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &zed_resource, 0));
            HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&zed_tex_array, zed_resource, 0, 0));
            HANDLE_CUDA_ERROR(cudaMemcpy2DToArray(
                    zed_tex_array, 0, 0,
                    zed_in_img.getPtr<sl::uchar4>(MEM_GPU),
                    zed_in_img.getStepBytes(MEM_GPU),
                    zed_in_img.getWidth() * sizeof(sl::uchar4),
                    zed_in_img.getHeight(),
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