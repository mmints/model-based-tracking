#include <CVK_2/CVK_Framework.h>

#include <ErrorHandling/HANDLE_CUDA_ERROR.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>

#include <cuda_gl_interop.h>
#include <sl/Camera.hpp>

#include "kernel.h"

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 16

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

void generateGlTexture(GLuint &tex)
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void renderParticleGrid(CVK::FBO &fbo, ShaderSimple &shaderSimple, mt::ParticleGenerator &particleGenerator,mt::ParticleGrid &particleGrid)
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
    particleGenerator.renderParticleTextureGrid(particleGrid.particles);
    glFinish(); // Wait until everything is done.
    fbo.unbind();

    particleGrid.texture = fbo.getColorTexture(0);
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

    mt::ParticleGenerator particleGenerator(RESOURCES_PATH "/rubiks_cube/rubiks_cube.obj", PARTICLE_COUNT, WIDTH, HEIGHT);
    CVK::FBO fbo( WIDTH, HEIGHT, 1, true);

    mt::ParticleGrid particleGrid;
    particleGenerator.initializeParticles(particleGrid.particles, 1.8f);
    renderParticleGrid(fbo, shaderSimple, particleGenerator, particleGrid);

    // CUDA interopertion part
    struct cudaGraphicsResource *particle_grid_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&particle_grid_resource, particleGrid.texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &particle_grid_resource));

    cudaArray *particle_grid_tex_array;
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&particle_grid_tex_array, particle_grid_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &particle_grid_resource));

    // zed interopertion
    Mat zed_in_img  =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat zed_out_img =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    GLuint zed_tex;
    generateGlTexture(zed_tex);
    cudaGraphicsResource* zed_resource;

    // Register ZED input image texture resource
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&zed_resource, zed_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaArray *zed_tex_array;

    // Set up global memory array for transferring weight for particles from GPU to CPU
    float global_weight_memory[PARTICLE_COUNT] = {0}; // with space for all particle weights
    for (int i = 0; i < PARTICLE_COUNT; i++) global_weight_memory[i] = 0.f; // fill array with 0.f

    // Allocate corresponding memory space on GPU
    float *dev_global_weight_memory;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_global_weight_memory, PARTICLE_COUNT * sizeof(float)));



    while(!glfwWindowShouldClose( window))
    {
        // Update particle translation and rotation
        particleGenerator.updateParticles(particleGrid.particles);
        renderParticleGrid(fbo, shaderSimple, particleGenerator, particleGrid);

        // Transfer ZED input image and particleGrid.texture to Kernel
        zed.grab();
        if (zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU) == SUCCESS) {

            // Connect with global_weight_memory with device
            HANDLE_CUDA_ERROR(cudaMemcpy(dev_global_weight_memory, global_weight_memory, PARTICLE_COUNT * sizeof(float), cudaMemcpyHostToDevice));

            callKernel(zed_in_img.getPtr<sl::uchar4>(MEM_GPU),
                    zed_out_img.getPtr<sl::uchar4>(MEM_GPU),
                    zed_in_img.getStep(MEM_GPU),
                    WIDTH, HEIGHT,
                    particle_grid_tex_array,
                    dev_global_weight_memory);

            // Copy the resulting weights back to CPU after calling the kernel
            HANDLE_CUDA_ERROR(cudaMemcpy(global_weight_memory, dev_global_weight_memory, PARTICLE_COUNT * sizeof(float), cudaMemcpyDeviceToHost));

            HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &zed_resource, 0));
            HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&zed_tex_array, zed_resource, 0, 0));
            HANDLE_CUDA_ERROR(cudaMemcpy2DToArray(
                    zed_tex_array, 0, 0,
                    zed_out_img.getPtr<sl::uchar1>(MEM_GPU),
                    zed_out_img.getStepBytes(MEM_GPU),
                    zed_out_img.getWidth() * sizeof(sl::uchar4),
                    zed_out_img.getHeight(),
                    cudaMemcpyDeviceToDevice));

            HANDLE_CUDA_ERROR( cudaGraphicsUnmapResources(1, &zed_resource, 0));
        }

        for (int i = 0; i < PARTICLE_COUNT; i++)
        {
            particleGrid.particles[i].setWeight(global_weight_memory[i]);
            global_weight_memory[i] = 0.f; // fill array with 0.f
        }

        renderTextureToScreen(zed_tex, simpleTextureShader);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    // LOG
    printf("[LOG] Particle Weights: \n");
    for (int i = 0; i < PARTICLE_COUNT; i++) printf("%i -> %f \n", i, particleGrid.particles[i].getWeight());

    HANDLE_CUDA_ERROR(cudaFree(dev_global_weight_memory));
    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}