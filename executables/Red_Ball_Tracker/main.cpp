#include <ModelTracker/ModelTracker.h>
#include <Shader/ShaderSimple.h>

#include <CVK_2/CVK_Framework.h>

#include <ErrorHandling/HANDLE_CUDA_ERROR.h>
#include <cuda_gl_interop.h>
#include <sl/Camera.hpp>
#include <ImageFilter/ImageFilter.h>

using namespace sl;
using namespace mt;
using namespace std;

// Same Resolution as the target ZED resolution
#define WINDOW_W 1280
#define WINDOW_H 720

#define PARTICLE_W WINDOW_W/2
#define PARTICLE_H WINDOW_H/2

#define PARTICLE_C 100 //Particle Count

GLFWwindow* window;

glm::mat4 viewMatrix;
glm::mat4 projectionMatrix;
GLuint viewMatrixHandle;
GLuint projectionMatrixHandle;
void initMatrices(ShaderSimple shaderSimple)
{
    viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WINDOW_W/WINDOW_H, 1.0f, 100.0f);
}

void renderParticleGrid(CVK::FBO &fbo, ShaderSimple &shaderSimple, mt::ParticleGenerator &particleGenerator,mt::ParticleGrid &particleGrid)
{
    CVK::State::getInstance()->setShader( &shaderSimple);

    fbo.bind();
    shaderSimple.useProgram();
    // pipe uniform variables to shader
    glUniformMatrix4fv(viewMatrixHandle, 1, GL_FALSE, value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionMatrixHandle, 1, GL_FALSE, value_ptr(projectionMatrix));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    particleGenerator.renderParticleTextureGrid(particleGrid.particles);
    fbo.unbind();

    particleGrid.texture = fbo.getColorTexture(0);
}

int main(int argc, char **argv)
{
    window = initGLWindow(window, WINDOW_W, WINDOW_H, "Red Ball Tracker", BLACK);
    printf("[LOG] Initialize GL Window \n");

    Camera zed;
    initZedCamera(zed, argv[1]); // If argv[1] is null than the hardware camera setting will be loaded
    printf("[LOG] Initialize ZED \n");

    const char *shadernamesTextureToScreen [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture textureToScreen( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesTextureToScreen);
    printf("[LOG] textureToScreen is loaded \n");

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    printf("[LOG] shaderSimple is loaded \n");

    initMatrices(shaderSimple);
    printf("[LOG] Initialize Matricess \n");

    // Set up ParticleGenerator
    // TODO: Create a initialization function for this
    mt::ParticleGenerator particleGenerator(PARTICLE_C, PARTICLE_W, PARTICLE_H);
    CVK::FBO fbo( PARTICLE_W * std::sqrt(PARTICLE_C), PARTICLE_H * std::sqrt(PARTICLE_C), 1, true);

    mt::ParticleGrid particleGrid; // TODO: Merge Particle Grid and Particle Generator together
    particleGenerator.initializeParticles(particleGrid.particles, 1.8f);
    renderParticleGrid(fbo, shaderSimple, particleGenerator, particleGrid);

    // Set up global memory array for transferring weight for particles from GPU to CPU
    float global_weight_memory[PARTICLE_C] = {0}; // with space for all particle weights
    for (int i = 0; i < PARTICLE_C; i++) global_weight_memory[i] = 0.f; // fill array with 0.f

    // Allocate corresponding memory space on GPU
    float *dev_global_weight_memory;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_global_weight_memory, PARTICLE_C * sizeof(float)));

    // CUDA interopertion part
    // TODO: Create a initialization function for this
    struct cudaGraphicsResource *particle_grid_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&particle_grid_resource, particleGrid.texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &particle_grid_resource));

    cudaArray *particle_grid_tex_array;
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&particle_grid_tex_array, particle_grid_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &particle_grid_resource));

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    GLuint zed_tex;
    generateGlTexture(zed_tex, WINDOW_W, WINDOW_H);
    cudaGraphicsResource* zed_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&zed_resource, zed_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaArray *zed_tex_array;
    printf("[LOG] Create and register GL texture for ZED frame \n");

    Mat zed_in_img  =  Mat(WINDOW_W, WINDOW_H, MAT_TYPE_8U_C4, MEM_GPU);
    Mat zed_out_img =  Mat(WINDOW_W, WINDOW_H, MAT_TYPE_8U_C4, MEM_GPU);

    // Render Loop
    while(!glfwWindowShouldClose( window))
    {
        zed.grab();
        // ZED interoperation routine
        if (zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU) == SUCCESS) {
            kernel::simpleRedDetector(zed_in_img.getPtr<sl::uchar4>(MEM_GPU), zed_out_img.getPtr<sl::uchar4>(MEM_GPU), 140, WINDOW_W, WINDOW_H, zed_in_img.getStep(MEM_GPU));

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

        // TODO: Resampling
        for (int i = 0; i < PARTICLE_C; i++) {
            particleGrid.particles[i].setWeight(global_weight_memory[i]);
            global_weight_memory[i] = 0.f; // fill array with 0.f - clean up
        }

        renderZEDTextureToScreen(zed_tex, WINDOW_W, WINDOW_H, textureToScreen);

        particleGenerator.updateParticles(particleGrid.particles);
        renderParticleGrid(fbo, shaderSimple, particleGenerator, particleGrid);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    printf("[LOG] Done! Cleaning up and shutting down \n");

    zed_in_img.free();

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;

}
