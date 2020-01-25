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

#define PARTICLE_SCALE 10

#define PARTICLE_W  WINDOW_W / PARTICLE_SCALE
#define PARTICLE_H  WINDOW_H / PARTICLE_SCALE

#define PARTICLE_C 16//Particle Count. Have to be a quad!

GLFWwindow* window;

// Transformation Matrices for Particle Rendering
glm::mat4 viewMatrix;
glm::mat4 projectionMatrix;
GLuint viewMatrixHandle;
GLuint projectionMatrixHandle;

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
    printf("[LOG] Particle Height: %i \n", PARTICLE_H);

    window = initGLWindow(window, WINDOW_W, WINDOW_H, "Red Ball Tracker", BLACK);
    printf("[LOG] Initialize GL Window \n");

    Camera zed;
    initZedCamera(zed, argv[1]); // If argv[1] is null than the hardware camera setting will be loaded
    printf("[LOG] Initialize ZED \n");

    const char *shadernamesTextureToScreen [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };      // Attention: ZED image will stay BGR! Make sure to switch texel values in Kernel
    CVK::ShaderSimpleTexture textureToScreen( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesTextureToScreen);
    printf("[LOG] textureToScreen is loaded \n");

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    printf("[LOG] shaderSimple is loaded \n");

    // init Matrices
    viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WINDOW_W/WINDOW_H, 1.0f, 100.0f);
    printf("[LOG] Initialize Matrices \n");

    // Set up ParticleGenerator
    // TODO: Create a initialization function for this
    mt::ParticleGenerator particleGenerator(PARTICLE_C, PARTICLE_W, PARTICLE_H);

    CVK::FBO fbo( (int) (PARTICLE_W * std::sqrt(PARTICLE_C)), (int)(PARTICLE_H * std::sqrt(PARTICLE_C)), 1, true);
    printf("***** %i \n", (int)(PARTICLE_W * std::sqrt(PARTICLE_C)) );
    printf("***** %i \n", (int)(PARTICLE_H * std::sqrt(PARTICLE_C)) );

    //CVK::FBO fbo( WINDOW_W, WINDOW_H, 1, true);
    printf("[LOG] Initialize Particle Generator and FBO \n");

    mt::ParticleGrid particleGrid; // TODO: Merge Particle Grid and Particle Generator together
    particleGenerator.initializeParticles(particleGrid.particles, 1.8f);
    renderParticleGrid(fbo, shaderSimple, particleGenerator, particleGrid);
    printf("[LOG] Initialize Particle Grid and generate particles. \n");
    printf("[LOG] Render ParticleGrid into FBO \n");


    // Set up global memory array for transferring weight for particles from GPU to CPU
    float global_weight_memory[PARTICLE_C] = {0}; // with space for all particle weights
    for (int i = 0; i < PARTICLE_C; i++) global_weight_memory[i] = 0.f; // fill array with 0.f
    printf("[LOG] Initialize and erase global weight memory for comunication between CPU and GPU\n");

    // Allocate corresponding memory space on GPU
    float *dev_global_weight_memory;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_global_weight_memory, PARTICLE_C * sizeof(float)));
    printf("[LOG] Allocate global weight memory on GPU \n");

    // CUDA interopertion part
    // TODO: Create a initialization function for this
    struct cudaGraphicsResource *particle_grid_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&particle_grid_resource, particleGrid.texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &particle_grid_resource));

    cudaArray *particle_grid_tex_array;
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&particle_grid_tex_array, particle_grid_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &particle_grid_resource));
    printf("[LOG] Register Particle Grid texture for ZED frame \n");

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    GLuint zed_tex;
    generateGlTexture(zed_tex, WINDOW_W, WINDOW_H);
    cudaGraphicsResource* zed_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&zed_resource, zed_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaArray *zed_tex_array;
    printf("[LOG] Create and register GL texture for ZED frame \n");

    Mat zed_in_img      =  Mat(WINDOW_W, WINDOW_H, MAT_TYPE_8U_C4, MEM_GPU);
    Mat zed_red_map_img =  Mat(WINDOW_W, WINDOW_H, MAT_TYPE_8U_C4, MEM_GPU);
    Mat zed_out_img     =  Mat(WINDOW_W, WINDOW_H, MAT_TYPE_8U_C4, MEM_GPU);
    printf("[LOG] Set ZED Mats \n");

    // Render Loop
    while(!glfwWindowShouldClose( window))
    {
        zed.grab();
        // ZED interoperation routine
        if (zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU) == SUCCESS) {

            // Call kernels
            kernel::simpleRedDetector(zed_in_img.getPtr<sl::uchar4>(MEM_GPU), zed_red_map_img.getPtr<sl::uchar4>(MEM_GPU), 140, WINDOW_W, WINDOW_H, zed_in_img.getStep(MEM_GPU));

            HANDLE_CUDA_ERROR(cudaMemcpy(dev_global_weight_memory, global_weight_memory, PARTICLE_C * sizeof(float), cudaMemcpyHostToDevice));

            mt::compareRedPixel(zed_red_map_img.getPtr<sl::uchar4>(MEM_GPU),
                                zed_red_map_img.getStep(MEM_GPU),
                                PARTICLE_SCALE,
                                (int) std::sqrt(PARTICLE_C),
                                PARTICLE_W, PARTICLE_H,
                                particle_grid_tex_array, dev_global_weight_memory,
                                zed_out_img.getPtr<sl::uchar4>(MEM_GPU), // For debugging!
                                zed_in_img.getPtr<sl::uchar4>(MEM_GPU)); // For debugging!

            HANDLE_CUDA_ERROR(cudaMemcpy(global_weight_memory, dev_global_weight_memory, PARTICLE_C * sizeof(float), cudaMemcpyDeviceToHost));

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
        // Transfer weights from global_weight_memory to particles
        for (int i = 0; i < PARTICLE_C; i++) {
            particleGrid.particles[i].setWeight(global_weight_memory[i]);
            global_weight_memory[i] = 0.f; // fill array with 0.f - clean up
        }

        // TODO: For now, this is only the red color map
        renderZEDTextureToScreen(zed_tex, WINDOW_W, WINDOW_H, textureToScreen); // TODO: Render best fitting particle on top
        //renderTextureToScreen(particleGrid.texture, WINDOW_W, WINDOW_H, textureToScreen);

        particleGenerator.updateParticles(particleGrid.particles);
        renderParticleGrid(fbo, shaderSimple, particleGenerator, particleGrid);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    printf("[LOG] Particle Weights: \n");
    for (int i = 0; i < PARTICLE_C; i++) printf("%i -> %f \n", i, particleGrid.particles[i].getWeight());


    printf("[LOG] Done! Cleaning up and shutting down \n");

    zed_in_img.free();
    zed_out_img.free();
    zed_red_map_img.free();

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;

}
