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

int main()
{
        window = initGLWindow(window, WIDTH, HEIGHT, "Test - Particle Filter", BLACK);

    Camera zed;
    mt::ZedAdapter zedAdapter(zed, RESOLUTION_HD720, "~/bachelor_thesis/HD720_BENCHMARK.svo");

    // Creating Color Particle Grid
    mt::ParticleGrid particleGrid(RESOURCES_PATH "/simple_cube/simple_cube.obj", PARTICLE_WIDTH, PARTICLE_HEIGHT, PARTICLE_COUNT);
    particleGrid.renderColorTexture();

    // Create particleFilter and map gl_texture on cuda_array
    cudaArray *tex_array;
    mt::ParticleFilter particleFilter;
    particleFilter.mapGLTextureToCudaArray(particleGrid.getColorTexture(), tex_array);

    // zed interopertion
    Mat zed_in_img  =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat zed_out_img =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);

    while(!glfwWindowShouldClose( window))
    {
        particleGrid.update(0.2f, 0.8f);
        particleGrid.renderColorTexture();

        HANDLE_ZED_ERROR(zed.grab());
        HANDLE_ZED_ERROR(zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU));
        // callKernel(zed_in_img.getPtr<sl::uchar4>(MEM_GPU), zed_out_img.getPtr<sl::uchar4>(MEM_GPU), zed_in_img.getStep(MEM_GPU), WIDTH, HEIGHT, tex_array);

        particleFilter.convertBGRtoRGB(zed_in_img, zed_out_img);

        zedAdapter.imageToGlTexture(zed_out_img);
        zedAdapter.renderImage();

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    printf("CLEAN UP... \n");
    zed_in_img.free();
    zed_out_img.free();
    zed.close();
    printf("DONE \n");

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}