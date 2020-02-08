//#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

//#include <cuda_gl_interop.h>
//#include <sl/Camera.hpp>

#include <ModelTracker/ModelTracker.h>
#include <ImageFilter/ImageFilter.h>
#include "kernel.h"

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 1

#define PARTICLE_WIDTH WIDTH   / 1
#define PARTICLE_HEIGHT HEIGHT / 1

using namespace sl;

GLFWwindow* window;

int main(int argc, char **argv)
{
        window = initGLWindow(window, WIDTH, HEIGHT, "Test - Particle Filter", BLACK);

    Camera zed;
    mt::ZedAdapter zedAdapter(zed, RESOLUTION_HD720, argv[1]);

    // Creating Color Particle Grid
    mt::ParticleGrid particleGrid(RESOURCES_PATH "/simple_cube/simple_cube.obj", PARTICLE_WIDTH, PARTICLE_HEIGHT, PARTICLE_COUNT);
    // particleGrid.renderColorTexture();

    printf("**** SIZE OF PARTICLE GRID ***** \n");
    printf("Dimension: %i \n", particleGrid.getParticleGridDimension());
    printf("Particle Resolution: %i x %i \n", particleGrid.getParticleWidth(), particleGrid.getParticleHeight());

    mt::ParticleFilter particleFilter(particleGrid);

    // Create particleFilter and map gl_texture on cuda_array
    cudaArray *tex_array; // Only for the testing Kernel
    particleFilter.mapGLTextureToCudaArray(particleGrid.getColorTexture(), tex_array); // Only for the testing Kernel

    // zed interopertion
    Mat img_raw  =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat img_rgb =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat img_depth =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat img_normals =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);


    Mat img_out =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU); // Only for the testing Kernel


    while(!glfwWindowShouldClose( window))
    {
        //particleGrid.update(0.2f, 0.0f);
        particleGrid.renderColorTexture();

        zed.grab();
        HANDLE_ZED_ERROR(zed.retrieveImage(img_raw, VIEW_LEFT, MEM_GPU));
        HANDLE_ZED_ERROR(zed.retrieveImage(img_depth, VIEW_DEPTH, MEM_GPU));        // MEASURE_DEPTH needs sl::MAT_TYPE_32F_C1
        HANDLE_ZED_ERROR(zed.retrieveImage(img_normals, VIEW_NORMALS, MEM_GPU));    // MEASURE_NORMALS needs sl::MAT_TYPE_32F_C4

        filter::convertBGRtoRGB(img_raw, img_rgb);

        // This Kernel Renders the selected texture from the particle grid into the ZED frame
        callKernel(img_rgb.getPtr<sl::uchar4>(MEM_GPU), img_out.getPtr<sl::uchar4>(MEM_GPU), img_rgb.getStep(MEM_GPU), WIDTH, HEIGHT, tex_array);

        // Calculate the weights
        particleFilter.calculateWeightColor(img_rgb, particleGrid);
        particleFilter.calculateWeightDepth(img_depth, particleGrid);
        particleFilter.calculateWeightNormals(img_normals, particleGrid);

        zedAdapter.imageToGlTexture(img_out);
        zedAdapter.renderImage();

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    printf("CLEAN UP... \n");
    img_raw.free();
    img_rgb.free();
    img_depth.free();
    img_normals.free();
    img_out.free();
    zed.close();
    HANDLE_CUDA_ERROR(cudaDeviceReset());
    printf("DONE \n");

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}