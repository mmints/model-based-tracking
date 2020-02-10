//#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

//#include <cuda_gl_interop.h>
//#include <sl/Camera.hpp>

#include <ModelTracker/ModelTracker.h>
#include <ImageFilter/ImageFilter.h>

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 1024

#define PARTICLE_WIDTH WIDTH   / 10
#define PARTICLE_HEIGHT HEIGHT / 10

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

    printf("\n **** SIZE OF PARTICLE GRID ***** \n");
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

    int measure_count = 0;
    while(!glfwWindowShouldClose( window))
    {
        particleGrid.update(0.4f, 0.2f);
        particleGrid.renderColorTexture();
        particleGrid.renderDepthTexture();
        particleGrid.renderNormalTexture();

        zed.grab();
        HANDLE_ZED_ERROR(zed.retrieveImage(img_raw, VIEW_LEFT, MEM_GPU));
        HANDLE_ZED_ERROR(zed.retrieveImage(img_depth, VIEW_DEPTH, MEM_GPU));        // MEASURE_DEPTH needs sl::MAT_TYPE_32F_C1
        HANDLE_ZED_ERROR(zed.retrieveImage(img_normals, VIEW_NORMALS, MEM_GPU));    // MEASURE_NORMALS needs sl::MAT_TYPE_32F_C4

        filter::convertBGRtoRGB(img_raw, img_rgb);

        zedAdapter.imageToGlTexture(img_rgb);
        zedAdapter.renderImage();

        // Calculate the weights
        particleFilter.calculateWeightColor(img_rgb, particleGrid);
        particleFilter.calculateWeightDepth(img_depth, particleGrid);
        particleFilter.calculateWeightNormals(img_normals, particleGrid);

        particleFilter.setParticleWeight(particleGrid);

         particleGrid.sortParticlesByWeight();

        if (particleGrid.m_particles[0].getWeight() > 100.f) {
            printf("\n *** FOUND %i*** \n", measure_count);
            printf("WEIGHT: %f \n", particleGrid.m_particles[0].getWeight());
            particleGrid.renderFirstParticleToScreen();
        }

        bool checker = false;
        printf("\n **** MEASUREMENT %i ***** \n", measure_count);
        for (int i = 0; i < PARTICLE_COUNT; i++) {
            if (particleGrid.m_particles[i].getWeight() > 0.f) {
                printf("I: %i - W: %f \n", i, particleGrid.m_particles[i].getWeight());
            }

            if (particleGrid.m_particles[i].getWeight() < 0.f) {
                printf("\n **** FUUUUUCK ***** \n");
                printf("I: %i - W: %f \n", i, particleGrid.m_particles[i].getWeight());
                //checker = true;
            }
        }
        measure_count++;

        particleFilter.resample(particleGrid, 60);

        for (int i = 0; i < PARTICLE_COUNT; i++) {
            if (particleGrid.m_particles[i].getWeight() < 0.f) {
                printf("\n **** RESAMPING FAILURE ***** \n");
                printf("I: %i - W: %f \n", i, particleGrid.m_particles[i].getWeight());
                checker = true;
            }
        }
        if (checker)
            break;



        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    particleGrid.sortParticlesByWeight();

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