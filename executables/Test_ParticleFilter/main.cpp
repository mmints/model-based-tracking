//#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

//#include <cuda_gl_interop.h>
//#include <sl/Camera.hpp>

#include <ModelTracker/ModelTracker.h>
#include <ImageFilter/ImageFilter.h>

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 16*16

#define PARTICLE_WIDTH WIDTH   / 2
#define PARTICLE_HEIGHT HEIGHT / 2

using namespace sl;

GLFWwindow* window;

int main(int argc, char **argv)
{
        window = initGLWindow(window, WIDTH, HEIGHT, "Test - Particle Filter", BLACK);

    Camera zed;
    mt::ZedAdapter zedAdapter(zed, RESOLUTION_HD720, argv[1]);
    CameraInformation info = zed.getCameraInformation();

    printf("Focal X: %f \n ", info.calibration_parameters.left_cam.fx);
    printf("Focal Y: %f \n ", info.calibration_parameters.left_cam.fy);
    printf("Center X: %f \n ", info.calibration_parameters.left_cam.cx);
    printf("Center Y: %f \n ", info.calibration_parameters.left_cam.cy);
    printf("Horizontal FOV: %f \n ", info.calibration_parameters.left_cam.h_fov);
    printf("Vertical FOV: %f \n ", info.calibration_parameters.left_cam.v_fov);
    printf("Diagonal FOV: %f \n ", info.calibration_parameters.left_cam.d_fov);

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

    Mat img_normals =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);

    Mat img_gs =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C1, MEM_GPU);
    Mat img_edge =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C1, MEM_GPU);

    while(!glfwWindowShouldClose( window))
    {
        particleGrid.update(0.2f, .4f);
        particleGrid.renderColorTexture();
        particleGrid.renderEdgeTexture();

        zed.grab();
        HANDLE_ZED_ERROR(zed.retrieveImage(img_raw, VIEW_LEFT, MEM_GPU));
        HANDLE_ZED_ERROR(zed.retrieveImage(img_gs, VIEW_LEFT_GRAY, MEM_GPU));
        HANDLE_ZED_ERROR(zed.retrieveImage(img_normals, VIEW_NORMALS, MEM_GPU));

        // Preprocessing ZED
        filter::convertBGRtoRGB(img_raw, img_rgb);
        filter::sobelFilter(img_gs, img_edge);

        // Render Camera Stream to Screen
        zedAdapter.imageToGlTexture(img_rgb);
        zedAdapter.renderImage();

        // Particle Filter routine
        //particleFilter.calculateWeightColor(img_rgb, particleGrid);
        particleFilter.calculateWeightNormals(img_normals, particleGrid);
        particleFilter.calculateWeightEdge(img_edge, particleGrid);
        particleFilter.setParticleWeight(particleGrid);
        particleFilter.resample(particleGrid, (int)PARTICLE_COUNT * 0.2f);

        particleGrid.renderFirstParticleToScreen();

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    particleGrid.sortParticlesByWeight();

    printf("CLEAN UP... \n");
    img_raw.free();
    img_rgb.free();
    img_gs.free();
    img_edge.free();
    zed.close();
    HANDLE_CUDA_ERROR(cudaDeviceReset());
    printf("DONE \n");

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}