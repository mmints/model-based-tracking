#include <ModelTracker/ModelTracker.h>

#define WIDTH 1280
#define HEIGHT 720
GLFWwindow* window;

using namespace mt;

int main(int argc, char **argv)
{
    window = initGLWindow(window, WIDTH, HEIGHT, "Test ZED Adapter", BLACK);

    Camera zed;
    initBasicZedCameraHD720(zed);
    //Mat zed_img;

    ZedAdapter zedAdapter(WIDTH, HEIGHT);

    while( !glfwWindowShouldClose(window))
    {
        zed.grab();
        //zed.retrieveImage(zed_img, VIEW_LEFT, MEM_GPU);
        //zedAdapter.imageToGlTexture(zed_img);
        zedAdapter.retrieveImage(zed);
        zedAdapter.imageToGlTexture();
        zedAdapter.renderImage();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    zedAdapter.freeImages();
    zed.close();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}