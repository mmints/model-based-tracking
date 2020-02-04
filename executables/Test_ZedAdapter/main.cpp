#include <ModelTracker/ModelTracker.h>

#define WIDTH 1280
#define HEIGHT 720
GLFWwindow* window;

using namespace mt;

int main(int argc, char **argv)
{
    window = initGLWindow(window, WIDTH, HEIGHT, "Test ZED Adapter", BLACK);

    Camera zed;
    Mat zed_img;

    ZedAdapter zedAdapter(zed, RESOLUTION_HD720);

    while( !glfwWindowShouldClose(window))
    {
        zed.grab();
        zed.retrieveImage(zed_img, VIEW_LEFT, MEM_GPU);
        zedAdapter.imageToGlTexture(zed_img);
        zedAdapter.renderImage();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    zed_img.free();
    zed.close();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}