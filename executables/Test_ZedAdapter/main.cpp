#include <ModelTracker/ModelTracker.h>

#define WIDTH 1280
#define HEIGHT 720
GLFWwindow* window;

using namespace mt;

int main(int argc, char **argv)
{
    window = initGLWindow(window, WIDTH, HEIGHT, "Test ZED Adapter", BLACK);

    ZedAdapter zedAdapter(WIDTH, HEIGHT);
    zedAdapter.initCamera();

    while( !glfwWindowShouldClose(window))
    {
        zedAdapter.grab();
        zedAdapter.retrieveRawImage();
        zedAdapter.imageToGlTexture();
        zedAdapter.renderImage();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    zedAdapter.clean();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}