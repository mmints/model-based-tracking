#ifndef MT_HELPER_FUNCTIONS_H
#define MT_HELPER_FUNCTIONS_H

// This header includes some helping functions that are mandatory for applications,
// that are based in the ModelTracker-Framework.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <CVK_2/CVK_Framework.h>

void initGL(const glm::vec3 backGroundColor)
{
    CVK::useOpenGL33CoreProfile();
    glewInit();
    glEnable(GL_DEPTH_TEST);

    CVK::State::getInstance()->setBackgroundColor(backGroundColor);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);
}

GLFWwindow* initWindow(GLFWwindow* window, const int width, const int height, const char* title)
{
    glfwInit();
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    return window;
}

/**
 * Initialize a GLFW Window with an already preset GL environment.
 * If the target applications has more than one window use initWindow for the additional one.
 * @param window The target window pointer
 * @param width Dimension of the window
 * @param height Dimension of the window
 * @param title Title of the window
 * @param backGroundColor Back ground color of the GL instance.
 * @return Pointer to the window for further use of glfw functions.
 */
GLFWwindow* initGLWindow(GLFWwindow* window, const int width, const int height, const char* title, const glm::vec3 backGroundColor)
{
    window = initWindow(window, width, height, title);
    initGL(backGroundColor);
    return window;
}


#endif //MT_HELPER_FUNCTIONS_H
