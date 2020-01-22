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

/**
 * Generates an OpenGL texture.
 * @param textureID Allocated texture ID
 * @param width Dimension of the texture
 * @param height Dimension of the texture
 */
void generateGlTexture(GLuint &textureID, const int width, const int height)
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/**
 * Renders a simple GL texture to a window frame of given dimensions by using a given Shader.
 * @param textureID Target Texture
 * @param width Dimension of the texture
 * @param height Dimension of the texture
 * @param simpleTextureShader Shader program, that is used to render the texture
 */
void renderTextureToScreen(GLuint textureID, const int width, const int height, CVK::ShaderSimpleTexture &simpleTextureShader)
{
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simpleTextureShader.setTextureInput(0, textureID);
    simpleTextureShader.useProgram();
    simpleTextureShader.update();
    simpleTextureShader.render();
}

/**
 * Renders a ZED video frame that was transform to a GL texture to a window frame of given dimensions by using a given Shader.
 * The difference to renderTextureToScreen is, that another render function implemented in the shader class is used.
 * @param textureID Target Texture
 * @param width Dimension of the texture
 * @param height Dimension of the texture
 * @param simpleTextureShader Shader program, that is used to render the texture
 */
 // TODO: Move this function to ParticleGrid after turning ParticleGrid into a Class
void renderZEDTextureToScreen(GLuint textureID, const int width, const int height, CVK::ShaderSimpleTexture &simpleTextureShader)
{
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simpleTextureShader.setTextureInput(0, textureID);
    simpleTextureShader.useProgram();
    simpleTextureShader.update();
    simpleTextureShader.renderZED();
}



#endif //MT_HELPER_FUNCTIONS_H
