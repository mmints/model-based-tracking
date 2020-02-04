#ifndef MT_HELPER_FUNCTIONS_H
#define MT_HELPER_FUNCTIONS_H

// This header includes some helping functions that are mandatory for applications,
// that are based in the ModelTracker-Framework.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <CVK_2/CVK_Framework.h>

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
GLFWwindow* initGLWindow(GLFWwindow* window, const int width, const int height, const char* title, const glm::vec3 backGroundColor);

/**
 * Generates an OpenGL texture.
 * @param textureID Allocated texture ID
 * @param width Dimension of the texture
 * @param height Dimension of the texture
 */
void generateGlTexture(GLuint &textureID, const int width, const int height);

/**
 * Renders a simple GL texture to a window frame of given dimensions by using a given Shader.
 * @param textureID Target Texture
 * @param width Dimension of the texture
 * @param height Dimension of the texture
 * @param simpleTextureShader Shader program, that is used to render the texture
 */
void renderTextureToScreen(GLuint textureID, const int width, const int height, CVK::ShaderSimpleTexture &simpleTextureShader);
/**
 * Renders a ZED video frame that was transform to a GL texture to a window frame of given dimensions by using a given Shader.
 * The difference to renderTextureToScreen is, that another render function implemented in the shader class is used.
 * @param textureID Target Texture
 * @param width Dimension of the texture
 * @param height Dimension of the texture
 * @param simpleTextureShader Shader program, that is used to render the texture
 */
void renderZEDTextureToScreen(GLuint textureID, const int width, const int height, CVK::ShaderSimpleTexture &simpleTextureShader);

// Helper for initGLWindow
void initGL(const glm::vec3 backGroundColor);
GLFWwindow* initWindow(GLFWwindow* window, const int width, const int height, const char* title);

#endif //MT_HELPER_FUNCTIONS_H
