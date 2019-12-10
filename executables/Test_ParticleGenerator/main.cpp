#include <vector>
#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>


// Window
const int width = 1280;
const int height = 720;
GLFWwindow* window;

// Resource declaration
GLuint imageTex;

int main()
{
    glfwInit(); // Initial GLFW object for using GLFW functionalities

    //Init Window
    window = glfwCreateWindow(width, height, "ZED Introduction", NULL, NULL);
    glfwSetWindowPos( window, 50, 50);
    glfwMakeContextCurrent(window);

    glewInit();

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    CVK::State::getInstance()->setShader( &shaderSimple);

    const char *shadernames2[1] = {SHADERS_PATH "/DisplayTexture.frag"};
    ShaderSimple displayTextureShader(FRAGMENT_SHADER_BIT, shadernames2);

    mt::ParticleGenerator particleGenerator(RESOURCES_PATH "/rubiks_cube/rubiks_cube.obj", 100, glm::vec2(width, height));
    std::vector<mt::Particle> particle;
    particleGenerator.initializeParticles(particle);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);

    shaderSimple.useProgram();
    CVK::FBO fbo(width, height, 1, true, false);
    fbo.bind();
    particleGenerator.renderParticleTextureGrid(10, 10, particle);
    fbo.getColorTexture(imageTex);
    fbo.unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    // Set the uniform variable for texImage (sampler2D) to the texture unit
    glUniform1i(glGetUniformLocation(displayTextureShader.getProgramID(), "texImage"), 0);

    while( !glfwWindowShouldClose(window))
    {
        ////  OpenGL rendering part ////
        glEnable(GL_DEPTH_TEST);
        glLoadIdentity(); // replace the current matrix with the identity matrix (Why?)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glBindTexture(GL_TEXTURE_2D, imageTex);
        displayTextureShader.useProgram();

        // Render the final texture
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 1.0);
        glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 1.0);
        glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 0.0);
        glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 0.0);
        glVertex2f(-1.0, 1.0);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteProgram(displayTextureShader.getProgramID());

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
