#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>

#define WIDTH 1920
#define HEIGHT 1080

#define COUNT 16

GLFWwindow* window;


glm::mat4 viewMatrix;
glm::mat4 projectionMatrix;
GLuint viewMatrixHandle;
GLuint projectionMatrixHandle;
void initMatrices(ShaderSimple shaderSimple)
{
    viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WIDTH/HEIGHT, 1.0f, 100.0f);
}

void renderParticleGrid(CVK::FBO &fbo, ShaderSimple &shaderSimple, mt::ParticleGenerator &particleGenerator,mt::ParticleGrid &particleGrid)
{
    CVK::State::getInstance()->setShader( &shaderSimple);

    fbo.bind();
    shaderSimple.useProgram();
    // pipe uniform variables to shader
    glUniformMatrix4fv(viewMatrixHandle, 1, GL_FALSE, value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionMatrixHandle, 1, GL_FALSE, value_ptr(projectionMatrix));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    particleGenerator.renderParticleTextureGrid(particleGrid.particles);
    fbo.unbind();

    particleGrid.texture = fbo.getColorTexture(0);
}

int main()
{
    window = initGLWindow(window, WIDTH, HEIGHT, "Render Red Particle Grid", BLACK);

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    //CVK::State::getInstance()->setShader( &shaderSimple);

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture );

    CVK::FBO fbo( WIDTH * std::sqrt(COUNT), HEIGHT * std::sqrt(COUNT), 1, true );

    mt::ParticleGrid particleGrid;
    mt::ParticleGenerator particleGenerator(COUNT, WIDTH, HEIGHT);
    particleGenerator.initializeParticles(particleGrid.particles, 1.8f);

    viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WIDTH/HEIGHT, 1.0f, 100.0f);

    while(!glfwWindowShouldClose( window))
    {
        renderParticleGrid(fbo,shaderSimple,particleGenerator,particleGrid);
        renderTextureToScreen(particleGrid.texture, WIDTH, HEIGHT, simpleTextureShader);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}