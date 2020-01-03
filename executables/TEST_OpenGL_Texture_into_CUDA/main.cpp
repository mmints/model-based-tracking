#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>
#include <cuda_gl_interop.h>
#include <sl/Camera.hpp>

#include "kernel.h"

#define WIDTH 1280
#define HEIGHT 720

using namespace sl;

GLFWwindow* window;


Camera zed;
Mat gpuLeftImage;
Mat outImage = Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
cudaGraphicsResource* pcuImageRes; // Cuda resources for CUDA-OpenGL interoperability

void initGL()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH, HEIGHT, "[TEST] OpenGL Texture into CUDA", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glewInit();
    glEnable(GL_DEPTH_TEST);

    CVK::State::getInstance()->setBackgroundColor(BLACK);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);
}

// Set up all necessary shader and use the Particle Generator to create a geometry, that will be
// rendered into tha Color Buffer of ang given FBO
void renderIntoFBO(CVK::FBO &fbo, int particle_count)
{
    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    CVK::State::getInstance()->setShader( &shaderSimple);

    mt::ParticleGenerator particleGenerator(RESOURCES_PATH "/rubiks_cube/rubiks_cube.obj", particle_count, WIDTH, HEIGHT);
    std::vector<mt::Particle> particles;
    particleGenerator.initializeParticles(particles, 1.8f);

    // set up the location of matrices in shader program
    GLuint viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    GLuint projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WIDTH/HEIGHT, 1.0f, 100.0f);

    fbo.bind();
    shaderSimple.useProgram();
    // pipe uniform variables to shader
    glUniformMatrix4fv(viewMatrixHandle, 1, GL_FALSE, value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionMatrixHandle, 1, GL_FALSE, value_ptr(projectionMatrix));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    particleGenerator.renderParticleTextureGrid(particles);
    glFinish(); // Wait until everything is done.
    fbo.unbind();
}

// Renders the first Color Buffer of the given FBO onto a screen filling quad.
void renderTextureToScreen(GLuint textureID)
{
    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture );

    glViewport(0, 0, WIDTH, HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    simpleTextureShader.setTextureInput(0, textureID);
    simpleTextureShader.useProgram();
    simpleTextureShader.update();
    simpleTextureShader.render();
}

int main()
{
    initGL();

    CVK::FBO fbo( WIDTH, HEIGHT, 1, true );
    renderIntoFBO(fbo, 1);

    // CUDA interopertion part
    struct cudaGraphicsResource *tex_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&tex_resource, fbo.getColorTexture(0), GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &tex_resource));

    cudaArray *tex_array;
    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&tex_array, tex_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &tex_resource));

    renderTextureToScreen(fbo.getColorTexture(0));
    callKernel(WIDTH, HEIGHT, tex_array);

    while(!glfwWindowShouldClose( window))
    {
        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}