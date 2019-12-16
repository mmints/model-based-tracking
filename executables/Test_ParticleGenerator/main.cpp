#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>

#define WIDTH 1920
#define HEIGHT 1080

GLFWwindow* window;

int main()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH, HEIGHT, "[TEST] Particle Generator", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glewInit();
    glEnable(GL_DEPTH_TEST);

    CVK::State::getInstance()->setBackgroundColor(BLACK);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    CVK::State::getInstance()->setShader( &shaderSimple);

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture );

    CVK::FBO fbo( WIDTH, HEIGHT, 1, true );

    mt::ParticleGenerator particleGenerator(RESOURCES_PATH "/rubiks_cube/rubiks_cube.obj", 256, WIDTH, HEIGHT);
    std::vector<mt::Particle> particles;
    particleGenerator.initializeParticles(particles, 1.8f);

    // set up the location of matrices in shader program
    GLuint viewMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "viewMatrix");
    GLuint projectionMatrixHandle = glGetUniformLocation(shaderSimple.getProgramID(), "projectionMatrix");

    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(40.0f), (float) WIDTH/HEIGHT, 1.0f, 100.0f);

    while(!glfwWindowShouldClose( window))
    {
        // First Renderpass to FBO
        fbo.bind();
        shaderSimple.useProgram();
        // pipe uniform variables to shader
        glUniformMatrix4fv(viewMatrixHandle, 1, GL_FALSE, value_ptr(viewMatrix));
        glUniformMatrix4fv(projectionMatrixHandle, 1, GL_FALSE, value_ptr(projectionMatrix));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        particleGenerator.renderParticleTextureGrid(particles);
        glFinish(); // Wait until everything is done.
        fbo.unbind();

        glViewport(0, 0, WIDTH, HEIGHT);
        // Second Renderpass to Screen Filling Quad
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        simpleTextureShader.setTextureInput( 0, fbo.getColorTexture ( 0 ) );
        simpleTextureShader.useProgram();
        simpleTextureShader.update ();
        simpleTextureShader.render ();

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}