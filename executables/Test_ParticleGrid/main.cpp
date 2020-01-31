#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ModelTracker.h>

#define PARTICLE_COUNT 16

#define PARTICLE_WIDTH 100
#define PARTICLE_HEIGHT 100

#define WIDTH 400
#define HEIGHT 400

GLFWwindow* window;

int main()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH * 4, HEIGHT, "[TEST] Particle Grid", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glewInit();
    glEnable(GL_DEPTH_TEST);

    CVK::State::getInstance()->setBackgroundColor(BLACK);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture );

    mt::ParticleGrid particleGrid(RESOURCES_PATH "/simple_cube/simple_cube.obj", PARTICLE_WIDTH, PARTICLE_HEIGHT, PARTICLE_COUNT);

    particleGrid.renderAllTextures();

    GLuint color_tex_id;
    GLuint normals_tex_id;
    GLuint depth_tex_id;
    GLuint edge_tex_id;

    while(!glfwWindowShouldClose( window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glViewport(0, 0, WIDTH, HEIGHT);
        simpleTextureShader.setTextureInput( 0, color_tex_id);
        simpleTextureShader.useProgram();
        simpleTextureShader.update ();
        simpleTextureShader.render ();

        glViewport(WIDTH, 0, WIDTH, HEIGHT);
        simpleTextureShader.setTextureInput( 0, normals_tex_id);
        simpleTextureShader.useProgram();
        simpleTextureShader.update ();
        simpleTextureShader.render ();


        glViewport(WIDTH * 2, 0, WIDTH, HEIGHT);
        simpleTextureShader.setTextureInput( 0, depth_tex_id);
        simpleTextureShader.useProgram();
        simpleTextureShader.update ();
        simpleTextureShader.render ();

        glViewport(WIDTH * 3, 0, WIDTH, HEIGHT);
        simpleTextureShader.setTextureInput( 0, edge_tex_id);
        simpleTextureShader.useProgram();
        simpleTextureShader.update ();
        simpleTextureShader.render ();

        particleGrid.updateParticleGrid(0.2f, 0.8f);
        particleGrid.renderAllTextures();
        color_tex_id   = particleGrid.getColorTexture();
        normals_tex_id = particleGrid.getNormalTexture();
        depth_tex_id   = particleGrid.getDepthTexture();
        edge_tex_id    = particleGrid.getEdgeTexture();

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}