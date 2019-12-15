#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>

#define WIDTH 512
#define HEIGHT 512

GLFWwindow* window = nullptr;

//define Camera (Trackball)
CVK::Trackball* cam_trackball  = nullptr;
CVK::Perspective projection( glm::radians(60.0f), WIDTH / (float) HEIGHT, 0.1f, 10.f);

void initCamera()
{
    cam_trackball = new CVK::Trackball(window, WIDTH, HEIGHT, &projection);
    glm::vec3 center(0.0f, 0.0f, 0.0f);
    cam_trackball->setCenter(&center);
    cam_trackball->setRadius(5.0f);
    CVK::State::getInstance()->setCamera(cam_trackball);
}

// resize the scene it the window is scaled
void resizeCallback( GLFWwindow *window, int w, int h)
{
    cam_trackball->setWidthHeight(w, h);
    cam_trackball->getProjection()->updateRatio(w / (float) h);
    glViewport( 0, 0, w, h);
}

CVK::Node *cube_node = nullptr;
CVK::Node *screen_filling_quad = nullptr;
void initScene()
{
    cube_node = new CVK::Node("rubiksCube", RESOURCES_PATH "/rubiks_cube/rubiks_cube.obj");
    cube_node->setModelMatrix(glm::scale(glm::mat4(1.f), glm::vec3(0.3f)));

    CVK::Plane quad(glm::vec3(-1.f, 1.f, 0.f),glm::vec3(-1.f, -1.f, 0.f), glm::vec3(1.f, -1.f, 0.f), glm::vec3(1.f, 1.f, 0.f));
    screen_filling_quad = new CVK::Node("screen_filling_quad");
    screen_filling_quad->setGeometry(&quad);

}

int main()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH, HEIGHT, "CVK_2 intro", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeCallback( window, resizeCallback);
    glfwSwapInterval(1); // vsync
    glewInit();

    CVK::State::getInstance()->setBackgroundColor(GRAY);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);

    CVK::FBO fbo( WIDTH, HEIGHT, 1, true );

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture );

    initCamera();
    initScene();

    glEnable(GL_DEPTH_TEST);

    double time = glfwGetTime();
    while(!glfwWindowShouldClose( window))
    {
        double deltaTime = glfwGetTime() - time;
        time = glfwGetTime();

        // First Renderpass to FBO
        fbo.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        cam_trackball->update(deltaTime);
        CVK::State::getInstance()->setShader( &shaderSimple);
        shaderSimple.update();
        cube_node->render();
        glFinish ( );
        fbo.unbind ( );

        // Second Renderpass to Screen Filling Quad
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        simpleTextureShader.setTextureInput ( 0, fbo.getColorTexture ( 0 ) );
        simpleTextureShader.useProgram ( );
        simpleTextureShader.update ( );
        simpleTextureShader.render ( );


        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    delete cam_trackball;
    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}