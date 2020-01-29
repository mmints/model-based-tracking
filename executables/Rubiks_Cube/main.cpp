#include <CVK_2/CVK_Framework.h>
#include <Shader/ShaderSimple.h>

#define WIDTH 512
#define HEIGHT 512

GLFWwindow* window = nullptr;

//define Camera (Trackball)
CVK::Perspective projection( glm::radians(60.0f), WIDTH / (float) HEIGHT, 0.1f, 10.f);

CVK::Trackball* cam_trackball  = nullptr;
CVK::Material *mat_red = nullptr;
CVK::Material *mat_blue = nullptr;

// resize the scene it the window is scaled
void resizeCallback( GLFWwindow *window, int w, int h)
{
    cam_trackball->setWidthHeight(w, h);
    cam_trackball->getProjection()->updateRatio(w / (float) h);
    glViewport( 0, 0, w, h);
}

void initCamera()
{
    cam_trackball = new CVK::Trackball(window, WIDTH, HEIGHT, &projection);
    glm::vec3 center(0.0f, 0.0f, 0.0f);
    cam_trackball->setCenter(&center);
    cam_trackball->setRadius(5.0f);
    CVK::State::getInstance()->setCamera(cam_trackball);
}

CVK::Node *cube_node = nullptr;
void initScene()
{
    cube_node = new CVK::Node("rubiksCube", RESOURCES_PATH "/simple_cube/simple_cube.obj");
    cube_node->setModelMatrix(glm::scale(glm::mat4(1.f), glm::vec3(1.f)));
}

void charCallback (GLFWwindow *window, unsigned int key)
{
    switch (key)
    {
        case 'W':
            glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
            break;
        case 'w':
            glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
            break;
        case 'r':
            cube_node->setMaterial(mat_red);
            break;
        case 'b':
            cube_node->setMaterial(mat_blue);
            break;
    }
}

int main()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH, HEIGHT, "CVK_2 intro", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glfwSetCharCallback (window, charCallback);
    glfwSetWindowSizeCallback( window, resizeCallback);
    glfwSwapInterval(1); // vsync
    glewInit();

    CVK::State::getInstance()->setBackgroundColor(GRAY);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    CVK::Material *mat_bg = new CVK::Material(1.f, BgCol);
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);
    glEnable(GL_DEPTH_TEST);

    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);
    CVK::State::getInstance()->setShader( &shaderSimple);

    initCamera();
    initScene();

    double time = glfwGetTime();
    while(!glfwWindowShouldClose( window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update Camera
        double deltaTime = glfwGetTime() - time;
        time = glfwGetTime();
        cam_trackball->update(deltaTime);

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        cube_node->setMaterial(mat_red);
        shaderSimple.update();
        cube_node->render();

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);

        cube_node->setMaterial(mat_bg);
        shaderSimple.update();
        cube_node->render();
        glDisable(GL_POLYGON_OFFSET_FILL);



        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    delete cam_trackball;
    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}