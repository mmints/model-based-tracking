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
void initScene()
{
    cube_node = new CVK::Node("rubiksCube", RESOURCES_PATH "/simple_cube/simple_cube.obj");
    cube_node->setModelMatrix(glm::scale(glm::mat4(1.f), glm::vec3(1.f)));
}

int main()
{
    glfwInit();
    CVK::useOpenGL33CoreProfile();
    window = glfwCreateWindow(WIDTH, HEIGHT, "Sobel on Cube", nullptr, nullptr);
    glfwSetWindowPos( window, 100, 50);
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeCallback( window, resizeCallback);
    glfwSwapInterval(1); // vsync
    glewInit();

    CVK::State::getInstance()->setBackgroundColor(GRAY);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);

    CVK::FBO fbo( WIDTH, HEIGHT, 1, true );
    CVK::FBO fbo2( WIDTH, HEIGHT, 1, true );


    // Simple rasterization of the 3D Model
    const char *shadernames[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple shaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernames);

    // Post-processing: Sobel
    const char *shadernamesSobelShader[2] = {SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SobelFilter.frag"};
    CVK::ShaderSimpleTexture sobelShader( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, shadernamesSobelShader);

    // Render texture to screen
    const char *shadernamesDisplayTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture displayTexture( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesDisplayTexture );

    initCamera();
    initScene();

    glEnable(GL_DEPTH_TEST);

    double time = glfwGetTime();
    while(!glfwWindowShouldClose( window))
    {
        double deltaTime = glfwGetTime() - time;
        time = glfwGetTime();

        // Render Model into a FBO
        fbo.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        cam_trackball->update(deltaTime);
        CVK::State::getInstance()->setShader( &shaderSimple);
        shaderSimple.update();
        cube_node->render();
        glFinish();
        fbo.unbind();

        // Execute post processing sobel filer shader and render back into FBO
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        sobelShader.setTextureInput(0,fbo.getColorTexture(0));
       // fbo2.bind();
        sobelShader.useProgram();
        sobelShader.update();
        sobelShader.render();
      //  fbo2.unbind();


/*        // Render Texture into a screen filling quad for visualization
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        displayTexture.setTextureInput(0, fbo2.getColorTexture(0));
        displayTexture.useProgram();
        displayTexture.update();
        displayTexture.render();*/

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    delete cam_trackball;
    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}