#include <ModelTracker/ModelTracker.h>
#include <ImageFilter/ImageFilter.h>

#define WIDTH 1280
#define HEIGHT 720

#define PARTICLE_COUNT 1

using namespace sl;

GLFWwindow* window;

int main(int argc, char **argv)
{
    window = initGLWindow(window, WIDTH, HEIGHT, "Test - Render Final", BLACK);
    mt::ParticleGrid particleGrid(RESOURCES_PATH "/simple_cube/simple_cube.obj", WIDTH, HEIGHT, PARTICLE_COUNT);

    const char *color_shader_paths[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    ShaderSimple simple_shader( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, color_shader_paths);

    // Set Matrices
    glm::mat4 view_matrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection_matrix = glm::perspective(glm::radians(40.0f), (float) WIDTH/HEIGHT, 1.0f, 100.0f);

    GLuint view_matrix_handler = glGetUniformLocation(simple_shader.getProgramID(), "viewMatrix");;
    GLuint projection_matrix_handler = glGetUniformLocation(simple_shader.getProgramID(), "projectionMatrix");;

    CVK::Node model("model", RESOURCES_PATH "/simple_cube/simple_cube.obj");

    Camera zed;
    mt::ZedAdapter zedAdapter(zed, RESOLUTION_HD720, argv[1]);

    // zed interopertion
    Mat img_raw  =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);
    Mat img_rgb =  Mat(WIDTH, HEIGHT, MAT_TYPE_8U_C4, MEM_GPU);

    while(!glfwWindowShouldClose( window))
    {
        zed.grab();
        HANDLE_ZED_ERROR(zed.retrieveImage(img_raw, VIEW_LEFT, MEM_GPU));
        filter::convertBGRtoRGB(img_raw, img_rgb);

        zedAdapter.imageToGlTexture(img_rgb);
        zedAdapter.renderImage();

/*        glClear(GL_DEPTH_BUFFER_BIT);

        CVK::State::getInstance()->setShader( &simple_shader);
        simple_shader.useProgram();

        glUniformMatrix4fv(view_matrix_handler, 1, GL_FALSE, value_ptr(view_matrix));
        glUniformMatrix4fv(projection_matrix_handler, 1, GL_FALSE, value_ptr(projection_matrix));

        glViewport(0, 0, WIDTH, HEIGHT);
        model.setModelMatrix(particleGrid.m_particles[0].getModelMatrix());
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        model.render();
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );*/

        particleGrid.renderFirstParticleToScreen();
        particleGrid.update(0.1f, 0.1f);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }


    printf("CLEAN UP... \n");
    img_raw.free();
    img_rgb.free();
    zed.close();
    HANDLE_CUDA_ERROR(cudaDeviceReset());
    printf("DONE \n");

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;
}