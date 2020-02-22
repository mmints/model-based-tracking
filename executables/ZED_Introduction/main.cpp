// OpenGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Cuda
#include <cuda_gl_interop.h>

// ZED
#include <sl/Camera.hpp>
#include <Shader/ShaderSimple.h>

using namespace sl;

// Window
const int width = 1280;
const int height = 720;
GLFWwindow* window;

// Resource declaration
GLuint imageTex;

// ZED SDK objects
sl::Camera zed;
sl::Mat gpuLeftImage;

// Cuda resources for CUDA-OpenGL interoperability
cudaGraphicsResource* pcuImageRes;

sl::VIEW view = VIEW_LEFT;

void charCallback (GLFWwindow *window, unsigned int key)
{
    switch (key)
    {
        case 'c':
            view = VIEW_LEFT;
            break;
        case 'd':
            view = VIEW_DEPTH;
            break;
        case 'n':
            view = VIEW_NORMALS;
            break;
    }
}

int main()
{
    glfwInit(); // Initial GLFW object for using GLFW functionalities

    //Init Window
    window = glfwCreateWindow(width, height, "ZED Introduction", NULL, NULL);
    glfwSetWindowPos( window, 50, 50);
    glfwMakeContextCurrent(window);
    glfwSetCharCallback (window, charCallback);
    glewInit();

    // Create shader program that uses the switchRedAndBlue fragment shader
    const char *shadernames[1] = {SHADERS_PATH "/SwitchRedAndBlue.frag"};
    ShaderSimple shaderSimple( FRAGMENT_SHADER_BIT, shadernames);

    // Init ZED Camera
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION_HD720;
    init_parameters.camera_fps = 30.f;
    init_parameters.depth_mode = DEPTH_MODE_PERFORMANCE;
    ERROR_CODE err = zed.open(init_parameters);

    // Set Runtime Parameters
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_FILL;

    // ERRCODE display
    if (err != SUCCESS) {
        zed.close();
        return -1;
    }

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Cuda resources for CUDA-OpenGL interoperability
    cudaError_t err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // If any error are triggered, exit the program
    if (err1 != 0) return -1;

    // Set the uniform variable for texImage (sampler2D) to the texture unit
    glUniform1i(glGetUniformLocation(shaderSimple.getProgramID(), "texImage"), 0);

    while( !glfwWindowShouldClose(window))
    {
        int res = zed.grab(runtime_parameters);

        if (zed.retrieveImage(gpuLeftImage, view, MEM_GPU) == SUCCESS) {
            cudaArray_t ArrIm;
            cudaGraphicsMapResources(1, &pcuImageRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrIm, 0, 0, gpuLeftImage.getPtr<sl::uchar1>(MEM_GPU), gpuLeftImage.getStepBytes(MEM_GPU), gpuLeftImage.getWidth() * sizeof(sl::uchar4), gpuLeftImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuImageRes, 0);
        }

        ////  OpenGL rendering part ////
        glEnable(GL_DEPTH_TEST);
        glLoadIdentity(); // replace the current matrix with the identity matrix (Why?)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glBindTexture(GL_TEXTURE_2D, imageTex);
        shaderSimple.useProgram();

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

    // Clean up
    gpuLeftImage.free();
    zed.close();
    glDeleteProgram(shaderSimple.getProgramID());

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
