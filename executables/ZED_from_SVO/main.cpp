#include <CVK_2/CVK_Framework.h>

// OpenGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Cuda
#include <cuda_gl_interop.h>

// ZED
#include <sl/Camera.hpp>
#include <Shader/ShaderSimple.h>
#include <ModelTracker/ZedAdapter.h>

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
// Cuda resources for CUDA-OpenGL interoperability
cudaGraphicsResource* pcuImageRes;

int main(int argc, char **argv)
{
    glfwInit(); // Initial GLFW object for using GLFW functionalities

    //Init Window
    window = glfwCreateWindow(width, height, "ZED from SVO", NULL, NULL);
    glfwSetWindowPos( window, 50, 50);
    glfwMakeContextCurrent(window);
    glfwSetCharCallback (window, charCallback);

    glewInit();

    const char *shadernamesSimpleTexture [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/BGRtoRGBSimpleTexture.frag" };
    CVK::ShaderSimpleTexture simpleTextureShader( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesSimpleTexture);

    // Init ZED Camera or SVO input file
    if (argv[1])
        mt::ZedAdapter zedAdapter(zed, RESOLUTION_HD720, argv[1]);
    else
        mt::ZedAdapter zedAdapter(zed, RESOLUTION_HD720);

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

    while( !glfwWindowShouldClose(window))
    {
        int res = zed.grab();

        if (zed.retrieveImage(gpuLeftImage, view, MEM_GPU) == SUCCESS) {
            cudaArray_t ArrIm;
            cudaGraphicsMapResources(1, &pcuImageRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrIm, 0, 0, gpuLeftImage.getPtr<sl::uchar1>(MEM_GPU), gpuLeftImage.getStepBytes(MEM_GPU), gpuLeftImage.getWidth() * sizeof(sl::uchar4), gpuLeftImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuImageRes, 0);
        }

        ////  OpenGL rendering part ////
        glEnable(GL_DEPTH_TEST);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        simpleTextureShader.setTextureInput(0, imageTex);
        simpleTextureShader.useProgram();
        simpleTextureShader.update();
        simpleTextureShader.renderZED();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    gpuLeftImage.free();
    zed.close();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
