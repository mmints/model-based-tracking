#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include <sl/Camera.hpp>
#include <ModelTracker/ModelTracker.h>
#include <Shader/ShaderSimple.h>
#include <ImageFilter/ImageFilter.h>

using namespace sl;

// Window
const int width = 1280;
const int height = 720;
GLFWwindow* window;

// ZED
Camera zed;
Mat inImage;
Mat tempImage = Mat(width, height, MAT_TYPE_8U_C1, MEM_GPU);
Mat outImage = Mat(width, height, MAT_TYPE_8U_C1, MEM_GPU);
// GL rendering
GLuint imageTex; // map the ZED camera frame onto this texture for displaying
cudaGraphicsResource* pcuImageRes; // Cuda resources for CUDA-OpenGL interoperability

int main(int argc, char **argv)
{
    glfwInit();
    //Init Window
    window = glfwCreateWindow(width, height, "ZED Sobel", NULL, NULL);
    glfwSetWindowPos( window, 50, 50);
    glfwMakeContextCurrent(window);

    // Init ZED with default parameter without depth vision
    mt::initZedCamera(zed, argv[1]);
    glewInit();

    // Create shader program that uses the switchRedAndBlue fragment shader
    const char *shadernames[1] = {SHADERS_PATH "/RedToGrayScale.frag"};
    ShaderSimple shaderSimple( FRAGMENT_SHADER_BIT, shadernames);

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // Use only the red chanel because the input image is a one channel gray scale
    glTexImage2D(GL_TEXTURE_2D, 0,  GL_RED, width, height, 0,  GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Cuda resources for CUDA-OpenGL interoperability
    cudaError_t err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // If any error are triggered, exit the program
    if (err1 != 0) return -1;

    shaderSimple.useProgram();

    while( !glfwWindowShouldClose(window))
    {
        int res = zed.grab();

        if (zed.retrieveImage(inImage, VIEW_LEFT_GRAY, MEM_GPU) == SUCCESS) {
            kernel::meanFilter(inImage.getPtr<sl::uchar1>(MEM_GPU), tempImage.getPtr<sl::uchar1>(MEM_GPU), width, height, inImage.getStep(MEM_GPU), 3);
            kernel::sobelFilter(tempImage.getPtr<sl::uchar1>(MEM_GPU), outImage.getPtr<sl::uchar1>(MEM_GPU), width, height, tempImage.getStep(MEM_GPU));
            cudaArray_t ArrIm;
            cudaGraphicsMapResources(1, &pcuImageRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrIm, 0, 0, outImage.getPtr<sl::uchar1>(MEM_GPU), outImage.getStepBytes(MEM_GPU), outImage.getWidth() * sizeof(sl::uchar1), outImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuImageRes, 0);
        }
        ////  OpenGL rendering part ////
        glEnable(GL_DEPTH_TEST);
        glLoadIdentity(); // replace the current matrix with the identity matrix (Why?)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glBindTexture(GL_TEXTURE_2D, imageTex);

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
    inImage.free();
    outImage.free();
    tempImage.free();
    zed.close();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}