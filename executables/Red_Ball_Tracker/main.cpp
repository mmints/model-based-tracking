#include <ModelTracker/ModelTracker.h>
#include <Shader/ShaderSimple.h>

#include <CVK_2/CVK_Framework.h>

#include <ErrorHandling/HANDLE_CUDA_ERROR.h>
#include <cuda_gl_interop.h>
#include <sl/Camera.hpp>

using namespace sl;
using namespace mt;
using namespace std;

// Same Resolution as the target ZED resolution
#define WINDOW_W 1280
#define WINDOW_H 720

GLFWwindow* window;

int main(int argc, char **argv)
{
    window = initGLWindow(window, WINDOW_W, WINDOW_H, "Red Ball Tracker", BLACK);
    printf("[LOG] Initialize GL Window \n");

    Camera zed;
    initZedCamera(zed, argv[1]); // If argv[1] is null than the hardware camera setting will be loaded
    printf("[LOG] Initialize ZED \n");

    const char *shadernamesTextureToScreen [ 2 ] = { SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SimpleTexture.frag" };
    CVK::ShaderSimpleTexture textureToScreen( VERTEX_SHADER_BIT | FRAGMENT_SHADER_BIT, shadernamesTextureToScreen);
    printf("[LOG] textureToScreen is loaded \n");

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    GLuint zed_tex;
    generateGlTexture(zed_tex, WINDOW_W, WINDOW_H);
    cudaGraphicsResource* zed_resource;
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&zed_resource, zed_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaArray *zed_tex_array;
    printf("[LOG] Create and register GL texture for ZED frame \n");

    Mat zed_in_img  =  Mat(WINDOW_W, WINDOW_H, MAT_TYPE_8U_C4, MEM_GPU);

    // Render Loop
    while(!glfwWindowShouldClose( window))
    {
        zed.grab();
        if (zed.retrieveImage(zed_in_img, VIEW_LEFT, MEM_GPU) == SUCCESS) {

            HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &zed_resource, 0));
            HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&zed_tex_array, zed_resource, 0, 0));
            HANDLE_CUDA_ERROR(cudaMemcpy2DToArray(
                    zed_tex_array, 0, 0,
                    zed_in_img.getPtr<sl::uchar1>(MEM_GPU),
                    zed_in_img.getStepBytes(MEM_GPU),
                    zed_in_img.getWidth() * sizeof(sl::uchar4),
                    zed_in_img.getHeight(),
                    cudaMemcpyDeviceToDevice));

            HANDLE_CUDA_ERROR( cudaGraphicsUnmapResources(1, &zed_resource, 0));
        }

        renderZEDTextureToScreen(zed_tex, WINDOW_W, WINDOW_H, textureToScreen);

        glfwSwapBuffers( window);
        glfwPollEvents();
    }

    printf("[LOG] Done! Cleaning up and shutting down \n");

    zed_in_img.free();

    glfwDestroyWindow( window);
    glfwTerminate();
    return 0;

}
