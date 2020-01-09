#include <iostream>
#include <cuda_runtime_api.h>
#include <ErrorHandling/HANDLE_CUDA_ERROR.h>

int main()
{
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));

    printf("General Information: \n");
    printf("Name:%s \n", prop.name);
    printf("Global Memory: %i \n", prop.totalGlobalMem);

    printf("\n Cuda Specific Information: \n");
    printf("Maximal Threads per Block: %i \n", prop.maxThreadsPerBlock);
    printf("Maximal Grid Size: X:%i, Y:%i, Z:%i \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximal Size of 2D Texture: X:%i, Y:%i \n", prop.maxTexture2D[0], prop.maxTexture2D[0]);

    printf("\n OpenGL Specific Information: \n");
    printf("echo glxinfo -l | grep MAX_TEXTURE \n [...] \n");
    printf("GL_MAX_TEXTURE_BUFFER_SIZE = 134217728 //  The value gives the maximum number of texels allowed in the texel array of a texture buffer object \n");
    printf("MAX_TEXTURE_SIZE = 32768 //  The value gives a rough estimate of the largest texture that the GL can handle \n [...]");
}