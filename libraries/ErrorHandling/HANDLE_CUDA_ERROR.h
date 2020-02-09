#ifndef HANDLE_CUDA_ERROR_H
#define HANDLE_CUDA_ERROR_H

#include <iostream>
#include <cuda_runtime_api.h>

static void handleCudaError( cudaError_t err,
                             const char *file,
                             int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_CUDA_ERROR( err ) (handleCudaError( err, __FILE__, __LINE__ ))

#endif //HANDLE_CUDA_ERROR_H
