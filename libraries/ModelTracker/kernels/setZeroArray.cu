
#include "ModelTracker/kernel_functions.h"

__global__ void setZeroArrayKernel(float *array, int count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < count)
    {
        array[index] = 0.f;

    }
}

void mt::setZeroArray(float *array, int count)
{
    setZeroArrayKernel<<<(count-127)/128, 128>>>(array, count);
}



