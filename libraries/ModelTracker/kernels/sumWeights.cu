#include "ModelTracker/kernel_functions.h"

__global__ void sumWeightKernel(float *color, float *depth, float *normals, float *edge, float *sum, int count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < count)
    {
        sum[index] = color[index] + depth[index] + normals[index] + edge[index];

    }
}

void mt::sumWeights(float *color, float *depth, float *normals, float *edge, float *sum, int count)
{
    sumWeightKernel<<<(count-127)/128, 128>>>(color, depth, normals, edge, sum, count);
}



