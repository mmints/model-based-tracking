#include <cstdio>
#include "General.h"

#define N 100000 // vector size

__global__
void addKernel(int *a, int *b, int *c)
{
    // calculate the index position in grid
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N)
    {
        c[index] = a[index] + b[index];
    }
}

void kernel::add()
{
    int a[N], b[N], c[N];       // voctors of size N
    int *dev_a, *dev_b, *dev_c; // pointer to device memory space

    // Allocate memory space on gpu
    cudaMalloc((void**) &dev_a, N * sizeof(int));
    cudaMalloc((void**) &dev_b, N * sizeof(int));
    cudaMalloc((void**) &dev_c, N * sizeof(int));

    // Just for getting some numbers
    // Fill vectors on CPU
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // copy vectors from CPU to GPU memory
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice); // a is the pointer to a[0]
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

    // call the kernel with as much as possible block with at least 128 threads each (utilize modulo for integer)
    addKernel<<<(N + 127)/128, 128>>>(dev_a, dev_b, dev_c);

    // copy result back from GPU to CPU memory
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // print out the results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    }

    // clear memory on gpu
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}