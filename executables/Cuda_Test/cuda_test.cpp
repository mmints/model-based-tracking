#include "General.h"

#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    cout << "Available CUDA Devices: " << devCount << endl;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    cout << "\nDevice Properties:" << endl;
    cout << "Name:" << devProp.name << endl;
    cout << "Max Grid Size: " << devProp.maxGridSize[3] << endl;
    cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << endl;

    printf(" ****** \n Further we will calculate sum of 2 vectors on the GPU: \n");

    kernel::add(); // Temporary, just for skipping errors

    return 0;
}
