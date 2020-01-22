#include <cstdint>

#include "ImageFilter.h"

__global__ void simpleRedDetectorKernel(sl::uchar4 *d_in, sl::uchar4 *d_out, size_t step, unsigned char threshold) {

    // Flat the 2D Coordinates to 1D
    uint32_t zed_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t zed_y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = zed_x + zed_y * step;

    sl::uchar4 pixel_color = d_in[offset];
    sl::uchar4 red = sl::uchar4(255, 0, 0, 0);

    // The values from ZED are coming in BGR
    if (pixel_color.z >= threshold && pixel_color.y < 100 && pixel_color.x < 100) {
        d_out[offset] = red;
    }
    else {
        d_out[offset] = sl::uchar4(0, 0, 0, 0);
    }
}

// Returns a red color map in d_out
void kernel::simpleRedDetector(sl::uchar4 *d_in, sl::uchar4 *d_out, unsigned char threshold, int width, int height, size_t step)
{
    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    // TODO: Add HSV transformation kernel to remove lightning artifacts
    simpleRedDetectorKernel<<<dimGrid, dimBlock>>>(d_in, d_out, step, threshold);
}