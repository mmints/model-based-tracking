#include <cstdint>

#include "ImageFilter.h"

__global__
void meanFilterKernel(sl::uchar1 *in, sl::uchar1 *out, int width, int height, size_t stepIn, int radius)
{
    uint32_t x_local = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y_local = blockIdx.y * blockDim.y + threadIdx.y;

    float pixelValue = 0;
    float pixelCount = 0;

    // get the average of the surrounding pixels
    for (int x = -radius; x < radius + 1; x++)
    {
        for (int y = -radius; y < radius + 1; y++)
        {
            int cur_x = x_local + x;
            int cur_y = y_local + y;

            // image boarder
            if (cur_x > -1 && cur_x < width && cur_y > -1 && cur_y < height)
            {
                pixelValue += in[cur_x + cur_y * stepIn];
                pixelCount++;
            }
        }
    }
    out[x_local + y_local * stepIn]  = (sl::uchar1) (pixelValue / pixelCount);
}

void kernel::meanFilter(sl::uchar1 *d_in, sl::uchar1 *d_out, int width, int height, size_t step, int radius)
{
    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    meanFilterKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, step, radius);
}