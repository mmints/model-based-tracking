#include <cstdint>

#include "ImageFilter.h"

__global__
void sobelFilterKernel(sl::uchar1 *in, sl::uchar1 *out, int width, int height, size_t stepIn) {
    uint32_t x_local = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y_local = blockIdx.y * blockDim.y + threadIdx.y;

    // ignore image boarder
    if (x_local == 0 || x_local == (width-1) || y_local == 0 || y_local == (height-1)) {
        return;
    }

    // Horizontal Gradients
    float pixelValueHorizontal = 0;

    pixelValueHorizontal += (float) in[(x_local -1) + (y_local +1) * stepIn] *  1.f;
    pixelValueHorizontal += (float) in[(x_local -1) + (y_local   ) * stepIn] *  2.f;
    pixelValueHorizontal += (float) in[(x_local -1) + (y_local -1) * stepIn] *  1.f;

    pixelValueHorizontal += (float) in[(x_local +1) + (y_local +1) * stepIn] * -1.f;
    pixelValueHorizontal += (float) in[(x_local +1) + (y_local   ) * stepIn] * -2.f;
    pixelValueHorizontal += (float) in[(x_local +1) + (y_local -1) * stepIn] * -1.f;

    // Vertical Gradient
    float pixelValueVertical = 0;

    pixelValueVertical += (float) in[(x_local -1) + (y_local +1) * stepIn] *  1.f;
    pixelValueVertical += (float) in[(x_local   ) + (y_local +1) * stepIn] *  2.f;
    pixelValueVertical += (float) in[(x_local +1) + (y_local +1) * stepIn] *  1.f;

    pixelValueVertical += (float) in[(x_local -1) + (y_local -1) * stepIn] * -1.f;
    pixelValueVertical += (float) in[(x_local   ) + (y_local -1) * stepIn] * -2.f;
    pixelValueVertical += (float) in[(x_local +1) + (y_local -1) * stepIn] * -1.f;

    float result = sqrtf(powf(pixelValueHorizontal, 2.0) + powf(pixelValueVertical, 2.0));
    out[x_local + y_local * stepIn] = (sl::uchar1) (result);
}

void filter::sobelFilter(const sl::Mat &in_zed, sl::Mat &out_zed)
{
    size_t width = in_zed.getWidth();
    size_t height = in_zed.getHeight();
    size_t step = in_zed.getStep(sl::MEM_GPU);

    sl::uchar4 *in_zed_ptr = in_zed.getPtr<sl::uchar4>(sl::MEM_GPU);
    sl::uchar4 *out_zed_ptr = out_zed.getPtr<sl::uchar4>(sl::MEM_GPU);

    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    sobelFilterKernel<<<dimGrid, dimBlock>>>(in_zed_ptr, out_zed_ptr, width, height, step);
}

// DEPRECATED
void kernel::sobelFilter(sl::uchar1 *d_in, sl::uchar1 *d_out, int width, int height, size_t step)
{
    const size_t BLOCKSIZE_X = 32;
    const size_t BLOCKSIZE_Y = 8;

    dim3 dimBlock{BLOCKSIZE_X,BLOCKSIZE_Y};
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    sobelFilterKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, step);
}