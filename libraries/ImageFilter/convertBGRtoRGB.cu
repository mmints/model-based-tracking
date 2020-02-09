
#include "ImageFilter.h"

__global__ void convertBGRtoRGBKernel(sl::uchar4 *in, sl::uchar4 *out, size_t step)
{
    // Flat the 2D Coordinates
    uint32_t zed_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t zed_y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

    out[offset].x = in[offset].z;
    out[offset].y = in[offset].y;
    out[offset].z = in[offset].x;
}

void filter::convertBGRtoRGB(const sl::Mat &in_zed, sl::Mat &out_zed)
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

    convertBGRtoRGBKernel<<<dimGrid, dimBlock>>>(in_zed_ptr, out_zed_ptr, step);
}