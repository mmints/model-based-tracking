#include "ImageFilter.h"

__global__ void normalizeColorKernel(sl::uchar4 *in, sl::uchar4 *out, size_t step)
{
    // Flat the 2D Coordinates
    uint32_t zed_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t zed_y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = zed_x + zed_y * step; // Flat coordinate to memory space

    sl::uchar4 pixel_color = in[offset];

    float s = (float)(pixel_color.x + pixel_color.y + pixel_color.z);

    float blue_n = pixel_color.x / s;
    float green_n = pixel_color.y / s;
    float red_n = pixel_color.z / s;

    out[offset].x = sl::uchar1 (blue_n * 255);
    out[offset].y = sl::uchar1 (green_n * 255);
    out[offset].z = sl::uchar1 (red_n * 255);
}

void filter::normalizeColor(const sl::Mat &in_zed, sl::Mat &out_zed)
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

    normalizeColorKernel<<<dimGrid, dimBlock>>>(in_zed_ptr, out_zed_ptr, step);
}