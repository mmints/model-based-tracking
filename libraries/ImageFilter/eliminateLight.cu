#include "ImageFilter.h"

__global__ void eliminateLightKernel(sl::uchar4 *in, sl::uchar4 *out, size_t step)
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


/* Just keep this stuff, because it might be useful and I will
 * be to lazy to reimplemented this stuff
 *
 * // convert 0..255 to 0..1.f
    float3 pixel_color_f;
    pixel_color_f.x = pixel_color.x / 255.f; // B
    pixel_color_f.y = pixel_color.y / 255.f; // G
    pixel_color_f.z = pixel_color.z / 255.f; // R

    // Calculate HLS values see: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    float h, s, l;

    float max_color = maxColor(pixel_color_f);
    float min_color = minColor(pixel_color_f);

    // Calculate H
    if (max_color == min_color)
        h = 0;

    else if (max_color == pixel_color_f.x) // B
        h = 60.f * (4 + ((pixel_color_f.z - pixel_color_f.y) / (max_color - min_color)));

    else if (max_color == pixel_color_f.y) // G
        h = 60.f * (2 + ((pixel_color_f.x - pixel_color_f.z) / (max_color - min_color)));

    else if (max_color == pixel_color_f.z) // R
        h = 60.f * (((pixel_color_f.x - pixel_color_f.y) / (max_color - min_color)));

    if (h < 0) // Stay in the HSV circle
        h += 360.f;

    // Calculate L
    l = (max_color + min_color) / 2.f;

    // Calculate S
    if (max_color == 0 || min_color == 0)
        s = 0;
    else if (l < 0.5)
        s = (max_color - min_color)/(max_color + min_color);
    else if (l >= 0.5)
        s = (max_color - min_color)/(2.f - (max_color + min_color));


    // eliminate saturation and lightness
    if (l == 1.f) // WHITE
        out[offset] = sl::uchar4(255, 255, 255, 255);

    else if (l == 0.f) // BLACK
        out[offset] = sl::uchar4(0, 0, 0, 255);

    else{

    }*/

}

void filter::eliminateLight(const sl::Mat &in_zed, sl::Mat &out_zed)
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

    eliminateLightKernel<<<dimGrid, dimBlock>>>(in_zed_ptr, out_zed_ptr, step);
}