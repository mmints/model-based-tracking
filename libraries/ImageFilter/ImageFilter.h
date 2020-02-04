//
// Created by mark on 28.10.19.
//

#ifndef MODEL_BASED_TRACKER_FILTER_H
#define MODEL_BASED_TRACKER_FILTER_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sl/Camera.hpp>


namespace filter
{
    void normalizeColor(const sl::Mat &in_zed, sl::Mat &out_zed);
    void sobelFilter(const sl::Mat &in_zed, sl::Mat &out_zed);
    void meanFilter(const sl::Mat &in_zed, sl::Mat &out_zed, int radius);
}

// DEPRECATED
namespace kernel
{
    void meanFilter(sl::uchar1 *d_in, sl::uchar1 *d_out, int width, int height, size_t step, int radius);
    void sobelFilter(sl::uchar1 *d_in, sl::uchar1 *d_out, int width, int height, size_t step);
    void simpleRedDetector(sl::uchar4 *d_in, sl::uchar4 *d_out, unsigned char threshold, int width, int height, size_t step);

}

#endif //MODEL_BASED_TRACKER_FILTER_H
