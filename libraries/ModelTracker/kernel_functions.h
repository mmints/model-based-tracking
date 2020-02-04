#ifndef MT_KERNEL_FUNCTIONS_H
#define MT_KERNEL_FUNCTIONS_H

/**
 * This is the collection of all kernel functions
 * that are in use in the ModelTracker.
 */

#include <cuda_runtime.h>
#include <sl/Camera.hpp>

namespace mt
{
    void calculateWeight(const sl::Mat &in_zed);
}

#endif //MT_KERNEL_FUNCTIONS_H
