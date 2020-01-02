#include <sl/Camera.hpp>

#ifndef MODEL_BASED_TRACKER_ZED_HELPER_H
#define MODEL_BASED_TRACKER_ZED_HELPER_H

namespace mt // model based tracking
{
    void initZedCamera(sl::Camera &zed, const char* path_to_file);

    void initBasicZedCameraHD720(sl::Camera &zed);
    void initSVOZedCamera(sl::Camera &zed, const char* path_to_file);

}

#endif //MODEL_BASED_TRACKER_ZED_HELPER_H
