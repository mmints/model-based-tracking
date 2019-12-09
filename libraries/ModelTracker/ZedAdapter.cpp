#include <iostream>
#include "ZedAdapter.h"

void mt::initBasicZedCameraHD720(sl::Camera* zed)
{
    // Init ZED Camera
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION_HD720;
    init_parameters.camera_fps = 30.f;
    sl::ERROR_CODE err = zed->open(init_parameters);

    // ERRCODE display
    if (err != sl::SUCCESS) {
        std::printf((const char *) err);
        zed->close();
    }
}