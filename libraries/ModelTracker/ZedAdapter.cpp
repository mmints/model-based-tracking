#include <iostream>
#include "ZedAdapter.h"




// *** Deprecated *** //
void mt::initBasicZedCameraHD720(sl::Camera &zed)
{
    // Init ZED Camera
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION_HD720;
    init_parameters.camera_fps = 30.f;
    sl::ERROR_CODE err = zed.open(init_parameters);

    // ERRCODE display
    if (err != sl::SUCCESS) {
        std::printf((const char *) err);
        zed.close();
    }
}

void mt::initSVOZedCamera(sl::Camera &zed, const char* path_to_file)
{
    // Init ZED Camera from SVO files
    sl::InitParameters initParameters;
    initParameters.svo_input_filename.set(path_to_file);
    initParameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
    sl::ERROR_CODE err = zed.open(initParameters);

    // ERRCODE display
    if (err != sl::SUCCESS) {
        zed.close();
    }
}

void mt::initZedCamera(sl::Camera &zed, const char *path_to_file) {

    if (path_to_file) {
        printf("##### Load from file \n");
        mt::initSVOZedCamera(zed, path_to_file);
    }
    else {
        printf("###### Load Camera \n");
        mt::initBasicZedCameraHD720(zed);
    }

}
