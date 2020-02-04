#ifndef HANDLE_ZED_ERROR_H
#define HANDLE_ZED_ERROR_H

#include <sl/Camera.hpp>

static void handleZedError( sl::ERROR_CODE err,
                             const char *file,
                             int line ) {
    if (err != sl::SUCCESS) {
        printf( "ZED ERROR [CODE: %i] in %s at line %d\n", err, file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ZED_ERROR( err ) (handleZedError( err, __FILE__, __LINE__ ))

#endif //HANDLE_ZED_ERROR_H
