cmake_minimum_required(VERSION 2.8)
IF (${CMAKE_MAJOR_VERSION} GREATER "2")
  cmake_policy(SET CMP0022 OLD)
  cmake_policy(SET CMP0038 OLD)
ENDIF ()

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} LANGUAGES CXX CUDA)

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${ZED_INCLUDE_DIRS}
        ${OpenGL3_INCLUDE_PATH}
        ${GLEW_INCLUDE_PATH}
        ${GLFW3_INCLUDE_PATH}
        ${ASSIMP_INCLUDE_PATH}
        ${GLM_INCLUDE_PATH}
        ${EXTERNAL_LIBRARY_PATHS}
        ${LIBRARIES_PATH}
)

file(GLOB_RECURSE SOURCES *.cpp *.cu)
file(GLOB_RECURSE HEADER *.h *.cuh)

add_definitions(-DSHADERS_PATH="${SHADERS_PATH}")
add_definitions(-DRESOURCES_PATH="${RESOURCES_PATH}")
add_definitions(-DGLFW_INCLUDE_GLCOREARB)
add_definitions(-DGLEW_STATIC)
# glew.h sets __gl_h_ which makes gl3.h think /gl.h (OpenGL 2) is included. Calm that warning:
add_definitions(-DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED)
# ... and really don't include GLU and GL(2)
add_definitions(-DGLEW_NO_GLU)

add_library(${ProjectId} ${SOURCES} ${HEADER})

target_link_libraries(
        ${ProjectId}
        ${ALL_LIBRARIES}
        ${GLFW3_LIBRARIES}
        ${ASSIMP_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${OpenGL3_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${ZED_LIBRARIES}
        ${CUDA_NPP_LIBRARIES_ZED}
)

