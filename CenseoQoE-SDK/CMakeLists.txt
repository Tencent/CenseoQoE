cmake_minimum_required(VERSION 3.10)
project(CenseoQoE)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++14 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -pthread -fPIC")

############################################################
# find dependent libtorch/opencv/libcurl library
############################################################
set(CMAKE_PREFIX_PATH "./third_party/libtorch" "./third_party/opencv_lib" "./third_party/libcurl" "./third_party/libzip")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(libzip REQUIRED)

include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/)
include_directories(${CMAKE_SOURCE_DIR}/third_party/spdlog/include)
link_directories(${CMAKE_SOURCE_DIR}/third_party/spdlog/lib)

include_directories(${CMAKE_SOURCE_DIR}/third_party/libcurl/include)
link_directories(${CMAKE_SOURCE_DIR}/third_party/libcurl/lib)

include_directories(${CMAKE_SOURCE_DIR}/third_party/libzip/include)
link_directories(${CMAKE_SOURCE_DIR}/third_party/libzip/lib)

############################################################
# find dependent FFmpeg library
############################################################

set(ffmpeg_build_include ${CMAKE_SOURCE_DIR}/third_party/ffmpeg_lib/include)
set(ffmpeg_build_lib ${CMAKE_SOURCE_DIR}/third_party/ffmpeg_lib/lib)

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h PATHS ${ffmpeg_build_include})
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h PATHS ${ffmpeg_build_include})
find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h PATHS ${ffmpeg_build_include})
find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h PATHS ${ffmpeg_build_include})
find_path(SWRESAMPLE_INCLUDE_DIR libswresample/swresample.h PATHS ${ffmpeg_build_include})
find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h PATHS ${ffmpeg_build_include})
find_path(X264_INCLUDE_DIR x264.h PATHS ${ffmpeg_build_include})

find_library(AVCODEC_LIBRARY avcodec PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)
find_library(AVFORMAT_LIBRARY avformat PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)
find_library(AVUTIL_LIBRARY avutil PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)
find_library(AVDEVICE_LIBRARY avdevice PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)
find_library(SWRESAMPLE_LIBRARY swresample PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)
find_library(SWSCALE_LIBRARY swscale PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)
find_library(X264_LIBRARY x264 PATHS ${ffmpeg_build_lib} NO_CMAKE_SYSTEM_PATH)

if (NOT AVCODEC_INCLUDE_DIR OR NOT AVCODEC_INCLUDE_DIR OR NOT AVUTIL_INCLUDE_DIR
        OR NOT AVDEVICE_INCLUDE_DIR OR NOT X264_INCLUDE_DIR)
    message(FATAL_ERROR " ## FFmpeg_INCLUDE_DIR include NOT found ")
endif ()

if (NOT AVCODEC_LIBRARY OR NOT AVFORMAT_LIBRARY OR NOT AVUTIL_LIBRARY
        OR NOT AVDEVICE_LIBRARY OR NOT X264_LIBRARY)
    message(FATAL_ERROR " ## FFmpeg_INCLUDE_DIR library NOT found ")
endif ()

file(GLOB_RECURSE SRC "src/*.cpp" "src/*.c" "src/*.cc")

add_executable(${PROJECT_NAME} main.cpp ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC curl zip ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY}
        ${SWRESAMPLE_LIBRARY} ${SWSCALE_LIBRARY}
        ${AVDEVICE_LIBRARY} ${X264_LIBRARY})
target_include_directories(${PROJECT_NAME} PUBLIC ${AVCODEC_INCLUDE_DIR}
        ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR}
        ${SWRESAMPLE_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR}
        ${AVDEVICE_INCLUDE_DIR} ${X264_INCLUDE_DIR})
