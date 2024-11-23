cmake_minimum_required(VERSION 3.10)
project(OpenCV_Project)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_INCLUDE_DIRS /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4)
set(OpenCV_LIBS 
    /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_core.dylib
    /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_imgcodecs.dylib
    /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_highgui.dylib
)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv opencv.cc)

target_link_libraries(opencv ${OpenCV_LIBS})
