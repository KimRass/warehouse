# CMakeLists.txt
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Chap02)

find_package(Torch REQUIRED)

add_executable(main main.cc)
target_link_libraries(main "${TORCH_LIBRARIES}")
# set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
