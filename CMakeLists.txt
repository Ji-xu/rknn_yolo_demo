cmake_minimum_required(VERSION 3.10)
project(rknn_yolo_demo)
set(CMAKE_CXX_STANDARD 14)

## ---- 输出目录设置 ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(RKNN_DIR ${CMAKE_SOURCE_DIR})
include_directories(
    ${RKNN_DIR}/include
    ${RKNN_DIR}/include/import
)

find_package(OpenCV REQUIRED)

add_executable(demo src/main.cpp src/yolo.cpp)
target_link_libraries(demo
    ${OpenCV_LIBS}
    ${RKNN_DIR}/lib/librknnrt.so
    ${RKNN_DIR}/lib/librga.so
)
