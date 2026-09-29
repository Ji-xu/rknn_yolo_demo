cmake_minimum_required(VERSION 3.10)
project(rknn_yolo_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include/import
    ${CMAKE_SOURCE_DIR}/include/inner
)

# 库文件路径
link_directories(
    ${CMAKE_SOURCE_DIR}/lib
)

# 源文件
file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_executable(rknn_yolo_demo ${SRC_FILES})

# 链接库：RKNN 运行时、RGA、pthread、dl
target_link_libraries(rknn_yolo_demo
    rknnrt
    rga
    pthread
    dl
    m
)

# 可选：如果用到 OpenCV，取消下面注释
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# target_link_libraries(rknn_yolo_demo ${OpenCV_LIBS})