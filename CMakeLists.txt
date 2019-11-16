cmake_minimum_required(VERSION 3.15)
project(test)

set(CMAKE_CXX_STANDARD 17)

add_executable(test main.cpp)


set(OpenCV_DIR "C:/Program Files/opencv/build2/install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(test ${OpenCV_LIBS})