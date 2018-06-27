cmake_minimum_required(VERSION 2.8.3)
project(marker_location)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    )

find_package(OpenCV REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

add_executable(marker_location
  ./src/markerLocation.cpp
  ./src/MarkerDetector.cpp
  ./src/Marker.cpp
)

target_link_libraries(marker_location
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}) 


