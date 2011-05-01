project(OpenCL CXX)
cmake_minimum_required(VERSION 2.8.0)
set(CMAKE_MODULE_PATH ${OpenCL_SOURCE_DIR}/cmake)
include_directories(SYSTEM include)

##### OpenCV
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

##### OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wfatal-errors")

add_executable(hellocl hellocl)
target_link_libraries(hellocl ${OPENCL_LIBRARIES} ${OpenCV_LIBS})
add_custom_target(hellocl_kernels ALL ${CMAKE_COMMAND} -E copy
    ${OpenCL_SOURCE_DIR}/hellocl_kernels.cl
    ${OpenCL_BINARY_DIR}/hellocl_kernels.cl)
