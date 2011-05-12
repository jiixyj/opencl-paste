project(OpenCL CXX)
cmake_minimum_required(VERSION 2.6.4)
set(CMAKE_MODULE_PATH ${OpenCL_SOURCE_DIR}/cmake)
include_directories(SYSTEM include)

##### OpenCV
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

##### OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS})
# link_directories("/home/jan/usr/AMD-APP-SDK-v2.4-lnx64/include")
# link_directories("/home/jan/usr/AMD-APP-SDK-v2.4-lnx64/lib/x86_64")

########## Compiler setup
if("${CMAKE_VERSION}" STRGREATER "2.7.9" AND NOT CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()

include("${CMAKE_MODULE_PATH}/CompilerFlags.txt")  # This sets VALID_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ${VALID_FLAGS}")

add_subdirectory(src)
