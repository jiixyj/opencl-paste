project(OpenCL)
cmake_minimum_required(VERSION 2.8.0)
set(CMAKE_MODULE_PATH ${OpenCL_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${OpenCL_BINARY_DIR})
include_directories(SYSTEM include)

##### OpenCV
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

##### OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS})

##### Qt
find_package(Qt4 COMPONENTS QtGui QtOpenGL REQUIRED)
include(${QT_USE_FILE})
include_directories(SYSTEM
  ${QT_INCLUDE_DIR}
  ${QT_QTGUI_INCLUDE_DIR}
  ${QT_QTOPENGL_INCLUDE_DIR}
)

########## Compiler setup
if(NOT CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()

include("${CMAKE_MODULE_PATH}/CompilerFlags.txt")  # This sets VALID_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ${VALID_FLAGS}")

add_subdirectory(src)
