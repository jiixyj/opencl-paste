project(OpenCL)
cmake_minimum_required(VERSION 2.6.4)
set(CMAKE_MODULE_PATH ${OpenCL_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${OpenCL_BINARY_DIR})
include_directories(SYSTEM include)

##### GLEW
include(FindPkgConfig)
pkg_check_modules(GLEW REQUIRED glew)

##### OpenGL
find_package(OpenGL REQUIRED)

##### OpenCV
find_package(OpenCV REQUIRED)

##### OpenCL
find_package(OpenCL REQUIRED)

##### Qt
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL REQUIRED)
include(${QT_USE_FILE})
include_directories(SYSTEM
  ${QT_INCLUDE_DIR}
  ${QT_QTCORE_INCLUDE_DIR}
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
