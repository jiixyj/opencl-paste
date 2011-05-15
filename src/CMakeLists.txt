add_custom_target(hellocl_kernels ALL ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/hellocl_kernels.cl
    ${OpenCL_BINARY_DIR}/hellocl_kernels.cl)

add_library(opencl_helper opencl)
target_link_libraries(opencl_helper ${OPENCL_LIBRARY})

add_library(pv_context context)
target_link_libraries(pv_context opencl_helper ${OpenCV_LIBS} GLEW)

qt4_automoc(window.cpp glwidget.cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(pv main window glwidget)
target_link_libraries(pv pv_context ${QT_LIBRARIES})
