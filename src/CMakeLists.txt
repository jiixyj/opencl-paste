add_custom_target(hellocl_kernels ALL ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/hellocl_kernels.cl
    ${OpenCL_BINARY_DIR}/hellocl_kernels.cl)

add_executable(poisson poisson_test)
target_link_libraries(poisson ${OPENCL_LIBRARY} ${OpenCV_LIBS} GLEW glut)
