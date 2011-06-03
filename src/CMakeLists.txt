add_custom_target(hellocl_kernels ALL ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/hellocl_kernels.cl
    ${OpenCL_BINARY_DIR}/hellocl_kernels.cl)

add_custom_target(gl_kernels ALL ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gl_kernels.cl
    ${OpenCL_BINARY_DIR}/gl_kernels.cl)

add_library(opencl_helper opencl)
target_link_libraries(
  opencl_helper
  ${OPENCL_LIBRARY}
  ${OpenCV_LIBS}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARIES}
)

add_library(pv_context context)
target_link_libraries(pv_context opencl_helper)

add_library(pv_gl_context gl_context)
target_link_libraries(pv_gl_context pv_context)

add_subdirectory(frontend)
add_subdirectory(tests)
