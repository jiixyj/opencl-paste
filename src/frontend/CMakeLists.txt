qt4_automoc(window.cpp glwidget.cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(pv main window glwidget)
target_link_libraries(pv pv_gl_context ${QT_LIBRARIES})
