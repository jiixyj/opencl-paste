#include <array>
#include <iostream>
#include <iterator>
#include <fstream>

#include <GL/glew.h>

#include "./helper.cpp"

std::vector<float> image_average() {
  float average[4];

  glGenerateMipmap(GL_TEXTURE_2D);

  int width, height;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
  int max_level = int(std::floor(std::log2(std::max(width, height))));
  glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_HEIGHT, &height);
  assert(width == 1 && height == 1);
  glGetTexImage(GL_TEXTURE_2D, max_level, GL_RGBA, GL_FLOAT, average);
  return std::vector<float>(average, average + 4);
}

int old_pos_x = 0;
int old_pos_y = 0;
int pos_x = 0;
int pos_y = 0;

bool draw_residual = false;

cl::Event event;

void display() {
  static double number_iterations = 10.0;
  static size_t frame_count = 0;
  static int current_time;
  // TODO: static int previous_time = glutGet(GLUT_ELAPSED_TIME);
  static int time_interval;
  static float fps;
  static float wanted_fps = 30.0f;

  // Wait for kernel calculations from last frame to finish
  try {
    event.wait();
  } catch (cl::Error err) { /* ignore */ }

  // Calculate average of residual image
  glBindTexture(GL_TEXTURE_2D, g_residual);
  std::vector<float> average = image_average();

  // draw image
  square();
  // TODO: glutSwapBuffers();

  // Jacobi iterations
  glFinish();
  std::vector<cl::Memory> gl_image{*cl_render_ptr, *cl_g_residual_ptr};
  queue.enqueueAcquireGLObjects(&gl_image);
  int number_iterations_int = int(number_iterations + 0.5);
  for (int i = 0; i < number_iterations_int; ++i) {
    jacobi.setArg<cl::Image2D>(1, *cl_image_ptr_1);
    jacobi.setArg<cl::Image2D>(2, *cl_image_ptr_2);
    jacobi.setArg<cl::Image2DGL>(3, *cl_render_ptr);
    jacobi.setArg<int>(4, i == number_iterations_int - 1);
    queue.enqueueNDRangeKernel(
      jacobi,
      cl::NullRange,
      cl::NDRange(cl_render_ptr->getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_render_ptr->getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange,
      NULL, NULL
    );
    std::swap(cl_image_ptr_1, cl_image_ptr_2);
  }
  // residual calculation
  if (draw_residual) {
    calculate_residual.setArg<cl::Image2D>(1, *cl_image_ptr_1);
    calculate_residual.setArg<cl::Image2D>(2, *cl_res_ptr);
    calculate_residual.setArg<cl::Image2DGL>(3, *cl_g_residual_ptr);
    queue.enqueueNDRangeKernel(
      calculate_residual,
      cl::NullRange,
      cl::NDRange(cl_image_ptr_1->getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_image_ptr_1->getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange,
      NULL, NULL
    );
  }
  queue.enqueueReleaseGLObjects(&gl_image, NULL, &event);

  // adjust frame rate
  frame_count++;
  // TODO: current_time = glutGet(GLUT_ELAPSED_TIME);
  // TODO: time_interval = current_time - previous_time;
  // TODO: if (time_interval > 200) {
  // TODO:   fps = float(frame_count) / (float(time_interval) / 1000.0f);
  //   std::cerr << "FPS: " << fps << " "
  //             << "iterations: " << number_iterations << " "
  //             << "avg: ";
  //   std::for_each(average.begin(),
  //                 average.end(),
  //                 [] (float a) { std::cout << a << " "; });
  //   std::cout << std::endl;

  //   if (fps >= wanted_fps) {
  //     number_iterations += fps - wanted_fps;
  //   } else {
  //     number_iterations -= wanted_fps - fps;
  //   }
  //   if (number_iterations < 10) number_iterations = 10;
  //   previous_time = current_time;
  //   frame_count = 0;
  // }
}

bool button_pressed = false;
int ww, wh;
int old_x, old_y;
// TODO: 
// void mouseEvent(int button, int state, int x, int y) {
//   if (button == GLUT_LEFT_BUTTON) {
//     y = wh - y;
//     if (state == GLUT_UP) {
//       button_pressed = false;
//       old_pos_x = pos_x;
//       old_pos_y = pos_y;
//     } else {
//       button_pressed = true;
//       old_x = x;
//       old_y = y;
//     }
//   }
// }
// 
// void setup_new_system(bool initialize);
// void mouseMoveEvent(int x, int y) {
//   if (button_pressed) {
//     y = wh - y;
//     pos_x = old_pos_x + x - old_x;
//     pos_y = old_pos_y + y - old_y;
//     setup_new_system(false);
//   }
// }
// void keyboardEvent(unsigned char key, int x, int y) {
//   switch (key) {
//     case 'q':
//     case '':
//       exit(0);
//       break;
//     case 'r':
//       draw_residual = !draw_residual;
//       break;
//   }
// }



int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "syntax: <exe> source.png mask.png target.png" << std::endl;
    return 1;
  }
  th = target.rows;
  tw = target.cols;

  // Init OpenGL
  // glutInit(&argc, argv);
  // glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  // ww = target.cols;
  // wh = target.rows;
  // glutInitWindowSize(ww, wh);
  // glutInitWindowPosition(100, 100);
  // glutCreateWindow("Poissonviz");
  // glutDisplayFunc(display);
  // glutIdleFunc(display);
  // glutMouseFunc(mouseEvent);
  // glutMotionFunc(mouseMoveEvent);
  // glutKeyboardFunc(keyboardEvent);




  cl::size_t<3> origin;
  origin.push_back(0);
  origin.push_back(0);
  origin.push_back(0);
  cl::size_t<3> region_source;
  region_source.push_back(size_t(source.cols));
  region_source.push_back(size_t(source.rows));
  region_source.push_back(1);
  cl::size_t<3> region_target;
  region_target.push_back(size_t(target.cols));
  region_target.push_back(size_t(target.rows));
  region_target.push_back(1);
  queue.enqueueWriteImage(cl_source, CL_FALSE,
                          origin, region_source, 0, 0,
                          source.data);
  queue.enqueueWriteImage(cl_target, CL_FALSE,
                          origin, region_target, 0, 0,
                          target.data);

  try {
    setup_system = cl::Kernel(program, "setup_system", NULL);
    setup_new_system(true);

    jacobi = cl::Kernel(program, "jacobi", NULL);
    jacobi.setArg<cl::Image2D>(0, cl_b);

    calculate_residual = cl::Kernel(program, "calculate_residual", NULL);
    calculate_residual.setArg<cl::Image2D>(0, cl_b);

  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }
  // TODO: glutMainLoop();
  return EXIT_SUCCESS;
}
