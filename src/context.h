#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <GL/glew.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cv.h>

namespace pv {

class Context {
 public:
  Context();
  void set_source(cv::Mat source, cv::Mat mask);
  void set_target(cv::Mat target);

  void init_gl();
  void init_cl();

  void start_calculation_async(double number_iterations);
  void wait_for_calculations();
  float get_residual_average();
  std::pair<int, int> get_gl_size();

  void draw_frame();
  void setup_new_system(bool initialize);

  void set_offset(int off_x, int off_y);
  void get_offset(int& off_x, int& off_y);

  bool draw_residual;
 private:
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Event main_loop_event_;

  cv::Mat source_;
  cv::Mat target_;

  cl::size_t<3> origin;
  cl::size_t<3> region_source;
  cl::size_t<3> region_target;


  GLuint g_texture;
  GLuint g_residual;
  GLuint g_target;
  cl::Program program_;
  cl::Kernel jacobi;
  cl::Kernel calculate_residual;
  cl::Kernel setup_system;
  cl::Kernel reset_image;
  cl::Kernel reduce;
  cl::Image2D cl_source;
  cl::Image2D cl_target;
  cl::Image2D cl_b;
  cl::Image2D cl_x1;
  cl::Image2D cl_x2;
  cl::Image2D cl_residual;
  cl::Image2DGL cl_g_render;
  cl::Image2DGL cl_g_residual;
  int pos_x;
  int pos_y;

  // helper functions
  GLuint load_texture(cv::Mat image, int width = -1, int height = -1);
  cv::Mat make_rgba(const cv::Mat& image, cv::Mat alpha = cv::Mat());
  cl::Program load_program(std::string program_name);
};

}

#endif
