#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <GL/glew.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cv.h>
#include <stack>

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

  void push_residual_stack();
  void pop_residual_stack();

  void set_offset(int off_x, int off_y);
  void get_offset(int& off_x, int& off_y);

  void toggle_residual_drawing() { draw_residual_ = !draw_residual_; }
 private:
  static const int X_CL_TYPE = CL_FLOAT;

  void build_multigrid();

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
  cl::Kernel copy_xyz;
  cl::Kernel add_images;
  cl::Kernel bilinear_filter;
  cl::Image2D cl_source;
  cl::Image2D cl_target;
  std::vector<cl::Image2D> a1_stack;
  std::vector<cl::Image2D> a2_stack;
  std::vector<cl::Image2D> a3_stack;
  std::stack<cl::Image2D> u_stack;
  std::stack<cl::Image2D> f_stack;
  cl::Image2D cl_a1;
  cl::Image2D cl_a2;
  cl::Image2D cl_a3;
  cl::Image2D cl_b;
  cl::Image2D cl_x1;
  cl::Image2D cl_x2;
  cl::Image2D cl_residual;
  cl::Image2DGL cl_g_render;
  cl::Image2DGL cl_g_residual;
  bool draw_residual_;
  int current_grid_;
  int pos_x;
  int pos_y;

  // helper functions
  GLuint load_texture(cv::Mat image, int width = -1, int height = -1);
  cv::Mat make_rgba(const cv::Mat& image, cv::Mat alpha = cv::Mat());
  cl::Program load_program(std::string program_name);
};

}

#endif
