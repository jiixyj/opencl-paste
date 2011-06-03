#ifndef CONTEXT_H_
#define CONTEXT_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <stack>
#include <cv.h>

namespace pv {

class SolverContext {
 public:
  SolverContext();
  void set_source(cv::Mat source, cv::Mat mask);
  void set_target(cv::Mat target);

  void init(cl::Context context,
            cl::CommandQueue queue);

  void jacobi_iterations(int iterations);
  void v_cycle(double number_iterations);
  void start_calculation_async(double number_iterations);
  float get_residual_average();

  void setup_new_system(bool initialize);

  void push_residual_stack();
  void pop_residual_stack();

  void set_offset(int off_x, int off_y);
  void get_offset(int& off_x, int& off_y);

  const cl::Image2D& current_solution() const { return x1_stack[0]; };
  const cl::Image2D& current_residual() const { return residual_stack[0]; };

 private:
  static const int X_CL_TYPE = CL_FLOAT;

  void build_multigrid(bool initialize);

  cl::Context context_;
  cl::CommandQueue queue_;

  cv::Mat source_;
  cv::Mat target_;

  cl::size_t<3> origin;
  cl::size_t<3> region_source;
  cl::size_t<3> region_target;

  cl::Program program_;
  cl::Kernel jacobi;
  cl::Kernel calculate_residual;
  cl::Kernel setup_system;
  cl::Kernel reset_image;
  cl::Kernel reduce;
  cl::Kernel copy_xyz;
  cl::Kernel add_images;
  cl::Kernel bilinear_interp;
  cl::Kernel bilinear_restrict;
  cl::Image2D cl_source;
  cl::Image2D cl_target;
  std::vector<cl::Image2D> b_stack;
  std::vector<cl::Image2D> x1_stack;
  std::vector<cl::Image2D> x2_stack;
  std::vector<cl::Image2D> residual_stack;
  size_t current_grid_;
  int pos_x;
  int pos_y;

  // kernel launchers
  void launch_reset_image(bool block, cl::Image2D image);
};

}

#endif
