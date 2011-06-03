#ifndef CONTEXT_H_
#define CONTEXT_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <stack>
#include <cv.h>

#include "solver.h"

namespace pv {

class SimpleVCycle : public Solver {
 public:
  SimpleVCycle();

  void set_source(cv::Mat source, cv::Mat mask);
  void set_target(cv::Mat target);

  void init(cl::Context context, cl::CommandQueue queue);
  void set_offset(int off_x, int off_y);

  void start_calculation_async(double number_iterations);
  float get_residual_average();

  const cl::Image2D& current_solution() { return x1_stack[0]; }
  const cl::Image2D& current_residual() { return residual_stack[0]; }

 private:
  static const int X_CL_TYPE = CL_FLOAT;

  void jacobi_iterations(int iterations);
  void v_cycle(double number_iterations);
  void setup_new_system(bool initialize);
  void build_multigrid(bool initialize);
  void push_residual_stack();
  void pop_residual_stack();

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
  std::vector<cl::Image2D> b_stack;
  std::vector<cl::Image2D> x1_stack;
  std::vector<cl::Image2D> x2_stack;
  std::vector<cl::Image2D> residual_stack;
  size_t current_grid_;

  // kernel launchers
  void launch_reset_image(bool block, cl::Image2D image);
};

}

#endif
