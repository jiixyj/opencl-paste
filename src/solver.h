#ifndef SOLVER_H_
#define SOLVER_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cv.h>

namespace pv {

class Solver {
 public:
  Solver();
  virtual ~Solver() {};

  virtual void set_source(cv::Mat source, cv::Mat mask);
  virtual void set_target(cv::Mat target);

  virtual void init(cl::Context context,
                    cl::CommandQueue queue);

  virtual void set_offset(int off_x, int off_y);
  void get_offset(int& off_x, int& off_y);

  virtual void start_calculation_async(double number_iterations) = 0;
  virtual float get_residual_average() = 0;

  virtual const cl::Image2D& current_solution() = 0;
  virtual const cl::Image2D& current_residual() = 0;

 protected:
  cl::Context context_;
  cl::CommandQueue queue_;

  cv::Mat source_;
  cv::Mat target_;

  cl::Image2D cl_source_;
  cl::Image2D cl_target_;

  cl::size_t<3> origin_;
  cl::size_t<3> region_source_;
  cl::size_t<3> region_target_;

  int pos_x_;
  int pos_y_;
};

}

#endif  // CONTEXT_H_
