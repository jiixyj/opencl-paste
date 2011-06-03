#include "opencl.h"
#include "solver.h"

namespace pv {

Solver::Solver() :
    context_(),
    queue_(),
    source_(),
    target_(),
    origin_(),
    cl_source_(),
    cl_target_(),
    region_source_(),
    region_target_(),
    pos_x_(),
    pos_y_() {
  origin_.push_back(0);
  origin_.push_back(0);
  origin_.push_back(0);
  region_source_.push_back(0);
  region_source_.push_back(0);
  region_source_.push_back(1);
  region_target_.push_back(0);
  region_target_.push_back(0);
  region_target_.push_back(1);
}

void Solver::set_source(cv::Mat source, cv::Mat mask) {
  source_ = pv::make_rgba(source, mask);
  cv::flip(source_, source_, 0);

  cl_source_ = cl::Image2D(context_, CL_MEM_READ_ONLY,
                              cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                              size_t(source_.cols), size_t(source_.rows));

  region_source_[0] = size_t(source_.cols);
  region_source_[1] = size_t(source_.rows);
  queue_.enqueueWriteImage(cl_source_, CL_TRUE,
                           origin_, region_source_, 0, 0,
                           source_.data);
}

void Solver::set_target(cv::Mat target) {
  target_ = pv::make_rgba(target);
  cv::flip(target_, target_, 0);

  cl_target_ = cl::Image2D(context_, CL_MEM_READ_ONLY,
                           cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                           size_t(target.cols), size_t(target.rows));

  region_target_[0] = size_t(target_.cols);
  region_target_[1] = size_t(target_.rows);
  queue_.enqueueWriteImage(cl_target_, CL_TRUE,
                           origin_, region_target_, 0, 0,
                           target_.data);
}

void Solver::init(cl::Context context,
                  cl::CommandQueue queue) {
  context_ = context;
  queue_ = queue;
}

void Solver::set_offset(int off_x, int off_y) {
  pos_x_ = off_x;
  pos_y_ = off_y;
}

void Solver::get_offset(int& off_x, int& off_y) {
  off_x = pos_x_;
  off_y = pos_y_;
}

}
