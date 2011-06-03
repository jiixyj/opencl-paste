#include "opencl.h"
#include "context.h"

#include <iostream>
#include <numeric>

namespace pv {

SolverContext::SolverContext() :
    context_(),
    queue_(),
    source_(),
    target_(),
    origin(),
    region_source(),
    region_target(),
    program_(),
    jacobi(),
    calculate_residual(),
    setup_system(),
    reset_image(),
    reduce(),
    copy_xyz(),
    add_images(),
    bilinear_interp(),
    bilinear_restrict(),
    cl_source(),
    cl_target(),
    b_stack(),
    x1_stack(),
    x2_stack(),
    residual_stack(),
    current_grid_(),
    pos_x(),
    pos_y() {
  origin.push_back(0);
  origin.push_back(0);
  origin.push_back(0);
  region_source.push_back(0);
  region_source.push_back(0);
  region_source.push_back(1);
  region_target.push_back(0);
  region_target.push_back(0);
  region_target.push_back(1);
}

void SolverContext::set_source(cv::Mat source, cv::Mat mask) {
  source_ = pv::make_rgba(source, mask);
  cv::flip(source_, source_, 0);

  cl_source = cl::Image2D(context_, CL_MEM_READ_ONLY,
                              cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                              size_t(source_.cols), size_t(source_.rows));
  b_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
  x1_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                   size_t(source_.cols), size_t(source_.rows)));
  x2_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                   size_t(source_.cols), size_t(source_.rows)));
  residual_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                     size_t(source_.cols), size_t(source_.rows)));

  region_source[0] = size_t(source_.cols);
  region_source[1] = size_t(source_.rows);
  queue_.enqueueWriteImage(cl_source, CL_TRUE,
                           origin, region_source, 0, 0,
                           source_.data);
  launch_reset_image(false, residual_stack[0]);
  launch_reset_image(false, x1_stack[0]);
  launch_reset_image(false, x2_stack[0]);
  launch_reset_image(true, b_stack[0]);
}

void SolverContext::set_target(cv::Mat target) {
  target_ = pv::make_rgba(target);
  cv::flip(target_, target_, 0);

  cl_target = cl::Image2D(context_, CL_MEM_READ_ONLY,
                              cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                              size_t(target.cols), size_t(target.rows));
  region_target[0] = size_t(target_.cols);
  region_target[1] = size_t(target_.rows);
  queue_.enqueueWriteImage(cl_target, CL_TRUE,
                          origin, region_target, 0, 0,
                          target_.data);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, target_.cols, 0, target_.rows, 0.5, 100);
  glMatrixMode(GL_MODELVIEW);

  setup_new_system(true);
  build_multigrid(true);
}

void SolverContext::init(cl::Context context,
                         cl::CommandQueue queue) {
  context_ = context;
  queue_ = queue;

  program_ = pv::load_program(context_, "hellocl_kernels");
  try {
    setup_system = cl::Kernel(program_, "setup_system", NULL);
    jacobi = cl::Kernel(program_, "jacobi", NULL);
    calculate_residual = cl::Kernel(program_, "calculate_residual", NULL);
    reset_image = cl::Kernel(program_, "reset_image", NULL);
    reduce = cl::Kernel(program_, "reduce", NULL);
    add_images = cl::Kernel(program_, "add_images", NULL);
    bilinear_interp = cl::Kernel(program_, "bilinear_interp", NULL);
    bilinear_restrict = cl::Kernel(program_, "bilinear_restrict", NULL);
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

float SolverContext::get_residual_average() {
  size_t global_size = 1024;
  size_t local_size = 16;
  size_t nr_groups = global_size / local_size;

  size_t nr_pixels = residual_stack[0].getImageInfo<CL_IMAGE_WIDTH>() *
                     residual_stack[0].getImageInfo<CL_IMAGE_HEIGHT>();

  cl::Buffer result(context_, CL_MEM_WRITE_ONLY, nr_groups * sizeof(cl_float));

  reduce.setArg<cl::Image2D>(0, residual_stack[0]);
  reduce.setArg<cl_ulong>(1, nr_pixels);
  reduce.setArg(2, local_size * sizeof(cl_float), NULL);
  reduce.setArg<cl::Buffer>(3, result);

  cl::Event ev;
  queue_.enqueueNDRangeKernel(
    reduce,
    cl::NullRange,
    cl::NDRange(global_size),
    cl::NDRange(local_size),
    NULL, &ev
  );
  ev.wait();
  std::vector<cl_float> result_host(nr_groups);
  queue_.enqueueReadBuffer(result, CL_TRUE, 0,
                           nr_groups * sizeof(cl_float),
                           &result_host[0]);
  return std::accumulate(result_host.begin(), result_host.end(), .0f) /
         float(nr_pixels);
  // return result_host[0];
}

void SolverContext::jacobi_iterations(int iterations) {
  size_t local_size = 8;
  size_t glob_width = x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>();
  size_t glob_height = x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>();
  glob_width = (glob_width / local_size + bool(glob_width % local_size))
             * local_size;
  glob_height = (glob_height / local_size + bool(glob_height % local_size))
              * local_size;
  for (int i = 0; i < iterations; ++i) {
    jacobi.setArg<cl::Image2D>(0, b_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(1, x1_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(2, x2_stack[current_grid_]);
    jacobi.setArg(3, (local_size + 2) *
                     (local_size + 2) * sizeof(cl_float4), NULL);
    queue_.enqueueNDRangeKernel(
      jacobi,
      cl::NullRange,
      cl::NDRange(glob_width, glob_height),
      cl::NDRange(local_size, local_size)
    );
    std::swap(x1_stack[current_grid_], x2_stack[current_grid_]);
  }
  // residual calculation
  calculate_residual.setArg<cl::Image2D>(0, b_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(1, x1_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(2, residual_stack[current_grid_]);
  queue_.enqueueNDRangeKernel(
    calculate_residual,
    cl::NullRange,
    cl::NDRange(x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
}

void SolverContext::v_cycle(double number_iterations) {
  if (current_grid_ == x1_stack.size() - 1) {
    return;
  }
  jacobi_iterations(1);
  push_residual_stack();
  v_cycle(number_iterations);
  pop_residual_stack();
  jacobi_iterations(1);
}
void SolverContext::start_calculation_async(double number_iterations) {
  // Jacobi iterations
  v_cycle(number_iterations);
}

void SolverContext::push_residual_stack() {
  ++current_grid_;

  bilinear_restrict.setArg<cl::Image2D>(0, residual_stack[current_grid_ - 1]);
  bilinear_restrict.setArg<cl::Image2D>(1, b_stack[current_grid_]);
  queue_.enqueueNDRangeKernel(
    bilinear_restrict,
    cl::NullRange,
    cl::NDRange(b_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                b_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
  launch_reset_image(false, residual_stack[current_grid_]);
  launch_reset_image(false, x1_stack[current_grid_]);
  launch_reset_image(false, x2_stack[current_grid_]);
}

void SolverContext::pop_residual_stack() {
  if (current_grid_ > 0) {
    --current_grid_;

    cl::Image2D cl_x1_copy(context_, CL_MEM_READ_WRITE,
                           cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                           x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                           x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>());

    bilinear_interp.setArg<cl::Image2D>(0, x1_stack[current_grid_ + 1]);
    bilinear_interp.setArg<cl::Image2D>(1, cl_x1_copy);
    cl::Event ev;
    queue_.enqueueNDRangeKernel(
      bilinear_interp,
      cl::NullRange,
      cl::NDRange(cl_x1_copy.getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_x1_copy.getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );

    cl::Image2D cl_current_x1_copy(context_, CL_MEM_READ_WRITE,
                        cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                        x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                        x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>());
    cl::size_t<3> size;
    size.push_back(x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>());
    size.push_back(x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>());
    size.push_back(1);
    queue_.enqueueCopyImage(x1_stack[current_grid_], cl_current_x1_copy, origin, origin, size);

    add_images.setArg<cl::Image2D>(0, cl_current_x1_copy);
    add_images.setArg<cl::Image2D>(1, cl_x1_copy);
    add_images.setArg<cl::Image2D>(2, b_stack[current_grid_]);
    add_images.setArg<cl::Image2D>(3, x1_stack[current_grid_]);
    queue_.enqueueNDRangeKernel(
      add_images,
      cl::NullRange,
      cl::NDRange(x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                  x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
  }
}

void SolverContext::build_multigrid(bool initialize) {
  size_t current_width = b_stack[0].getImageInfo<CL_IMAGE_WIDTH>();
  size_t current_height = b_stack[0].getImageInfo<CL_IMAGE_HEIGHT>();
  if (initialize) {
    b_stack.resize(1);
    x1_stack.resize(1);
    x2_stack.resize(1);
    residual_stack.resize(1);
  }
  while (current_height != 1 && current_width != 1) {
    current_width = (current_width + 1) / 2;
    current_height = (current_height + 1) / 2;
    if (initialize) {
      b_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                                    cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                    current_width, current_height));
      x1_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                                     cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                                     current_width, current_height));
      x2_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                                     cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                                     current_width, current_height));
      residual_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                                           cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                           current_width, current_height));
    }
  }
}

void SolverContext::setup_new_system(bool initialize) {
  setup_system.setArg<cl::Image2D>(0, cl_source);
  setup_system.setArg<cl::Image2D>(1, cl_target);
  setup_system.setArg<cl::Image2D>(2, b_stack[0]);
  setup_system.setArg<cl::Image2D>(3, x1_stack[0]);
  setup_system.setArg<cl_int>(4, pos_x);
  setup_system.setArg<cl_int>(5, pos_y);
  setup_system.setArg<cl_int>(6, initialize);

  queue_.enqueueNDRangeKernel(
    setup_system,
    cl::NullRange,
    cl::NDRange(cl_source.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_source.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
}

void SolverContext::launch_reset_image(bool block, cl::Image2D image) {
  cl::Event ev;
  reset_image.setArg<cl::Image2D>(0, image);
  queue_.enqueueNDRangeKernel(
    reset_image,
    cl::NullRange,
    cl::NDRange(image.getImageInfo<CL_IMAGE_WIDTH>(),
                image.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange,
    NULL, &ev
  );
  if (block) {
    ev.wait();
  }
}

void SolverContext::set_offset(int off_x, int off_y) {
  pos_x = off_x;
  pos_y = off_y;
  setup_new_system(false);
  build_multigrid(false);
}

void SolverContext::get_offset(int& off_x, int& off_y) {
  off_x = pos_x;
  off_y = pos_y;
}

}
