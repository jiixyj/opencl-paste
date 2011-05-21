#include "opencl.h"
#include "context.h"

#include <iostream>
#include <numeric>

namespace pv {

Context::Context() :
    context_(),
    queue_(),
    main_loop_event_(),
    source_(),
    target_(),
    origin(),
    region_source(),
    region_target(),
    g_texture(),
    g_residual(),
    g_target(),
    program_(),
    jacobi(),
    calculate_residual(),
    setup_system(),
    reset_image(),
    reduce(),
    copy_xyz(),
    add_images(),
    nearest_interp(),
    bilinear_interp(),
    bilinear_restrict(),
    cl_source(),
    cl_target(),
    a1_stack(),
    a2_stack(),
    a3_stack(),
    b_stack(),
    x1_stack(),
    x2_stack(),
    residual_stack(),
    cl_g_render(),
    cl_g_residual(),
    draw_residual_(),
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

void Context::set_source(cv::Mat source, cv::Mat mask) {
  source_ = pv::make_rgba(source, mask);
  cv::flip(source_, source_, 0);

  try {
    g_texture = load_texture(cv::Mat(), source_.cols, source_.rows);
    g_residual = load_texture(cv::Mat(), source_.cols, source_.rows);
    cl_g_render = cl::Image2DGL(context_, CL_MEM_WRITE_ONLY,
                                      GL_TEXTURE_2D, 0, g_texture);
    cl_g_residual = cl::Image2DGL(context_, CL_MEM_WRITE_ONLY,
                                          GL_TEXTURE_2D, 0, g_residual);
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  cl_source = cl::Image2D(context_, CL_MEM_READ_ONLY,
                              cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                              size_t(source_.cols), size_t(source_.rows));
  a1_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
  a2_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
  a3_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
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
}

void Context::set_target(cv::Mat target) {
  target_ = pv::make_rgba(target);
  cv::flip(target_, target_, 0);

  g_target = load_texture(target_);

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

void Context::init_cl() {
  pv::init_cl(context_, queue_, true);

  program_ = pv::load_program(context_, "hellocl_kernels");
  try {
    setup_system = cl::Kernel(program_, "setup_system", NULL);
    jacobi = cl::Kernel(program_, "jacobi", NULL);
    calculate_residual = cl::Kernel(program_, "calculate_residual", NULL);
    reset_image = cl::Kernel(program_, "reset_image", NULL);
    reduce = cl::Kernel(program_, "reduce", NULL);
    add_images = cl::Kernel(program_, "add_images", NULL);
    bilinear_interp = cl::Kernel(program_, "bilinear_interp", NULL);
    nearest_interp = cl::Kernel(program_, "nearest_interp", NULL);
    bilinear_restrict = cl::Kernel(program_, "bilinear_restrict", NULL);
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void Context::init_gl() {
  GLenum err = glewInit();
  if (err != GLEW_OK) {
    std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "Status: Using GLEW "
            << glewGetString(GLEW_VERSION) << std::endl;

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glLoadIdentity();
}

void Context::draw_frame() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  int w = int(cl_g_render.getImageInfo<CL_IMAGE_WIDTH>());
  int h = int(cl_g_render.getImageInfo<CL_IMAGE_HEIGHT>());
  int tw = target_.cols;
  int th = target_.rows;

  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, g_target);
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex3i( 0,  0, -10);
  glTexCoord2d(1.0, 0.0); glVertex3i(tw,  0, -10);
  glTexCoord2d(1.0, 1.0); glVertex3i(tw, th, -10);
  glTexCoord2d(0.0, 1.0); glVertex3i( 0, th, -10);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, g_texture);
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex3i(pos_x, pos_y, -9);
  glTexCoord2d(1.0, 0.0); glVertex3i(pos_x + w, pos_y, -9);
  glTexCoord2d(1.0, 1.0); glVertex3i(pos_x + w, pos_y + h, -9);
  glTexCoord2d(0.0, 1.0); glVertex3i(pos_x, pos_y + h, -9);
  glEnd();

  if (draw_residual_) {
    glBlendFunc(GL_ONE, GL_ONE);
  } else {
    glBlendFunc(GL_ZERO, GL_ONE);
  }

  glBindTexture(GL_TEXTURE_2D, g_residual);
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex3i(pos_x, pos_y, -8);
  glTexCoord2d(1.0, 0.0); glVertex3i(pos_x + w, pos_y, -8);
  glTexCoord2d(1.0, 1.0); glVertex3i(pos_x + w, pos_y + h, -8);
  glTexCoord2d(0.0, 1.0); glVertex3i(pos_x, pos_y + h, -8);
  glEnd();
}

void Context::wait_for_calculations() {
  // Wait for kernel calculations from last frame to finish
  try {
    main_loop_event_.wait();
  } catch (cl::Error err) { /* ignore */ }
}

float Context::get_residual_average() {
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

std::pair<int, int> Context::get_gl_size() {
  return std::make_pair(target_.cols, target_.rows);
}

void Context::jacobi_iterations(int iterations) {
  for (int i = 0; i < iterations; ++i) {
    jacobi.setArg<cl::Image2D>(0, a1_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(1, a2_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(2, a3_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(3, b_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(4, x1_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(5, x2_stack[current_grid_]);
    jacobi.setArg<cl::Image2DGL>(6, cl_g_render);
    jacobi.setArg<int>(7, (i == iterations - 1) ?
                            (current_grid_ > 0 ? 0 : 1) : 0);
    queue_.enqueueNDRangeKernel(
      jacobi,
      cl::NullRange,
      cl::NDRange(x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                  x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
    std::swap(x1_stack[current_grid_], x2_stack[current_grid_]);
  }
  // residual calculation
  calculate_residual.setArg<cl::Image2D>(0, a1_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(1, a2_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(2, a3_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(3, b_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(4, x1_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(5, residual_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2DGL>(6, cl_g_residual);
  calculate_residual.setArg<int>(7, current_grid_ == 0);
  queue_.enqueueNDRangeKernel(
    calculate_residual,
    cl::NullRange,
    cl::NDRange(x1_stack[current_grid_].getImageInfo<CL_IMAGE_WIDTH>(),
                x1_stack[current_grid_].getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
}

void Context::v_cycle(double number_iterations) {
  if (current_grid_ == x1_stack.size() - 1) {
    return;
  }
  jacobi_iterations(int(number_iterations / 2.0 + 0.5) * 2);
  push_residual_stack();
  v_cycle(number_iterations);
  pop_residual_stack();
  jacobi_iterations(int(number_iterations / 2.0 + 0.5) * 2);
}
void Context::start_calculation_async(double number_iterations) {
  // Jacobi iterations
  glFinish();
  std::vector<cl::Memory> gl_image{cl_g_render, cl_g_residual};
  queue_.enqueueAcquireGLObjects(&gl_image);
  v_cycle(number_iterations);
  queue_.enqueueReleaseGLObjects(&gl_image, NULL, &main_loop_event_);
}

void Context::push_residual_stack() {
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

void Context::pop_residual_stack() {
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
    add_images.setArg<cl::Image2D>(2, a2_stack[current_grid_]);
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

void Context::build_multigrid(bool initialize) {
  size_t current_width = a1_stack[0].getImageInfo<CL_IMAGE_WIDTH>();
  size_t current_height = a1_stack[0].getImageInfo<CL_IMAGE_HEIGHT>();
  if (initialize) {
    a1_stack.resize(1);
    a2_stack.resize(1);
    a3_stack.resize(1);
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
    std::vector<cl::Image2D>* arr[3] = {&a1_stack, &a2_stack, &a3_stack};
    for (int i = 0; i < 3; ++i) {
      if (initialize) {
        arr[i]->push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                          cl::ImageFormat(CL_RGBA, CL_FLOAT),
                          current_width, current_height));
      }
      nearest_interp.setArg<cl::Image2D>(0, arr[i]->at(a1_stack.size() - 2));
      nearest_interp.setArg<cl::Image2D>(1, arr[i]->back());
      queue_.enqueueNDRangeKernel(nearest_interp, cl::NullRange,
        cl::NDRange(arr[i]->back().getImageInfo<CL_IMAGE_WIDTH>(),
                    arr[i]->back().getImageInfo<CL_IMAGE_HEIGHT>()),
        cl::NullRange
      );
    }
  }
}

void Context::setup_new_system(bool initialize) {
  setup_system.setArg<cl::Image2D>(0, cl_source);
  setup_system.setArg<cl::Image2D>(1, cl_target);
  setup_system.setArg<cl::Image2D>(2, a1_stack[0]);
  setup_system.setArg<cl::Image2D>(3, a2_stack[0]);
  setup_system.setArg<cl::Image2D>(4, a3_stack[0]);
  setup_system.setArg<cl::Image2D>(5, b_stack[0]);
  setup_system.setArg<cl::Image2D>(6, x1_stack[0]);
  setup_system.setArg<cl_int>(7, pos_x);
  setup_system.setArg<cl_int>(8, pos_y);
  setup_system.setArg<cl_int>(9, initialize);

  queue_.enqueueNDRangeKernel(
    setup_system,
    cl::NullRange,
    cl::NDRange(cl_source.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_source.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
}

void Context::launch_reset_image(bool block, cl::Image2D image) {
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

void Context::set_offset(int off_x, int off_y) {
  pos_x = off_x;
  pos_y = off_y;
  setup_new_system(false);
  build_multigrid(false);
}

void Context::get_offset(int& off_x, int& off_y) {
  off_x = pos_x;
  off_y = pos_y;
}

GLuint Context::load_texture(cv::Mat image, int width, int height) {
  std::vector<uint8_t> data(width * height * 4, 0);
  GLuint texture;

  glGenTextures(1, &texture); //generate the texture with the loaded data
  glBindTexture(GL_TEXTURE_2D, texture); //bind the texture to it’s array

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
               !image.empty() ? image.cols : width,
               !image.empty() ? image.rows : height,
               0, GL_RGBA, GL_UNSIGNED_BYTE,
               !image.empty() ? image.data : data.data());

  return texture;
}

}
