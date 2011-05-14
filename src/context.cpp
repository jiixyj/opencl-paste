#include "context.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <numeric>

#include <sys/stat.h>
#include <GL/glx.h>

namespace pv {

Context::Context()
          : context_(),
            queue_(),
            source_(),
            target_(),
            pos_x(0),
            pos_y(0),
            main_loop_event_(),
            draw_residual_(false),
            current_grid_(0) {
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
  source_ = make_rgba(source, mask);
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
                     cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
  a2_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
  a3_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                     size_t(source_.cols), size_t(source_.rows)));
  cl_b = cl::Image2D(context_, CL_MEM_READ_WRITE,
                         cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                         size_t(source_.cols), size_t(source_.rows));
  cl_x1 = cl::Image2D(context_, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                   size_t(source_.cols), size_t(source_.rows));
  cl_x2 = cl::Image2D(context_, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                   size_t(source_.cols), size_t(source_.rows));
  cl_residual = cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                     size_t(source_.cols), size_t(source_.rows));

  region_source[0] = size_t(source_.cols);
  region_source[1] = size_t(source_.rows);
  queue_.enqueueWriteImage(cl_source, CL_TRUE,
                          origin, region_source, 0, 0,
                          source_.data);
  reset_image.setArg<cl::Image2D>(0, cl_residual);
  queue_.enqueueNDRangeKernel(
    reset_image,
    cl::NullRange,
    cl::NDRange(cl_residual.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_residual.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
}

void Context::set_target(cv::Mat target) {
  target_ = make_rgba(target);
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
  build_multigrid();
}

void Context::init_cl() {
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform size 0" << std::endl;
      exit(EXIT_FAILURE);
    }
    cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, cl_context_properties((platforms[0])()),
              CL_GLX_DISPLAY_KHR, cl_context_properties(glXGetCurrentDisplay()),
              CL_GL_CONTEXT_KHR,  cl_context_properties(glXGetCurrentContext()),
              0 };
    try {
      context_ = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    } catch (cl::Error) {
      context_ = cl::Context(CL_DEVICE_TYPE_CPU, properties);
    }
    std::vector<cl::Device> devices(context_.getInfo<CL_CONTEXT_DEVICES>());
    queue_ = cl::CommandQueue(context_, devices[0], CL_QUEUE_PROFILING_ENABLE);
    // std::cerr << devices[0].getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  program_ = load_program("hellocl_kernels");
  try {
    setup_system = cl::Kernel(program_, "setup_system", NULL);
    jacobi = cl::Kernel(program_, "jacobi", NULL);
    calculate_residual = cl::Kernel(program_, "calculate_residual", NULL);
    reset_image = cl::Kernel(program_, "reset_image", NULL);
    reduce = cl::Kernel(program_, "reduce", NULL);
    add_images = cl::Kernel(program_, "add_images", NULL);
    bilinear_filter = cl::Kernel(program_, "bilinear_filter", NULL);
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
  int global_size = 1024;
  int local_size = 16;
  int nr_groups = global_size / local_size;

  int nr_pixels = int(cl_source.getImageInfo<CL_IMAGE_WIDTH>()) *
                  int(cl_source.getImageInfo<CL_IMAGE_HEIGHT>());

  cl::Buffer result(context_, CL_MEM_WRITE_ONLY, nr_groups * sizeof(cl_float));

  reduce.setArg<cl::Image2D>(0, cl_residual);
  reduce.setArg<int>(1, nr_pixels);
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
  cl_float result_host[nr_groups];
  queue_.enqueueReadBuffer(result, CL_TRUE, 0,
                           nr_groups * sizeof(cl_float), result_host);
  return std::accumulate(result_host, result_host + nr_groups, .0f) / nr_pixels;
  // return result_host[0];
}

std::pair<int, int> Context::get_gl_size() {
  return std::make_pair(target_.cols, target_.rows);
}

void Context::start_calculation_async(double number_iterations) {
  // Jacobi iterations
  glFinish();
  std::vector<cl::Memory> gl_image{cl_g_render, cl_g_residual};
  queue_.enqueueAcquireGLObjects(&gl_image);
  int number_iterations_int = int(number_iterations / 2 + 0.5) * 2;
  for (int i = 0; i < number_iterations_int; ++i) {
    jacobi.setArg<cl::Image2D>(0, a1_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(1, a2_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(2, a3_stack[current_grid_]);
    jacobi.setArg<cl::Image2D>(3, cl_b);
    jacobi.setArg<cl::Image2D>(4, cl_x1);
    jacobi.setArg<cl::Image2D>(5, cl_x2);
    jacobi.setArg<cl::Image2DGL>(6, cl_g_render);
    jacobi.setArg<int>(7, (i == number_iterations_int - 1) ?
                            (!u_stack.empty() ? 1 : 1) : 0);
    queue_.enqueueNDRangeKernel(
      jacobi,
      cl::NullRange,
      cl::NDRange(cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_x1.getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
    std::swap(cl_x1, cl_x2);
  }
  // residual calculation
  calculate_residual.setArg<cl::Image2D>(0, a1_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(1, a2_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(2, a3_stack[current_grid_]);
  calculate_residual.setArg<cl::Image2D>(3, cl_b);
  calculate_residual.setArg<cl::Image2D>(4, cl_x1);
  calculate_residual.setArg<cl::Image2D>(5, cl_residual);
  calculate_residual.setArg<cl::Image2DGL>(6, cl_g_residual);
  queue_.enqueueNDRangeKernel(
    calculate_residual,
    cl::NullRange,
    cl::NDRange(cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_x1.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
  queue_.enqueueReleaseGLObjects(&gl_image, NULL, &main_loop_event_);
}

void Context::push_residual_stack() {
  ++current_grid_;
  cl::Image2D old_cl_b(context_, CL_MEM_READ_WRITE,
                       cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                       cl_b.getImageInfo<CL_IMAGE_WIDTH>(),
                       cl_b.getImageInfo<CL_IMAGE_HEIGHT>());
  cl::Image2D old_cl_x1(context_, CL_MEM_READ_WRITE,
                        cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                        cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                        cl_x1.getImageInfo<CL_IMAGE_HEIGHT>());

  cl::size_t<3> region_cl_b;
  region_cl_b.push_back(cl_x1.getImageInfo<CL_IMAGE_WIDTH>());
  region_cl_b.push_back(cl_x1.getImageInfo<CL_IMAGE_HEIGHT>());
  region_cl_b.push_back(1);
  queue_.enqueueCopyImage(cl_b, old_cl_b, origin, origin, region_cl_b);
  queue_.enqueueCopyImage(cl_x1, old_cl_x1, origin, origin, region_cl_b);
  u_stack.push(old_cl_x1);
  f_stack.push(old_cl_b);

  cl_b = cl::Image2D(context_, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                     (cl_b.getImageInfo<CL_IMAGE_WIDTH>() + 1) / 2,
                     (cl_b.getImageInfo<CL_IMAGE_HEIGHT>() + 1) / 2);
  bilinear_filter.setArg<cl::Image2D>(0, cl_residual);
  bilinear_filter.setArg<cl::Image2D>(1, cl_b);
  queue_.enqueueNDRangeKernel(
    bilinear_filter,
    cl::NullRange,
    cl::NDRange(cl_b.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_b.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
  cl_residual = cl::Image2D(context_, CL_MEM_READ_WRITE,
                            cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                            cl_b.getImageInfo<CL_IMAGE_WIDTH>(),
                            cl_b.getImageInfo<CL_IMAGE_HEIGHT>());
  reset_image.setArg<cl::Image2D>(0, cl_residual);
  queue_.enqueueNDRangeKernel(
    reset_image,
    cl::NullRange,
    cl::NDRange(cl_residual.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_residual.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );

  cl::Event ev;
  cl_x1 = cl::Image2D(context_, CL_MEM_READ_WRITE,
                      cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                      (cl_x1.getImageInfo<CL_IMAGE_WIDTH>() + 1) / 2,
                      (cl_x1.getImageInfo<CL_IMAGE_HEIGHT>() + 1) / 2);
  cl_x2 = cl::Image2D(context_, CL_MEM_READ_WRITE,
                      cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                      cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                      cl_x1.getImageInfo<CL_IMAGE_HEIGHT>());
  reset_image.setArg<cl::Image2D>(0, cl_x1);
  queue_.enqueueNDRangeKernel(
    reset_image,
    cl::NullRange,
    cl::NDRange(cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_x1.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange,
    NULL, &ev
  );
  ev.wait();
}

void Context::pop_residual_stack() {
  if (!u_stack.empty() && !f_stack.empty()) {
    --current_grid_;
    cl::Image2D old_cl_x1 = u_stack.top();
    cl_b = f_stack.top();
    u_stack.pop();
    f_stack.pop();

    cl::Image2D cl_x1_copy(context_, CL_MEM_READ_WRITE,
                           cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                           old_cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                           old_cl_x1.getImageInfo<CL_IMAGE_HEIGHT>());
    cl_residual = cl::Image2D(context_, CL_MEM_READ_WRITE,
                              cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                              cl_b.getImageInfo<CL_IMAGE_WIDTH>(),
                              cl_b.getImageInfo<CL_IMAGE_HEIGHT>());
    reset_image.setArg<cl::Image2D>(0, cl_residual);
    queue_.enqueueNDRangeKernel(
      reset_image,
      cl::NullRange,
      cl::NDRange(cl_residual.getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_residual.getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
    bilinear_filter.setArg<cl::Image2D>(0, cl_x1);
    bilinear_filter.setArg<cl::Image2D>(1, cl_x1_copy);
    cl::Event ev;
    queue_.enqueueNDRangeKernel(
      bilinear_filter,
      cl::NullRange,
      cl::NDRange(cl_x1_copy.getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_x1_copy.getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange,
      NULL, &ev
    );
    ev.wait();

    cl_x1 = cl::Image2D(context_, CL_MEM_READ_WRITE,
                        cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                        old_cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                        old_cl_x1.getImageInfo<CL_IMAGE_HEIGHT>());
    cl_x2 = cl::Image2D(context_, CL_MEM_READ_WRITE,
                        cl::ImageFormat(CL_RGBA, X_CL_TYPE),
                        cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                        cl_x1.getImageInfo<CL_IMAGE_HEIGHT>());
    add_images.setArg<cl::Image2D>(0, old_cl_x1);
    add_images.setArg<cl::Image2D>(1, cl_x1_copy);
    add_images.setArg<cl::Image2D>(2, cl_x1);
    queue_.enqueueNDRangeKernel(
      add_images,
      cl::NullRange,
      cl::NDRange(cl_x1.getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_x1.getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange,
      NULL, &ev
    );
    ev.wait();
  }
}

void Context::build_multigrid() {
  int current_width = a1_stack[0].getImageInfo<CL_IMAGE_WIDTH>();
  int current_height = a1_stack[0].getImageInfo<CL_IMAGE_HEIGHT>();
  a1_stack.resize(1);
  a2_stack.resize(1);
  a3_stack.resize(1);
  while (current_height != 1 && current_width != 1) {
    current_width = (current_width + 1) / 2;
    current_height = (current_height + 1) / 2;
    a1_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                       cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                       current_width, current_height));
    a2_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                       cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                       current_width, current_height));
    a3_stack.push_back(cl::Image2D(context_, CL_MEM_READ_WRITE,
                       cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT),
                       current_width, current_height));
    bilinear_filter.setArg<cl::Image2D>(0, a1_stack[a1_stack.size() - 2]);
    bilinear_filter.setArg<cl::Image2D>(1, a1_stack.back());
    queue_.enqueueNDRangeKernel(bilinear_filter, cl::NullRange,
      cl::NDRange(a1_stack.back().getImageInfo<CL_IMAGE_WIDTH>(),
                  a1_stack.back().getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
    bilinear_filter.setArg<cl::Image2D>(0, a2_stack[a2_stack.size() - 2]);
    bilinear_filter.setArg<cl::Image2D>(1, a2_stack.back());
    queue_.enqueueNDRangeKernel(bilinear_filter, cl::NullRange,
      cl::NDRange(a2_stack.back().getImageInfo<CL_IMAGE_WIDTH>(),
                  a2_stack.back().getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
    bilinear_filter.setArg<cl::Image2D>(0, a3_stack[a3_stack.size() - 2]);
    bilinear_filter.setArg<cl::Image2D>(1, a3_stack.back());
    queue_.enqueueNDRangeKernel(bilinear_filter, cl::NullRange,
      cl::NDRange(a3_stack.back().getImageInfo<CL_IMAGE_WIDTH>(),
                  a3_stack.back().getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange
    );
  }
}

void Context::setup_new_system(bool initialize) {
  setup_system.setArg<cl::Image2D>(0, cl_source);
  setup_system.setArg<cl::Image2D>(1, cl_target);
  setup_system.setArg<cl::Image2D>(2, a1_stack[0]);
  setup_system.setArg<cl::Image2D>(3, a2_stack[0]);
  setup_system.setArg<cl::Image2D>(4, a3_stack[0]);
  setup_system.setArg<cl::Image2D>(5, cl_b);
  setup_system.setArg<cl::Image2D>(6, cl_x1);
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

void Context::set_offset(int off_x, int off_y) {
  pos_x = off_x;
  pos_y = off_y;
  setup_new_system(false);
  build_multigrid();
}

void Context::get_offset(int& off_x, int& off_y) {
  off_x = pos_x;
  off_y = pos_y;
}

cv::Mat Context::make_rgba(const cv::Mat& image, cv::Mat alpha) {
  if (alpha.empty()) {
    alpha = cv::Mat(cv::Mat::ones(image.size(), CV_8U) * 255);
  }
  static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
  cv::Mat with_alpha(image.size(), CV_8UC4);
  cv::Mat images[] = {image, alpha};
  cv::mixChannels(images, 2, &with_alpha, 1, fromto, 4);
  return with_alpha;
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

cl::Program Context::load_program(std::string program_name) {
  bool load_binary = true;

  time_t so_time = 0;
  time_t cl_time = 1;
  struct stat stat_buf_so;
  struct stat stat_buf_cl;
  int stat_status_so = stat((program_name + ".so").c_str(), &stat_buf_so);
  int stat_status_cl = stat((program_name + ".cl").c_str(), &stat_buf_cl);
  if (!stat_status_so && !stat_status_cl) {
    so_time = stat_buf_so.st_mtime;
    cl_time = stat_buf_cl.st_mtime;
  }

  std::ifstream ifs(program_name + ".so");
  if (!ifs || so_time < cl_time) {
    if (ifs) ifs.close();
    load_binary = false;
    ifs.open(program_name + ".cl");
  }
  std::string src((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());
  cl::Program program;
  std::vector<cl::Device> devices(context_.getInfo<CL_CONTEXT_DEVICES>());
  try {
    if (load_binary) {
      cl::Program::Binaries bins {
        std::make_pair(src.c_str(), src.size())
      };
      program = cl::Program(context_, devices, bins);
    } else {
      cl::Program::Sources source {
        std::make_pair(src.c_str(), src.size())
      };
      program = cl::Program(context_, source);
    }
    std::stringstream options;
    if (!strcmp(devices[0].getInfo<CL_DEVICE_NAME>().c_str(),
                "GeForce 8800 GT")) {
      options << "-D FIX_BROKEN_IMAGE_WRITING";
    }
    program.build(devices, options.str().c_str());

    if (!load_binary) {
      std::string log;
      program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &log);
      std::cerr << log;

      std::vector<size_t> sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
      std::vector<char*> bins = program.getInfo<CL_PROGRAM_BINARIES>(NULL);
      if (bins.size()) {
        std::ofstream out(program_name + ".so");
        std::copy(bins[0], bins[0] + sizes[0],
                  std::ostream_iterator<char>(out));
      }
    }
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    std::string log;
    program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &log);
    std::cerr << log;
    exit(EXIT_FAILURE);
  }
  return program;
}

}
