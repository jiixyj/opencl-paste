#include "opencl.h"
#include "gl_context.h"

#include <iostream>

namespace pv {

GLContext::GLContext()
    : gl_context_(),
      queue_(),
      gl_kernels_(),
      gpu_write_solution(),
      gpu_write_residual(),
      solver_(new SimpleVCycle),
      g_texture(),
      g_residual(),
      g_target(),
      target_width_(),
      target_height_(),
      cl_g_render(),
      cl_g_residual(),
      draw_residual_(false) {
}

void GLContext::init() {
  // Initialize OpenGL
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

  pv::init_cl(gl_context_, queue_, true);
  solver_->init(gl_context_, queue_);

  gl_kernels_ = pv::load_program(gl_context_, "gl_kernels");
  gpu_write_solution = cl::Kernel(gl_kernels_, "gpu_write_solution", NULL);
  gpu_write_residual = cl::Kernel(gl_kernels_, "gpu_write_residual", NULL);
}

void GLContext::set_source(cv::Mat source, cv::Mat mask) {
  g_texture = load_texture(cv::Mat(), source.cols, source.rows);
  g_residual = load_texture(cv::Mat(), source.cols, source.rows);
  cl_g_render = cl::Image2DGL(gl_context_, CL_MEM_WRITE_ONLY,
                              GL_TEXTURE_2D, 0, g_texture);
  cl_g_residual = cl::Image2DGL(gl_context_, CL_MEM_WRITE_ONLY,
                                GL_TEXTURE_2D, 0, g_residual);
  solver_->set_source(source, mask);
}

void GLContext::set_target(cv::Mat target) {
  cv::Mat target_rgba = pv::make_rgba(target);
  cv::flip(target_rgba, target_rgba, 0);

  g_target = load_texture(target_rgba);
  target_width_ = target.cols;
  target_height_ = target.rows;

  solver_->set_target(target);
}

void GLContext::draw_frame() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  int w = int(cl_g_render.getImageInfo<CL_IMAGE_WIDTH>());
  int h = int(cl_g_render.getImageInfo<CL_IMAGE_HEIGHT>());
  int tw = target_width_;
  int th = target_height_;

  int pos_x, pos_y;
  solver_->get_offset(pos_x, pos_y);

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

std::pair<int, int> GLContext::get_gl_size() {
  return std::make_pair(target_width_, target_height_);
}

Solver* GLContext::solver() {
  return solver_.get();
}

void GLContext::toggle_residual_drawing() {
  draw_residual_ = !draw_residual_;
}

void GLContext::lock_gl() {
  glFinish();
  std::vector<cl::Memory> gl_image{cl_g_render, cl_g_residual};
  queue_.enqueueAcquireGLObjects(&gl_image);
}

void GLContext::unlock_gl() {
  std::vector<cl::Memory> gl_image{cl_g_render, cl_g_residual};
  queue_.enqueueReleaseGLObjects(&gl_image);
}

void GLContext::prepare_images_for_drawing() {
  gpu_write_solution.setArg<cl::Image2D>(0, solver_->current_solution());
  gpu_write_solution.setArg<cl::Image2D>(1, cl_g_render);

  queue_.enqueueNDRangeKernel(
    gpu_write_solution,
    cl::NullRange,
    cl::NDRange(cl_g_render.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_g_render.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange,
    NULL, NULL
  );

  gpu_write_residual.setArg<cl::Image2D>(0, solver_->current_residual());
  gpu_write_residual.setArg<cl::Image2D>(1, cl_g_residual);

  queue_.enqueueNDRangeKernel(
    gpu_write_residual,
    cl::NullRange,
    cl::NDRange(cl_g_residual.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_g_residual.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange,
    NULL, NULL
  );
}

GLuint GLContext::load_texture(cv::Mat image, int width, int height) {
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
               !image.empty() ? image.data :
                            std::vector<uint8_t>(width * height * 4, 0).data());

  return texture;
}

}
