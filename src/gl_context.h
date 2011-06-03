#ifndef GL_CONTEXT_H_
#define GL_CONTEXT_H_

#include "context.h"

namespace pv {

class GLContext {
 public:
  GLContext();

  void init();

  void set_source(cv::Mat source, cv::Mat mask);
  void set_target(cv::Mat target);

  void draw_frame();
  std::pair<int, int> get_gl_size();

  SolverContext& solver() { return solver_context_; }
  void toggle_residual_drawing() { draw_residual_ = !draw_residual_; }

  void lock_gl() {
    glFinish();
    std::vector<cl::Memory> gl_image{cl_g_render, cl_g_residual};
    queue_.enqueueAcquireGLObjects(&gl_image);
  }
  void unlock_gl() {
    std::vector<cl::Memory> gl_image{cl_g_render, cl_g_residual};
    queue_.enqueueReleaseGLObjects(&gl_image);
  }
  void prepare_images_for_drawing();

 private:
  cl::Context gl_context_;
  cl::CommandQueue queue_;

  cl::Program gl_kernels_;
  cl::Kernel gpu_write_solution;
  cl::Kernel gpu_write_residual;

  SolverContext solver_context_;

  GLuint g_texture;
  GLuint g_residual;
  GLuint g_target;
  int target_width_;
  int target_height_;

  cl::Image2DGL cl_g_render;
  cl::Image2DGL cl_g_residual;
  bool draw_residual_;

  static GLuint load_texture(cv::Mat image, int width = -1, int height = -1);
};

}

#endif  // GL_CONTEXT_H_
