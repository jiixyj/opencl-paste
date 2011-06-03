#ifndef GL_CONTEXT_H_
#define GL_CONTEXT_H_

#include <memory>

#include "context.h"

namespace pv {

class GLContext {
 public:
  GLContext();

  void init();

  void set_source(cv::Mat source, cv::Mat mask);
  void set_target(cv::Mat target);

  void draw_frame();
  void lock_gl();
  void prepare_images_for_drawing();
  void unlock_gl();

  std::pair<int, int> get_gl_size();
  Solver* solver();
  void toggle_residual_drawing();

 private:
  cl::Context gl_context_;
  cl::CommandQueue queue_;

  cl::Program gl_kernels_;
  cl::Kernel gpu_write_solution;
  cl::Kernel gpu_write_residual;

  std::shared_ptr<Solver> solver_;

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
