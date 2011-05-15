#include "opencl.h"

#include <iostream>
#include <GL/glx.h>

namespace pv {

void init_cl(cl::Context& context_, cl::CommandQueue& queue_) {
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
}

}
