#ifndef OPENCL_H_
#define OPENCL_H_

#include <GL/glew.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cv.h>

namespace pv {

void init_cl(cl::Context& context_, cl::CommandQueue& queue_, bool with_gl);
cl::Program load_program(cl::Context& context_, std::string program_name);
cv::Mat make_rgba(const cv::Mat& image, cv::Mat alpha = cv::Mat());
// FIXME: Doesn't work for GPU images
void save_cl_image(std::string filename,
                   cl::CommandQueue const& queue,
                   cl::Image2D const& cl_image);

}

#endif
