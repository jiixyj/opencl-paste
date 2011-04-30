#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <array>
#include <iostream>

#include <cv.h>
#include <highgui.h>

void init_cl(cl::Context& context,
             cl::Program& program,
             cl::CommandQueue& queue,
             std::string const& src) {
  cl_int err = CL_SUCCESS;
  std::vector<cl::Device> devices;
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform size 0" << std::endl;
      exit(EXIT_FAILURE);
    }
    cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
    devices = std::vector<cl::Device>(context.getInfo<CL_CONTEXT_DEVICES>());

    queue = cl::CommandQueue(context, devices[0], 0, &err);

    cl::Program::Sources source {
      std::make_pair(src.c_str(), src.size())
    };
    program = cl::Program(context, source);
    program.build(devices);
    std::string log;
    program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &log);
    std::cerr << log;
  } catch (cl::Error err) {
    std::cerr << "ERROR: "
              << err.what()
              << "(" << err.err() << ")"
              << std::endl;
    std::string log;
    program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &log);
    std::cerr << log;
    exit(EXIT_FAILURE);
  }
}

#define STRINGIFY(a) #a

std::string helloStr = STRINGIFY(

// #pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void hello(__read_only image2d_t in,
                    __write_only image2d_t out) {

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_FILTER_NEAREST |
                            CLK_ADDRESS_CLAMP_TO_EDGE;
  const int kernel_size = 3;

  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  uint4 pixel = (uint4)(0);
  for (int i = -kernel_size + coord.y; i <= kernel_size + coord.y; ++i) {
    for (int j = -kernel_size + coord.x; j <= kernel_size + coord.x; ++j) {
      pixel += read_imageui(in, sampler, (int2)(j, i));
    }
  }
  pixel /= 49;
  write_imageui(out, (int2)(coord.x, coord.y), pixel);
}

);

cv::Mat add_alpha_channel(const cv::Mat& image) {
  static int fromto[] = {0, 0,  1, 1,  2, 2,  3, 3};
  cv::Mat with_alpha(image.size(), CV_8UC4);
  cv::Mat images[] = {image, cv::Mat::ones(image.size(), CV_8U) * 255};
  cv::mixChannels(images, 2, &with_alpha, 1, fromto, 4);
  return with_alpha;
}

int main() {
  cl_int err = CL_SUCCESS;

  // Init OpenCL
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
  init_cl(context, program, queue, helloStr);

  // Load image
  cv::Mat image = cv::imread("RGB.png");
  cv::Mat with_alpha = add_alpha_channel(image);

  cl::Image2D cl_img_i(context, CL_MEM_READ_ONLY,
                       cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                       image.rows, image.cols);
  cl::Image2D cl_img_o(context, CL_MEM_WRITE_ONLY,
                       cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                       image.rows, image.cols);

  cl::size_t<3> origin;
  origin.push_back(0);
  origin.push_back(0);
  origin.push_back(0);
  cl::size_t<3> region;
  region.push_back(image.rows);
  region.push_back(image.cols);
  region.push_back(1);
  queue.enqueueWriteImage(cl_img_i, CL_TRUE,
                          origin, region, 0, 0,
                          with_alpha.data);

  cl::Kernel kernel(program, "hello", &err);
  kernel.setArg<cl::Image2D>(0, cl_img_i);
  kernel.setArg<cl::Image2D>(1, cl_img_o);

  cl::Event event;
  queue.enqueueNDRangeKernel(kernel,
                             cl::NullRange,
                             cl::NDRange(image.rows, image.cols),
                             cl::NullRange,
                             NULL, &event);
  event.wait();

  cv::Mat out_mat(image.size(), CV_8UC4);
  queue.enqueueReadImage(cl_img_o, CL_TRUE,
                         origin, region, 0, 0,
                         out_mat.data);

  // Write result image
  cv::imwrite("1.png", out_mat);

  return EXIT_SUCCESS;
}
