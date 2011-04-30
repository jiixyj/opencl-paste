#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <array>
#include <fstream>
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

cv::Mat make_rgba(const cv::Mat& image) {
  static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
  cv::Mat with_alpha(image.size(), CV_8UC4);
  cv::Mat images[] = {image, cv::Mat::ones(image.size(), CV_8U) * 255};
  cv::mixChannels(images, 2, &with_alpha, 1, fromto, 4);
  return with_alpha;
}

int main() {
  // Load image
  cv::Mat image = cv::imread("RGB.png");
  cv::Mat with_alpha = make_rgba(image);

  std::ifstream ifs("hellocl_kernels.cl");
  std::string hellocl_kernels((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());

  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
  init_cl(context, program, queue, hellocl_kernels);


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

  cl::Kernel kernel(program, "hello", NULL);
  kernel.setArg<cl::Image2D>(0, cl_img_i);
  kernel.setArg<cl::Image2D>(1, cl_img_o);
  kernel.setArg<cl_int>(2, 0);

  cl::Event event;
  queue.enqueueNDRangeKernel(kernel,
                             cl::NullRange,
                             cl::NDRange(image.rows, image.cols),
                             cl::NullRange,
                             NULL, &event);
  event.wait();
  queue.enqueueCopyImage(cl_img_o, cl_img_i, origin, origin, region, NULL, &event);
  event.wait();
  kernel.setArg<cl_int>(2, 1);
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
  {
    cv::Mat tmp(image.size(), CV_8UC4);
    static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
    cv::mixChannels(&out_mat, 1, &tmp, 1, fromto, 4);
    out_mat = tmp;
  }
#if 0
  cv::Mat out_mat(image.size(), CV_8UC4);
  cv::blur(with_alpha, out_mat, cv::Size(101, 101));

  {
    cv::Mat tmp(image.size(), CV_8UC4);
    static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
    cv::mixChannels(&out_mat, 1, &tmp, 1, fromto, 4);
    out_mat = tmp;
  }
#endif
  // Write result image
  cv::imwrite("1.png", out_mat);

  return EXIT_SUCCESS;
}
