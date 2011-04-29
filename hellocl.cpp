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

    cl::Program::Sources source{
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

std::string helloStr = R"(

__kernel void hello(__global uchar* in,
                    __global uchar* out) {
  // const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
  //                           CLK_FILTER_NEAREST |
  //                           CLK_ADDRESS_CLAMP;

  // int x = get_global_id(0);
  // int y = get_global_id(1);
  int i = get_global_id(0);
  out[i] = in[i];
  // uint4 pixel;
  // pixel = read_imageui(in, sampler, (int2)(x, y));
  // pixel.s3 = 255;
  // write_imageui(out, (int2)(x, y), pixel);
  // out[y * 300 * 4 + x * 4] = 0;
  // out[y * 300 * 4 + x * 4 + 1] = 0;
  // out[y * 300 * 4 + x * 4 + 2] = 0;
  // out[y * 300 * 4 + x * 4 + 3] = 255;

  // write_imageui(out, (int2)(x, y), (uint4)(0, 0, 0, 0));
  // uint4 pixel = read_imageui(in, sampler, (int2)(x, y));
  // write_imageui(out, (int2)(x, y), pixel);

  // if (y < 150) {
  //   write_imageui(out, (int2)(x, y), (uint4)(255,255,0,255));
  // } else {
  //   write_imageui(out, (int2)(x, y), (uint4)(255,0,255,255));
  // }
}

)";

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
  cv::Mat with_alpha = add_alpha_channel(image).clone();
  std::cerr << image.rows << " " << image.cols << std::endl;

  // cl::Image2D cl_img_i(context, CL_MEM_READ_ONLY,
  //                      cl::ImageFormat(CL_BGRA, CL_UNORM_INT8),
  //                      image.rows, image.cols);
  // cl::Image2D cl_img_o(context, CL_MEM_WRITE_ONLY,
  //                      cl::ImageFormat(CL_BGRA, CL_UNORM_INT8),
  //                      image.rows, image.cols);
  cl::Buffer buffer_in (context, CL_MEM_READ_ONLY,  image.rows * image.cols * 4);
  cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, image.rows * image.cols * 4);
  queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0,
                           image.rows * image.cols * 4, with_alpha.data);

  // cl::size_t<3> origin;
  // origin.push_back(0);
  // origin.push_back(0);
  // origin.push_back(0);
  // cl::size_t<3> region;
  // region.push_back(image.rows);
  // region.push_back(image.cols);
  // region.push_back(1);
  // unsigned char test[300*300*4];
  // for (int i = 0; i < 300*300*4; ++i) {
  //   test[i] = -1;
  //   // if (i % 4 == 2) {
  //   //   test[i] = 255;
  //   // }
  // }
  // queue.enqueueWriteImage(cl_img_i, CL_TRUE,
  //                         origin, region, 0, 0,
  //                         with_alpha.data);

  cl::Kernel kernel(program, "hello", &err);
  kernel.setArg<cl::Buffer>(0, buffer_in);
  kernel.setArg<cl::Buffer>(1, buffer_out);

  cl::Event event;
  queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(image.rows * image.cols * 4),
      cl::NullRange,
      NULL,
      &event);
  event.wait();

  cv::Mat out_mat(image.size(), CV_8UC4);
  queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0,
                          image.rows * image.cols * 4,
                          out_mat.data);

  // Write result image
  cv::imwrite("1.png", out_mat);

  return EXIT_SUCCESS;
}
