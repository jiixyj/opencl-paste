#include <iostream>
#include <iterator>
#include <fstream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cv.h>
#include <highgui.h>

void init_cl(cl::Context& context,
             cl::CommandQueue& queue) {
  cl_int err = CL_SUCCESS;
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform size 0" << std::endl;
      exit(EXIT_FAILURE);
    }
    cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, cl_context_properties((platforms[0])()), 0 };
    try {
      context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    } catch (cl::Error) {
      context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
    }
    std::vector<cl::Device> devices(context.getInfo<CL_CONTEXT_DEVICES>());
    queue = cl::CommandQueue(context, devices[0], 0, &err);
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl::Program load_program(const cl::Context& context,
                         std::string program_name) {
  bool load_binary = true;
  std::ifstream ifs(program_name + ".so");
  if (!ifs) {
    load_binary = false;
    ifs.open(program_name + ".cl");
  }
  std::string src((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());
  cl::Program program;
  std::vector<cl::Device> devices(context.getInfo<CL_CONTEXT_DEVICES>());
  try {
    if (load_binary) {
      cl::Program::Binaries bins {
        std::make_pair(src.c_str(), src.size())
      };
      program = cl::Program(context, devices, bins);
    } else {
      cl::Program::Sources source {
        std::make_pair(src.c_str(), src.size())
      };
      program = cl::Program(context, source);
    }
    program.build(devices);

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

cv::Mat make_rgba(const cv::Mat& image, cv::Mat alpha = cv::Mat()) {
  if (alpha.empty()) {
    alpha = cv::Mat(cv::Mat::ones(image.size(), CV_8U) * 255);
  }
  static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
  cv::Mat with_alpha(image.size(), CV_8UC4);
  cv::Mat images[] = {image, alpha};
  cv::mixChannels(images, 2, &with_alpha, 1, fromto, 4);
  return with_alpha;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "syntax: <exe> source.png mask.png target.png" << std::endl;
    return 1;
  }
  cv::Mat mask = cv::imread(argv[2], 0);
  cv::Mat source = make_rgba(cv::imread(argv[1]), mask);
  cv::Mat target = make_rgba(cv::imread(argv[3]));
  // Load image

  cl::Context context;
  cl::CommandQueue queue;
  init_cl(context, queue);

  cl::Program program = load_program(context, "hellocl_kernels");

  size_t nr_pixels = size_t(source.rows) * size_t(source.cols);
  cl::Buffer cl_a(context, CL_MEM_READ_WRITE, nr_pixels * sizeof(cl_uchar));
  // queue.enqueueWriteBuffer(cl_a, CL_TRUE, 0,
  //                          SIZE * sizeof(float), &in[0]);
  cl::Image2D cl_source(context, CL_MEM_READ_ONLY,
                        cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                        size_t(source.cols), size_t(source.rows));
  cl::Image2D cl_target(context, CL_MEM_READ_ONLY,
                        cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                        size_t(target.cols), size_t(target.rows));
  cl::Image2D cl_b(context, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, CL_FLOAT),
                   size_t(source.cols), size_t(source.rows));

  cl::size_t<3> origin;
  origin.push_back(0);
  origin.push_back(0);
  origin.push_back(0);
  cl::size_t<3> region_source;
  region_source.push_back(size_t(source.cols));
  region_source.push_back(size_t(source.rows));
  region_source.push_back(1);
  cl::size_t<3> region_target;
  region_target.push_back(size_t(target.cols));
  region_target.push_back(size_t(target.rows));
  region_target.push_back(1);
  queue.enqueueWriteImage(cl_source, CL_FALSE,
                          origin, region_source, 0, 0,
                          source.data);
  queue.enqueueWriteImage(cl_target, CL_FALSE,
                          origin, region_target, 0, 0,
                          target.data);

  cl::Kernel kernel(program, "setup_system", NULL);
  kernel.setArg<cl::Image2D>(0, cl_source);
  kernel.setArg<cl::Image2D>(1, cl_target);
  kernel.setArg<cl::Buffer>(2, cl_a);
  kernel.setArg<cl::Image2D>(3, cl_b);

  cl::Event event;
  queue.enqueueNDRangeKernel(kernel,
                             cl::NullRange,
                             cl::NDRange(size_t(source.cols),
                                         size_t(source.rows)),
                             cl::NullRange,
                             NULL, &event);
  event.wait();
  // kernel.setArg<cl::Image2D>(0, cl_img_o);
  // kernel.setArg<cl::Image2D>(1, cl_img_i);
  // kernel.setArg<cl_int>(2, 1);
  // cl::Event event;
  // queue.enqueueNDRangeKernel(kernel,
  //                            cl::NullRange,
  //                            cl::NDRange(size_t(image.rows), size_t(image.cols)),
  //                            cl::NullRange,
  //                            NULL, &event);
  // event.wait();

  // cv::Mat out_mat(image.size(), CV_8UC4);
  // queue.enqueueReadImage(cl_img_i, CL_TRUE,
  //                        origin, region, 0, 0,
  //                        out_mat.data);
  // {
  //   cv::Mat tmp(image.size(), CV_8UC4);
  //   static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
  //   cv::mixChannels(&out_mat, 1, &tmp, 1, fromto, 4);
  //   out_mat = tmp;
  // }
  // // Write result image
  // cv::imwrite("1.png", out_mat);
  cv::Mat out_mat(source.size(), CV_32FC4);
  queue.enqueueReadImage(cl_b, CL_TRUE,
                         origin, region_source, 0, 0,
                         out_mat.data);
  {
    cv::Mat tmp(source.size(), CV_32FC4);
    static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
    cv::mixChannels(&out_mat, 1, &tmp, 1, fromto, 4);
    out_mat = tmp;
  }
  // Write result image
  cv::imwrite("1.png", out_mat);

  return EXIT_SUCCESS;
}