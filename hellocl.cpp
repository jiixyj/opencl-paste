#include <stdcl.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <array>
#include <iostream>

#include <cv.h>
#include <highgui.h>

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

  CONTEXT* cp = (stdgpu) ? stdgpu : stdcpu;
  cl::Context context(cp->ctx);
  cl::CommandQueue queue(cp->cmdq[0]);

  void* clh = clopen(cp, "hellocl_kernels.cl", CLLD_NOW);
  cl::Kernel kernel(clsym(cp, clh, "hello", CLLD_NOW));

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
  kernel.setArg<cl_int>(2, 1);
  queue.enqueueCopyImage(cl_img_o, cl_img_i, origin, origin, region, NULL, &event);
  event.wait();
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
