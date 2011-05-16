#include "opencl.h"

#include <highgui.h>

int main() {
  cl::Context context;
  cl::CommandQueue queue;
  pv::init_cl(context, queue, false);

  cl::Program program = pv::load_program(context, "hellocl_kernels");
  cv::Mat lena_original = cv::imread("lena.png");
  cv::Mat lena = pv::make_rgba(lena_original);
  lena.convertTo(lena, CV_32F);
  cv::flip(lena, lena, 0);
  cv::Mat lena_256(lena.rows / 2, lena.cols / 2, lena.type());

  cl::Image2D cl_lena(context, CL_MEM_READ_ONLY,
                      cl::ImageFormat(CL_RGBA, CL_FLOAT),
                      size_t(lena.cols), size_t(lena.rows));
  cl::Image2D cl_lena_256(context, CL_MEM_WRITE_ONLY,
                          cl::ImageFormat(CL_RGBA, CL_FLOAT),
                          size_t(lena.cols) / 2, size_t(lena.rows) / 2);

  cl::Kernel bilinear_restrict(program, "bilinear_restrict", NULL);
  cl::size_t<3> origin;
  origin.push_back(0);
  origin.push_back(0);
  origin.push_back(0);
  cl::size_t<3> region_source;
  region_source.push_back(cl_lena.getImageInfo<CL_IMAGE_WIDTH>());
  region_source.push_back(cl_lena.getImageInfo<CL_IMAGE_HEIGHT>());
  region_source.push_back(1);
  queue.enqueueWriteImage(cl_lena, CL_FALSE,
                          origin, region_source, 0, 0,
                          lena.data);
  bilinear_restrict.setArg<cl::Image2D>(0, cl_lena);
  bilinear_restrict.setArg<cl::Image2D>(1, cl_lena_256);
  queue.enqueueNDRangeKernel(
    bilinear_restrict,
    cl::NullRange,
    cl::NDRange(cl_lena_256.getImageInfo<CL_IMAGE_WIDTH>(),
                cl_lena_256.getImageInfo<CL_IMAGE_HEIGHT>()),
    cl::NullRange
  );
  pv::save_cl_image("blub.png", queue, cl_lena_256);
  cv::Mat cv_reference;
  cv::resize(lena_original, cv_reference, cv::Size(256, 256));
  cv::imwrite("blub_ref.png", cv_reference);
  cv::Mat filter_result;
  cv::Mat_<float> bilinear_kernel(3, 3);
  bilinear_kernel(0, 0) = 0.25f / 4.0f  ;
  bilinear_kernel(0, 1) = 0.5f  / 4.0f  ;
  bilinear_kernel(0, 2) = 0.25f / 4.0f  ;
  bilinear_kernel(1, 0) = 0.5f  / 4.0f  ;
  bilinear_kernel(1, 1) = 1.0f  / 4.0f  ;
  bilinear_kernel(1, 2) = 0.5f  / 4.0f  ;
  bilinear_kernel(2, 0) = 0.25f / 4.0f  ;
  bilinear_kernel(2, 1) = 0.5f  / 4.0f  ;
  bilinear_kernel(2, 2) = 0.25f / 4.0f  ;
  cv::filter2D(lena_original, filter_result, -1, bilinear_kernel);
  cv::imwrite("blub_ref_filter.png", filter_result);
  cv::Mat filter_result_small;
  cv::resize(filter_result, filter_result_small, cv::Size(256, 256), 0, 0,
             cv::INTER_LINEAR);
  cv::imwrite("blub_ref_filter_small.png", filter_result_small);
  return 0;
}
