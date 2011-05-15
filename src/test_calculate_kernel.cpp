#include <iostream>
#include <cv.h>

cv::Mat_<float> coarser_grid_kernel(cv::Mat_<float> kernel,
                                    cv::Mat_<float> bilinear_kernel) {
  cv::Mat_<float> tmp = cv::Mat::zeros(7, 7, CV_32F);
  cv::Mat_<float> sub = tmp(cv::Rect(2, 2, 3, 3));
  kernel.copyTo(sub);
  cv::filter2D(tmp, tmp, -1, bilinear_kernel);
  cv::filter2D(tmp, tmp, -1, bilinear_kernel);
  cv::Mat_<float> ret(3, 3);
  ret(0, 0) = tmp(1, 1);
  ret(0, 1) = tmp(1, 3);
  ret(0, 2) = tmp(1, 5);
  ret(1, 0) = tmp(3, 1);
  ret(1, 1) = tmp(3, 3);
  ret(1, 2) = tmp(3, 5);
  ret(2, 0) = tmp(5, 1);
  ret(2, 1) = tmp(5, 3);
  ret(2, 2) = tmp(5, 5);
  return ret;
}


int main() {
  cv::Mat_<float> bilinear_kernel(3, 3);
  bilinear_kernel(0, 0) = 0.25f;
  bilinear_kernel(0, 1) = 0.5f;
  bilinear_kernel(0, 2) = 0.25f;
  bilinear_kernel(1, 0) = 0.5f;
  bilinear_kernel(1, 1) = 1.0f;
  bilinear_kernel(1, 2) = 0.5f;
  bilinear_kernel(2, 0) = 0.25f;
  bilinear_kernel(2, 1) = 0.5f;
  bilinear_kernel(2, 2) = 0.25f;

  cv::Mat_<float> laplace_kernel(3, 3);
  // laplace_kernel(0, 0) = 0.0f;
  // laplace_kernel(0, 1) = 0.0f;
  // laplace_kernel(0, 2) = 0.0f;
  // laplace_kernel(1, 0) = 0.0f;
  // laplace_kernel(1, 1) = 1.0f;
  // laplace_kernel(1, 2) = 0.0f;
  // laplace_kernel(2, 0) = 0.0f;
  // laplace_kernel(2, 1) = 0.0f;
  // laplace_kernel(2, 2) = 0.0f;
  laplace_kernel(0, 0) = 0.0f;
  laplace_kernel(0, 1) = .0f;
  laplace_kernel(0, 2) = 0.0f;
  laplace_kernel(1, 0) = -1.0f;
  laplace_kernel(1, 1) = 3.0f;
  laplace_kernel(1, 2) = -1.0f;
  laplace_kernel(2, 0) = 0.0f;
  laplace_kernel(2, 1) = -1.0f;
  laplace_kernel(2, 2) = 0.0f;

  cv::Mat_<float> cgk = coarser_grid_kernel(laplace_kernel, bilinear_kernel);
  cgk = coarser_grid_kernel(cgk, bilinear_kernel);
  cgk = coarser_grid_kernel(cgk, bilinear_kernel);
  std::cout << cgk(0, 0) << " ";
  std::cout << cgk(0, 1) << " ";
  std::cout << cgk(0, 2) << std::endl;
  std::cout << cgk(1, 0) << " ";
  std::cout << cgk(1, 1) << " ";
  std::cout << cgk(1, 2) << std::endl;
  std::cout << cgk(2, 0) << " ";
  std::cout << cgk(2, 1) << " ";
  std::cout << cgk(2, 2) << std::endl;
  return 0;
}
