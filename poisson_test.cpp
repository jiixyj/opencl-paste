#include <array>
#include <iostream>
#include <iterator>
#include <fstream>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cv.h>
#include <highgui.h>

#include <sys/stat.h>
#include <sys/time.h>

#include "./helper.cpp"

GLuint g_texture;
GLuint g_residual;
cl::Context context;
cl::CommandQueue queue;
cl::Kernel jacobi;
cl::Kernel calculate_residual;
cl::Image2D* cl_image_ptr_1;
cl::Image2D* cl_image_ptr_2;
cl::Image2D* cl_res_ptr;
cl::Image2DGL* cl_render_ptr = NULL;
cl::Image2DGL* cl_g_residual_ptr;

std::vector<float> image_average() {
  float average[4];

  glGenerateMipmap(GL_TEXTURE_2D);

  int width, height;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
  int max_level = int(std::floor(std::log2(std::max(width, height))));
  glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_HEIGHT, &height);
  assert(width == 1 && height == 1);
  glGetTexImage(GL_TEXTURE_2D, max_level, GL_RGBA, GL_FLOAT, average);
  return std::vector<float>(average, average + 4);
}

void square() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glDisable(GL_TEXTURE_2D);
  glColor4f(1.0f, 0.0f, 1.0f, 1.0f);
  glBegin(GL_TRIANGLES);
  glVertex3d(-1.0, -1.0, -0.1);
  glVertex3d( 1.0, -1.0, -0.1);
  glVertex3d( 1.0,  1.0, -0.1);
  glEnd();

  glEnable(GL_TEXTURE_2D);
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glBindTexture(GL_TEXTURE_2D, g_texture);
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex3d(-1.0, -1.0, 0.0);
  glTexCoord2d(1.0, 0.0); glVertex3d( 1.0, -1.0, 0.0);
  glTexCoord2d(1.0, 1.0); glVertex3d( 1.0,  1.0, 0.0);
  glTexCoord2d(0.0, 1.0); glVertex3d(-1.0,  1.0, 0.0);
  glEnd();


  glBlendFunc(GL_ONE, GL_ONE);

  glBindTexture(GL_TEXTURE_2D, g_residual);
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex3d(-1.0, -1.0, 0.1);
  glTexCoord2d(1.0, 0.0); glVertex3d( 1.0, -1.0, 0.1);
  glTexCoord2d(1.0, 1.0); glVertex3d( 1.0,  1.0, 0.1);
  glTexCoord2d(0.0, 1.0); glVertex3d(-1.0,  1.0, 0.1);
  glEnd();
}

void display() {
  static int number_iterations = 10;
  static cl::Event event;

  static int frame_count = 0;
  static int current_time;
  static int previous_time = glutGet(GLUT_ELAPSED_TIME);
  static int time_interval;
  static float fps;
  static float wanted_fps = 10.0f;

  static std::vector<float> average;

  try {
    event.wait();
  } catch (cl::Error err) { /* ignore */ }

  square();
  glutSwapBuffers();

  glFinish();
  std::vector<cl::Memory> gl_image{*cl_render_ptr};
  queue.enqueueAcquireGLObjects(&gl_image);
  for (int i = 0; i < number_iterations; ++i) {
    jacobi.setArg<cl::Image2D>(1, *cl_image_ptr_1);
    jacobi.setArg<cl::Image2D>(2, *cl_image_ptr_2);
    jacobi.setArg<cl::Image2DGL>(3, *cl_render_ptr);
    jacobi.setArg<int>(4, i == number_iterations - 1);
    queue.enqueueNDRangeKernel(
      jacobi,
      cl::NullRange,
      cl::NDRange(cl_render_ptr->getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_render_ptr->getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange,
      NULL, NULL
    );
    std::swap(cl_image_ptr_1, cl_image_ptr_2);
  }
  queue.enqueueReleaseGLObjects(&gl_image, NULL, &event);

  frame_count++;
  current_time = glutGet(GLUT_ELAPSED_TIME);
  time_interval = current_time - previous_time;
  if (time_interval > 500) {
    fps = frame_count / (time_interval / 1000.0f);

    std::vector<cl::Memory> gl_residual{*cl_g_residual_ptr};
    queue.enqueueAcquireGLObjects(&gl_residual);
    calculate_residual.setArg<cl::Image2D>(1, *cl_image_ptr_1);
    calculate_residual.setArg<cl::Image2D>(2, *cl_res_ptr);
    calculate_residual.setArg<cl::Image2DGL>(3, *cl_g_residual_ptr);
    queue.enqueueNDRangeKernel(
      calculate_residual,
      cl::NullRange,
      cl::NDRange(cl_image_ptr_1->getImageInfo<CL_IMAGE_WIDTH>(),
                  cl_image_ptr_1->getImageInfo<CL_IMAGE_HEIGHT>()),
      cl::NullRange,
      NULL, NULL
    );
    queue.enqueueReleaseGLObjects(&gl_residual, NULL, &event);
    event.wait();

    glBindTexture(GL_TEXTURE_2D, g_residual);
    average = image_average();
    // average = std::vector<float>();
    // if (cl_g_residual_ptr)
    //   delete cl_g_residual_ptr;
    // cl_g_residual_ptr = new cl::Image2DGL(context, CL_MEM_WRITE_ONLY,
    //                                   GL_TEXTURE_2D, 0, g_residual);

    std::cerr << "FPS: " << fps << " "
              << "iterations: " << number_iterations << " "
              << "avg: ";
    std::for_each(average.begin(),
                  average.end(),
                  [] (float a) { std::cout << a << " "; });
    std::cout << std::endl;

    if (fps >= wanted_fps + 0.1f) {
      number_iterations += fps - wanted_fps + 0.25f;
    } else if (fps <= wanted_fps - 0.1f) {
      number_iterations -= wanted_fps - fps + 0.25f;
    }
    if (number_iterations < 10) number_iterations = 10;
    previous_time = current_time;
    frame_count = 0;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "syntax: <exe> source.png mask.png target.png" << std::endl;
    return 1;
  }
  cv::Mat source = make_rgba(cv::imread(argv[1]), cv::imread(argv[2], 0));
  cv::Mat target = make_rgba(cv::imread(argv[3]));
  cv::flip(source, source, 0);
  cv::flip(target, target, 0);

  // Init OpenGL
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(source.cols, source.rows);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("A basic OpenGL Window");
  glutDisplayFunc(display);
  glutIdleFunc(display);
  // glutReshapeFunc(reshape);

  GLenum err = glewInit();
  if (err != GLEW_OK) {
    std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "Status: Using GLEW "
            << glewGetString(GLEW_VERSION) << std::endl;

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(-1, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  g_texture = load_texture(cv::Mat(), source.cols, source.rows);
  g_residual = load_texture(cv::Mat(), source.cols, source.rows);

  // Init OpenCL
  init_cl(context, queue);

  cl::Program program = load_program(context, "hellocl_kernels");

  size_t nr_pixels = size_t(source.rows) * size_t(source.cols);
  cl::Buffer cl_a(context, CL_MEM_READ_WRITE, nr_pixels * sizeof(cl_uchar));
  cl::Image2D cl_source(context, CL_MEM_READ_ONLY,
                        cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                        size_t(source.cols), size_t(source.rows));
  cl::Image2D cl_target(context, CL_MEM_READ_ONLY,
                        cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                        size_t(target.cols), size_t(target.rows));
  cl::Image2D cl_b(context, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, CL_FLOAT),
                   size_t(source.cols), size_t(source.rows));
  cl::Image2D cl_x1(context, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, CL_FLOAT),
                   size_t(source.cols), size_t(source.rows));
  cl::Image2D cl_x2(context, CL_MEM_READ_WRITE,
                   cl::ImageFormat(CL_RGBA, CL_FLOAT),
                   size_t(source.cols), size_t(source.rows));
  cl::Image2D cl_res(context, CL_MEM_READ_WRITE,
                     cl::ImageFormat(CL_RGBA, CL_FLOAT),
                     size_t(source.cols), size_t(source.rows));
  cl_render_ptr = new cl::Image2DGL(context, CL_MEM_WRITE_ONLY,
                                    GL_TEXTURE_2D, 0, g_texture);
  cl_g_residual_ptr = new cl::Image2DGL(context, CL_MEM_WRITE_ONLY,
                                        GL_TEXTURE_2D, 0, g_residual);
  cl_image_ptr_1 = &cl_x1;
  cl_image_ptr_2 = &cl_x2;
  cl_res_ptr = &cl_res;

  cl::Event event;
  // cl::Kernel reset_image(program, "reset_image", NULL);
  // std::vector<cl::Memory> gl_residual{*cl_g_residual_ptr};
  // queue.enqueueAcquireGLObjects(&gl_residual);
  // reset_image.setArg<cl::Image2DGL>(0, *cl_g_residual_ptr);
  // queue.enqueueNDRangeKernel(
  //   reset_image,
  //   cl::NullRange,
  //   cl::NDRange(cl_g_residual_ptr->getImageInfo<CL_IMAGE_WIDTH>(),
  //               cl_g_residual_ptr->getImageInfo<CL_IMAGE_HEIGHT>()),
  //   cl::NullRange,
  //   NULL, NULL
  // );
  // queue.enqueueReleaseGLObjects(&gl_residual, NULL, &event);
  // event.wait();

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

  try {
    cl::Kernel kernel(program, "setup_system", NULL);
    kernel.setArg<cl::Image2D>(0, cl_source);
    kernel.setArg<cl::Image2D>(1, cl_target);
    kernel.setArg<cl::Image2D>(2, cl_b);
    kernel.setArg<cl::Image2D>(3, *cl_image_ptr_1);

    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(size_t(source.cols),
                                           size_t(source.rows)),
                               cl::NullRange,
                               NULL, &event);
    event.wait();

    jacobi = cl::Kernel(program, "jacobi", NULL);
    jacobi.setArg<cl::Image2D>(0, cl_b);

    calculate_residual = cl::Kernel(program, "calculate_residual", NULL);
    calculate_residual.setArg<cl::Image2D>(0, cl_b);

  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    return EXIT_FAILURE;
  }
  glutMainLoop();
  return EXIT_SUCCESS;
}
