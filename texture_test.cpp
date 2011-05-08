#include <GL/glew.h>
#include <GL/glut.h>

#include <cv.h>
#include <highgui.h>

#include <iostream>

GLuint g_texture; //the array for our texture

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

GLuint LoadTexture(cv::Mat image) {
  GLuint texture;

  glGenTextures(1, &texture); //generate the texture with the loaded data
  glBindTexture(GL_TEXTURE_2D, texture); //bind the texture to it’s array
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); //set texture environment parameters

  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  cv::Mat image_flipped;
  cv::flip(image, image_flipped, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
               image_flipped.cols, image_flipped.rows,
               0, GL_RGBA, GL_FLOAT, image_flipped.data);
  glGenerateMipmap(GL_TEXTURE_2D);
  int width, height;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
  int max_level = int(std::floor(std::log2(std::max(width, height))));
  glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_HEIGHT, &height);
  assert(width == 1 && height == 1);
  float average[4];
  glGetTexImage(GL_TEXTURE_2D, max_level, GL_RGBA, GL_FLOAT, average);
  std::cout << average[0] << " "
            << average[1] << " "
            << average[2] << " "
            << average[3] << std::endl;

  return texture;
}

void square() {
  glBindTexture(GL_TEXTURE_2D, g_texture);
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, -1.0);
  glTexCoord2d(1.0, 0.0); glVertex2d( 1.0, -1.0);
  glTexCoord2d(1.0, 1.0); glVertex2d( 1.0,  1.0);
  glTexCoord2d(0.0, 1.0); glVertex2d(-1.0,  1.0);
  glEnd();
}

void display() {
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();
  glEnable(GL_TEXTURE_2D);
  gluLookAt(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  square();
  glutSwapBuffers();
}

cv::Size image_size;

void reshape(int w, int h) {
  glViewport(0, 0, GLsizei(w), GLsizei(h));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(90, GLfloat(w) / GLfloat(h) *
                     GLfloat(image_size.height) / GLfloat(image_size.width),
                 1.0, 100.0);
  glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char* argv[]) {
  cv::Mat image;
  make_rgba(cv::imread(argv[1])).convertTo(image, CV_32F, 1.0 / 255.0);
  image_size = image.size();

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE);
  glutInitWindowSize(image.cols, image.rows);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("A basic OpenGL Window");
  glutDisplayFunc(display);
  glutIdleFunc(display);
  glutReshapeFunc(reshape);

  GLenum err = glewInit();
  if (err != GLEW_OK) {
    std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "Status: Using GLEW "
            << glewGetString(GLEW_VERSION) << std::endl;

  g_texture = LoadTexture(image);

  glutMainLoop();

  glDeleteTextures(1, &g_texture);

  return EXIT_SUCCESS;
}
