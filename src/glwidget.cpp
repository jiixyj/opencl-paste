#include "glwidget.moc"

#include <iostream>

#include <highgui.h>

GLWidget::GLWidget(QWidget* _parent, pv::Context* context)
          : QGLWidget(_parent),
            context_(context),
            width(0),
            height(0),
            min_width(0),
            min_height(0),
            button_pressed(false),
            old_pos_x(),
            old_pos_y() {
  connect(&idle_timer, SIGNAL(timeout()), this, SLOT(updateGL()));
  idle_timer.start(0);
  setFocusPolicy(Qt::ClickFocus);
}

GLWidget::~GLWidget() {
}

QSize GLWidget::sizeHint() const {
  return QSize(width, height);
}

QSize GLWidget::minimumSizeHint() const {
  return QSize(min_width, min_height);
}

void GLWidget::initializeGL() {
  context_->init_gl();
  context_->init_cl();
  frame_time.start();
}

extern std::string source;
extern std::string mm;
extern std::string target;

void GLWidget::set_images() {
  std::string s = source;
  std::string m = mm;
  std::string t = target;
  context_->set_source(cv::imread(s), cv::imread(m));
  context_->set_target(cv::imread(t));
  std::pair<int, int> p = context_->get_gl_size();
  min_width = width = p.first;
  min_height = height = p.second;
  updateGeometry();
  QCoreApplication::processEvents();
  min_width = min_height = 0;
  updateGeometry();
}

void GLWidget::paintGL() {
  static double number_iterations = 10.0;
  static size_t frame_count = 0;
  static int time_interval;
  static float fps;
  static float wanted_fps = 30.0f;

  context_->wait_for_calculations();
  float average = context_->get_residual_average();
  context_->draw_frame();
  context_->start_calculation_async(number_iterations);

  frame_count++;
  time_interval = frame_time.elapsed();
  if (time_interval > 200) {
    fps = float(frame_count) / (float(time_interval) / 1000.0f);
    std::cerr << "FPS: " << fps << " "
              << "iterations: " << number_iterations << " "
              << "avg: " << average << std::endl;

    if (fps >= wanted_fps) {
      number_iterations += fps - wanted_fps;
    } else {
      number_iterations -= wanted_fps - fps;
    }
    if (number_iterations < 4) number_iterations = 4;
    frame_time.restart();
    frame_count = 0;
  }
}

void GLWidget::resizeGL(int width, int height) {
  glViewport(0, 0, (GLint) width, (GLint) height);
}

void GLWidget::mousePressEvent(QMouseEvent* mevent) {
  switch (mevent->button()) {
    case Qt::LeftButton:
      button_pressed = true;
      old_x = mevent->pos().x();
      old_y = height - mevent->pos().y();
      break;
    default:
      break;
  }
}

void GLWidget::mouseReleaseEvent(QMouseEvent* mevent) {
  switch (mevent->button()) {
    case Qt::LeftButton:
      button_pressed = false;
      context_->get_offset(old_pos_x, old_pos_y);
      break;
    default:
      break;
  }
}

void GLWidget::mouseMoveEvent(QMouseEvent* mevent) {
  if (button_pressed) {
    context_->set_offset(old_pos_x + mevent->pos().x() - old_x,
                         old_pos_y + height - mevent->pos().y() - old_y);
  }
}

void GLWidget::keyPressEvent(QKeyEvent* kevent) {
  switch (kevent->key()) {
    case Qt::Key_R:
      context_->toggle_residual_drawing();
      break;
    case Qt::Key_P:
      idle_timer.stop();
      context_->wait_for_calculations();
      context_->push_residual_stack();
      idle_timer.start();
      break;
    case Qt::Key_O:
      idle_timer.stop();
      context_->wait_for_calculations();
      context_->pop_residual_stack();
      idle_timer.start();
      break;
  }
}
