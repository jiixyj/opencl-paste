#include "glwidget.moc"

#include <iostream>

#include <highgui.h>

GLWidget::GLWidget(QWidget* _parent, pv::GLContext* _context)
          : QGLWidget(_parent),
            context_(_context),
            width(),
            height(),
            min_width(),
            min_height(),
            idle_timer(),
            frame_time(),
            old_x(),
            old_y(),
            old_pos_x(),
            old_pos_y(),
            button_pressed() {
  connect(&idle_timer, SIGNAL(timeout()), this, SLOT(updateGL()));
  idle_timer.start(0);
  setFocusPolicy(Qt::StrongFocus);
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
  context_->init();
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
  static size_t frame_count = 0;
  static int time_interval;
  static double fps;

  float average = context_->solver().get_residual_average();
  context_->draw_frame();
  context_->lock_gl();
  context_->solver().start_calculation_async(1);
  context_->prepare_images_for_drawing();
  context_->unlock_gl();

  frame_count++;
  time_interval = frame_time.elapsed();
  if (time_interval > 200) {
    fps = float(frame_count) / (float(time_interval) / 1000.0f);
    std::cerr << "FPS: " << fps << " "
              << "avg: " << average << std::endl;
    frame_time.restart();
    frame_count = 0;
  }
}

void GLWidget::resizeGL(int w, int h) {
  glViewport(0, 0, w, h);
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
      context_->solver().get_offset(old_pos_x, old_pos_y);
      break;
    default:
      break;
  }
}

void GLWidget::mouseMoveEvent(QMouseEvent* mevent) {
  if (button_pressed) {
    context_->solver().set_offset(old_pos_x + mevent->pos().x() - old_x,
                              old_pos_y + height - mevent->pos().y() - old_y);
  }
}

void GLWidget::keyPressEvent(QKeyEvent* kevent) {
  switch (kevent->key()) {
    case Qt::Key_R:
      context_->toggle_residual_drawing();
      break;
    case Qt::Key_Q:
      std::exit(0);
      break;
    default:
      break;
  }
}
