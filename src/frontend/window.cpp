#include "window.moc"

#include "glwidget.h"

Window::Window()
          : context_(),
            gl_widget_(new GLWidget(this, &context_)) {
  QHBoxLayout* main_layout = new QHBoxLayout;
  gl_widget_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  main_layout->addWidget(gl_widget_);
  setLayout(main_layout);
  setWindowTitle(tr("Hello GL"));
}

void Window::set_images() {
  gl_widget_->set_images();
}

void Window::set_source(cv::Mat source, cv::Mat _mask) {
  context_.set_source(source, _mask);
}

void Window::set_target(cv::Mat target) {
  context_.set_target(target);
}
