#ifndef WINDOW_H
#define WINDOW_H

#include <QtGui>

#include "gl_context.h"

class GLWidget;

class Window : public QWidget {
  Q_OBJECT

 public:
  Window();
  void set_images();
  void set_source(cv::Mat source, cv::Mat mask);
  void set_target(cv::Mat target);

 private:
  pv::GLContext context_;
  GLWidget* gl_widget_;

  // don't copy
  Window(const Window&);
  Window* operator=(const Window&);
};

#endif
