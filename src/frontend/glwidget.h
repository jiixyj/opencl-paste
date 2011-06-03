#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "gl_context.h"
#include <QtOpenGL>

class GLWidget : public QGLWidget {
  Q_OBJECT

 public:
  GLWidget(QWidget *parent, pv::GLContext* _context);
  ~GLWidget();
  QSize sizeHint() const;
  QSize minimumSizeHint() const;

  void set_images();

 protected:
  pv::GLContext* context_;

  int width, height, min_width, min_height;

  void initializeGL();
  void paintGL();
  void resizeGL(int width, int height);
  void mousePressEvent(QMouseEvent* mevent);
  void mouseReleaseEvent(QMouseEvent* mevent);
  void mouseMoveEvent(QMouseEvent* mevent);
  void keyPressEvent(QKeyEvent* kevent);

 private:
  QTimer idle_timer;
  QTime frame_time;
  int old_x, old_y, old_pos_x, old_pos_y;
  int button_pressed;

  GLWidget(const GLWidget&);
  GLWidget* operator=(const GLWidget&);
};

#endif
