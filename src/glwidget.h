#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "context.h"
#include <QtOpenGL>

class GLWidget : public QGLWidget {
  Q_OBJECT

 public:
  GLWidget(QWidget *parent, pv::Context* _context);
  ~GLWidget();
  QSize sizeHint() const;
  QSize minimumSizeHint() const;

  void set_images();

 protected:
  pv::Context* context_;

  int width, height, min_width, min_height;

  void initializeGL();
  void paintGL();
  void resizeGL(int width, int height);
  void mousePressEvent(QMouseEvent* mevent);
  void mouseReleaseEvent(QMouseEvent* mevent);
  void mouseMoveEvent(QMouseEvent* mevent);

 private:
  QTimer idle_timer;
  QTime frame_time;
  int old_x, old_y, old_pos_x, old_pos_y;
  int button_pressed;
};

#endif
