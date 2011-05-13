#include <QApplication>

#include "window.h"

std::string source;
std::string mm;
std::string target;

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  source = argv[1];
  mm = argv[2];
  target = argv[3];
  Window window;
  window.show();
  window.set_images();
  return app.exec();
}
