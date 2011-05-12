void init_cl(cl::Context& context,
             cl::CommandQueue& queue) {
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform size 0" << std::endl;
      exit(EXIT_FAILURE);
    }
    cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, cl_context_properties((platforms[0])()),
              CL_GLX_DISPLAY_KHR, cl_context_properties(glXGetCurrentDisplay()),
              CL_GL_CONTEXT_KHR,  cl_context_properties(glXGetCurrentContext()),
              0 };
    try {
      context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    } catch (cl::Error) {
      context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
    }
    std::vector<cl::Device> devices(context.getInfo<CL_CONTEXT_DEVICES>());
    queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
    // std::cerr << devices[0].getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl::Program load_program(const cl::Context& context,
                         std::string program_name) {
  bool load_binary = true;

  time_t so_time = 0;
  time_t cl_time = 1;
  struct stat stat_buf_so;
  struct stat stat_buf_cl;
  int stat_status_so = stat((program_name + ".so").c_str(), &stat_buf_so);
  int stat_status_cl = stat((program_name + ".cl").c_str(), &stat_buf_cl);
  if (!stat_status_so && !stat_status_cl) {
    so_time = stat_buf_so.st_mtime;
    cl_time = stat_buf_cl.st_mtime;
  }

  std::ifstream ifs(program_name + ".so");
  if (!ifs || so_time < cl_time) {
    if (ifs) ifs.close();
    load_binary = false;
    ifs.open(program_name + ".cl");
  }
  std::string src((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());
  cl::Program program;
  std::vector<cl::Device> devices(context.getInfo<CL_CONTEXT_DEVICES>());
  try {
    if (load_binary) {
      cl::Program::Binaries bins {
        std::make_pair(src.c_str(), src.size())
      };
      program = cl::Program(context, devices, bins);
    } else {
      cl::Program::Sources source {
        std::make_pair(src.c_str(), src.size())
      };
      program = cl::Program(context, source);
    }
    std::stringstream options;
    if (!strcmp(devices[0].getInfo<CL_DEVICE_NAME>().c_str(),
                "GeForce 8800 GT")) {
      options << "-D FIX_BROKEN_IMAGE_WRITING";
    }
    program.build(devices, options.str().c_str());

    if (!load_binary) {
      std::string log;
      program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &log);
      std::cerr << log;

      std::vector<size_t> sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
      std::vector<char*> bins = program.getInfo<CL_PROGRAM_BINARIES>(NULL);
      if (bins.size()) {
        std::ofstream out(program_name + ".so");
        std::copy(bins[0], bins[0] + sizes[0],
                  std::ostream_iterator<char>(out));
      }
    }
  } catch (cl::Error error) {
    std::cerr << "ERROR: "
              << error.what()
              << "(" << error.err() << ")"
              << std::endl;
    std::string log;
    program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &log);
    std::cerr << log;
    exit(EXIT_FAILURE);
  }
  return program;
}

GLuint load_texture(cv::Mat image, int width = -1, int height = -1) {
  std::vector<uint8_t> data(width * height * 4, 0);
  GLuint texture;

  glGenTextures(1, &texture); //generate the texture with the loaded data
  glBindTexture(GL_TEXTURE_2D, texture); //bind the texture to it’s array

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
               !image.empty() ? image.cols : width,
               !image.empty() ? image.rows : height,
               0, GL_RGBA, GL_UNSIGNED_BYTE,
               !image.empty() ? image.data : data.data());

  return texture;
}

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

// FIXME: Doesn't work for GPU/flipped images
void save_cl_image(std::string filename,
                   cl::CommandQueue const& queue,
                   cl::Image2D const& cl_image) {
  size_t width = cl_image.getImageInfo<CL_IMAGE_WIDTH>();
  size_t height = cl_image.getImageInfo<CL_IMAGE_HEIGHT>();

  cv::Mat out_mat(cv::Size(int(width), int(height)), CV_32FC4);

  cl::size_t<3> origin;
  origin.push_back(0);
  origin.push_back(0);
  origin.push_back(0);
  cl::size_t<3> region;
  region.push_back(width);
  region.push_back(height);
  region.push_back(1);
  queue.enqueueReadImage(cl_image, CL_TRUE,
                         origin, region, 0, 0,
                         out_mat.data);
  {
    cv::Mat tmp(out_mat.size(), CV_32FC4);
    static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
    cv::mixChannels(&out_mat, 1, &tmp, 1, fromto, 4);
    out_mat = tmp;
  }
  // Write result image
  cv::imwrite(filename, out_mat);
}
