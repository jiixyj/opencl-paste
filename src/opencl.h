#ifndef OPENCL_H_
#define OPENCL_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace pv {

void init_cl(cl::Context& context_, cl::CommandQueue& queue_);

}

#endif
