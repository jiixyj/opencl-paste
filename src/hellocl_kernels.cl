#pragma OPENCL EXTENSION cl_amd_printf : enable

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_FILTER_NEAREST |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void setup_system(read_only image2d_t source,
                         read_only image2d_t target,
                         write_only image2d_t a1,
                         write_only image2d_t a2,
                         write_only image2d_t a3,
                         write_only image2d_t b,
                         write_only image2d_t x,
                         int ox, int oy,
                         int initialize) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  size_t size_x = get_global_size(0);

  uint4 pixel = read_imageui(source, sampler, coord);
  float4 a1_val = 0.0f;
  float4 a2_val = 0.0f;
  float4 a3_val = 0.0f;
  if (pixel.w) {
    uint4 source_m = read_imageui(source, sampler, coord);
    uint4 source_u = read_imageui(source, sampler, coord + (int2)( 0,  1));
    uint4 source_l = read_imageui(source, sampler, coord + (int2)(-1,  0));
    uint4 source_d = read_imageui(source, sampler, coord + (int2)( 0, -1));
    uint4 source_r = read_imageui(source, sampler, coord + (int2)( 1,  0));
    bool u_o = source_u.w;
    bool l_o = source_l.w;
    bool d_o = source_d.w;
    bool r_o = source_r.w;
    uint4 target_u = read_imageui(target, sampler, coord + (int2)(ox, oy + 1));
    uint4 target_l = read_imageui(target, sampler, coord + (int2)(ox - 1, oy));
    uint4 target_d = read_imageui(target, sampler, coord + (int2)(ox, oy - 1));
    uint4 target_r = read_imageui(target, sampler, coord + (int2)(ox + 1, oy));
    bool u_s = target_u.w;
    bool l_s = target_l.w;
    bool d_s = target_d.w;
    bool r_s = target_r.w;

    a2_val.y =  4.0f;
    a1_val.y = -1.0f;
    a2_val.x = -1.0f;
    a3_val.y = -1.0f;
    a2_val.z = -1.0f;

    uint4 laplace = (uint4)(0);
    if (!u_o && u_s) {
      laplace += target_u;
    }
    if (!l_o && l_s) {
      laplace += target_l;
    }
    if (!d_o && d_s) {
      laplace += target_d;
    }
    if (!r_o && r_s) {
      laplace += target_r;
    }
    if (u_s) {
      laplace += source_m - source_u;
    }
    if (l_s) {
      laplace += source_m - source_l;
    }
    if (d_s) {
      laplace += source_m - source_d;
    }
    if (r_s) {
      laplace += source_m - source_r;
    }

    float4 laplacef = convert_float4(as_int4(laplace));
#ifdef FIX_BROKEN_IMAGE_WRITING
    coord.x = coord.x * 2;
#endif
    write_imagef(b, coord, laplacef);
    if (initialize) {
      // write_imagef(x, coord, 0);
      write_imagef(x, coord, convert_float4(pixel));
    }
    write_imagef(a1, coord, a1_val);
    write_imagef(a2, coord, a2_val);
    write_imagef(a3, coord, a3_val);
  }
}

kernel void jacobi(read_only image2d_t a1,
                   read_only image2d_t a2,
                   read_only image2d_t a3,
                   read_only image2d_t b,
                   read_only image2d_t x_in,
                   write_only image2d_t x_out,
                   write_only image2d_t render,
                   local float4* cache,
                   int write_to_image) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  int2 image_dim = get_image_dim(x_in);

  int2 lc = (int2)(get_local_id(0), get_local_id(1)) + (int2)(1);
  int lw = get_local_size(0) + 2;

  cache[lw * lc.y + lc.x] = read_imagef(x_in, sampler, coord);
  if (lc.x == 1) {
    cache[lw * lc.y + lc.x - 1] =
                        read_imagef(x_in, sampler, coord + (int2)(-1,  0));
  } else if (lc.x == lw - 2) {
    cache[lw * lc.y + lc.x + 1] =
                        read_imagef(x_in, sampler, coord + (int2)( 1,  0));
  }
  if (lc.y == 1) {
    cache[lw * (lc.y - 1) + lc.x] =
                        read_imagef(x_in, sampler, coord + (int2)( 0, -1));
  } else if (lc.y == lw - 2) {
    cache[lw * (lc.y + 1) + lc.x] =
                        read_imagef(x_in, sampler, coord + (int2)( 0,  1));
  }

  float4 a1_val = read_imagef(a1, sampler, coord);
  float4 a2_val = read_imagef(a2, sampler, coord);
  float4 a3_val = read_imagef(a3, sampler, coord);
  float4 sigma = read_imagef(b, sampler, coord);

  barrier(CLK_LOCAL_MEM_FENCE);  // Cache has been written at this point

  for (int i = 0; i < 10; ++i) {
    sigma -= a1_val.y * cache[lw * (lc.y + 1) + lc.x];
    sigma -= a2_val.x * cache[lw * lc.y + lc.x - 1];
    sigma -= a3_val.y * cache[lw * (lc.y - 1) + lc.x];
    sigma -= a2_val.z * cache[lw * lc.y + lc.x + 1];
    sigma /= a2_val.y ? a2_val.y : 1;  // prevent division by zero

    cache[lw * lc.y + lc.x] = sigma;
    sigma = read_imagef(b, sampler, coord);

    barrier(CLK_LOCAL_MEM_FENCE);  // Cache has been updated here
  }

  if (a2_val.y && coord.x < image_dim.x && coord.y < image_dim.y) {
  #ifdef FIX_BROKEN_IMAGE_WRITING
    write_imagef(x_out, coord * (int2)(2, 1), cache[lw * lc.y + lc.x]);
  #else
    write_imagef(x_out, coord, cache[lw * lc.y + lc.x]);
  #endif
    if (write_to_image == 1) {
      cache[lw * lc.y + lc.x].w = 255.0f;
      write_imagef(render, coord, cache[lw * lc.y + lc.x] / 255.0f);
    } else if (write_to_image == 2) {
      write_imagef(render, coord, cache[lw * lc.y + lc.x] / 64.0f + 0.5f);
    }
  }
}

kernel void calculate_residual(read_only image2d_t a1,
                               read_only image2d_t a2,
                               read_only image2d_t a3,
                               read_only image2d_t b,
                               read_only image2d_t x,
                               write_only image2d_t res,
                               write_only image2d_t gpu_abs,
                               int write_to_image) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  float4 a2_val = read_imagef(a2, sampler, coord);
  if (a2_val.y < 1) {
    return;
  }
  float4 a1_val = read_imagef(a1, sampler, coord);
  float4 a3_val = read_imagef(a3, sampler, coord);
  float4 sigma = read_imagef(b, sampler, coord);
  sigma -= a1_val.y * read_imagef(x, sampler, coord + (int2)( 0,  1));
  sigma -= a2_val.x * read_imagef(x, sampler, coord + (int2)(-1,  0));
  sigma -= a3_val.y * read_imagef(x, sampler, coord + (int2)( 0, -1));
  sigma -= a2_val.z * read_imagef(x, sampler, coord + (int2)( 1,  0));

  sigma -= a2_val.y * read_imagef(x, sampler, coord);
  sigma.w = 0.0f;
  if (write_to_image) {
    write_imagef(gpu_abs, coord, sigma * sigma * 128.0f);
  }
#ifdef FIX_BROKEN_IMAGE_WRITING
  coord.x = coord.x * 2;
#endif
  write_imagef(res, coord, sigma);
}

kernel void reset_image(write_only image2d_t out) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
#ifdef FIX_BROKEN_IMAGE_WRITING
  coord.x = coord.x * 2;
#endif
  write_imagef(out, coord, 0.0f);
}



const sampler_t bilinear_sampler = CLK_NORMALIZED_COORDS_TRUE |
                                   CLK_FILTER_LINEAR |
                                   CLK_ADDRESS_CLAMP_TO_EDGE;
const sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_TRUE |
                                  CLK_FILTER_NEAREST |
                                  CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void nearest_interp(read_only image2d_t source,
                           write_only image2d_t output) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

#ifdef FIX_BROKEN_IMAGE_WRITING
  write_imagef(output, coord * (int2)(2, 1),
#else
  write_imagef(output, coord,
#endif
    read_imagef(source, nearest_sampler,
                (convert_float2(coord) + (float2)(0.5f)) /
                convert_float2(get_image_dim(output))));
}

kernel void bilinear_interp(read_only image2d_t source,
                            write_only image2d_t output) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

#ifdef FIX_BROKEN_IMAGE_WRITING
  write_imagef(output, coord * (int2)(2, 1),
#else
  write_imagef(output, coord,
#endif
    read_imagef(source, bilinear_sampler,
                (convert_float2(coord) + (float2)(0.5f)) /
                convert_float2(get_image_dim(output))));
}

kernel void bilinear_restrict(read_only image2d_t source,
                              write_only image2d_t output) {

  int2 coord = (int2)(get_global_id(0), get_global_id(1));

#ifdef FIX_BROKEN_IMAGE_WRITING
  write_imagef(output, coord * (int2)(2, 1),
#else
  write_imagef(output, coord,
#endif
              (read_imagef(source, bilinear_sampler,
                           (convert_float2(coord)) /
                           convert_float2(get_image_dim(output)))
             + read_imagef(source, bilinear_sampler,
                           (convert_float2(coord) + (float2)(1.0f, 0.0f)) /
                           convert_float2(get_image_dim(output)))
             + read_imagef(source, bilinear_sampler,
                           (convert_float2(coord) + (float2)(0.0f, 1.0f)) /
                           convert_float2(get_image_dim(output)))
             + read_imagef(source, bilinear_sampler,
                           (convert_float2(coord) + (float2)(1.0f, 1.0f)) /
                           convert_float2(get_image_dim(output)))
                         ) / 2.0f);
}

kernel void reduce(read_only image2d_t buffer,
                   const long length,
                   local float* scratch,
                   global float* result) {
  int global_index = get_global_id(0);
  int2 size = get_image_dim(buffer);
  float accumulator = 0.0f;

  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float4 element = read_imagef(buffer, sampler,
                                         (int2)(global_index % size.x,
                                                global_index / size.x));
    accumulator += dot(element * element, (float4)(1.0f));
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = other + mine;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}

kernel void add_images(read_only image2d_t lhs,
                       read_only image2d_t rhs,
                       read_only image2d_t a2,
                       write_only image2d_t result) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  float4 result_val;
  if (read_imagef(a2, sampler, coord).y < 1) {
    result_val = 0.0f;
  } else {
    result_val = read_imagef(lhs, sampler, coord) +
                 read_imagef(rhs, sampler, coord);
  }
#ifdef FIX_BROKEN_IMAGE_WRITING
    write_imagef(result, coord * (int2)(2, 1),
#else
    write_imagef(result, coord,
#endif
                 result_val);
}
