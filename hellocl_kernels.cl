#pragma OPENCL EXTENSION cl_amd_printf : enable

float sum(float4 in) {
  return dot(in, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_FILTER_NEAREST |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void setup_system(read_only image2d_t source,
                         read_only image2d_t target,
                         write_only image2d_t b,
                         write_only image2d_t x) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  size_t size_x = get_global_size(0);

  uint4 pixel = read_imageui(source, sampler, coord);
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
    uint4 target_u = read_imageui(target, sampler, coord + (int2)( 0,  1));
    uint4 target_l = read_imageui(target, sampler, coord + (int2)(-1,  0));
    uint4 target_d = read_imageui(target, sampler, coord + (int2)( 0, -1));
    uint4 target_r = read_imageui(target, sampler, coord + (int2)( 1,  0));
    bool u_s = target_u.w;
    bool l_s = target_l.w;
    bool d_s = target_d.w;
    bool r_s = target_r.w;

    uchar a_val = u_s + l_s + d_s + r_s;
    if (u_o && u_s) {
      a_val |= 1 << 7;
    }
    if (l_o && l_s) {
      a_val |= 1 << 6;
    }
    if (d_o && d_s) {
      a_val |= 1 << 5;
    }
    if (r_o && r_s) {
      a_val |= 1 << 4;
    }

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

#ifdef FIX_BROKEN_IMAGE_WRITING
    coord.x = coord.x * 2;
#endif

    float4 laplacef = convert_float4(as_int4(laplace));
    laplacef.w = a_val;
    // is OK because OpenCL has two's complement
    write_imagef(b, coord, laplacef);
    write_imagef(x, coord, convert_float4(pixel));
  } else {
    uint4 tmp = read_imageui(target, sampler, coord);
    tmp.w = 1;
    float4 tmpf = convert_float4(tmp);
#ifdef FIX_BROKEN_IMAGE_WRITING
    coord.x = coord.x * 2;
#endif
    write_imagef(b, coord, tmpf);
    write_imagef(x, coord, tmpf);
  }
}

kernel void jacobi(read_only image2d_t b,
                   read_only image2d_t x_in,
                   write_only image2d_t x_out,
                   write_only image2d_t render,
                   int write_to_image) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  float4 sigma = read_imagef(b, sampler, coord);
  uchar a_val = sigma.w;
  sigma.w = 255.0f;
  if (a_val & (1 << 7)) {
    sigma += read_imagef(x_in, sampler, coord + (int2)( 0,  1));
  }
  if (a_val & (1 << 6)) {
    sigma += read_imagef(x_in, sampler, coord + (int2)(-1,  0));
  }
  if (a_val & (1 << 5)) {
    sigma += read_imagef(x_in, sampler, coord + (int2)( 0, -1));
  }
  if (a_val & (1 << 4)) {
    sigma += read_imagef(x_in, sampler, coord + (int2)( 1,  0));
  }

  float4 result = sigma / (a_val & 0x0F);
  if (write_to_image) {
    write_imagef(render, coord, result / 255.0f);
  }
#ifdef FIX_BROKEN_IMAGE_WRITING
  coord.x = coord.x * 2;
#endif
  write_imagef(x_out, coord, result);
}

kernel void calculate_residual(read_only image2d_t b,
                               read_only image2d_t x,
                               write_only image2d_t res,
                               write_only image2d_t gpu_abs) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  float4 sigma = read_imagef(b, sampler, coord);
  uchar a_val = sigma.w;
  if (a_val & (1 << 7)) {
    sigma += read_imagef(x, sampler, coord + (int2)( 0,  1));
  }
  if (a_val & (1 << 6)) {
    sigma += read_imagef(x, sampler, coord + (int2)(-1,  0));
  }
  if (a_val & (1 << 5)) {
    sigma += read_imagef(x, sampler, coord + (int2)( 0, -1));
  }
  if (a_val & (1 << 4)) {
    sigma += read_imagef(x, sampler, coord + (int2)( 1,  0));
  }
  sigma -= (a_val & 0x0F) * read_imagef(x, sampler, coord);
  sigma.w = 0.0f;
  write_imagef(gpu_abs, coord, sigma * sigma * 1024.0f);
#ifdef FIX_BROKEN_IMAGE_WRITING
  coord.x = coord.x * 2;
#endif
  write_imagef(res, coord, sigma);
}

kernel void reset_image(write_only image2d_t out) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  write_imagef(out, coord, 0.0f);
}

kernel void bilinear_filter(read_only image2d_t source,
                            write_only image2d_t output) {

  const sampler_t bilinear_sampler = CLK_NORMALIZED_COORDS_TRUE |
                                     CLK_FILTER_LINEAR |
                                     CLK_ADDRESS_CLAMP_TO_EDGE;
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  write_imagef(output, coord,
               read_imagef(source, bilinear_sampler,
                           (convert_float2(coord) + (float2)(0.5f)) /
                           convert_float2(get_image_dim(output))));
}
