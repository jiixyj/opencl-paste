float sum(float4 in) {
  return dot(in, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_FILTER_NEAREST |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void setup_system(read_only image2d_t source,
                         read_only image2d_t target,
                         write_only global uchar* a,
                         write_only image2d_t b) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  size_t size_x = get_global_size(0);

  uint4 pixel = read_imageui(source, sampler, coord);
  if (pixel.w) {
    uint4 source_m = read_imageui(source, sampler, coord);
    uint4 source_u = read_imageui(source, sampler, coord + (int2)( 0, -1));
    uint4 source_l = read_imageui(source, sampler, coord + (int2)(-1,  0));
    uint4 source_d = read_imageui(source, sampler, coord + (int2)( 0,  1));
    uint4 source_r = read_imageui(source, sampler, coord + (int2)( 1,  0));
    bool u_o = source_u.w;
    bool l_o = source_l.w;
    bool d_o = source_d.w;
    bool r_o = source_r.w;
    uint4 target_u = read_imageui(target, sampler, coord + (int2)( 0, -1));
    uint4 target_l = read_imageui(target, sampler, coord + (int2)(-1,  0));
    uint4 target_d = read_imageui(target, sampler, coord + (int2)( 0,  1));
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
    a[coord.y * size_x + coord.x] = a_val;

    pixel = (uint4)(0);
    if (!u_o && u_s) {
      pixel += target_u;
    }
    if (!l_o && l_s) {
      pixel += target_l;
    }
    if (!d_o && d_s) {
      pixel += target_d;
    }
    if (!r_o && r_s) {
      pixel += target_r;
    }
    if (u_s) {
      pixel -= source_u;
      pixel += source_m;
    }
    if (l_s) {
      pixel -= source_l;
      pixel += source_m;
    }
    if (d_s) {
      pixel -= source_d;
      pixel += source_m;
    }
    if (r_s) {
      pixel -= source_r;
      pixel += source_m;
    }

    pixel.w = 255;
    write_imagef(b, coord, convert_float4(pixel));

  } else {
    a[coord.y * size_x + coord.x] = 1;
    uint4 tmp = read_imageui(target, sampler, coord);
    tmp.w = 255;
    write_imagef(b, coord, convert_float4(tmp));
    // write_imagef(b, coord, (float4)(0.0f));
  }
}

kernel void hello(read_only image2d_t in,
                  write_only image2d_t out,
                  int vertical) {
  const int kernel_size = 50;

  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  uint4 pixel = (uint4)(0);
  if (vertical) {
    for (int i = -kernel_size + coord.y; i <= kernel_size + coord.y; ++i) {
      pixel += read_imageui(in, sampler, (int2)(coord.x, i));
    }
  } else {
    for (int j = -kernel_size + coord.x; j <= kernel_size + coord.x; ++j) {
      pixel += read_imageui(in, sampler, (int2)(j, coord.y));
    }
  }

  pixel /= (kernel_size * 2 + 1);
  write_imageui(out, (int2)(coord.x, coord.y), pixel);
}
