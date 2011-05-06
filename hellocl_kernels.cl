// #pragma OPENCL EXTENSION cl_amd_printf : enable

float sum(float4 in) {
  return dot(in, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_FILTER_NEAREST |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

kernel void setup_system(read_only image2d_t source,
                         read_only image2d_t target,
                         write_only global uchar* a,
                         write_only image2d_t b,
                         write_only image2d_t x) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  int2 source_size = get_image_dim(source);
  if (coord.x >= source_size.x || coord.y >= source_size.y) {
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
  }
  size_t size_x = get_global_size(0);


  int2 coord_local = (int2)(get_local_id(0), get_local_id(1));
  local uint4 source_cache[BLOCKSIZE_X + 2][BLOCKSIZE_Y + 2];
  source_cache[coord_local.x + 1][coord_local.y + 1] =
                             read_imageui(source, sampler, coord);
  local uint4 target_cache[BLOCKSIZE_X + 2][BLOCKSIZE_Y + 2];
  target_cache[coord_local.x + 1][coord_local.y + 1] =
                             read_imageui(target, sampler, coord);

  uint4 pixel = source_cache[coord_local.x + 1][coord_local.y + 1];
  if (pixel.w) {
    // Cache images in local memory
    if (coord_local.x == 0) {
      source_cache[0][coord_local.y + 1] =
                             read_imageui(source, sampler, coord + (int2)(-1, 0));
    } else if (coord_local.x == BLOCKSIZE_X - 1) {
      source_cache[BLOCKSIZE_X + 1][coord_local.y + 1] =
                             read_imageui(source, sampler, coord + (int2)(1, 0));
    }
    if (coord_local.y == 0) {
      source_cache[coord_local.x + 1][0] =
                             read_imageui(source, sampler, coord + (int2)(0, -1));
    } else if (coord_local.y == BLOCKSIZE_Y - 1) {
      source_cache[coord_local.x + 1][BLOCKSIZE_Y + 1] =
                             read_imageui(source, sampler, coord + (int2)(0, 1));
    }
    if (coord_local.x == 0) {
      target_cache[0][coord_local.y + 1] =
                             read_imageui(target, sampler, coord + (int2)(-1, 0));
    } else if (coord_local.x == BLOCKSIZE_X - 1) {
      target_cache[BLOCKSIZE_X + 1][coord_local.y + 1] =
                             read_imageui(target, sampler, coord + (int2)(1, 0));
    }
    if (coord_local.y == 0) {
      target_cache[coord_local.x + 1][0] =
                             read_imageui(target, sampler, coord + (int2)(0, -1));
    } else if (coord_local.y == BLOCKSIZE_Y - 1) {
      target_cache[coord_local.x + 1][BLOCKSIZE_Y + 1] =
                             read_imageui(target, sampler, coord + (int2)(0, 1));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint4 source_m = source_cache[coord_local.x + 1][coord_local.y + 1];
    uint4 source_u = source_cache[coord_local.x + 1][coord_local.y];
    uint4 source_l = source_cache[coord_local.x][coord_local.y + 1];
    uint4 source_d = source_cache[coord_local.x + 1][coord_local.y + 2];
    uint4 source_r = source_cache[coord_local.x + 2][coord_local.y + 1];
    bool u_o = source_u.w;
    bool l_o = source_l.w;
    bool d_o = source_d.w;
    bool r_o = source_r.w;
    uint4 target_u = target_cache[coord_local.x + 1][coord_local.y];
    uint4 target_l = target_cache[coord_local.x][coord_local.y + 1];
    uint4 target_d = target_cache[coord_local.x + 1][coord_local.y + 2];
    uint4 target_r = target_cache[coord_local.x + 2][coord_local.y + 1];
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
    uint4 tmp = target_cache[coord_local.x + 1][coord_local.y + 1];
    tmp.w = 255;
    write_imagef(b, coord, convert_float4(tmp));

    barrier(CLK_LOCAL_MEM_FENCE);
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
