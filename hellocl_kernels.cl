__kernel void hello(__read_only image2d_t in,
                    __write_only image2d_t out,
                    int vertical) {

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_FILTER_NEAREST |
                            CLK_ADDRESS_CLAMP_TO_EDGE;
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
