const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_FILTER_NEAREST |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void gpu_write_solution(read_only image2d_t x,
                               write_only image2d_t gpu) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  float4 pixel = read_imagef(x, sampler, coord);
  pixel /= 255.0f;
  write_imagef(gpu, coord, pixel);
}

kernel void gpu_write_residual(read_only image2d_t residual,
                               write_only image2d_t gpu) {
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  float4 pixel = read_imagef(residual, sampler, coord);
  pixel.w = 0.0f;
  pixel = pixel * pixel * 255.0f;
  write_imagef(gpu, coord, pixel);
}
