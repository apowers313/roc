extern "C" __global__ void saxpy(float a, float *x, float *y, float *out,
                                 size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  int xxx = 3;
  // there's a huge number of threads, so this printf effectively hangs the
  // program
  //   printf("foo\n");

  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}