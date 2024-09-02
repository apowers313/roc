__global__ void k1() {
  printf("kernel 1 starting...\n");
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  printf("kernel 1 done.\n");
}

__global__ void k2() {
  printf("kernel 2 starting...\n");
  printf("kernel 2 done.\n");
}
