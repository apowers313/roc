#include "test.h"

__global__ void test_kernel() {
  printf("(%d, %d, %d): Block (%d, %d, %d), Thread (%d, %d, %d) -- %d\n", MY_X,
         MY_Y, MY_Z, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
         threadIdx.y, threadIdx.z, MY_THING);
}