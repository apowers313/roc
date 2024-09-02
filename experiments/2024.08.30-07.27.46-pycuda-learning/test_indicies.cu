#include "test.h"

__global__ void test_indicies(int width, int out[])
{
    int idx = MY_X << 8 | MY_Y;
    out[IDX_2D(MY_X, MY_Y, width)] = idx;
    printf("(%d, %d): out[%d] = %d\n", MY_X, MY_Y, IDX_2D(MY_X, MY_Y, width), idx);
    // printf("width is %d\n", width);
    // printf("out is %p\n", out);
}