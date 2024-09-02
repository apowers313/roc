#include "test.h"

__global__ void dump_buf(int width, int height, char *buf)
{
    // printf("width %d, height %d\n", width, height);
    char val = buf[MY_X + MY_Y * width];
    printf("(%d, %d): val %d\n", MY_X, MY_Y, val);
}