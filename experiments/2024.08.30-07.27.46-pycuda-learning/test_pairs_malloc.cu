#include "test.h"

__global__ void test_pairs(int2 *out)
{
    out = malloc(&out, 6 * sizeof(int2))
    int idx = MY_X;
    out[idx].x = MY_X + 10;
    out[idx].y = MY_Y + 100;
    printf("%d: (%d, %d)\n", idx, out[idx].x, out[idx].y);
}