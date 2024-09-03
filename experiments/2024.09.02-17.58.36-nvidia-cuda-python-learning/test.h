#define MY_X (threadIdx.x + blockIdx.x * blockDim.x)
#define MY_Y (threadIdx.y + blockIdx.y * blockDim.y)
#define MY_Z (threadIdx.z + blockIdx.z * blockDim.z)
#define MY_THING 42
#define IDX_2D(x,y,width) ((x) + (y)*width)