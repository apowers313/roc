#include "test.h"

#define GET_XY_VAL(buf, width, x, y) ((buf)[(x) + ((y) * (width))])
#define SET_XY_VAL(buf, width, x, y, val) ((buf)[(x) + ((y) * (width))] = (val))

__device__ bool is_unique_from_neighbors(short *buf, int width, int height,
                                         int x, int y, short val) {
  const int max_width = width - 1;
  const int max_height = height - 1;
  // up left
  if (((x > 0 && y > 0) && (GET_XY_VAL(buf, width, x - 1, y - 1)) == val))
    return false;
  // up
  if ((y > 0) && (GET_XY_VAL(buf, width, x, y - 1) == val))
    return false;
  // up right
  if ((x < max_width) && (y > 0) &&
      (GET_XY_VAL(buf, width, x + 1, y - 1) == val))
    return false;
  // left
  if ((x > 0) && (GET_XY_VAL(buf, width, x - 1, y) == val))
    return false;
  // right
  if ((x < max_width) && (GET_XY_VAL(buf, width, x + 1, y) == val))
    return false;
  // down left
  if ((x > 0) && (y < max_height) &&
      (GET_XY_VAL(buf, width, x - 1, y + 1) == val))
    return false;
  // down right
  if ((x < max_width) && (y < max_height) &&
      (GET_XY_VAL(buf, width, x + 1, y + 1) == val))
    return false;

  return true;
}

__global__ void fe_single(short *out, int width, int height, short *in) {
  short val = GET_XY_VAL(in, width, MY_X, MY_Y);
  bool unique = is_unique_from_neighbors(in, width, height, MY_X, MY_Y, val);
  printf("(%d, %d) val: %d, unique %d\n", MY_X, MY_Y, val, unique);
  if (unique)
    SET_XY_VAL(out, width, MY_X, MY_Y, 1);
}

// def is_unique_from_neighbors(data: IntGrid, point: Point) -> bool:
//     """Helper function to determine if a point in a matrix has the same value
//     as any points around it.

//     Args:
//         data (VisionData): The matrix / Grid to evaluate
//         point (Point): The point to see if any of its neighbors have the same
//         value

//     Returns:
//         bool: Returns True if the point is different from all surrounding
//         points, False otherwise.
//     """
//     max_width = data.width - 1
//     max_height = data.height - 1
//     # up left
//     if point.x > 0 and point.y > 0 and data.get_val(point.x - 1, point.y - 1)
//     == point.val:
//         return False
//     # up
//     if point.y > 0 and data.get_val(point.x, point.y - 1) == point.val:
//         return False
//     # up right
//     if point.x < max_width and point.y > 0 and data.get_val(point.x + 1,
//     point.y - 1) == point.val:
//         return False
//     # left
//     if point.x > 0 and data.get_val(point.x - 1, point.y) == point.val:
//         return False
//     # right
//     if point.x < max_width and data.get_val(point.x + 1, point.y) ==
//     point.val:
//         return False
//     # down left
//     if point.x > 0 and point.y < max_height and data.get_val(point.x - 1,
//     point.y + 1) == point.val:
//         return False
//     # down
//     if point.y < max_height and data.get_val(point.x, point.y + 1) ==
//     point.val:
//         return False
//     # down right
//     if (
//         point.x < max_width
//         and point.y < max_height
//         and data.get_val(point.x + 1, point.y + 1) == point.val
//     ):
//         return False
//     return True
