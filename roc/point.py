from __future__ import annotations

from typing import Any, Iterator

ValList = list[list[int]]


class Point:
    def __init__(self, x: int, y: int, val: int) -> None:
        self.x = x
        self.y = y
        self.val = val

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}): {self.val}"

    def __eq__(self, p: Any) -> bool:
        if not isinstance(p, Point):
            return False
        return self.x == p.x and self.y == p.y and self.val == p.val


PointList = list[Point]


class ChangedPoint(Point):
    def __init_(self, x: int, y: int, val: int, old_val: int) -> None:
        super().__init__(x, y, val)
        self.old_val = old_val

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}): {self.old_val} -> {self.val}"


class Grid:
    """A rectangular array of points"""

    def __init__(self, val_list: ValList) -> None:
        self.val_list = val_list

    def __iter__(self) -> Iterator[Point]:
        """Iterate over all the points in the grid"""

        for y in range(self.height):
            for x in range(self.width):
                yield self.get_point(x, y)

    def get_point(self, x: int, y: int) -> Point:
        """Returns the Point located at (x, y)"""

        return Point(x, y, self.get_val(x, y))

    def get_val(self, x: int, y: int) -> int:
        """Returns the value located at (x, y)"""

        return self.val_list[y][x]

    def set_val(self, x: int, y: int, val: int) -> None:
        self.val_list[y][x] = val

    @property
    def width(self) -> int:
        return len(self.val_list[0])

    @property
    def height(self) -> int:
        return len(self.val_list)

    def __repr__(self) -> str:
        ret = ""
        for line in self.val_list:
            for ch in line:
                ret += chr(ch)
            ret += "\n"
        return ret

    @staticmethod
    def filled(val: int, width: int, height: int) -> Grid:
        cols = [val for x in range(width)]
        rows = [cols.copy() for x in range(height)]
        return Grid(rows)


class PointCollection:
    """A collection of abitrary points"""

    def __init__(self, point_list: PointList) -> None:
        self._point_hash: dict[int, Point] = {}
        for p in point_list:
            self.add(p)

    def add(self, p: Point) -> None:
        """Adds a new point to the collection"""

        hash_val = self.do_hash(p)
        self._point_hash[hash_val] = p

    def contains(self, p: Point) -> bool:
        """Verifies if a point is already in the collection or not"""

        hash_val = self.do_hash(p)
        return hash_val in self._point_hash

    def do_hash(self, p: Point) -> int:
        """Returns the hash value of a point. Mostly for internal use."""

        return hash((p.x, p.y))

    @property
    def size(self) -> int:
        return len(self._point_hash)

    @property
    def points(self) -> PointList:
        return list(self._point_hash.values())


class TypedPointCollection(PointCollection):
    """A collection of points that all have the same type"""

    def __init__(self, type: int, point_list: PointList) -> None:
        self.type = type
        super().__init__(point_list)

    def do_hash(self, p: Point) -> int:
        return hash(p)

    def add(self, p: Point) -> None:
        """Add a new point to the collection and enforce that it is the same
        type as the collection"""
        if p.val != self.type:
            raise TypeError(
                f"Trying to add '{p.val}' to TypedPointCollection with type '{self.type}"
            )

        super().add(p)