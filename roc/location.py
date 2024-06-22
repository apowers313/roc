from __future__ import annotations

from colorsys import hsv_to_rgb
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar, overload

from colored import Back, Fore, Style


class Point:
    def __init__(self, x: int, y: int, val: int) -> None:
        self.x = x
        self.y = y
        self.val = val

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}): {self.val} '{chr(self.val)}'"

    def __eq__(self, p: Any) -> bool:
        if not isinstance(p, Point):
            return False
        return self.x == p.x and self.y == p.y and self.val == p.val

    @overload
    @staticmethod
    def isadjacent(*, x1: int, y1: int, x2: int, y2: int) -> bool: ...

    @overload
    @staticmethod
    def isadjacent(*, p1: Point, x2: int, y2: int) -> bool: ...

    @overload
    @staticmethod
    def isadjacent(*, p1: Point, p2: Point) -> bool: ...

    @staticmethod
    def isadjacent(
        # o1: int | Point, o2: int | Point, o3: int | None = None, o4: int | None = None
        *,
        x1: int | None = None,
        y1: int | None = None,
        x2: int | None = None,
        y2: int | None = None,
        p1: Point | None = None,
        p2: Point | None = None,
    ) -> bool:
        if isinstance(p1, Point):
            x1 = p1.x
            y1 = p1.y
        elif not isinstance(x1, int) or not isinstance(y1, int):
            raise TypeError("bad p1 arguments in isadjacent()")

        if isinstance(p2, Point):
            x2 = p2.x
            y2 = p2.y
        elif not isinstance(x2, int) or not isinstance(y2, int):
            raise TypeError("bad p2 arguments in isadjacent()")

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx == 0 and dy == 0:
            return False

        return dx <= 1 and dy <= 1


PointList = list[Point]


class ChangedPoint(Point):
    def __init_(self, x: int, y: int, val: int, old_val: int) -> None:
        super().__init__(x, y, val)
        self.old_val = old_val

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}): {self.old_val} '{chr(self.old_val)}' -> {self.val} '{chr(self.val)}'"  # noqa: E501


GridVal = TypeVar("GridVal")
ValList = list[list[GridVal]]


class GenericGrid(Generic[GridVal]):
    """A rectangular array of points"""

    def __init__(self, val_list: ValList[GridVal]) -> None:
        self.val_list = val_list

    def __iter__(self) -> Iterator[GridVal]:
        """Iterate over all the points in the grid"""

        for y in range(self.height):
            for x in range(self.width):
                yield self.get_val(x, y)

    def get_val(self, x: int, y: int) -> GridVal:
        """Returns the value located at (x, y)"""

        return self.val_list[y][x]

    def set_val(self, x: int, y: int, val: GridVal) -> None:
        self.val_list[y][x] = val

    @property
    def width(self) -> int:
        return len(self.val_list[0])

    @property
    def height(self) -> int:
        return len(self.val_list)

    @staticmethod
    def filled(val: int, width: int, height: int) -> Grid:
        cols = [val for x in range(width)]
        rows = [cols.copy() for x in range(height)]
        return Grid(rows)


class Grid(GenericGrid[int]):
    def get_point(self, x: int, y: int) -> Point:
        """Returns the Point located at (x, y)"""

        return Point(x, y, self.get_val(x, y))

    def __repr__(self) -> str:
        ret = ""
        for line in self.val_list:
            for ch in line:
                ret += chr(ch)
            ret += "\n"
        return ret

    def points(self) -> Iterator[Point]:
        """Iterate over all the points in the grid"""

        for y in range(self.height):
            for x in range(self.width):
                yield self.get_point(x, y)


@dataclass
class GridStyle:
    front_hue: float
    front_saturation: float
    front_brightness: float
    back_hue: float
    back_saturation: float
    back_brightness: float
    style: str
    val: str


class DebugGrid(GenericGrid[GridStyle]):
    def __init__(self, grid: Grid) -> None:
        width = grid.width
        height = grid.height
        map: list[list[GridStyle]] = [
            [
                GridStyle(
                    front_hue=0,
                    front_saturation=0,
                    front_brightness=1,
                    back_hue=0,
                    back_saturation=1,
                    back_brightness=0,
                    val=" ",
                    style="none",
                )
                for col in range(width)
            ]
            for row in range(height)
        ]
        super().__init__(map)

        # copy over all the values from the grid
        for p in grid.points():
            s = self.get_val(p.x, p.y)
            s.val = chr(p.val)

    def set_style(self, x: int, y: int, *, style: str | None = None, **kwargs: float) -> None:
        s = self.get_val(x, y)

        if style:
            s.style = style

        for key, value in kwargs.items():
            if value < 0 or value > 1:
                raise Exception(
                    f"set_style expects values to be floats between 0 and 1, '{key}' was '{value}'"
                )

            match key:
                case "front_hue":
                    s.front_hue = value
                case "front_brightness":
                    s.front_brightness = value
                case "front_saturation":
                    s.front_saturation = value
                case "back_hue":
                    s.back_hue = value
                case "back_brightness":
                    s.back_brightness = value
                case "back_saturation":
                    s.back_saturation = value
                case _:
                    raise Exception(f"unknown key '{key}' in set_style")

    def get_front_rgb(self, x: int, y: int) -> tuple[int, int, int]:
        s = self.get_val(x, y)
        rgb = hsv_to_rgb(s.front_hue, s.front_saturation, s.front_brightness)
        ret = tuple(map(lambda c: round(c * 255), rgb))
        assert len(ret) == 3
        return ret

    def get_back_rgb(self, x: int, y: int) -> tuple[int, int, int]:
        s = self.get_val(x, y)
        rgb = hsv_to_rgb(s.back_hue, s.back_saturation, s.back_brightness)
        ret = tuple(map(lambda c: round(c * 255), rgb))
        assert len(ret) == 3
        return ret

    def __str__(self) -> str:
        ret = ""
        for y in range(self.height):
            for x in range(self.width):
                s = self.get_val(x, y)
                fr, fg, fb = self.get_front_rgb(x, y)
                br, bg, bb = self.get_back_rgb(x, y)
                ret += f"{Fore.rgb(fr,fg,fb)}{Back.rgb(br,bg,bb)}{s.val}{Style.reset}"
            ret += "\n"
        return ret


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

    def __repr__(self) -> str:
        return f"{len(self._point_hash)} Points: {self.type} ({chr(self.type)})"

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
