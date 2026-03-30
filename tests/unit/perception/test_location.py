# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/location.py."""

import numpy as np
import pytest

from roc.perception.location import (
    ChangedPoint,
    DebugGrid,
    Grid,
    GridStyle,
    IntGrid,
    Point,
    PointCollection,
    TextGrid,
    TypedPointCollection,
    XLoc,
    YLoc,
)


class TestPoint:
    def test_constructor(self):
        p = Point(XLoc(1), YLoc(2), 65)
        assert p.x == 1
        assert p.y == 2
        assert p.val == 65

    def test_hash(self):
        p1 = Point(XLoc(1), YLoc(2), 65)
        p2 = Point(XLoc(1), YLoc(2), 66)
        # Same location -> same hash
        assert hash(p1) == hash(p2)

    def test_hash_different_location(self):
        p1 = Point(XLoc(1), YLoc(2), 65)
        p2 = Point(XLoc(3), YLoc(4), 65)
        assert hash(p1) != hash(p2)

    def test_repr(self):
        p = Point(XLoc(1), YLoc(2), 65)
        r = repr(p)
        assert "(1, 2): 65 'A'" in r

    def test_eq_same(self):
        p1 = Point(XLoc(1), YLoc(2), 65)
        p2 = Point(XLoc(1), YLoc(2), 65)
        assert p1 == p2

    def test_eq_different_val(self):
        p1 = Point(XLoc(1), YLoc(2), 65)
        p2 = Point(XLoc(1), YLoc(2), 66)
        assert p1 != p2

    def test_eq_non_point(self):
        p = Point(XLoc(1), YLoc(2), 65)
        assert p != "not a point"
        assert p != 42

    def test_from_valloc(self):
        p = Point.from_valloc((XLoc(3), YLoc(4), 66))
        assert p.x == 3
        assert p.y == 4
        assert p.val == 66


class TestPointIsadjacent:
    def test_adjacent_horizontal(self):
        assert Point.isadjacent(xy1=(XLoc(0), YLoc(0)), xy2=(XLoc(1), YLoc(0))) is True

    def test_adjacent_vertical(self):
        assert Point.isadjacent(xy1=(XLoc(0), YLoc(0)), xy2=(XLoc(0), YLoc(1))) is True

    def test_adjacent_diagonal(self):
        assert Point.isadjacent(xy1=(XLoc(0), YLoc(0)), xy2=(XLoc(1), YLoc(1))) is True

    def test_same_point_not_adjacent(self):
        assert Point.isadjacent(xy1=(XLoc(0), YLoc(0)), xy2=(XLoc(0), YLoc(0))) is False

    def test_far_apart_not_adjacent(self):
        assert Point.isadjacent(xy1=(XLoc(0), YLoc(0)), xy2=(XLoc(3), YLoc(3))) is False

    def test_with_p1_and_xy2(self):
        p1 = Point(XLoc(5), YLoc(5), 0)
        assert Point.isadjacent(p1=p1, xy2=(XLoc(6), YLoc(5))) is True

    def test_with_p1_and_p2(self):
        p1 = Point(XLoc(5), YLoc(5), 0)
        p2 = Point(XLoc(6), YLoc(6), 0)
        assert Point.isadjacent(p1=p1, p2=p2) is True

    def test_bad_p1_args_raises(self):
        with pytest.raises(TypeError, match="bad p1"):
            Point.isadjacent(xy2=(XLoc(0), YLoc(0)))  # type: ignore

    def test_bad_p2_args_raises(self):
        with pytest.raises(TypeError, match="bad p2"):
            Point.isadjacent(xy1=(XLoc(0), YLoc(0)))  # type: ignore


class TestChangedPoint:
    def test_repr(self):
        cp = ChangedPoint(XLoc(1), YLoc(2), 66, 65)
        r = repr(cp)
        assert "(1, 2):" in r
        assert "65" in r
        assert "66" in r
        assert "->" in r


class TestGrid:
    def test_new_2d(self):
        g: Grid[int] = Grid(np.array([[1, 2], [3, 4]]))
        assert g.shape == (2, 2)

    def test_new_1d_raises(self):
        with pytest.raises(AssertionError):
            Grid(np.array([1, 2, 3]))

    def test_iter(self):
        g: Grid[int] = Grid(np.array([[10, 20], [30, 40]]))
        points = list(g)
        # Should yield (x, y, val) tuples
        assert (0, 0, 10) in points
        assert (1, 0, 20) in points
        assert (0, 1, 30) in points
        assert (1, 1, 40) in points

    def test_get_val(self):
        g: Grid[int] = Grid(np.array([[10, 20], [30, 40]]))
        assert g.get_val(0, 0) == 10
        assert g.get_val(1, 0) == 20
        assert g.get_val(0, 1) == 30

    def test_set_val(self):
        g: Grid[int] = Grid(np.array([[10, 20], [30, 40]]))
        g.set_val(1, 1, 99)
        assert g.get_val(1, 1) == 99

    def test_width_height(self):
        g: Grid[int] = Grid(np.array([[1, 2, 3], [4, 5, 6]]))
        assert g.width == 3
        assert g.height == 2


class TestIntGrid:
    def test_get_point(self):
        g = IntGrid(np.array([[65, 66], [67, 68]]))
        p = g.get_point(XLoc(1), YLoc(0))
        assert isinstance(p, Point)
        assert p.x == 1
        assert p.y == 0
        assert p.val == 66

    def test_points_iterator(self):
        g = IntGrid(np.array([[65, 66], [67, 68]]))
        pts = list(g.points())
        assert len(pts) == 4
        assert all(isinstance(p, Point) for p in pts)


class TestTextGrid:
    def test_str(self):
        # 'AB' on first row, 'CD' on second
        g = TextGrid(np.array([[65, 66], [67, 68]]))
        s = str(g)
        assert "AB" in s
        assert "CD" in s


class TestGridStyle:
    def test_dataclass_fields(self):
        gs = GridStyle(
            front_hue=0.5,
            front_saturation=0.5,
            front_brightness=0.5,
            back_hue=0.0,
            back_saturation=1.0,
            back_brightness=0.0,
            style="bold",
            val="X",
        )
        assert gs.front_hue == pytest.approx(0.5)
        assert gs.style == "bold"
        assert gs.val == "X"


class TestDebugGrid:
    @pytest.fixture()
    def debug_grid(self):
        ig = IntGrid(np.array([[65, 66], [67, 68]]))
        return DebugGrid(ig)

    def test_new(self, debug_grid):
        dg = debug_grid
        assert dg.width == 2
        assert dg.height == 2
        s = dg.get_val(0, 0)
        assert isinstance(s, GridStyle)
        assert s.val == "A"

    def test_set_style(self, debug_grid):
        dg = debug_grid
        dg.set_style(0, 0, front_hue=0.5, back_brightness=0.8)
        s = dg.get_val(0, 0)
        assert s.front_hue == pytest.approx(0.5)
        assert s.back_brightness == pytest.approx(0.8)

    def test_set_style_out_of_range(self, debug_grid):
        with pytest.raises(Exception, match="between 0 and 1"):
            debug_grid.set_style(0, 0, front_hue=1.5)

    def test_set_style_unknown_key(self, debug_grid):
        with pytest.raises(Exception, match="unknown key"):
            debug_grid.set_style(0, 0, bogus_key=0.5)

    def test_set_style_with_style_str(self, debug_grid):
        debug_grid.set_style(0, 0, style="bold")
        s = debug_grid.get_val(0, 0)
        assert s.style == "bold"

    def test_get_front_rgb(self, debug_grid):
        # Default: hue=0, sat=0, brightness=1 -> white (255, 255, 255)
        r, g, b = debug_grid.get_front_rgb(0, 0)
        assert r == 255
        assert g == 255
        assert b == 255

    def test_get_back_rgb(self, debug_grid):
        # Default: hue=0, sat=1, brightness=0 -> black (0, 0, 0)
        r, g, b = debug_grid.get_back_rgb(0, 0)
        assert r == 0
        assert g == 0
        assert b == 0

    def test_str(self, debug_grid):
        s = str(debug_grid)
        # Should contain the characters
        assert "A" in s
        assert "B" in s

    def test_to_html_vals(self, debug_grid):
        result = debug_grid.to_html_vals()
        assert "chars" in result
        assert "fg" in result
        assert "bg" in result
        assert len(result["chars"]) == 2  # 2 rows
        assert len(result["chars"][0]) == 2  # 2 cols
        # chars should be ord values
        assert result["chars"][0][0] == 65  # 'A'

    def test_blue_to_red_hue(self):
        # val=0 -> 2/3, val=1 -> 1.0
        assert DebugGrid.blue_to_red_hue(0.0) == pytest.approx(2 / 3)
        assert DebugGrid.blue_to_red_hue(1.0) == pytest.approx(1.0)

    def test_five_color_hue(self):
        # val=0 -> 2/3 (blue), val=1 -> 0 (red)
        assert DebugGrid.five_color_hue(0.0) == pytest.approx(2 / 3)
        assert DebugGrid.five_color_hue(1.0) == pytest.approx(0.0)


class TestPointCollection:
    def test_add_and_contains(self):
        pc = PointCollection([])
        p = Point(XLoc(1), YLoc(2), 65)
        pc.add(p)
        assert pc.contains(p) is True

    def test_contains_false(self):
        pc = PointCollection([])
        p = Point(XLoc(1), YLoc(2), 65)
        assert pc.contains(p) is False

    def test_init_with_points(self):
        p1 = Point(XLoc(0), YLoc(0), 65)
        p2 = Point(XLoc(1), YLoc(1), 66)
        pc = PointCollection([p1, p2])
        assert pc.size == 2

    def test_do_hash(self):
        pc = PointCollection([])
        p = Point(XLoc(3), YLoc(4), 65)
        assert pc.do_hash(p) == hash((3, 4))

    def test_size(self):
        pc = PointCollection([Point(XLoc(i), YLoc(i), 65) for i in range(5)])
        assert pc.size == 5

    def test_points_property(self):
        p1 = Point(XLoc(0), YLoc(0), 65)
        pc = PointCollection([p1])
        assert p1 in pc.points


class TestTypedPointCollection:
    def test_add_correct_type(self):
        tpc = TypedPointCollection(65, [])
        p = Point(XLoc(0), YLoc(0), 65)
        tpc.add(p)
        assert tpc.size == 1

    def test_add_wrong_type_raises(self):
        tpc = TypedPointCollection(65, [])
        p = Point(XLoc(0), YLoc(0), 66)
        with pytest.raises(TypeError, match="Trying to add"):
            tpc.add(p)

    def test_repr(self):
        tpc = TypedPointCollection(65, [Point(XLoc(0), YLoc(0), 65)])
        r = repr(tpc)
        assert "1 Points" in r
        assert "65" in r
        assert "A" in r  # chr(65)

    def test_do_hash(self):
        tpc = TypedPointCollection(65, [])
        p = Point(XLoc(1), YLoc(2), 65)
        # TypedPointCollection uses hash(p) not hash((x,y))
        assert tpc.do_hash(p) == hash(p)
