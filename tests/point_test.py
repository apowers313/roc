# mypy: disable-error-code="no-untyped-def"

import pytest
from helpers.nethack_screens import screens

from roc.point import Grid, Point, PointCollection, TypedPointCollection


class TestPoint:
    def test_equality(self) -> None:
        p1 = Point(42, 69, 255)
        p2 = Point(42, 69, 255)
        assert p1 == p2

    def test_hash_equality(self) -> None:
        p1 = Point(42, 69, 255)
        p2 = Point(42, 69, 255)
        h1 = hash(p1)
        h2 = hash(p2)
        assert h1 == h2

    def test_hash_inequality(self) -> None:
        p1 = Point(42, 69, 255)
        p2 = Point(43, 69, 255)
        h1 = hash(p1)
        h2 = hash(p2)
        assert h1 != h2

    def test_repr(self) -> None:
        p1 = Point(42, 69, 255)
        assert repr(p1) == "(42, 69): 255"


class TestGrid:
    def test_grid(self) -> None:
        screen0 = Grid(screens[0]["chars"])
        assert isinstance(screen0, Grid)
        assert screen0.width == 79
        assert screen0.height == 21
        assert screen0.get_point(0, 0) == Point(0, 0, 32)
        for p in screen0:
            assert p == Point(0, 0, 32)
            break

    def test_grid_repr(self) -> None:
        val = [
            [32, 32, 32],
            [49, 50, 51],
            [97, 98, 99],
        ]
        g = Grid(val)
        assert repr(g) == "   \n123\nabc\n"

        screen0 = Grid(screens[0]["chars"])
        print(screen0)


class TestPointCollection:
    def test_create(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 128)
        p3 = Point(5, 6, 64)
        pc = PointCollection([p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)

    def test_add(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 128)
        p3 = Point(5, 6, 64)
        pc = PointCollection([p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)
        pc.add(p3)
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert pc.contains(p3)

    def test_get_points(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 128)
        p3 = Point(5, 6, 64)
        pc = PointCollection([p1, p2, p3])
        assert pc.points == [p1, p2, p3]


class TestTypedPointCollection:
    def test_create(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 255)
        p3 = Point(5, 6, 255)
        pc = TypedPointCollection(255, [p1, p2])
        assert pc.type == 255
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)

    def test_create_wrong_type(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 255)
        p3 = Point(5, 6, 64)
        with pytest.raises(TypeError):
            TypedPointCollection(255, [p1, p2, p3])

    def test_add(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 255)
        p3 = Point(5, 6, 255)
        pc = TypedPointCollection(255, [p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)
        pc.add(p3)
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert pc.contains(p3)

    def test_add_wrong_type(self) -> None:
        p1 = Point(1, 2, 255)
        p2 = Point(3, 4, 255)
        p3 = Point(5, 6, 64)
        pc = TypedPointCollection(255, [p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)
        with pytest.raises(TypeError):
            pc.add(p3)
