# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens

from roc.point import Grid, Point


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
