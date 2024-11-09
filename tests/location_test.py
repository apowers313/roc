# mypy: disable-error-code="no-untyped-def"

import pytest
from colored import Back, Fore, Style
from helpers.nethack_screens import screens

from roc.location import (
    DebugGrid,
    IntGrid,
    Point,
    PointCollection,
    TextGrid,
    TypedPointCollection,
    XLoc,
    YLoc,
)


class TestPoint:
    def test_equality(self) -> None:
        p1 = Point(XLoc(42), YLoc(69), 255)
        p2 = Point(XLoc(42), YLoc(69), 255)
        assert p1 == p2

    def test_hash_equality(self) -> None:
        p1 = Point(XLoc(42), YLoc(69), 255)
        p2 = Point(XLoc(42), YLoc(69), 255)
        h1 = hash(p1)
        h2 = hash(p2)
        assert h1 == h2

    def test_hash_inequality(self) -> None:
        p1 = Point(XLoc(42), YLoc(69), 255)
        p2 = Point(XLoc(43), YLoc(69), 255)
        h1 = hash(p1)
        h2 = hash(p2)
        assert h1 != h2

    def test_repr(self) -> None:
        p1 = Point(XLoc(42), YLoc(69), 65)
        assert repr(p1) == "(42, 69): 65 'A'"

    def test_isadjacent(self) -> None:
        p1 = Point(XLoc(42), YLoc(69), 255)
        p2 = Point(XLoc(42), YLoc(70), 255)
        p3 = Point(XLoc(42), YLoc(71), 255)

        assert Point.isadjacent(p1=p1, p2=p2)
        assert not Point.isadjacent(p1=p1, p2=p3)

    def test_isadjacent_args(self) -> None:
        p1 = Point(XLoc(42), YLoc(69), 255)
        # p2 = Point(XLoc(42), 70, 255)
        # p3 = Point(XLoc(42), 71, 255)

        assert Point.isadjacent(p1=p1, x2=XLoc(42), y2=YLoc(70))
        assert Point.isadjacent(x1=XLoc(42), y1=YLoc(69), x2=XLoc(42), y2=YLoc(70))
        # with pytest.raises(TypeError):
        #     assert Point.isadjacent(p1=p1, x1=42, y1=70)

    def test_adjacent(self) -> None:
        origin = Point(x=XLoc(0), y=YLoc(0), val=0)
        res = Point.isadjacent(p1=origin, p2=Point(x=XLoc(0), y=YLoc(1), val=120))
        assert res is True
        res = Point.isadjacent(p1=origin, p2=Point(x=XLoc(0), y=YLoc(2), val=120))
        assert res is False
        res = Point.isadjacent(p1=origin, p2=Point(x=XLoc(1), y=YLoc(0), val=120))
        assert res is True
        res = Point.isadjacent(p1=origin, p2=Point(x=XLoc(2), y=YLoc(0), val=120))
        assert res is False
        res = Point.isadjacent(p1=origin, p2=Point(x=XLoc(1), y=YLoc(1), val=120))
        assert res is True
        res = Point.isadjacent(p1=origin, p2=Point(x=XLoc(2), y=YLoc(1), val=120))
        assert res is False
        res = Point.isadjacent(p1=origin, p2=origin)
        assert res is False


class TestGrid:
    def test_grid(self) -> None:
        screen0 = IntGrid(screens[0]["chars"])
        assert isinstance(screen0, IntGrid)
        assert screen0.width == 79
        assert screen0.height == 21
        assert screen0.get_point(XLoc(0), YLoc(0)) == Point(XLoc(0), YLoc(0), 32)
        # for p in screen0:
        #     assert p == 32
        #     break

    def test_text_grid_str(self) -> None:
        val = [
            [32, 32, 32],
            [49, 50, 51],
            [97, 98, 99],
        ]
        g = TextGrid(val)
        assert str(g) == "   \n123\nabc\n"


class TestDebugGrid:
    def test_basic(self) -> None:
        val = [
            [32, 32, 32],
            [49, 50, 51],
            [97, 98, 99],
        ]
        g = IntGrid(val)
        dg = DebugGrid(g)
        # print(str(dg))
        # white on black
        assert str(dg) == (
            f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)} {Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)} {Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)} {Style.reset}"
            + "\n"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}1{Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}2{Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}3{Style.reset}"
            + "\n"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}a{Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}b{Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}c{Style.reset}"
            + "\n"
        )

    def test_set_style(self) -> None:
        val = [
            [32, 32, 32],
            [49, 50, 51],
            [97, 98, 99],
        ]
        g = IntGrid(val)
        dg = DebugGrid(g)
        dg.set_style(1, 1, front_brightness=1, front_saturation=1, front_hue=0.5)  # fore blue
        dg.set_style(
            0,
            2,
            front_brightness=1,
            front_saturation=1,
            front_hue=0.5,
            back_hue=(2 / 3),
            back_brightness=1,
            back_saturation=1,
        )  # fore red, back blue
        dg.set_style(2, 2, back_hue=(2 / 3), back_brightness=1, back_saturation=1)  # back blue
        # print(str(dg))
        # middle blue, bottom left blue on blue, bottom right blue
        assert str(dg) == (
            f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)} {Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)} {Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)} {Style.reset}"
            + "\n"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}1{Style.reset}"
            + f"{Fore.rgb(0, 255, 255)}{Back.rgb(0, 0, 0)}2{Style.reset}"  # fore blue
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}3{Style.reset}"
            + "\n"
            + f"{Fore.rgb(0, 255, 255)}{Back.rgb(0, 0, 255)}a{Style.reset}"  # blue on blue
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 0)}b{Style.reset}"
            + f"{Fore.rgb(255, 255, 255)}{Back.rgb(0, 0, 255)}c{Style.reset}"  # back blue
            + "\n"
        )


class TestPointCollection:
    def test_create(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 128)
        p3 = Point(XLoc(5), YLoc(6), 64)
        pc = PointCollection([p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)

    def test_add(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 128)
        p3 = Point(XLoc(5), YLoc(6), 64)
        pc = PointCollection([p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)
        pc.add(p3)
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert pc.contains(p3)

    def test_get_points(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 128)
        p3 = Point(XLoc(5), YLoc(6), 64)
        pc = PointCollection([p1, p2, p3])
        assert pc.points == [p1, p2, p3]


class TestTypedPointCollection:
    def test_create(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 255)
        p3 = Point(XLoc(5), YLoc(6), 255)
        pc = TypedPointCollection(255, [p1, p2])
        assert pc.type == 255
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)

    def test_create_wrong_type(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 255)
        p3 = Point(XLoc(5), YLoc(6), 64)
        with pytest.raises(TypeError):
            TypedPointCollection(255, [p1, p2, p3])

    def test_add(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 255)
        p3 = Point(XLoc(5), YLoc(6), 255)
        pc = TypedPointCollection(255, [p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)
        pc.add(p3)
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert pc.contains(p3)

    def test_add_wrong_type(self) -> None:
        p1 = Point(XLoc(1), YLoc(2), 255)
        p2 = Point(XLoc(3), YLoc(4), 255)
        p3 = Point(XLoc(5), YLoc(6), 64)
        pc = TypedPointCollection(255, [p1, p2])
        assert pc.contains(p1)
        assert pc.contains(p2)
        assert not pc.contains(p3)
        with pytest.raises(TypeError):
            pc.add(p3)
