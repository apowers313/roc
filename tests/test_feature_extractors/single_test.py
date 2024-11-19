# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.single import (
    Single,
    SingleFeature,
    is_unique_from_neighbors,
)
from roc.location import IntGrid, Point, XLoc, YLoc
from roc.perception import Settled, VisionData


class TestSingle:
    def test_single_exists(self) -> None:
        Single()

    def test_basic(self, empty_components) -> None:
        c = Component.get("single", "perception")
        assert isinstance(c, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 2

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.feature_name == "Single"
        assert e.data.point == (1, 1)
        assert e.data.type == 1

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_two_singles(self, empty_components) -> None:
        c = Component.get("single", "perception")
        assert isinstance(c, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2],
                ]
            )
        )

        assert s.output.call_count == 3

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (0, 0)
        assert e.data.type == 1

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (4, 4)
        assert e.data.type == 2

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_screen0(self, empty_components) -> None:
        c = Component.get("single", "perception")
        assert isinstance(c, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))

        assert s.output.call_count == 14

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (15, 3)
        assert e.data.type == 2362  # -

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (19, 3)
        assert e.data.type == 2363  # -

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (15, 4)
        assert e.data.type == 2360  # |

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (19, 4)
        assert e.data.type == 2360  # |

        # event 5
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (15, 5)
        assert e.data.type == 2371  # .

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (16, 5)
        assert e.data.type == 2017  # [

        # event 7
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (17, 5)
        assert e.data.type == 333  # @

        # event 8
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (18, 5)
        assert e.data.type == 115  # x

        # event 9
        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (19, 5)
        assert e.data.type == 2374  # +

        # event 10
        e = s.output.call_args_list[9].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (16, 6)
        assert e.data.type == 413  # f

        # event 11
        e = s.output.call_args_list[10].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (15, 8)
        assert e.data.type == 2364  # -

        # event 12
        e = s.output.call_args_list[11].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (18, 8)
        assert e.data.type == 2373  # |)

        # event 13
        e = s.output.call_args_list[12].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        assert e.data.point == (19, 8)
        assert e.data.type == 2365  # -

        # event 14
        e = s.output.call_args_list[13].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_unique(self) -> None:
        d = IntGrid(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(0), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(1), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(2), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(0), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(2), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(0), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(1), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        )
        assert is_unique_from_neighbors(d, Point(XLoc(2), YLoc(2), 1))

    def test_not_unique(self) -> None:
        d = IntGrid(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(0), 1))

        d = IntGrid(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(0), 1))

        d = IntGrid(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(0), 1))

        d = IntGrid(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(0), 1))

        d = IntGrid(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(0), 1))

        d = IntGrid(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(1), 1))

        d = IntGrid(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(1), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(0), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 1],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(1), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(2), 1))

        d = IntGrid(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 1],
            ]
        )
        assert not is_unique_from_neighbors(d, Point(XLoc(2), YLoc(2), 1))
