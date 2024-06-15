# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.motion import (
    Diff,
    Motion,
    MotionFeature,
    adjacent_direction,
)
from roc.perception import ElementOrientation, NewFeature, Settled, VisionData

screen0 = VisionData(screens[0]["chars"])
screen1 = VisionData(screens[1]["chars"])
screen4 = VisionData(screens[4]["chars"])
screen6 = VisionData(screens[6]["chars"])


class TestMotion:
    def test_exists(self, empty_components) -> None:
        Motion()

    # def test_print_screen(self) -> None:
    #     for line in screen0.screen:
    #         for ch in line:
    #             print(chr(ch), end="")
    #         print("")

    def test_direction(self) -> None:
        origin = Diff(x=0, y=0, old_val=120, new_val=0)
        d = adjacent_direction(origin, Diff(x=0, y=1, old_val=0, new_val=120))
        assert d == "DOWN"
        d = adjacent_direction(origin, Diff(x=0, y=-1, old_val=0, new_val=120))
        assert d == "UP"
        d = adjacent_direction(origin, Diff(x=-1, y=0, old_val=0, new_val=120))
        assert d == "LEFT"
        d = adjacent_direction(origin, Diff(x=1, y=0, old_val=0, new_val=120))
        assert d == "RIGHT"
        d = adjacent_direction(origin, Diff(x=-1, y=1, old_val=0, new_val=120))
        assert d == "DOWN_LEFT"
        d = adjacent_direction(origin, Diff(x=1, y=1, old_val=0, new_val=120))
        assert d == "DOWN_RIGHT"
        d = adjacent_direction(origin, Diff(x=-1, y=-1, old_val=0, new_val=120))
        assert d == "UP_LEFT"
        d = adjacent_direction(origin, Diff(x=1, y=-1, old_val=0, new_val=120))
        assert d == "UP_RIGHT"

    def test_basic(self, empty_components) -> None:
        d = Component.get("delta", "perception")
        assert isinstance(d, Delta)
        m = Component.get("motion", "perception")
        assert isinstance(m, Motion)
        s = StubComponent(
            input_bus=d.pb_conn.attached_bus,
            output_bus=m.pb_conn.attached_bus,
            filter=lambda e: e.src.name == "motion",
        )

        s.input_conn.send(screen0)
        s.input_conn.send(screen1)
        s.input_conn.send(screen4)
        s.input_conn.send(screen6)

        assert s.output.call_count == 8

        # screen 0
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # screen 1

        # ([EVENT: motion >>> perception]: 102 RIGHT: (16, 6) -> (17, 6),)
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, MotionFeature)
        assert e.data.get_type() == ord("f")
        dir = e.data.get_feature("Direction")
        assert isinstance(dir, ElementOrientation)
        assert dir.orientation == "RIGHT"
        v = e.data.get_type()
        assert v == 102
        end_pt = e.data.get_point()
        assert end_pt == (17, 6)
        origin = e.data.get_feature("Origin")
        assert isinstance(origin, NewFeature)
        start_pt = origin.get_point()
        assert start_pt == (16, 6)

        # # ([EVENT: motion >>> perception]: 46 LEFT: (17, 6) -> (16, 6),)
        # e = s.output.call_args_list[2].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, MotionFeature)
        # assert isinstance(e.data.feature, MotionVector)
        # v = e.data.feature
        # assert v.start_x == 17
        # assert v.start_y == 6
        # assert v.end_x == 16
        # assert v.end_y == 6
        # assert v.direction == "LEFT"
        # assert v.val == 46

        # e = s.output.call_args_list[3].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, Settled)

        # # screen 4

        # # ([EVENT: motion >>> perception]: 102 UP_RIGHT: (17, 6) -> (18, 5),)
        # e = s.output.call_args_list[4].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, MotionFeature)
        # assert isinstance(e.data.feature, MotionVector)
        # v = e.data.feature
        # assert v.start_x == 17
        # assert v.start_y == 6
        # assert v.end_x == 18
        # assert v.end_y == 5
        # assert v.direction == "UP_RIGHT"
        # assert v.val == 102

        # e = s.output.call_args_list[5].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, Settled)

        # # screen 6

        # # ([EVENT: motion >>> perception]: 102 DOWN: (18, 5) -> (18, 6),)
        # e = s.output.call_args_list[6].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, MotionFeature)
        # assert isinstance(e.data.feature, MotionVector)
        # v = e.data.feature
        # assert v.start_x == 18
        # assert v.start_y == 5
        # assert v.end_x == 18
        # assert v.end_y == 6
        # assert v.direction == "DOWN"
        # assert v.val == 102

        # e = s.output.call_args_list[7].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, Settled)
