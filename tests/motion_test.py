# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.feature_extractors.delta import Delta, Diff
from roc.feature_extractors.motion import Motion, adjacent_direction
from roc.perception import VisionData

screen0 = VisionData(screen=screens[0]["chars"])
screen1 = VisionData(screen=screens[1]["chars"])
screen4 = VisionData(screen=screens[4]["chars"])
screen6 = VisionData(screen=screens[6]["chars"])


class TestMotion:
    def test_exists(self, empty_components) -> None:
        Motion()

    def test_direction(self) -> None:
        origin = Diff(x=0, y=0, old_val=120, new_val=0)
        d = adjacent_direction(origin, Diff(x=0, y=1, old_val=0, new_val=120))
        assert d == "UP"
        d = adjacent_direction(origin, Diff(x=0, y=-1, old_val=0, new_val=120))
        assert d == "DOWN"
        d = adjacent_direction(origin, Diff(x=-1, y=0, old_val=0, new_val=120))
        assert d == "LEFT"
        d = adjacent_direction(origin, Diff(x=1, y=0, old_val=0, new_val=120))
        assert d == "RIGHT"
        d = adjacent_direction(origin, Diff(x=-1, y=1, old_val=0, new_val=120))
        assert d == "UP_LEFT"
        d = adjacent_direction(origin, Diff(x=1, y=1, old_val=0, new_val=120))
        assert d == "UP_RIGHT"
        d = adjacent_direction(origin, Diff(x=-1, y=-1, old_val=0, new_val=120))
        assert d == "DOWN_LEFT"
        d = adjacent_direction(origin, Diff(x=1, y=-1, old_val=0, new_val=120))
        assert d == "DOWN_RIGHT"

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

        # assert s.output.call_count == 4

        print("output call count", s.output.call_count)
        print("output 0", s.output.call_args_list[0][0])
        print("output 1", s.output.call_args_list[1][0])
        print("output 2", s.output.call_args_list[2][0])
        print("output 3", s.output.call_args_list[2][0])
        print("output 4", s.output.call_args_list[2][0])
