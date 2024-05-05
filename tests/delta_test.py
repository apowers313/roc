# mypy: disable-error-code="no-untyped-def"


from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta, DeltaFeature
from roc.perception import NONE_FEATURE, VisionData

screen0 = VisionData(screen=screens[0]["chars"])
screen1 = VisionData(screen=screens[1]["chars"])


class TestDelta:
    def test_basic(self, empty_components) -> None:
        c = Component.get("delta", "perception")
        assert isinstance(c, Delta)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(screen0)
        s.input_conn.send(screen1)

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert e.data.feature is NONE_FEATURE

        # second event
        assert len(s.output.call_args_list) == 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data.feature, DeltaFeature)
        diff_list = e.data.feature.diff_list
        assert len(diff_list) == 2
        # Diff(x=6, y=16, val1=102, val2=46), Diff(x=6, y=17, val1=46, val2=102)])
        assert diff_list[0].x == 6
        assert diff_list[0].y == 16
        assert diff_list[0].val1 == 102  # f
        assert diff_list[0].val2 == 46  # .
        assert diff_list[1].x == 6
        assert diff_list[1].y == 17
        assert diff_list[1].val1 == 46  # .
        assert diff_list[1].val2 == 102  # f
        # kitten moved right!
