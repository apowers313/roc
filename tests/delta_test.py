# mypy: disable-error-code="no-untyped-def"


from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta, DeltaFeature
from roc.perception import Settled, VisionData

screen0 = VisionData(screen=screens[0]["chars"])
screen1 = VisionData(screen=screens[1]["chars"])
screen2 = VisionData(screen=screens[2]["chars"])
screen3 = VisionData(screen=screens[3]["chars"])
screen4 = VisionData(screen=screens[4]["chars"])


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

        assert s.output.call_count == 4

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff
        # Diff(x=6, y=16, val1=102, val2=46), Diff(x=6, y=17, val1=46, val2=102)])
        assert d.x == 6
        assert d.y == 16
        assert d.old_val == 102  # f
        assert d.new_val == 46  # .

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff
        assert d.x == 6
        assert d.y == 17
        assert d.old_val == 46  # .
        assert d.new_val == 102  # f
        # kitten moved right!

        # fourth event
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_none(self, empty_components) -> None:
        c = Component.get("delta", "perception")
        assert isinstance(c, Delta)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        # same screen, no delta
        s.input_conn.send(screen0)
        s.input_conn.send(screen0)
        s.input_conn.send(screen0)

        assert s.output.call_count == 3

        # all events are none
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_multiple(self, empty_components) -> None:
        c = Component.get("delta", "perception")
        assert isinstance(c, Delta)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        # send multiple screen
        s.input_conn.send(screen0)
        s.input_conn.send(screen1)
        s.input_conn.send(screen2)
        s.input_conn.send(screen3)
        s.input_conn.send(screen4)

        assert s.output.call_count == 10

        # first screen
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # second screen
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff

        # Diff(x=6, y=16, val1=102, val2=46), Diff(x=6, y=17, val1=46, val2=102)])
        assert d.x == 6
        assert d.y == 16
        assert d.old_val == 102  # f
        assert d.new_val == 46  # .

        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff
        assert d.x == 6
        assert d.y == 17
        assert d.old_val == 46  # .
        assert d.new_val == 102  # f

        # second screen settled
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # third screen (nothing)
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # fourth screen
        e = s.output.call_args_list[5].args[0]
        print("fourth event", e)
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff
        # (5, 18): 120 -> 36
        assert d.x == 5
        assert d.y == 18
        assert d.old_val == 120  # x
        assert d.new_val == 36  # $
        # grid bug dies and drops money

        # fourth screen settled
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # fifth screen
        e = s.output.call_args_list[7].args[0]
        print("fifth event", e)
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff

        # (5, 18): 36 -> 102
        assert d.x == 5
        assert d.y == 18
        assert d.old_val == 36  # $
        assert d.new_val == 102  # f

        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        d = e.data.diff
        # (6, 17): 102 -> 46
        assert d.x == 6
        assert d.y == 17
        assert d.old_val == 102  # f
        assert d.new_val == 46  # .

        # fifth screen settled
        e = s.output.call_args_list[9].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
