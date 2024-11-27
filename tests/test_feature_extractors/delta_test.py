# mypy: disable-error-code="no-untyped-def"


from helpers.nethack_screens import screens
from helpers.nethack_screens2 import screens as screens2
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta, DeltaFeature, DeltaNode
from roc.location import XLoc, YLoc
from roc.perception import Settled, VisionData


class TestDelta:
    def test_to_nodes(self, fake_component) -> None:
        f = DeltaFeature(origin_id=("foo", "bar"), old_val=13, new_val=14, point=(XLoc(1), YLoc(2)))
        n = f.to_nodes()
        assert isinstance(n, DeltaNode)
        assert n.labels == {"Feature", "Delta"}
        assert n.old_val == 13
        assert n.new_val == 14

    def test_basic(self, empty_components) -> None:
        c = Component.get("delta", "perception")
        assert isinstance(c, Delta)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert s.output.call_count == 4

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.feature_name == "Delta"
        assert e.data.point == (16, 6)
        assert e.data.old_val == 413  # f
        assert e.data.new_val == 2378  # .

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (17, 6)
        assert e.data.old_val == 2378  # .
        assert e.data.new_val == 413  # f

        # fourth event
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_basic2(self, empty_components) -> None:
        c = Component.get("delta", "perception")
        assert isinstance(c, Delta)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens2[0]))
        s.input_conn.send(VisionData.from_dict(screens2[1]))

        assert s.output.call_count == 4

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (4, 14)
        assert e.data.old_val == 2378  # .
        assert e.data.new_val == 397  # d

        # # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (5, 14)
        assert e.data.old_val == 397  # d
        assert e.data.new_val == 2378  # .

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
        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[0]))

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
        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))
        s.input_conn.send(VisionData.from_dict(screens[2]))
        s.input_conn.send(VisionData.from_dict(screens[3]))
        s.input_conn.send(VisionData.from_dict(screens[4]))

        assert s.output.call_count == 10

        # first screen
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # second screen
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (16, 6)
        assert e.data.old_val == 413  # f
        assert e.data.new_val == 2378  # .

        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (17, 6)
        assert e.data.old_val == 2378  # .
        assert e.data.new_val == 413  # f

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
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (18, 5)
        assert e.data.old_val == 115
        assert e.data.new_val == 2316

        # fourth screen settled
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # fifth screen
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (18, 5)
        assert e.data.old_val == 2316  # $
        assert e.data.new_val == 413  # f

        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert e.data.point == (17, 6)
        assert e.data.old_val == 413  # f
        assert e.data.new_val == 2378  # .

        # fifth screen settled
        e = s.output.call_args_list[9].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_str(self) -> None:
        c = Component.get("delta", "perception")
        assert isinstance(c, Delta)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert s.output.call_count == 4

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert str(e.data) == "(16, 6): 413 -> 2378\n"

        # # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaFeature)
        assert str(e.data) == "(17, 6): 2378 -> 413\n"
