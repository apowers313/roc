# mypy: disable-error-code="no-untyped-def"


from helpers.nethack_screens import screens
from helpers.nethack_screens2 import screens as screens2
from helpers.util import StubComponent, check_num_src_edges, check_points, check_type

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta
from roc.perception import Feature, Settled, VisionData


class TestDelta:
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
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 2378)  # .
        check_points(e.data, {(16, 6)})
        old = e.data.get_feature("Past")
        check_type(old, 413)  # f

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 413)  # f
        check_points(e.data, {(17, 6)})
        old = e.data.get_feature("Past")
        check_type(old, 2378)  # .

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
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 397)  # d
        check_points(e.data, {(4, 14)})
        old = e.data.get_feature("Past")
        check_type(old, 2378)  # .

        # # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 2378)  # .
        check_points(e.data, {(5, 14)})
        old = e.data.get_feature("Past")
        check_type(old, 397)  # d

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
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 2378)  # .
        check_points(e.data, {(16, 6)})
        old = e.data.get_feature("Past")
        check_type(old, 413)  # f

        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 413)  # f
        check_points(e.data, {(17, 6)})
        old = e.data.get_feature("Past")
        check_type(old, 2378)  # .

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
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 2316)
        check_points(e.data, {(18, 5)})
        old = e.data.get_feature("Past")
        check_type(old, 115)

        # fourth screen settled
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # fifth screen
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 413)  # f
        check_points(e.data, {(18, 5)})
        old = e.data.get_feature("Past")
        check_type(old, 2316)  # $

        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, 2378)  # .
        check_points(e.data, {(17, 6)})
        old = e.data.get_feature("Past")
        check_type(old, 413)  # f

        # fifth screen settled
        e = s.output.call_args_list[9].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_repr(self) -> None:
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
        assert isinstance(e.data, Feature)
        assert str(e.data) == "(16, 6): 413 -> 2378\n"

        # # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        assert str(e.data) == "(17, 6): 2378 -> 413\n"
