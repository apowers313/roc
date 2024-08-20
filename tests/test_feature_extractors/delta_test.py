# mypy: disable-error-code="no-untyped-def"


from helpers.nethack_screens import screens
from helpers.util import StubComponent, check_num_src_edges, check_points, check_type

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta
from roc.perception import Feature, Settled, VisionData

screen0 = VisionData.from_dict(screens[0])
screen1 = VisionData.from_dict(screens[1])
screen2 = VisionData.from_dict(screens[2])
screen3 = VisionData.from_dict(screens[3])
screen4 = VisionData.from_dict(screens[4])


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
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, ord("."))
        check_points(e.data, {(16, 6)})
        old = e.data.get_feature("Past")
        check_type(old, ord("f"))

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, ord("f"))
        check_points(e.data, {(17, 6)})
        old = e.data.get_feature("Past")
        check_type(old, ord("."))

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
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, ord("."))
        check_points(e.data, {(16, 6)})
        old = e.data.get_feature("Past")
        check_type(old, ord("f"))

        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, ord("f"))
        check_points(e.data, {(17, 6)})
        old = e.data.get_feature("Past")
        check_type(old, ord("."))

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
        check_type(e.data, ord("$"))
        check_points(e.data, {(18, 5)})
        old = e.data.get_feature("Past")
        check_type(old, ord("x"))

        # fourth screen settled
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # fifth screen
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, ord("f"))
        check_points(e.data, {(18, 5)})
        old = e.data.get_feature("Past")
        check_type(old, ord("$"))

        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        check_num_src_edges(e.data, 3)
        check_type(e.data, ord("."))
        check_points(e.data, {(17, 6)})
        old = e.data.get_feature("Past")
        check_type(old, ord("f"))

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
        assert isinstance(e.data, Feature)
        assert str(e.data) == "(16, 6): 102 'f' -> 46 '.'\n"

        # # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Feature)
        assert str(e.data) == "(17, 6): 46 '.' -> 102 'f'\n"
