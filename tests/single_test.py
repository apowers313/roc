# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent, check_num_src_edges, check_points, check_type

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.single import Single, SingleFeature
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
            VisionData(
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
        check_num_src_edges(e.data, 2)
        check_type(e.data, 1)
        check_points(e.data, {(1, 1)})

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_screen0(self, empty_components) -> None:
        c = Component.get("single", "perception")
        assert isinstance(c, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData(screens[0]["chars"]))

        assert s.output.call_count == 9

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("|"))
        check_points(e.data, {(15, 4)})

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("|"))
        check_points(e.data, {(19, 4)})

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("["))
        check_points(e.data, {(16, 5)})

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("@"))
        check_points(e.data, {(17, 5)})

        # event 5
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("x"))
        check_points(e.data, {(18, 5)})

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("+"))
        check_points(e.data, {(19, 5)})

        # event 7
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("f"))
        check_points(e.data, {(16, 6)})

        # event 8
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SingleFeature)
        check_type(e.data, ord("-"))
        check_points(e.data, {(19, 8)})

        # event 9
        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
