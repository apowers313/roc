# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import (
    StubComponent,
    check_num_src_edges,
    check_points,
    check_size,
    check_type,
)

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.flood import Flood, FloodFeature
from roc.perception import Settled, VisionData


class TestFlood:
    def test_flood_exists(self) -> None:
        Flood()

    def test_flood_vertical(self, empty_components) -> None:
        c = Component.get("flood", "perception")
        assert isinstance(c, Flood)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 4

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 12)
        check_size(e.data, 10)
        check_type(e.data, 0)
        check_points(
            e.data,
            {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)},
        )

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 7)
        check_size(e.data, 5)
        check_type(e.data, 1)
        check_points(
            e.data,
            {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)},
        )

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 12)
        check_size(e.data, 10)
        check_type(e.data, 0)
        check_points(
            e.data,
            {(3, 0), (4, 0), (3, 1), (4, 1), (3, 2), (4, 2), (3, 3), (4, 3), (3, 4), (4, 4)},
        )

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_flood_horizontal(self, empty_components) -> None:
        c = Component.get("flood", "perception")
        assert isinstance(c, Flood)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 4

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 12)
        check_size(e.data, 10)
        check_type(e.data, 0)
        check_points(
            e.data,
            {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 1), (4, 1), (2, 1), (1, 1), (0, 1)},
        )

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 7)
        check_size(e.data, 5)
        check_type(e.data, 1)

        check_points(
            e.data,
            {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)},
        )

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 12)
        check_size(e.data, 10)
        check_type(e.data, 0)
        check_points(
            e.data,
            {(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (2, 4), (1, 4), (0, 4)},
        )

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_flood_diagonal1(self, empty_components) -> None:
        c = Component.get("flood", "perception")
        assert isinstance(c, Flood)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
        )

        assert s.output.call_count == 3

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 7)
        check_size(e.data, 5)
        check_type(e.data, 1)
        check_points(
            e.data,
            {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)},
        )

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 22)
        check_size(e.data, 20)
        check_type(e.data, 0)
        check_points(
            e.data,
            {
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (3, 1),
                (4, 1),
                (3, 2),
                (4, 2),
                (4, 3),
                (3, 4),
                (2, 3),
                (1, 4),
                (2, 4),
                (2, 1),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
                (0, 1),
                (0, 2),
            },
        )

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_flood_diagonal2(self, empty_components) -> None:
        c = Component.get("flood", "perception")
        assert isinstance(c, Flood)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 3

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 22)
        check_size(e.data, 20)
        check_type(e.data, 0)
        check_points(
            e.data,
            {
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (2, 1),
                (1, 2),
                (0, 3),
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (2, 3),
                (3, 3),
                (4, 3),
                (3, 2),
                (4, 2),
                (4, 1),
                (1, 1),
                (0, 2),
                (0, 1),
            },
        )

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 7)
        check_size(e.data, 5)
        check_type(e.data, 1)
        check_points(
            e.data,
            {(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)},
        )

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_flood_screen0(self, empty_components) -> None:
        c = Component.get("flood", "perception")
        assert isinstance(c, Flood)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))

        assert s.output.call_count == 3

        # # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 1631)
        check_size(e.data, 1629)
        check_type(e.data, 2359)  # ' '

        # # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 7)
        check_size(e.data, 5)
        check_type(e.data, 2378)  # .
        check_points(
            e.data,
            {(17, 6), (18, 6), (17, 7), (18, 7), (16, 7)},
        )

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_flood_two_screens(self, empty_components) -> None:
        """This test found some cache thrashing issues because of the number of Nodes it creates"""

        c = Component.get("flood", "perception")
        assert isinstance(c, Flood)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        assert s.output.call_count == 5

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 1631)
        check_size(e.data, 1629)
        check_type(e.data, 2359)  # ' '

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        check_num_src_edges(e.data, 7)
        check_size(e.data, 5)
        check_type(e.data, 2378)  # .
        check_points(
            e.data,
            {(17, 6), (18, 6), (17, 7), (18, 7), (16, 7)},
        )

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
