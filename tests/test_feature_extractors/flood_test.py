# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import (
    StubComponent,
)

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.flood import Flood, FloodFeature, FloodNode
from roc.location import XLoc, YLoc
from roc.perception import Settled, VisionData


class TestFlood:
    def test_flood_exists(self) -> None:
        Flood()

    def test_to_nodes(self, fake_component) -> None:
        f = FloodFeature(origin_id=("foo", "bar"), type=3, size=1, points={(XLoc(1), YLoc(2))})
        n = f.to_nodes()
        assert isinstance(n, FloodNode)
        assert n.labels == {"FeatureNode", "FloodNode"}
        assert n.type == 3
        assert n.size == 1

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
        assert e.data.size == 10
        assert e.data.type == 0
        assert e.data.points == {
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (0, 4),
            (1, 4),
        }

        # event 2
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}

        # event 3
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 10
        assert e.data.type == 0
        assert e.data.points == {
            (3, 0),
            (4, 0),
            (3, 1),
            (4, 1),
            (3, 2),
            (4, 2),
            (3, 3),
            (4, 3),
            (3, 4),
            (4, 4),
        }

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
        assert e.data.size == 10
        assert e.data.type == 0
        assert e.data.points == {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (3, 1),
            (4, 1),
            (2, 1),
            (1, 1),
            (0, 1),
        }

        # event 2
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}

        # event 3
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 10
        assert e.data.type == 0
        assert e.data.points == {
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (3, 4),
            (4, 4),
            (2, 4),
            (1, 4),
            (0, 4),
        }

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
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.feature_name == "Flood"
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}

        # event 2
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 20
        assert e.data.type == 0
        assert e.data.points == {
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
        }

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
        assert e.data.size == 20
        assert e.data.type == 0
        assert e.data.points == {
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
        }

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)}

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
        assert e.data.size == 1629
        assert e.data.type == 2359  # ' '

        # # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 5
        assert e.data.type == 2378  # .
        assert e.data.points == {(17, 6), (18, 6), (17, 7), (18, 7), (16, 7)}

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

        assert s.output.call_count == 6

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 1629
        assert e.data.type == 2359  # ' '

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 5
        assert e.data.type == 2378  # .
        assert e.data.points == {(17, 6), (18, 6), (17, 7), (18, 7), (16, 7)}

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 1629
        assert e.data.type == 2359  # ' '

        # # event 5
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        assert e.data.size == 5
        assert e.data.type == 2378  # .
        assert e.data.points == {(16, 7), (17, 7), (18, 7), (16, 6), (18, 6)}

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
