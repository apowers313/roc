# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent

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
            VisionData(
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
        flood = e.data.feature
        assert flood.size == 10
        assert flood.type == 0
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 0

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 5
        assert flood.type == 1
        p0 = flood.points[0]
        assert p0.x == 2 and p0.y == 0

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 10
        assert flood.type == 0
        p0 = flood.points[0]
        assert p0.x == 3 and p0.y == 0

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
            VisionData(
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
        flood = e.data.feature
        assert flood.size == 10
        assert flood.type == 0
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 0

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 5
        assert flood.type == 1
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 2

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 10
        assert flood.type == 0
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 3

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
            VisionData(
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
        flood = e.data.feature
        assert flood.size == 5
        assert flood.type == 1
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 0

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 20
        assert flood.type == 0
        p0 = flood.points[0]
        assert p0.x == 1 and p0.y == 0

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
            VisionData(
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
        flood = e.data.feature
        assert flood.size == 20
        assert flood.type == 0
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 0

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 5
        assert flood.type == 1
        p0 = flood.points[0]
        assert p0.x == 4 and p0.y == 0

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

        s.input_conn.send(VisionData(screens[0]["chars"]))

        assert s.output.call_count == 4

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 1629
        assert flood.type == 32
        p0 = flood.points[0]
        assert p0.x == 0 and p0.y == 0

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 5
        assert flood.type == ord("-")
        p0 = flood.points[0]
        assert p0.x == 15 and p0.y == 3

        # event 2
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, FloodFeature)
        flood = e.data.feature
        assert flood.size == 5
        assert flood.type == ord(".")
        p0 = flood.points[0]
        assert p0.x == 17 and p0.y == 6

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
