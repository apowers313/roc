# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.distance import Distance, DistanceFeature
from roc.feature_extractors.single import Single
from roc.perception import Settled, VisionData


class TestDistance:
    def test_distance_exists(self) -> None:
        Distance()

    def test_basic(self, empty_components) -> None:
        c = Component.get("distance", "perception")
        assert isinstance(c, Distance)
        single = Component.get("single", "perception")
        assert isinstance(single, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
            filter=lambda e: e.src.name == "distance",
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2],
                ]
            )
        )

        assert s.output.call_count == 2

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.feature_name == "Distance"
        assert e.data.start_point == (0, 0)
        assert e.data.end_point == (4, 4)
        assert e.data.size == 4

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_multi(self, empty_components) -> None:
        c = Component.get("distance", "perception")
        assert isinstance(c, Distance)
        single = Component.get("single", "perception")
        assert isinstance(single, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
            filter=lambda e: e.src.name == "distance",
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 3, 0, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 4

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.size == 3
        assert e.data.start_point == (0, 0)
        assert e.data.end_point == (3, 3)

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.size == 4
        assert e.data.start_point == (0, 0)
        assert e.data.end_point == (1, 4)

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.size == 2
        assert e.data.start_point == (3, 3)
        assert e.data.end_point == (1, 4)

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_screen0(self, empty_components) -> None:
        c = Component.get("distance", "perception")
        assert isinstance(c, Distance)
        single = Component.get("single", "perception")
        assert isinstance(single, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
            filter=lambda e: e.src.name == "distance",
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))

        assert s.output.call_count == 79

        ### single point: (15, 3)
        ### single point: (19, 3)
        # distance: (15, 3) (19, 3) 4
        ### single point: (15, 4)
        # distance: (15, 3) (15, 4) 1
        # distance: (19, 3) (15, 4) 4
        ### single point: (19, 4)
        # distance: (15, 3) (19, 4) 4
        # distance: (19, 3) (19, 4) 1
        # distance: (15, 4) (19, 4) 4
        ### single point: (15, 5)
        # distance: (15, 3) (15, 5) 2
        # distance: (19, 3) (15, 5) 4
        # distance: (15, 4) (15, 5) 1
        # distance: (19, 4) (15, 5) 4
        ### single point: (16, 5)
        # distance: (15, 3) (16, 5) 2
        # distance: (19, 3) (16, 5) 3
        # distance: (15, 4) (16, 5) 1
        # distance: (19, 4) (16, 5) 3
        # distance: (15, 5) (16, 5) 1
        ### single point: (17, 5)
        # distance: (15, 3) (17, 5) 2
        # distance: (19, 3) (17, 5) 2
        # distance: (15, 4) (17, 5) 2
        # distance: (19, 4) (17, 5) 2
        # distance: (15, 5) (17, 5) 2
        # distance: (16, 5) (17, 5) 1
        ### single point: (18, 5)
        # distance: (15, 3) (18, 5) 3
        # distance: (19, 3) (18, 5) 2
        # distance: (15, 4) (18, 5) 3
        # distance: (19, 4) (18, 5) 1
        # distance: (15, 5) (18, 5) 3
        # distance: (16, 5) (18, 5) 2
        # distance: (17, 5) (18, 5) 1
        ### single point: (19, 5)
        # distance: (15, 3) (19, 5) 4
        # distance: (19, 3) (19, 5) 2
        # distance: (15, 4) (19, 5) 4
        # distance: (19, 4) (19, 5) 1
        # distance: (15, 5) (19, 5) 4
        # distance: (16, 5) (19, 5) 3
        # distance: (17, 5) (19, 5) 2
        # distance: (18, 5) (19, 5) 1
        ### single point: (16, 6)
        # distance: (15, 3) (16, 6) 3
        # distance: (19, 3) (16, 6) 3
        # distance: (15, 4) (16, 6) 2
        # distance: (19, 4) (16, 6) 3
        # distance: (15, 5) (16, 6) 1
        # distance: (16, 5) (16, 6) 1
        # distance: (17, 5) (16, 6) 1
        # distance: (18, 5) (16, 6) 2
        # distance: (19, 5) (16, 6) 3
        ### single point: (15, 8)
        # distance: (15, 3) (15, 8) 5
        # distance: (19, 3) (15, 8) 5
        # distance: (15, 4) (15, 8) 4
        # distance: (19, 4) (15, 8) 4
        # distance: (15, 5) (15, 8) 3
        # distance: (16, 5) (15, 8) 3
        # distance: (17, 5) (15, 8) 3
        # distance: (18, 5) (15, 8) 3
        # distance: (19, 5) (15, 8) 4
        # distance: (16, 6) (15, 8) 2
        ### single point: (18, 8)
        # distance: (15, 3) (18, 8) 5
        # distance: (19, 3) (18, 8) 5
        # distance: (15, 4) (18, 8) 4
        # distance: (19, 4) (18, 8) 4
        # distance: (15, 5) (18, 8) 3
        # distance: (16, 5) (18, 8) 3
        # distance: (17, 5) (18, 8) 3
        # distance: (18, 5) (18, 8) 3
        # distance: (19, 5) (18, 8) 3
        # distance: (16, 6) (18, 8) 2
        # distance: (15, 8) (18, 8) 3
        ### single point: (19, 8)
        # distance: (15, 3) (19, 8) 5
        # distance: (19, 3) (19, 8) 5
        # distance: (15, 4) (19, 8) 4
        # distance: (19, 4) (19, 8) 4
        # distance: (15, 5) (19, 8) 4
        # distance: (16, 5) (19, 8) 3
        # distance: (17, 5) (19, 8) 3
        # distance: (18, 5) (19, 8) 3
        # distance: (19, 5) (19, 8) 3
        # distance: (16, 6) (19, 8) 3
        # distance: (15, 8) (19, 8) 4
        # distance: (18, 8) (19, 8) 1

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.size == 4
        assert e.data.start_point == (15, 3)
        assert e.data.end_point == (19, 3)

        # # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.size == 1
        assert e.data.start_point == (15, 3)
        assert e.data.end_point == (15, 4)

        # [...]

        # event 78
        e = s.output.call_args_list[77].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        assert e.data.size == 1
        assert e.data.start_point == (18, 8)
        assert e.data.end_point == (19, 8)

        # event 79
        e = s.output.call_args_list[78].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
