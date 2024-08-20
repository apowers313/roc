# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent, check_num_src_edges, check_points, check_size

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
        check_num_src_edges(e.data, 3)
        check_size(e.data, 4)
        check_points(e.data, {(0, 0), (4, 4)})

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
        check_num_src_edges(e.data, 3)
        check_size(e.data, 3)
        check_points(e.data, {(0, 0), (3, 3)})

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        check_num_src_edges(e.data, 3)
        check_size(e.data, 4)
        check_points(e.data, {(0, 0), (1, 4)})

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DistanceFeature)
        check_num_src_edges(e.data, 3)
        check_size(e.data, 2)
        check_points(e.data, {(3, 3), (1, 4)})

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

        assert s.output.call_count == 29

        ##### single point (15, 4): 124 '|'
        ##### single point (19, 4): 124 '|'
        # distance (19, 4) (15, 4) 4
        ##### single point (16, 5): 91 '['
        # distance (16, 5) (15, 4) 1
        # distance (16, 5) (19, 4) 3
        ##### single point (17, 5): 64 '@'
        # distance (17, 5) (15, 4) 2
        # distance (17, 5) (19, 4) 2
        # distance (17, 5) (16, 5) 1
        ##### single point (18, 5): 120 'x'
        # distance (18, 5) (15, 4) 3
        # distance (18, 5) (19, 4) 1
        # distance (18, 5) (16, 5) 2
        # distance (18, 5) (17, 5) 1
        ##### single point (19, 5): 43 '+'
        # distance (19, 5) (15, 4) 4
        # distance (19, 5) (19, 4) 1
        # distance (19, 5) (16, 5) 3
        # distance (19, 5) (17, 5) 2
        # distance (19, 5) (18, 5) 1
        ##### single point (16, 6): 102 'f'
        # distance (16, 6) (15, 4) 2
        # distance (16, 6) (19, 4) 3
        # distance (16, 6) (16, 5) 1
        # distance (16, 6) (17, 5) 1
        # distance (16, 6) (18, 5) 2
        # distance (16, 6) (19, 5) 3
        ##### single point (19, 8): 45 '-'
        # distance (19, 8) (15, 4) 4
        # distance (19, 8) (19, 4) 4
        # distance (19, 8) (16, 5) 3
        # distance (19, 8) (17, 5) 3
        # distance (19, 8) (18, 5) 3
        # distance (19, 8) (19, 5) 3
        # distance (19, 8) (16, 6) 3

        # # event 1
        # e = s.output.call_args_list[0].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, DistanceFeature)
        # check_num_src_edges(e.data, 3)
        # check_size(e.data, 3)
        # check_points(e.data, {(0, 0), (3, 3)})

        # # event 2
        # e = s.output.call_args_list[1].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, DistanceFeature)
        # check_num_src_edges(e.data, 3)
        # check_size(e.data, 4)
        # check_points(e.data, {(0, 0), (1, 4)})

        # # event 3
        # e = s.output.call_args_list[2].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, DistanceFeature)
        # check_num_src_edges(e.data, 3)
        # check_size(e.data, 2)
        # check_points(e.data, {(3, 3), (1, 4)})

        # event 4
        e = s.output.call_args_list[28].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
