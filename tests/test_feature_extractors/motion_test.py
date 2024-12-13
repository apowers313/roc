# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from helpers.nethack_screens import screens
from helpers.util import (
    StubComponent,
)

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.delta import Delta, DeltaFeature
from roc.feature_extractors.motion import (
    Motion,
    MotionFeature,
    MotionNode,
    adjacent_direction,
)
from roc.graphdb import GraphDB, Node
from roc.location import XLoc, YLoc
from roc.perception import Direction, Settled, VisionData, cache_registry

screen0 = VisionData.from_dict(screens[0])
screen1 = VisionData.from_dict(screens[1])
screen4 = VisionData.from_dict(screens[4])
screen6 = VisionData.from_dict(screens[6])


class TestMotion:
    def test_exists(self, empty_components) -> None:
        Motion()

    # def test_print_screen(self) -> None:
    #     for line in screen0.screen:
    #         for ch in line:
    #             print(chr(ch), end="")
    #         print("")

    def test_to_nodes(self, fake_component) -> None:
        f = MotionFeature(
            origin_id=("foo", "bar"),
            type=8008,
            start_point=(XLoc(1), YLoc(2)),
            end_point=(XLoc(3), YLoc(4)),
            direction=Direction.down,
        )
        n = f.to_nodes()
        assert isinstance(n, MotionNode)
        assert n.labels == {"Feature", "Motion"}
        assert n.type == 8008
        assert n.direction == Direction.down

    def test_to_nodes_dbfetch(self, mocker) -> None:
        # setup
        spy: MagicMock = mocker.spy(GraphDB, "raw_fetch")
        cache = Node.get_cache()
        cache_registry.clear()
        assert len(cache) == 0
        assert len(cache_registry) == 0

        # create node, save, clear cache
        f = MotionFeature(
            origin_id=("foo", "bar"),
            type=8008,
            start_point=(XLoc(1), YLoc(2)),
            end_point=(XLoc(3), YLoc(4)),
            direction=Direction.down,
        )
        n = f.to_nodes()
        assert spy.call_count == 1  # tried to load, not found
        assert len(cache_registry) == 1
        assert "Motion" in cache_registry
        assert len(cache) == 1
        Node.save(n)
        assert spy.call_count == 2  # saving node
        del n
        cache.clear()
        cache_registry.clear()

        # reload node from database
        f = MotionFeature(
            origin_id=("foo", "bar"),
            type=8008,
            start_point=(XLoc(1), YLoc(2)),
            end_point=(XLoc(3), YLoc(4)),
            direction=Direction.down,
        )
        n = f.to_nodes()
        assert len(cache) == 1
        assert len(cache_registry) == 1
        assert "Motion" in cache_registry
        assert spy.call_count == 3
        assert spy.call_args_list[2][1] == {"params": {"type": "8008", "direction": "DOWN"}}
        assert n.type == 8008
        assert n.direction == Direction.down

    def test_direction(self) -> None:
        o = Component()
        origin = DeltaFeature(origin_id=o.id, point=(XLoc(0), YLoc(0)), old_val=120, new_val=0)
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(0), YLoc(1)), old_val=0, new_val=120)
        )
        assert d == "DOWN"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(0), YLoc(-1)), old_val=0, new_val=120)
        )
        assert d == "UP"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(-1), YLoc(0)), old_val=0, new_val=120)
        )
        assert d == "LEFT"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(1), YLoc(0)), old_val=0, new_val=120)
        )
        assert d == "RIGHT"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(-1), YLoc(1)), old_val=0, new_val=120)
        )
        assert d == "DOWN_LEFT"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(1), YLoc(1)), old_val=0, new_val=120)
        )
        assert d == "DOWN_RIGHT"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(-1), YLoc(-1)), old_val=0, new_val=120)
        )
        assert d == "UP_LEFT"
        d = adjacent_direction(
            origin, DeltaFeature(origin_id=o.id, point=(XLoc(1), YLoc(-1)), old_val=0, new_val=120)
        )
        assert d == "UP_RIGHT"

    def test_basic(self, empty_components) -> None:
        d = Component.get("delta", "perception")
        assert isinstance(d, Delta)
        m = Component.get("motion", "perception")
        assert isinstance(m, Motion)
        s = StubComponent(
            input_bus=d.pb_conn.attached_bus,
            output_bus=m.pb_conn.attached_bus,
            filter=lambda e: e.src_id.name == "motion",
        )

        s.input_conn.send(screen0)
        s.input_conn.send(screen1)
        s.input_conn.send(screen4)
        s.input_conn.send(screen6)

        assert s.output.call_count == 8

        # screen 0
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # screen 1

        # ([EVENT: motion >>> perception]: 102 RIGHT: (16, 6) -> (17, 6),)
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, MotionFeature)
        assert e.data.feature_name == "Motion"
        assert e.data.type == 413  # f
        assert e.data.direction == Direction("RIGHT")
        assert e.data.end_point == (17, 6)
        assert e.data.start_point == (16, 6)

        # # ([EVENT: motion >>> perception]: 46 LEFT: (17, 6) -> (16, 6),)
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, MotionFeature)
        assert e.data.type == 2378  # .
        assert e.data.direction == Direction("LEFT")
        assert e.data.start_point == (17, 6)
        assert e.data.end_point == (16, 6)

        # Settled
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # # screen 4

        # # ([EVENT: motion >>> perception]: 102 UP_RIGHT: (17, 6) -> (18, 5),)
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, MotionFeature)
        assert e.data.type == 413  # f
        assert e.data.direction == Direction("UP_RIGHT")
        assert e.data.start_point == (17, 6)
        assert e.data.end_point == (18, 5)

        # Settled
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

        # # screen 6

        # # ([EVENT: motion >>> perception]: 102 DOWN: (18, 5) -> (18, 6),)
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, MotionFeature)
        assert e.data.type == 413  # f
        assert e.data.direction == Direction("DOWN")
        assert e.data.start_point == (18, 5)
        assert e.data.end_point == (18, 6)

        # Settled
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
