"""Converts Delta events into Motion events that signify an object moving in a direction"""

from __future__ import annotations

from dataclasses import dataclass

from ...db.graphdb import FindQueryOpts
from ..location import Point, XLoc, YLoc
from ..base import (
    Direction,
    FeatureExtractor,
    FeatureNode,
    PerceptionEvent,
    Settled,
    VisualFeature,
)
from .delta import DeltaFeature


class MotionNode(FeatureNode):
    """Graph node representing a motion event with type and direction."""

    type: int
    direction: Direction

    @property
    def attr_strs(self) -> list[str]:
        """Returns type and direction as strings."""
        return [str(self.type), str(self.direction)]


@dataclass(kw_only=True)
class MotionFeature(VisualFeature[MotionNode]):
    """A vector describing a motion, including the start point, end point,
    direction and value of the thing moving
    """

    feature_name: str = "Motion"
    start_point: tuple[XLoc, YLoc]
    end_point: tuple[XLoc, YLoc]
    type: int
    direction: Direction

    def __str__(self) -> str:
        return f"{self.type} '{chr(self.type)}' {self.direction}: ({self.start_point[0]}, {self.start_point[1]}) -> ({self.end_point[0]}, {self.end_point[1]})"

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        """Returns the destination point of the motion."""
        return {self.end_point}

    def node_hash(self) -> int:
        """Hashes by type and direction."""
        return hash((self.type, self.direction))

    def _create_nodes(self) -> MotionNode:
        """Creates a new MotionNode with type and direction."""
        return MotionNode(type=self.type, direction=self.direction)

    def _dbfetch_nodes(self) -> MotionNode | None:
        """Looks up an existing MotionNode by type and direction."""
        return MotionNode.find_one(
            "src.type = $type AND src.direction = $direction",
            query_opts=FindQueryOpts(params={"type": self.type, "direction": self.direction}),
        )


DeltaList = list[DeltaFeature]


class Motion(FeatureExtractor[MotionFeature]):
    """Component that consumes Delta events and produces Motion events"""

    name: str = "motion"
    type: str = "perception"

    def __init__(self) -> None:
        super().__init__()
        self.delta_list: DeltaList = []

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Only process events from the delta perception component."""
        # only listen to delta:perception events
        if e.src_id.name == "delta" and e.src_id.type == "perception":
            return True
        return False

    def get_feature(self, e: PerceptionEvent) -> None:
        """Matches adjacent deltas to detect motion and emits MotionFeatures."""
        if isinstance(e.data, Settled):
            self.delta_list.clear()
            self.settled()
            return None

        assert isinstance(e.data, DeltaFeature)
        d1 = e.data

        for d2 in self.delta_list:
            if Point.isadjacent(xy1=(d1.point[0], d1.point[1]), xy2=(d2.point[0], d2.point[1])):
                if d2.old_val == d1.new_val:
                    emit_motion(self, d2, d1)
                if d1.old_val == d2.new_val:
                    emit_motion(self, d1, d2)

        self.delta_list.append(d1)


def adjacent_direction(d1: DeltaFeature, d2: DeltaFeature) -> Direction:
    """Helper function to convert two positions into a direction such as 'UP' or
    'DOWN_LEFT'
    """
    lr_str = ""
    if d1.point[0] < d2.point[0]:
        lr_str = "RIGHT"
    elif d1.point[0] > d2.point[0]:
        lr_str = "LEFT"

    ud_str = ""
    # XXX: top left is 0,0
    if d1.point[1] > d2.point[1]:
        ud_str = "UP"
    if d1.point[1] < d2.point[1]:
        ud_str = "DOWN"

    join_str = ""
    if len(lr_str) and len(ud_str):
        join_str = "_"

    return Direction(f"{ud_str}{join_str}{lr_str}")


def emit_motion(mc: Motion, old_delta: DeltaFeature, new_delta: DeltaFeature) -> None:
    """Emits a MotionFeature from a pair of adjacent deltas that indicate movement."""
    mc.pb_conn.send(
        MotionFeature(
            origin_id=mc.id,
            start_point=old_delta.point,
            end_point=new_delta.point,
            type=new_delta.new_val,
            direction=adjacent_direction(old_delta, new_delta),
        )
    )
