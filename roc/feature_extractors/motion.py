from __future__ import annotations

from dataclasses import dataclass

from ..component import Component, register_component
from ..perception import (
    ComplexFeature,
    Direction,
    ElementOrientation,
    ElementPoint,
    ElementType,
    Feature,
    FeatureExtractor,
    OldLocation,
    PerceptionEvent,
    Settled,
    Transmogrifier,
)
from ..point import Point
from .delta import DeltaFeature, Diff


@dataclass
class MotionVector(Transmogrifier):
    direction: Direction
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    val: int

    class Config:
        use_enum_values = True

    def __str__(self) -> str:
        return f"{self.val} '{chr(self.val)}' {self.direction}: ({self.start_x}, {self.start_y}) -> ({self.end_x}, {self.end_y})"

    def add_to_feature(self, n: Feature) -> None:
        n.add_type(self.val)
        n.add_point(self.end_x, self.end_y)
        n.add_orientation(self.direction)
        ol = OldLocation(n.origin, self.start_x, self.start_y, self.val)
        n.add_feature("Origin", ol)

    @classmethod
    def from_feature(self, n: Feature) -> MotionVector:
        orig = n.get_feature("Origin")
        assert isinstance(orig, Feature)
        start_loc = orig.get_feature("Location")
        assert isinstance(start_loc, ElementPoint)
        val = n.get_feature("Type")
        assert isinstance(val, ElementType)
        end_loc = n.get_feature("Location")
        assert isinstance(end_loc, ElementPoint)
        dir = n.get_feature("Direction")
        assert isinstance(dir, ElementOrientation)
        return MotionVector(
            direction=dir.orientation,
            start_x=start_loc.x,
            start_y=start_loc.y,
            end_x=end_loc.x,
            end_y=end_loc.y,
            val=val.type,
        )


class MotionFeature(ComplexFeature[MotionVector]):
    def __init__(self, origin: Component, mv: MotionVector):
        super().__init__("Motion", origin, mv)


DiffList = list[Diff]


@register_component("motion", "perception")
class Motion(FeatureExtractor[MotionFeature]):
    def __init__(self) -> None:
        super().__init__()
        self.diff_list: DiffList = []

    def event_filter(self, e: PerceptionEvent) -> bool:
        # only listen to delta:perception events
        if e.src.name == "delta" and e.src.type == "perception":
            return True
        return False

    def get_feature(self, e: PerceptionEvent) -> None:
        if isinstance(e.data, Settled):
            self.diff_list.clear()
            self.settled()
            return None

        assert isinstance(e.data, DeltaFeature)
        d1 = Diff.from_feature(e.data)

        for d2 in self.diff_list:
            if Point.isadjacent(x1=d1.x, y1=d1.y, x2=d2.x, y2=d2.y):
                if d2.old_val == d1.new_val:
                    emit_motion(self, d2, d1)
                if d1.old_val == d2.new_val:
                    emit_motion(self, d1, d2)

        self.diff_list.append(d1)

        return None


def adjacent_direction(d1: Diff, d2: Diff) -> Direction:
    lr_str = ""
    if d1.x < d2.x:
        lr_str = "RIGHT"
    elif d1.x > d2.x:
        lr_str = "LEFT"

    ud_str = ""
    # XXX: top left is 0,0
    if d1.y > d2.y:
        ud_str = "UP"
    if d1.y < d2.y:
        ud_str = "DOWN"

    join_str = ""
    if len(lr_str) and len(ud_str):
        join_str = "_"

    return Direction(f"{ud_str}{join_str}{lr_str}")


def emit_motion(mc: Motion, old_diff: Diff, new_diff: Diff) -> None:
    mc.pb_conn.send(
        MotionFeature(
            mc,
            MotionVector(
                start_x=old_diff.x,
                start_y=old_diff.y,
                end_x=new_diff.x,
                end_y=new_diff.y,
                val=new_diff.new_val,
                direction=adjacent_direction(old_diff, new_diff),
            ),
        )
    )
