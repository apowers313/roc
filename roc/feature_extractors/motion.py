from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, PerceptionEvent, Settled
from ..point import Point
from .delta import DeltaFeature, Diff


class MotionVector(BaseModel):
    direction: Direction
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    val: str | int

    class Config:
        use_enum_values = True


class MotionFeature(Feature[MotionVector]):
    def __init__(self, origin: Component, v: MotionVector) -> None:
        super().__init__(origin, v)

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")

    def __repr__(self) -> str:
        v = self.feature
        return f"{v.val} {v.direction}: ({v.start_x}, {v.start_y}) -> ({v.end_x}, {v.end_y})"


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
        d1 = e.data.diff

        for d2 in self.diff_list:
            if Point.isadjacent(x1=d1.x, y1=d1.y, x2=d2.x, y2=d2.y):
                if d2.old_val == d1.new_val:
                    emit_motion(self, d2, d1)
                if d1.old_val == d2.new_val:
                    emit_motion(self, d1, d2)

        self.diff_list.append(d1)

        return None


class Direction(str, Enum):
    up = "UP"
    down = "DOWN"
    left = "LEFT"
    right = "RIGHT"
    up_right = "UP_RIGHT"
    up_left = "UP_LEFT"
    down_right = "DOWN_RIGHT"
    down_left = "DOWN_LEFT"


def adjacent_direction(d1: Diff, d2: Diff) -> str:
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

    # return Direction(f"{ud_str}{join_str}{lr_str}")
    dir = f"{ud_str}{join_str}{lr_str}"
    return dir


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
