from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, PerceptionEvent, Settled
from .delta import DeltaFeature, Diff


class MotionFeature(Feature):
    def __init__(self, origin: Component, v: MotionVector) -> None:
        super().__init__(origin)
        self.motion_vector = v

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")

    def __repr__(self) -> str:
        v = self.motion_vector
        return f"{v.val} {v.direction}: ({v.startX}, {v.startY}) -> ({v.endX}, {v.endY})"


DiffMap = Dict[str | int, list[Diff]]


@register_component("motion", "perception")
class Motion(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.prev_diff_map: DiffMap = {}

    def event_filter(self, e: PerceptionEvent) -> bool:
        # only listen to delta:perception events
        if e.src.name == "delta" and e.src.type == "perception":
            return True
        return False

    def get_feature(self, e: PerceptionEvent) -> MotionFeature | None:
        print("motion got event", e)
        if isinstance(e.data, Settled):
            print("delta settled, all done")
            self.prev_diff_map.clear()
            self.settled()
            return None

        assert isinstance(e.data, DeltaFeature)
        diff_map = self.prev_diff_map

        d = e.data.diff

        if d.old_val not in diff_map:
            print("d.old_val not in diff_map")

        # if not isinstance(diff_map[d.old_val], list):
        #     print("not isinstance(diff_map[d.old_val], list)")

        if d.old_val not in diff_map or not isinstance(diff_map[d.old_val], list):
            diff_map[d.old_val] = []

        diff_map[d.old_val].append(d)

        if d.new_val in diff_map:
            old_vals = diff_map[d.new_val]
        else:
            old_vals = []

        adjacent_diff: Diff | None = None
        for old in old_vals:
            if isadjacent(old, d):
                adjacent_diff = d
                break

        print("adjacent diff", adjacent_diff)

        if adjacent_diff:
            print("diff", d)
            print("adjacent_diff", adjacent_diff)
        return None
        # v = MotionVector(
        #     startX=adjacent_diff,
        #     startY=2,
        #     endX=3,
        #     endY=4,
        #     val=102,
        #     direction=adjacent_direction,
        # )

        # return MotionFeature(self, v)


class Direction(str, Enum):
    up = "UP"
    down = "DOWN"
    left = "LEFT"
    right = "RIGHT"
    up_right = "UP_RIGHT"
    up_left = "UP_LEFT"
    down_right = "DOWN_RIGHT"
    down_left = "DOWN_LEFT"


class MotionVector(BaseModel):
    direction: Direction
    startX: int
    startY: int
    endX: int
    endY: int
    val: str | int

    class Config:
        use_enum_values = True


def adjacent_direction(d1: Diff, d2: Diff) -> Direction:
    lr_str = ""
    if d1.x < d2.x:
        lr_str = "RIGHT"
    elif d1.x > d2.x:
        lr_str = "LEFT"

    ud_str = ""
    if d1.y < d2.y:
        ud_str = "UP"
    if d1.y > d2.y:
        ud_str = "DOWN"

    join_str = ""
    if len(lr_str) and len(ud_str):
        join_str = "_"

    return Direction(f"{ud_str}{join_str}{lr_str}")


def isadjacent(d1: Diff, d2: Diff) -> bool:
    dx = abs(d1.x - d2.x)
    dy = abs(d1.y - d2.y)
    return dx <= 1 and dy <= 1
