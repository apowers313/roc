from __future__ import annotations

from pydantic import BaseModel

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, Grid, PerceptionEvent, VisionData


class DeltaFeature(Feature):
    """A feature for representing vision changes (deltas)"""

    def __init__(self, origin: Component, diff: Diff) -> None:
        super().__init__(origin)
        # self.diff_list = diff_list
        self.diff = diff

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")

    def __repr__(self) -> str:
        # diff_str = "\n"
        # for d in self.diff_list:
        #     diff_str += f"({d.x}, {d.y}): {d.old_val} -> {d.new_val}\n"

        # return diff_str

        d = self.diff
        return f"({d.x}, {d.y}): {d.old_val} -> {d.new_val}\n"


@register_component("delta", "perception")
class Delta(FeatureExtractor):
    """A component for detecting changes in vision."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_grid: Grid | None = None

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> Feature | None:
        # assert isinstance(e, VisionData)
        # reveal_type(e)
        # reveal_type(e.data)
        data = e.data
        assert isinstance(data, VisionData)

        prev = self.prev_grid
        self.prev_grid = curr = data.screen

        if prev is None:
            self.settled()
            return None

        # roughly make sure that things are the same height
        assert len(prev) == len(curr)
        assert len(prev[0]) == len(curr[0])

        width = len(curr)
        height = len(curr[0])
        for x in range(width):
            for y in range(height):
                if prev[x][y] != curr[x][y]:
                    d = Diff(
                        x=x,
                        y=y,
                        old_val=prev[x][y],
                        new_val=curr[x][y],
                    )
                    f = DeltaFeature(self, d)
                    # diff_list.append(d)
                    self.pb_conn.send(f)

        # if len(diff_list) == 0:
        #     return None

        # return DeltaFeature(self, diff_list)
        self.settled()
        return None


class Diff(BaseModel):
    """A Pydantic model for representing a changes in vision."""

    x: int
    y: int
    old_val: str | int
    new_val: str | int


# DiffList = list[Diff]
