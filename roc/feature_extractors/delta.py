from __future__ import annotations

from pydantic import BaseModel

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, Grid, PerceptionEvent, VisionData


class DeltaFeature(Feature):
    """A feature for representing vision changes (deltas)"""

    def __init__(self, origin: Component, diff_list: DiffList) -> None:
        super().__init__(origin)
        self.diff_list = diff_list

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")


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
            return None

        # roughly make sure that things are the same height
        assert len(prev) == len(curr)
        assert len(prev[0]) == len(curr[0])

        width = len(curr)
        height = len(curr[0])
        diff_list: DiffList = []
        for x in range(width):
            for y in range(height):
                if prev[x][y] != curr[x][y]:
                    diff_list.append(
                        Diff(
                            x=x,
                            y=y,
                            val1=prev[x][y],
                            val2=curr[x][y],
                        )
                    )

        return DeltaFeature(self, diff_list)


class Diff(BaseModel):
    """A Pydantic model for representing a changes in vision."""

    x: int
    y: int
    val1: str | int
    val2: str | int


DiffList = list[Diff]
