from __future__ import annotations

from pydantic import BaseModel

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, PerceptionEvent, VisionData


class Diff(BaseModel):
    """A Pydantic model for representing a changes in vision."""

    x: int
    y: int
    old_val: str | int
    new_val: str | int


class DeltaFeature(Feature[Diff]):
    """A feature for representing vision changes (deltas)"""

    def __init__(self, origin: Component, diff: Diff) -> None:
        super().__init__(origin, diff)
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
class Delta(FeatureExtractor[Diff]):
    """A component for detecting changes in vision."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_viz: VisionData | None = None

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> Feature[Diff] | None:
        # assert isinstance(e, VisionData)
        # reveal_type(e)
        # reveal_type(e.data)
        data = e.data
        assert isinstance(data, VisionData)

        prev = self.prev_viz
        self.prev_viz = curr = data

        if prev is None:
            # can't get difference when there was nothing before this
            self.settled()
            return None

        # roughly make sure that things are the same size
        assert prev.height == curr.height
        assert prev.width == curr.width

        # height = len(curr)
        # width = len(curr[0])
        # for y in range(height):
        #     for x in range(width):
        #         if prev[y][x] != curr[y][x]:
        #             d = Diff(
        #                 x=x,
        #                 y=y,
        #                 old_val=prev[y][x],
        #                 new_val=curr[y][x],
        #             )
        #             f = DeltaFeature(self, d)
        #             # diff_list.append(d)
        #             self.pb_conn.send(f)
        for new_point in curr:
            old_point = prev.get_point(new_point.x, new_point.y)
            if old_point.val != new_point.val:
                d = Diff(
                    x=new_point.x,
                    y=new_point.y,
                    old_val=old_point.val,
                    new_val=new_point.val,
                )
                f = DeltaFeature(self, d)
                # diff_list.append(d)
                self.pb_conn.send(f)
        # if len(diff_list) == 0:
        #     return None

        # return DeltaFeature(self, diff_list)
        self.settled()
        return None


# DiffList = list[Diff]
