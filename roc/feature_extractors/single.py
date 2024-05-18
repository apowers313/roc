from typing import Any

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, PerceptionEvent, VisionData
from ..point import Point


class SingleFeature(Feature[Point]):
    """A collection of points representing similar values that are all adjacent to each other"""

    def __init__(self, origin: Component, points: Point) -> None:
        super().__init__(origin, points)

    def __hash__(self) -> int:
        raise NotImplementedError("SingleFeature hash not implemented")


@register_component("single", "perception")
class Single(FeatureExtractor[Point]):
    """A component for identifying single, isolated visual features"""

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> Feature[Any] | None:
        data = e.data
        assert isinstance(data, VisionData)

        ## iterate points
        for point in data:
            if (
                # up left
                data.get_val(point.x - 1, point.y - 1) != point.val
                and
                # up
                data.get_val(point.x, point.y - 1) != point.val
                and
                # up right
                data.get_val(point.x + 1, point.y - 1) != point.val
                and
                # left
                data.get_val(point.x - 1, point.y) != point.val
                and
                # right
                data.get_val(point.x + 1, point.y) != point.val
                and
                # down left
                data.get_val(point.x - 1, point.y + 1) != point.val
                and
                # down
                data.get_val(point.x, point.y + 1) != point.val
                and
                # down right
                data.get_val(point.x + 1, point.y + 1) != point.val
            ):
                print("emit val", point)
                self.pb_conn.send(SingleFeature(self, point))
        self.settled()
        return None
