"""Generates Features for things that aren't like their neighbors"""

from dataclasses import dataclass

from ..component import register_component
from ..location import IntGrid, Point
from ..perception import (
    FeatureExtractor,
    PerceptionEvent,
    PointFeature,
    VisionData,
)


@dataclass(kw_only=True)
class SingleFeature(PointFeature):
    """A single isolated feature with no similar features around it."""

    feature_name: str = "Single"

    # def __hash__(self) -> int:
    #     raise NotImplementedError("SingleFeature hash not implemented")


@register_component("single", "perception")
class Single(FeatureExtractor[Point]):
    """A component for identifying single, isolated visual features"""

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Filters out non-VisionData

        Args:
            e (PerceptionEvent): Any event on the perception bus

        Returns:
            bool: Returns True if the event is VisionData to keep processing it,
            False otherwise.
        """
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> None:
        """Emits the shape features.

        Args:
            e (PerceptionEvent): The VisionData

        Returns:
            Feature | None: None
        """
        vd = e.data
        assert isinstance(vd, VisionData)
        data = IntGrid(vd.glyphs)

        ## iterate points
        for x, y, v in data:
            point = Point(x, y, v)
            if is_unique_from_neighbors(data, point):
                self.pb_conn.send(
                    SingleFeature(
                        origin_id=self.id,
                        point=(x, y),
                        type=v,
                    )
                )
        self.settled()


def is_unique_from_neighbors(data: IntGrid, point: Point) -> bool:
    """Helper function to determine if a point in a matrix has the same value as
    any points around it.

    Args:
        data (VisionData): The matrix / Grid to evaluate
        point (Point): The point to see if any of its neighbors have the same value

    Returns:
        bool: Returns True if the point is different from all surrounding
        points, False otherwise.
    """
    max_width = data.width - 1
    max_height = data.height - 1
    # up left
    if point.x > 0 and point.y > 0 and data.get_val(point.x - 1, point.y - 1) == point.val:
        return False
    # up
    if point.y > 0 and data.get_val(point.x, point.y - 1) == point.val:
        return False
    # up right
    if point.x < max_width and point.y > 0 and data.get_val(point.x + 1, point.y - 1) == point.val:
        return False
    # left
    if point.x > 0 and data.get_val(point.x - 1, point.y) == point.val:
        return False
    # right
    if point.x < max_width and data.get_val(point.x + 1, point.y) == point.val:
        return False
    # down left
    if point.x > 0 and point.y < max_height and data.get_val(point.x - 1, point.y + 1) == point.val:
        return False
    # down
    if point.y < max_height and data.get_val(point.x, point.y + 1) == point.val:
        return False
    # down right
    if (
        point.x < max_width
        and point.y < max_height
        and data.get_val(point.x + 1, point.y + 1) == point.val
    ):
        return False
    return True
