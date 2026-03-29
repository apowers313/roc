"""Generates Features for things that aren't like their neighbors"""

from dataclasses import dataclass

from ..graphdb import FindQueryOpts
from ..location import IntGrid, Point
from ..perception import (
    FeatureExtractor,
    FeatureKind,
    FeatureNode,
    PerceptionEvent,
    PointFeature,
    VisionData,
)


class SingleNode(FeatureNode):
    """Graph node representing a single isolated feature by its glyph type."""

    kind = FeatureKind.PHYSICAL
    type: int

    @property
    def attr_strs(self) -> list[str]:
        """Returns the type as a string."""
        return [str(self.type)]


@dataclass(kw_only=True)
class SingleFeature(PointFeature[SingleNode]):
    """A single isolated feature with no similar features around it."""

    feature_name: str = "Single"

    def _create_nodes(self) -> SingleNode:
        """Creates a new SingleNode for this feature's type."""
        return SingleNode(type=self.type)

    def _dbfetch_nodes(self) -> SingleNode | None:
        """Looks up an existing SingleNode by type in the database."""
        return SingleNode.find_one(
            "src.type = $type", query_opts=FindQueryOpts(params={"type": self.type})
        )


class Single(FeatureExtractor[Point]):
    """A component for identifying single, isolated visual features"""

    name: str = "single"
    type: str = "perception"

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


_NEIGHBOR_OFFSETS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


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
    for dx, dy in _NEIGHBOR_OFFSETS:
        nx, ny = point.x + dx, point.y + dy
        if 0 <= nx <= max_width and 0 <= ny <= max_height:
            if data.get_val(nx, ny) == point.val:
                return False
    return True
