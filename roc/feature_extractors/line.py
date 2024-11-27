from dataclasses import dataclass

from ..component import register_component
from ..graphdb import Node
from ..location import IntGrid, Point, PointList, TypedPointCollection, XLoc, YLoc
from ..perception import (
    AreaFeature,
    FeatureExtractor,
    PerceptionEvent,
    VisionData,
)

MIN_LINE_COUNT = 4


class LineNode(Node):
    type: int
    size: int


@dataclass(kw_only=True)
class LineFeature(AreaFeature[LineNode]):
    """A collection of points representing a line"""

    feature_name: str = "Line"

    def _create_nodes(self) -> LineNode:
        return LineNode(type=self.type, size=self.size)

    def _dbfetch_nodes(self) -> LineNode | None:
        nodes = LineNode.find(
            "src.type = $type AND src.size = $size", params={"type": self.type, "size": self.size}
        )
        return Node.list_to_single(nodes)


@register_component("line", "perception")
class Line(FeatureExtractor[TypedPointCollection]):
    """A component for identifying similar values located along a vertical or
    horizontal line
    """

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> None:
        assert isinstance(e.data, VisionData)
        data = IntGrid(e.data.glyphs)

        points: PointList = []

        def try_emit(ln: Line) -> None:
            nonlocal points

            if len(points) >= MIN_LINE_COUNT:
                point_set = set([(p.x, p.y) for p in points])
                ln.pb_conn.send(
                    LineFeature(
                        origin_id=self.id,
                        points=point_set,
                        type=points[0].val,
                        size=len(points),
                    ),
                )
            points = []

        ## iterate points by 'x' to identify horizontal lines
        for y in range(data.height):
            for x in range(data.width):
                val = data.get_val(x, y)
                points.append(Point(XLoc(x), YLoc(y), val))
                if val != points[0].val:
                    p = points.pop()
                    try_emit(self)
                    points = [p]
            try_emit(self)

        ## iterate points by 'y' to identify vertical lines
        for x in range(data.width):
            for y in range(data.height):
                val = data.get_val(x, y)
                points.append(Point(XLoc(x), YLoc(y), val))
                if val != points[0].val:
                    p = points.pop()
                    try_emit(self)
                    points = [p]
            try_emit(self)

        self.settled()
