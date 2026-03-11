"""Identifies horizontal and vertical lines of repeated glyph values."""

from dataclasses import dataclass

from ..location import IntGrid, Point, PointList, TypedPointCollection, XLoc, YLoc
from ..perception import (
    AreaFeature,
    FeatureExtractor,
    FeatureKind,
    FeatureNode,
    PerceptionEvent,
    VisionData,
)

MIN_LINE_COUNT = 4


class LineNode(FeatureNode):
    """Graph node representing a line feature by type, length, color, and shape."""

    kind = FeatureKind.PHYSICAL
    type: int
    size: int
    color: int
    shape: int

    @property
    def attr_strs(self) -> list[str]:
        """Returns type, size, color, and shape as strings."""
        return [str(self.type), str(self.size), str(self.color), chr(self.shape)]


@dataclass(kw_only=True)
class LineFeature(AreaFeature[LineNode]):
    """A collection of points representing a line"""

    feature_name: str = "Line"
    color: int = 0
    shape: int = 0

    def node_hash(self) -> int:
        """Hashes by type, size, color, and shape."""
        return hash((self.type, self.size, self.color, self.shape))

    def _create_nodes(self) -> LineNode:
        """Creates a new LineNode with type, length, color, and shape."""
        return LineNode(type=self.type, size=self.size, color=self.color, shape=self.shape)

    def _dbfetch_nodes(self) -> LineNode | None:
        """Looks up an existing LineNode by type, size, color, and shape."""
        return LineNode.find_one(
            "src.type = $type AND src.size = $size AND src.color = $color AND src.shape = $shape",
            params={
                "type": self.type,
                "size": self.size,
                "color": self.color,
                "shape": self.shape,
            },
        )


class Line(FeatureExtractor[TypedPointCollection]):
    """A component for identifying similar values located along a vertical or
    horizontal line
    """

    name: str = "line"
    type: str = "perception"

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Only process VisionData events."""
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> None:
        """Scans for horizontal and vertical lines of repeated values."""
        assert isinstance(e.data, VisionData)
        vd = e.data
        data = IntGrid(vd.glyphs)

        points: PointList = []

        def try_emit(ln: Line) -> None:
            nonlocal points

            if len(points) >= MIN_LINE_COUNT:
                point_set = {(p.x, p.y) for p in points}
                rep = points[0]
                ln.pb_conn.send(
                    LineFeature(
                        origin_id=self.id,
                        points=point_set,
                        type=rep.val,
                        size=len(points),
                        color=int(vd.colors[rep.y, rep.x]),
                        shape=int(vd.chars[rep.y, rep.x]),
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
