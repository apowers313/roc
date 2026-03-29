"""Identifies horizontal and vertical lines of repeated glyph values."""

from dataclasses import dataclass

from ..graphdb import FindQueryOpts
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
            query_opts=FindQueryOpts(
                params={
                    "type": self.type,
                    "size": self.size,
                    "color": self.color,
                    "shape": self.shape,
                }
            ),
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

        # scan horizontal lines (iterate by x within each y)
        for y in range(data.height):
            coords = [(x, y) for x in range(data.width)]
            self._scan_line(coords, data, vd)

        # scan vertical lines (iterate by y within each x)
        for x in range(data.width):
            coords = [(x, y) for y in range(data.height)]
            self._scan_line(coords, data, vd)

        self.settled()

    def _scan_line(
        self,
        coords: list[tuple[int, int]],
        data: IntGrid,
        vd: VisionData,
    ) -> None:
        """Scan a sequence of coordinates for runs of identical values."""
        points: PointList = []
        for x, y in coords:
            val = data.get_val(x, y)
            points.append(Point(XLoc(x), YLoc(y), val))
            if val != points[0].val:
                p = points.pop()
                self._emit_line(points, vd)
                points = [p]
        self._emit_line(points, vd)

    def _emit_line(self, points: PointList, vd: VisionData) -> None:
        """Emit a LineFeature if the run meets the minimum length."""
        if len(points) >= MIN_LINE_COUNT:
            point_set = {(p.x, p.y) for p in points}
            rep = points[0]
            self.pb_conn.send(
                LineFeature(
                    origin_id=self.id,
                    points=point_set,
                    type=rep.val,
                    size=len(points),
                    color=int(vd.colors[rep.y, rep.x]),
                    shape=int(vd.chars[rep.y, rep.x]),
                ),
            )
