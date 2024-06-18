from ..component import Component, register_component
from ..location import Point, PointList, TypedPointCollection
from ..perception import Feature, FeatureExtractor, PerceptionEvent, VisionData

MIN_LINE_COUNT = 4


class LineFeature(Feature):
    """A collection of points representing a line"""

    def __init__(self, origin: Component, point_list: PointList, type: int) -> None:
        super().__init__(origin, "Line")
        self.add_type(type)
        self.add_size(len(point_list))
        for point in point_list:
            self.add_point(point.x, point.y)

    def __hash__(self) -> int:
        raise NotImplementedError("LineFeature hash not implemented")


@register_component("line", "perception")
class Line(FeatureExtractor[TypedPointCollection]):
    """A component for identifying similar values located along a vertical or
    horizontal line"""

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> Feature | None:
        data = e.data
        assert isinstance(data, VisionData)

        points: PointList = []

        def try_emit(ln: Line) -> None:
            nonlocal points

            if len(points) >= MIN_LINE_COUNT:
                ln.pb_conn.send(LineFeature(self, points, points[0].val))
            points = []

        ## iterate points by 'x' to identify horizontal lines
        for y in range(data.height):
            for x in range(data.width):
                val = data.get_val(x, y)
                points.append(Point(x, y, val))
                if val != points[0].val:
                    p = points.pop()
                    try_emit(self)
                    points = [p]
            try_emit(self)

        ## iterate points by 'y' to identify vertical lines
        for x in range(data.width):
            for y in range(data.height):
                val = data.get_val(x, y)
                points.append(Point(x, y, val))
                if val != points[0].val:
                    p = points.pop()
                    try_emit(self)
                    points = [p]
            try_emit(self)

        self.settled()
        return None
