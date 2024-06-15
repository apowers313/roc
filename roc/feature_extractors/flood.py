from ..component import Component, register_component
from ..perception import FeatureExtractor, NewFeature, PerceptionEvent, VisionData
from ..point import Grid, Point, PointList, TypedPointCollection

MIN_FLOOD_SIZE = 5


class FloodFeature(NewFeature):
    """A collection of points representing similar values that are all adjacent to each other"""

    def __init__(self, origin: Component, point_list: PointList, type: int) -> None:
        super().__init__(origin, "Flood")
        self.add_type(type)
        self.add_size(len(point_list))
        for point in point_list:
            self.add_point(point.x, point.y)

    def __hash__(self) -> int:
        raise NotImplementedError("FloodFeature hash not implemented")


class CheckMap:
    """Internal utility class for tracking which points in a flood have already been checked"""

    def __init__(self, width: int, height: int) -> None:
        self.grid = Grid.filled(0, width, height)

    def find_first_unused_point(self) -> Point | None:
        ret: Point | None = None
        for p in self.grid:
            if p.val == 0:
                ret = p
                break

        return ret

    def set(self, x: int, y: int) -> None:
        self.grid.set_val(x, y, 1)

    def checked(self, x: int, y: int) -> bool:
        return self.grid.get_val(x, y) == 1


@register_component("flood", "perception")
class Flood(FeatureExtractor[TypedPointCollection]):
    """A component for creating Flood features -- collections of adjacent points
    that all have the same value"""

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> NewFeature | None:
        data = e.data
        assert isinstance(data, VisionData)

        check_map = CheckMap(data.width, data.height)

        def recursive_flood_check(val: int, x: int, y: int, point_list: PointList) -> PointList:
            if x < 0 or y < 0 or x >= data.width or y >= data.height:
                # out of bounds... move on
                return point_list

            if check_map.checked(x, y):
                # we already checked this point... move on
                return point_list

            if data.get_val(x, y) != val:
                # not the right value, for this flood... move on
                return point_list

            check_map.set(x, y)
            point_list.append(Point(x, y, val))

            # explanation:
            # (0, 0) is top left, algorithm is iterating from left to right and
            # top to bottom
            # check left: already done
            # check up *: already done

            # check right
            recursive_flood_check(val, x + 1, y, point_list)
            # check left down
            recursive_flood_check(val, x - 1, y + 1, point_list)
            # check down
            recursive_flood_check(val, x, y + 1, point_list)
            # check right down
            recursive_flood_check(val, x + 1, y + 1, point_list)

            return point_list

        while p := check_map.find_first_unused_point():
            val = data.get_val(p.x, p.y)
            point_list = recursive_flood_check(val, p.x, p.y, [])
            if len(point_list) >= MIN_FLOOD_SIZE:
                self.pb_conn.send(FloodFeature(self, point_list, val))
            check_map.set(p.x, p.y)

        self.settled()
        return None
