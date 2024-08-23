"""Calculates distance between Singles. Note: this is probably post-attention in
humans, only calculated for a subset of features, and based on the saccades of the eyes.
"""

from __future__ import annotations

from ..component import Component, register_component
from ..perception import Feature, FeatureExtractor, PerceptionEvent, Settled, VisionData
from .single import SingleFeature


class DistanceFeature(Feature):
    """The distance between two features"""

    def __init__(
        self, origin: Component, start: tuple[int, int], end: tuple[int, int], sz: int
    ) -> None:
        super().__init__(origin, "Distance")
        self.add_size(sz)
        self.add_point(start[0], start[1])
        self.add_point(end[0], end[1])

    def __hash__(self) -> int:
        raise NotImplementedError("DistanceFeature hash not implemented")


@register_component("distance", "perception")
class Distance(FeatureExtractor[DistanceFeature]):
    def __init__(self) -> None:
        super().__init__()
        self.prev_features: list[SingleFeature] = []

    def event_filter(self, e: PerceptionEvent) -> bool:
        # only listen to single:perception events
        if e.src.name == "single" and e.src.type == "perception":
            return True
        return False

    def get_feature(self, e: PerceptionEvent) -> Feature | None:
        data = e.data

        if isinstance(data, Settled):
            self.prev_features.clear()
            self.settled()
            return None

        # calculate distance between all previous elements and emit event for each
        assert isinstance(data, SingleFeature)
        curr_pt = data.get_point()
        # print("### single point:", curr_pt)
        for f in self.prev_features:
            prev_pt = f.get_point()
            dist = chebyshev_distance(curr_pt, prev_pt)
            # print(f"# distance: ({prev_pt[0]}, {prev_pt[1]}) ({curr_pt[0]}, {curr_pt[1]}) {dist}")
            self.pb_conn.send(DistanceFeature(self, curr_pt, prev_pt, dist))

        # add to previous elements
        self.prev_features.append(data)

        return None


def chebyshev_distance(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy)
