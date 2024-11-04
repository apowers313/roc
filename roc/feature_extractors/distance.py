"""Calculates distance between Singles. Note: this is probably post-attention in
humans, only calculated for a subset of features, and based on the saccades of the eyes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..component import register_component
from ..location import XLoc, YLoc
from ..perception import FeatureExtractor, NewFeature, PerceptionEvent, Settled
from .single import SingleFeature


@dataclass(kw_only=True)
class DistanceFeature(NewFeature):
    """The distance between two features"""

    feature_name: str = "Distance"
    start_point: tuple[XLoc, YLoc]
    end_point: tuple[XLoc, YLoc]
    size: int

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        return {self.start_point, self.end_point}

    # def __hash__(self) -> int:
    #     raise NotImplementedError("DistanceFeature hash not implemented")


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

    def get_feature(self, e: PerceptionEvent) -> None:
        data = e.data

        if isinstance(data, Settled):
            self.prev_features.clear()
            self.settled()
            return

        # calculate distance between all previous elements and emit event for each
        assert isinstance(data, SingleFeature)
        curr_pt = data.point
        # print("### single point:", curr_pt)
        for f in self.prev_features:
            prev_pt = f.point
            dist = chebyshev_distance(curr_pt, prev_pt)
            # print(f"# distance: ({prev_pt[0]}, {prev_pt[1]}) ({curr_pt[0]}, {curr_pt[1]}) {dist}")
            self.pb_conn.send(
                DistanceFeature(
                    origin=self,
                    start_point=prev_pt,
                    end_point=curr_pt,
                    size=dist,
                )
            )

        # add to previous elements
        self.prev_features.append(data)


def chebyshev_distance(p1: tuple[XLoc, YLoc], p2: tuple[XLoc, YLoc]) -> int:
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy)
