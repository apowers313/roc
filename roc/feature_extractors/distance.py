"""Calculates distance between Singles. Note: this is probably post-attention in
humans, only calculated for a subset of features, and based on the saccades of the eyes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..location import XLoc, YLoc
from ..perception import FeatureExtractor, FeatureNode, PerceptionEvent, Settled, VisualFeature
from .single import SingleFeature


class DistanceNode(FeatureNode):
    """Graph node representing a Chebyshev distance between two features."""

    size: int

    @property
    def attr_strs(self) -> list[str]:
        """Returns the distance size as a string."""
        return [str(self.size)]


@dataclass(kw_only=True)
class DistanceFeature(VisualFeature[DistanceNode]):
    """The distance between two features"""

    feature_name: str = "Distance"
    start_point: tuple[XLoc, YLoc]
    end_point: tuple[XLoc, YLoc]
    size: int

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        """Returns both endpoints of the distance measurement."""
        return {self.start_point, self.end_point}

    def node_hash(self) -> int:
        """Hashes by distance size."""
        return hash(self.size)

    def _create_nodes(self) -> DistanceNode:
        """Creates a new DistanceNode with the measured size."""
        return DistanceNode(size=self.size)

    def _dbfetch_nodes(self) -> DistanceNode | None:
        """Looks up an existing DistanceNode by size."""
        return DistanceNode.find_one("src.size = $size", params={"size": self.size})


class Distance(FeatureExtractor[DistanceFeature]):
    """Component that calculates Chebyshev distances between single features."""

    name: str = "distance"
    type: str = "perception"

    def __init__(self) -> None:
        super().__init__()
        self.prev_features: list[SingleFeature] = []

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Only process events from the single perception component."""
        # only listen to single:perception events
        if e.src_id.name == "single" and e.src_id.type == "perception":
            return True
        return False

    def get_feature(self, e: PerceptionEvent) -> None:
        """Calculates distances between the current feature and all previous features."""
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
                    origin_id=self.id,
                    start_point=prev_pt,
                    end_point=curr_pt,
                    size=dist,
                )
            )

        # add to previous elements
        self.prev_features.append(data)


def chebyshev_distance(p1: tuple[XLoc, YLoc], p2: tuple[XLoc, YLoc]) -> int:
    """Computes the Chebyshev distance (max of dx, dy) between two points."""
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy)
