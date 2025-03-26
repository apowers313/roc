from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from ..location import IntGrid, TypedPointCollection
from ..perception import (
    AreaFeature,
    FeatureExtractor,
    FeatureNode,
    PerceptionEvent,
    VisionData,
)

MIN_FLOOD_SIZE = 5
NONZERO_VAL = -1


class FloodNode(FeatureNode):
    type: int
    size: int

    @property
    def attr_strs(self) -> list[str]:
        return [str(self.type), str(self.size)]


@dataclass(kw_only=True)
class FloodFeature(AreaFeature[FloodNode]):
    """A collection of points representing similar values that are all adjacent to each other"""

    feature_name: str = "Flood"

    def _create_nodes(self) -> FloodNode:
        return FloodNode(type=self.type, size=self.size)

    def _dbfetch_nodes(self) -> FloodNode | None:
        return FloodNode.find_one(
            "src.type = $type AND src.size = $size", params={"type": self.type, "size": self.size}
        )


class Flood(FeatureExtractor[TypedPointCollection]):
    """A component for creating Flood features -- collections of adjacent points
    that all have the same value
    """

    name: str = "flood"
    type: str = "perception"

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> None:
        assert isinstance(e.data, VisionData)
        data = IntGrid(e.data.glyphs)
        indices = np.indices(data.T.shape).T
        structure = np.ones((3, 3), dtype=int)

        unique_nums, unique_count = np.unique(data, return_counts=True)
        for idx in range(len(unique_nums)):
            if unique_count[idx] >= MIN_FLOOD_SIZE:
                val = unique_nums[idx]
                d2 = np.copy(data)
                # label only finds non-zero values
                # this handles the case when value is zero
                if val == 0:
                    d2[d2 == val] = NONZERO_VAL
                    val = NONZERO_VAL

                d2[d2 != val] = 0
                labeled, ncomponents = label(d2, structure)
                for n in range(1, ncomponents + 1):
                    point_list = indices[labeled == n]
                    if val == NONZERO_VAL:
                        val = 0
                    if len(point_list) >= MIN_FLOOD_SIZE:
                        point_set = {(p[0], p[1]) for p in point_list}
                        self.pb_conn.send(
                            FloodFeature(
                                origin_id=self.id,
                                points=point_set,
                                type=val,
                                size=len(point_set),
                            )
                        )

        self.settled()
