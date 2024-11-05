"""Generates Features for things that aren't like their neighbors"""

from dataclasses import dataclass

from ..component import register_component
from ..location import Point
from ..perception import (
    FeatureExtractor,
    PerceptionEvent,
    PointFeature,
    Settled,
    VisionData,
)
from .single import SingleFeature


@dataclass(kw_only=True)
class ShapeFeature(PointFeature):
    """The shape of a single feature."""

    feature_name: str = "Shape"

    def __hash__(self) -> int:
        raise NotImplementedError("ShapeFeature hash not implemented")


@register_component("shape", "perception")
class Shape(FeatureExtractor[Point]):
    """A component for simulating the shape of features based on the character
    value.
    """

    def __init__(self) -> None:
        super().__init__()
        self.queue: list[SingleFeature] = list()
        self.vd: VisionData | None = None
        self.single_settled = False

    def event_filter(self, e: PerceptionEvent) -> bool:
        """Filters out non-SingleFeatures and non-VisionData

        Args:
            e (PerceptionEvent): Any event on the perception bus

        Returns:
            bool: Returns True if the event is a SingleFeature or VisionData to
            keep processing it, False otherwise.
        """
        return isinstance(e.data, VisionData) or e.src.name == "single"

    def get_feature(self, e: PerceptionEvent) -> None:
        """Emits the shape features.

        Args:
            e (PerceptionEvent): The VisionData or SingleFeature

        Returns:
            Feature | None: None
        """
        if isinstance(e.data, SingleFeature):
            self.queue.append(e.data)
            return

        if isinstance(e.data, Settled):
            self.single_settled = True

        if isinstance(e.data, VisionData):
            self.vd = e.data

        if self.single_settled and self.vd is not None:
            for s in self.queue:
                x, y = s.point
                p = Point(x, y, self.vd.chars[y, x])
                self.pb_conn.send(
                    ShapeFeature(
                        origin_id=self.id,
                        point=(x, y),
                        type=self.vd.chars[y, x],
                    )
                )

            self.settled()
            self.queue.clear()
            self.single_settled = False
            self.vd = None
