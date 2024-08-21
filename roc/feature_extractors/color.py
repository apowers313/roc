"""Generates Features for things that aren't like their neighbors"""

from typing import Any

import numpy.typing as npt

from ..component import Component, register_component
from ..location import IntGrid, Point
from ..perception import Feature, FeatureExtractor, PerceptionEvent, Settled, VisionData
from .single import SingleFeature


class ColorFeature(Feature):
    """The color of a single feature."""

    def __init__(self, origin: Component, point: Point) -> None:
        super().__init__(origin, "Color")
        self.add_point(point.x, point.y)
        self.add_type(point.val)

    def __hash__(self) -> int:
        raise NotImplementedError("ColorFeature hash not implemented")


@register_component("color", "perception")
class Color(FeatureExtractor[Point]):
    """A component for simulating the color of features based on the character
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

    def get_feature(self, e: PerceptionEvent) -> Feature | None:
        """Emits the color features.

        Args:
            e (PerceptionEvent): The VisionData or SingleFeature

        Returns:
            Feature | None: None
        """
        if isinstance(e.data, SingleFeature):
            self.queue.append(e.data)
            return None

        if isinstance(e.data, Settled):
            self.single_settled = True

        if isinstance(e.data, VisionData):
            self.vd = e.data

        if self.single_settled and self.vd is not None:
            for s in self.queue:
                x, y = s.get_point()
                p = Point(x, y, self.vd.colors[y, x])
                self.pb_conn.send(ColorFeature(self, p))

            self.settled()
            self.queue.clear()
            self.single_settled = False
            self.vd = None

        return None
