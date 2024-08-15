"""A component for generating Features that represent differences in vision"""

from __future__ import annotations

from dataclasses import dataclass

from ..component import Component, register_component
from ..location import IntGrid
from ..perception import (
    ComplexFeature,
    Feature,
    FeatureExtractor,
    OldLocation,
    PerceptionEvent,
    Transmogrifier,
    VisionData,
)


@dataclass
class Diff(Transmogrifier):
    """A dataclass for representing a changes in vision."""

    x: int
    y: int
    old_val: int
    new_val: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y}): {self.old_val} '{chr(self.old_val)}' -> {self.new_val} '{chr(self.new_val)}'\n"

    def add_to_feature(self, n: Feature) -> None:
        """Adds a set of Diff nodes to a Feature"""
        n.add_type(self.new_val)
        n.add_point(self.x, self.y)
        ol = OldLocation(n.origin, self.x, self.y, self.old_val)
        n.add_feature("Past", ol)

    @classmethod
    def from_feature(self, n: Feature) -> Diff:
        """Creates a Diff from a Feature that has all the right Nodes"""
        x, y = n.get_point()
        new_val = n.get_type()
        old = n.get_feature("Past")
        assert isinstance(old, Feature)
        old_val = old.get_type()
        return Diff(x=x, y=y, old_val=old_val, new_val=new_val)


class DeltaFeature(ComplexFeature[Diff]):
    """A Feature that describes changes in vision"""

    def __init__(self, origin: Component, d: Diff) -> None:
        super().__init__("Delta", origin, d)


@register_component("delta", "perception")
class Delta(FeatureExtractor[DeltaFeature]):
    """A component for detecting changes in vision."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_viz: IntGrid | None = None

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> None:
        data = e.data
        assert isinstance(data, VisionData)

        prev = self.prev_viz
        self.prev_viz = curr = IntGrid(data.chars)  # TODO: glyphs

        if prev is None:
            # can't get difference when there was nothing before this
            self.settled()
            return None

        # roughly make sure that things are the same size
        assert prev.height == curr.height
        assert prev.width == curr.width

        for new_point in curr.points():
            old_point = prev.get_point(new_point.x, new_point.y)
            if old_point.val != new_point.val:
                self.pb_conn.send(
                    DeltaFeature(
                        self,
                        Diff(
                            x=new_point.x,
                            y=new_point.y,
                            old_val=old_point.val,
                            new_val=new_point.val,
                        ),
                    )
                )

        self.settled()
        return None
