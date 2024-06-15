from __future__ import annotations

from ..component import Component, register_component
from ..perception import (
    ElementPoint,
    ElementType,
    FeatureExtractor,
    NewFeature,
    OldLocation,
    PerceptionEvent,
    VisionData,
)


class DeltaFeature(NewFeature):
    """A feature for representing vision changes (deltas)"""

    def __init__(self, origin: Component, x: int, y: int, old_val: int, new_val: int) -> None:
        super().__init__(origin, "Delta")
        self.add_type(new_val)
        self.add_point(x, y)
        ol = OldLocation(origin, x, y, old_val)
        self.add_feature("Past", ol)

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")

    def __str__(self) -> str:
        old = self.get_feature("Past")
        assert isinstance(old, NewFeature)
        old_val = old.get_feature("Type")
        assert isinstance(old_val, ElementType)
        new_val = self.get_feature("Type")
        assert isinstance(new_val, ElementType)
        loc = self.get_feature("Location")
        assert isinstance(loc, ElementPoint)

        return f"({loc.x}, {loc.y}): {old_val.type} '{chr(old_val.type)}' -> {new_val.type} '{chr(new_val.type)}'\n"


@register_component("delta", "perception")
class Delta(FeatureExtractor[DeltaFeature]):
    """A component for detecting changes in vision."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_viz: VisionData | None = None

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> None:
        data = e.data
        assert isinstance(data, VisionData)

        prev = self.prev_viz
        self.prev_viz = curr = data

        if prev is None:
            # can't get difference when there was nothing before this
            self.settled()
            return None

        # roughly make sure that things are the same size
        assert prev.height == curr.height
        assert prev.width == curr.width

        for new_point in curr:
            old_point = prev.get_point(new_point.x, new_point.y)
            if old_point.val != new_point.val:
                self.pb_conn.send(
                    DeltaFeature(self, new_point.x, new_point.y, old_point.val, new_point.val)
                )

        self.settled()
        return None
