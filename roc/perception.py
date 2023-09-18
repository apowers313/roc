"""The Perception system breaks down the environment into features that can be
re-assembled as concepts."""

from __future__ import annotations

from collections.abc import Hashable

from pydantic import BaseModel

from .component import Component, register_component
from .event import Event, EventBus
from .logger import logger

Grid = tuple[tuple[int | str, ...], ...]


# TODO: vision input
# TODO: sound input
# TODO: other input
class VisionData(BaseModel):
    """A Pydantic model for the vision perception data."""

    # spectrum: tuple[tuple[tuple[int | str, ...], ...], ...]
    screen: Grid
    # spectrum: tuple[int | str, ...]


class DeltaData(BaseModel):
    """A Pydantic model for delta features."""

    diff_list: "DiffList"


PerceptionData = VisionData | DeltaData

PerceptionEvent = Event[PerceptionData]

perception_bus = EventBus[PerceptionData]("perception")


class Perception(Component):
    """The abstract class for Perception components. Handles perception bus
    connections and corresponding clean-up."""

    def __init__(self) -> None:
        super().__init__()
        self.pb_conn = self.connect_bus(perception_bus)
        self.pb_conn.listen(self.do_perception)

    def do_perception(self, e: PerceptionEvent) -> None:
        lambda e: logger.info(f"Perception got {e}")

    @staticmethod
    def init() -> None:
        global perception_bus
        perception_bus = EventBus[PerceptionData]("perception")


class Feature(Hashable):
    """An abstract feature for communicating features that have been detected."""

    pass


class DeltaFeature(Feature):
    """A feature for representing vision changes (deltas)"""

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")


@register_component("delta", "perception")
class Delta(Perception):
    """A component for detecting changes in vision."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_grid: Grid | None = None

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def do_perception(self, e: PerceptionEvent) -> None:
        logger.debug(f"got perception event {e}")
        # assert isinstance(e, VisionData)
        # reveal_type(e)
        # reveal_type(e.data)
        data = e.data
        assert isinstance(data, VisionData)

        prev = self.prev_grid
        self.prev_grid = curr = data.screen

        if prev is None:
            return

        # roughly make sure that things are the same height
        assert len(prev) == len(curr)
        assert len(prev[0]) == len(curr[0])

        width = len(curr)
        height = len(curr[0])
        diff_list: DiffList = []
        for x in range(width):
            for y in range(height):
                if prev[x][y] != curr[x][y]:
                    diff_list.append(
                        Diff(
                            x=x,
                            y=y,
                            val1=prev[x][y],
                            val2=curr[x][y],
                        )
                    )

        self.pb_conn.send(DeltaData(diff_list=diff_list))


class Diff(BaseModel):
    """A Pydantic model for representing a changes in vision."""

    x: int
    y: int
    val1: str | int
    val2: str | int


DiffList = list[Diff]
