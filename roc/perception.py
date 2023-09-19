"""The Perception system breaks down the environment into features that can be
re-assembled as concepts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import cast

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


# class DeltaData(BaseModel):
#     """A Pydantic model for delta features."""

#     diff_list: "DiffList"


@dataclass
class FeatureData:
    feature: Feature


PerceptionData = VisionData | FeatureData

PerceptionEvent = Event[PerceptionData]

perception_bus = EventBus[PerceptionData]("perception")


class Perception(Component, ABC):
    """The abstract class for Perception components. Handles perception bus
    connections and corresponding clean-up."""

    def __init__(self) -> None:
        super().__init__()
        self.pb_conn = self.connect_bus(perception_bus)
        self.pb_conn.listen(self.do_perception)

    @abstractmethod
    def do_perception(self, e: PerceptionEvent) -> None:
        ...

    @staticmethod
    def init() -> None:
        global perception_bus
        perception_bus = EventBus[PerceptionData]("perception")


class FeatureExtractor(Perception, ABC):
    def __init__(self) -> None:
        super().__init__()

    def do_perception(self, e: PerceptionEvent) -> None:
        f = self.get_feature(e)
        if f is None:
            f = NONE_FEATURE

        feature_data = FeatureData(feature=f)
        self.pb_conn.send(feature_data)

    @abstractmethod
    def get_feature(self, e: PerceptionEvent) -> Feature | None:
        ...


class Feature(Hashable):
    """An abstract feature for communicating features that have been detected."""

    origin: Component

    def __init__(self, origin: Component) -> None:
        self.origin = origin


class HashingNoneFeature(Exception):
    pass


class NoneFeature:
    def __hash__(self) -> int:
        raise HashingNoneFeature("Attempting to hash None feature")


# sentinal
NONE_FEATURE: Feature = cast(Feature, NoneFeature())


class DeltaFeature(Feature):
    """A feature for representing vision changes (deltas)"""

    def __init__(self, origin: Component, diff_list: DiffList) -> None:
        super().__init__(origin)
        self.diff_list = diff_list

    def __hash__(self) -> int:
        raise NotImplementedError("DeltaFeature hash not implemented")


@register_component("delta", "perception")
class Delta(FeatureExtractor):
    """A component for detecting changes in vision."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_grid: Grid | None = None

    def event_filter(self, e: PerceptionEvent) -> bool:
        return isinstance(e.data, VisionData)

    def get_feature(self, e: PerceptionEvent) -> Feature | None:
        logger.debug(f"got perception event {e}")
        # assert isinstance(e, VisionData)
        # reveal_type(e)
        # reveal_type(e.data)
        data = e.data
        assert isinstance(data, VisionData)

        prev = self.prev_grid
        self.prev_grid = curr = data.screen

        if prev is None:
            return None

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

        return DeltaFeature(self, diff_list)


class Diff(BaseModel):
    """A Pydantic model for representing a changes in vision."""

    x: int
    y: int
    val1: str | int
    val2: str | int


DiffList = list[Diff]
