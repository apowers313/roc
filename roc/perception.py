"""The Perception system breaks down the environment into features that can be
re-assembled as concepts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from .component import Component
from .event import Event, EventBus

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


class Perception(Component, ABC):
    """The abstract class for Perception components. Handles perception bus
    connections and corresponding clean-up."""

    bus = EventBus[PerceptionData]("perception")

    def __init__(self) -> None:
        super().__init__()
        self.pb_conn = self.connect_bus(Perception.bus)
        self.pb_conn.listen(self.do_perception)

    @abstractmethod
    def do_perception(self, e: PerceptionEvent) -> None: ...

    @classmethod
    def init(cls) -> None:
        global perception_bus
        cls.bus = EventBus[PerceptionData]("perception")


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
    def get_feature(self, e: PerceptionEvent) -> Feature | None: ...


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
