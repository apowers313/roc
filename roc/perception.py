"""The Perception system breaks down the environment into features that can be
re-assembled as concepts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

from .component import Component
from .event import Event, EventBus
from .point import Grid

# Grid = tuple[tuple[int | str, ...], ...]
# Screen = tuple[tuple[int | str, ...], ...]
# Screen = list[list[int]]

# val, x, y
# GridTriple = tuple[str | int, int, int]


# class VisionData:
#     def __init__(self, screen: Screen) -> None:
#         self.screen = screen

#     def __iter__(self) -> Iterator[GridTriple]:
#         for y in range(self.height):
#             for x in range(self.width):
#                 val = self.screen[y][x]
#                 yield (val, x, y)

#     def get(self, x: int, y: int) -> int | str:
#         return self.screen[y][x]

#     @property
#     def width(self) -> int:
#         return len(self.screen[0])

#     @property
#     def height(self) -> int:
#         return len(self.screen)

VisionData = Grid

# TODO: vision input
# TODO: sound input
# TODO: other input
# class VisionData(BaseModel):
#     """A Pydantic model for the vision perception data."""

#     # spectrum: tuple[tuple[tuple[int | str, ...], ...], ...]
#     screen: Grid
#     # spectrum: tuple[int | str, ...]


@dataclass
class Settled:
    pass


class Feature(Hashable):
    """An abstract feature for communicating features that have been detected."""

    origin: Component

    def __init__(self, origin: Component) -> None:
        self.origin = origin


PerceptionData = VisionData | Feature | Settled
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
            return

        self.pb_conn.send(f)

    def settled(self) -> None:
        self.pb_conn.send(Settled())

    @abstractmethod
    def get_feature(self, e: PerceptionEvent) -> Feature | None: ...


class HashingNoneFeature(Exception):
    pass


# @dataclass
# class FeatureData:
#     feature: Feature


# class SettledData:
#     def __hash__(self) -> int:
#         raise HashingNoneFeature("Attempting to hash None feature")

#     def __repr__(self) -> str:
#         return "<<SETTLED>>"
