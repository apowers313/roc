"""The Perception system breaks down the environment into features that can be
re-assembled as concepts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Self, TypeVar

from .component import Component
from .event import Event, EventBus
from .graphdb import Node
from .point import Grid

VisionData = Grid
# TODO: sound input
# TODO: other input


@dataclass
class Settled:
    pass


FeatureType = TypeVar("FeatureType")


class ElementSize(Node, extra="forbid"):
    size: int


class ElementType(Node, extra="forbid"):
    type: int


class ElementPoint(Node, extra="forbid"):
    x: int
    y: int


class ElementTypedPoint(Node, extra="forbid"):
    type: int
    x: int
    y: int


class Direction(str, Enum):
    up = "UP"
    down = "DOWN"
    left = "LEFT"
    right = "RIGHT"
    up_right = "UP_RIGHT"
    up_left = "UP_LEFT"
    down_right = "DOWN_RIGHT"
    down_left = "DOWN_LEFT"


class ElementOrientation(Node, extra="forbid"):
    orientation: Direction


class Feature(Node, ABC):
    _origin: str

    @property
    def origin(self) -> str:
        return self._origin

    def __init__(self, origin: Component | str, label: str) -> None:
        super().__init__(labels={"Feature", label})
        if isinstance(origin, Component):
            origin = f"{origin.name}:{origin.type}"
        # XXX: don't store a reference to the actual component or you may end up
        # with circular references and memory leaks
        self._origin = origin

    def add_type(self, type: int) -> ElementType:
        f = ElementType(type=type)
        Node.connect(self, f, "Type")
        return f

    def add_point(self, x: int, y: int) -> ElementPoint:
        p = ElementPoint(x=x, y=y)
        Node.connect(self, p, "Location")
        return p

    def add_size(self, size: int) -> ElementSize:
        s = ElementSize(size=size)
        Node.connect(self, s, "Size")
        return s

    def add_orientation(self, orientation: Direction) -> ElementOrientation:
        o = ElementOrientation(orientation=orientation)
        Node.connect(self, o, "Direction")
        return o

    def add_feature(self, type: str, feature: Feature) -> Feature:
        Node.connect(self, feature, type)
        return feature

    def get_feature(self, type: str) -> Node | None:
        nodes = self.get_features(type)
        if len(nodes) < 1:
            return None

        return nodes[0]

    def get_features(self, type: str) -> list[Node]:
        edge_iter = self.src_edges.get_edges(type)
        return list(map(lambda e: e.dst, edge_iter))

    def get_point(self) -> tuple[int, int]:
        pt = self.get_feature("Location")
        if not pt:
            raise Exception("no Location in get_point()")

        assert isinstance(pt, ElementPoint)
        return (pt.x, pt.y)

    def get_type(self) -> int:
        t = self.get_feature("Type")
        if not t:
            raise Exception("no Type in get_type()")

        assert isinstance(t, ElementType)
        return t.type

    def get_orientation(self) -> Direction:
        o = self.get_feature("Direction")
        if not o:
            raise Exception("no Orientation in get_orientation()")

        assert isinstance(o, ElementOrientation)
        return o.orientation


@dataclass
class Transmogrifier(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def add_to_feature(self, n: Feature) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_feature(self, n: Feature) -> Self:
        pass


FeatureTransmogrifier = TypeVar("FeatureTransmogrifier", bound=Transmogrifier)


class ComplexFeature(Feature, Generic[FeatureTransmogrifier]):
    def __init__(self, name: str, origin: Component, trans: FeatureTransmogrifier) -> None:
        super().__init__(origin, name)
        self._transmogrifier = trans
        self._transmogrifier.add_to_feature(self)

    def __str__(self) -> str:
        f = self._transmogrifier.__class__.from_feature(self)
        return str(f)


class OldLocation(Feature):
    """A feature for describing an old location and value"""

    def __init__(self, origin: str, x: int, y: int, val: int) -> None:
        super().__init__(origin, "Old")
        self.add_type(val)
        self.add_point(x, y)


PerceptionData = VisionData | Settled | Feature
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


class FeatureExtractor(Perception, Generic[FeatureType], ABC):
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
