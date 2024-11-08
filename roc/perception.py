"""The Perception system breaks down the environment into features that can be
re-assembled as concepts.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .component import Component
from .event import Event, EventBus
from .location import XLoc, YLoc


class VisionData:
    """Vision data received from the environment."""

    def __init__(
        self,
        glyphs: npt.NDArray[Any],
        chars: npt.NDArray[Any],
        colors: npt.NDArray[Any],
    ) -> None:
        self.glyphs = glyphs
        self.chars = chars
        self.colors = colors

    @staticmethod
    def from_dict(d: dict[str, Any]) -> VisionData:
        """Creates VisionData from an arbitrary dictionary

        Args:
            d (dict[str, Any]): The dictionary to create VisionData from. Must
            have 'chars', 'glyphs', and 'colors' members.

        Returns:
            VisionData: The newly created vision data.
        """

        def to_numpy(d: dict[str, Any], k: str) -> np.ndarray[Any, Any]:
            if not k in d:
                raise Exception(f"Expected '{k}' to exist in dict for VisionData.from_dict()")

            v = d[k]
            if not isinstance(v, np.ndarray):
                return np.array(v)
            return v

        glyphs = to_numpy(d, "glyphs")
        chars = to_numpy(d, "chars")
        colors = to_numpy(d, "colors")
        return VisionData(glyphs, chars, colors)

    @staticmethod
    def for_test(test_data: list[list[int]]) -> VisionData:
        """Creates VisionData for a test case, using a static 2D list of values
        to create all aspects of the VisionData

        Args:
            test_data (list[list[int]]): The test data to convert into VisionData

        Returns:
            VisionData: The created VisionData
        """
        a = np.array(test_data)
        return VisionData(a.copy(), a.copy(), a.copy())


# TODO: sound input
# TODO: other input


class Settled:
    pass


# class ElementSize(Node, extra="forbid"):
#     size: int


# class ElementType(Node, extra="forbid"):
# #     type: int


# class ElementPoint(Node, extra="forbid"):
#     x: int
#     y: int


# class ElementTypedPoint(Node, extra="forbid"):
# #     type: int
#     x: int
#     y: int


# class ElementOrientation(Node, extra="forbid"):
#     orientation: Direction


class Direction(str, Enum):
    up = "UP"
    down = "DOWN"
    left = "LEFT"
    right = "RIGHT"
    up_right = "UP_RIGHT"
    up_left = "UP_LEFT"
    down_right = "DOWN_RIGHT"
    down_left = "DOWN_LEFT"


@dataclass(kw_only=True)
class NewFeature(ABC):
    # feature_name: str | None = field(default=None)
    feature_name: str
    origin_id: tuple[str, str]

    @abstractmethod
    def get_points(self) -> set[tuple[XLoc, YLoc]]: ...

    # def to_nodes(self) -> Node:
    #     n = Node(labels={"Feature", self.feature_name})
    #     # TODO: create nodes for each feature value; steal from transmogrifier
    #     return n

    # @staticmethod
    # def from_nodes(n: Node) -> NewFeature:
    #     labels = n.labels - set("Feature")
    #     name = next(iter(labels))
    #     assert hasattr(n, "_origin")
    #     assert isinstance(n._origin, Component)
    #     f = NewFeature(origin=n._origin, feature_name=name)
    #     # TODO: add feature values from nodes; steal from transmogrifier
    #     return f

    # def add_type(self, type: int) -> ElementType:
    #     """Add a type Node to the feature that describes the type of feature"""
    #     f = ElementType(type=type)
    #     Node.connect(self, f, "Type")
    #     return f

    # def add_point(self, x: int, y: int) -> ElementPoint:
    #     """Add a location Node to the feature that describes one or more places
    #     where this feature exists.
    #     """
    #     p = ElementPoint(x=x, y=y)
    #     Node.connect(self, p, "Location")
    #     return p

    # def add_size(self, size: int) -> ElementSize:
    #     """Add a size Node to the feature that describes the magnitude of the feature"""
    #     s = ElementSize(size=size)
    #     Node.connect(self, s, "Size")
    #     return s

    # def add_orientation(self, orientation: Direction) -> ElementOrientation:
    #     """Add a direction Node to the feature that describes a direction or
    #     orientation of the feature
    #     """
    #     o = ElementOrientation(orientation=orientation)
    #     Node.connect(self, o, "Direction")
    #     return o

    # def add_feature(self, type: str, feature: Feature) -> Feature:
    #     """Add an arbitrary Feature Node"""
    #     Node.connect(self, feature, type)
    #     return feature

    # def get_feature(self, type: str) -> Node | None:
    #     """Get an Feature Node of the type described, or None if it doesn't
    #     exist. If multiple matching features are found, the first one is returned.
    #     """
    #     nodes = self.get_features(type)
    #     if len(nodes) < 1:
    #         return None

    #     return nodes[0]

    # def get_features(self, type: str) -> list[Node]:
    #     """Return all matching Feature Nodes of the type described."""
    #     edge_iter = self.src_edges.get_edges(type)
    #     return list(map(lambda e: e.dst, edge_iter))

    # def get_point(self) -> tuple[XLoc, YLoc]:
    #     pt = self.get_feature("Location")
    #     if pt is None:
    #         raise Exception("no Location in get_point()")

    #     assert isinstance(pt, ElementPoint)
    #     return (XLoc(pt.x), YLoc(pt.y))

    # def get_type(self) -> int:
    #     t = self.get_feature("Type")
    #     if not t:
    #         raise Exception("no Type in get_type()")

    #     assert isinstance(t, ElementType)
    #     return t.type

    # def get_size(self) -> int:
    #     sz = self.get_feature("Size")
    #     if not sz:
    #         raise Exception("no Size in get_size()")

    #     assert isinstance(sz, ElementSize)
    #     return sz.size

    # def get_orientation(self) -> Direction:
    #     o = self.get_feature("Direction")
    #     if not o:
    #         raise Exception("no Orientation in get_orientation()")

    #     assert isinstance(o, ElementOrientation)
    #     return o.orientation

    # @staticmethod
    # def find_parent_feature(n: Node) -> Feature | None:
    #     """Searches up the graph of nodes that point to this node until it finds
    #     an instance of Feature. Most Features are trees / DAGs, so this is good
    #     for taking a child node of a Feature and figuring out what Feature it
    #     belongs to. Note that bad things may happen if this is used on a Node
    #     that isn't part of a Feature AND if the parent graph has cycles.

    #     Args:
    #         n (Node): The node to find the parent feature of.

    #     Returns:
    #         Feature | None: The Feature that was found, or None if no feature was found.
    #     """
    #     if isinstance(n, Feature):
    #         return n

    #     for edge in n.dst_edges:
    #         f = Feature.find_parent_feature(edge.src)
    #         if f is not None:
    #             return f

    #     return None


# FeatureCls = TypeVar("FeatureCls", bound=type[NewFeature])
# def feature_definition(name: str) -> Callable[[type], type]:
#     def feature_definition_decorator(cls: type) -> type:
#         # set the type of 'feature_name' on the new class, so that dataclass
#         # finds it
#         cls.__annotations__["feature_name"] = str
#         # set the default value of 'feature_name' as a class property
#         cls.feature_name = name  # type: ignore
#         # call the dataclass decorator on our new class
#         cls = dataclass(kw_only=True)(cls)

#         return cls

#     return feature_definition_decorator


@dataclass(kw_only=True)
class AreaFeature(NewFeature):
    type: int
    points: set[tuple[XLoc, YLoc]]
    size: int

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        return self.points


@dataclass(kw_only=True)
class PointFeature(NewFeature):
    type: int
    point: tuple[XLoc, YLoc]

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        return {self.point}


PerceptionData = VisionData | Settled | NewFeature
PerceptionEvent = Event[PerceptionData]


class Perception(Component, ABC):
    """The abstract class for Perception components. Handles perception bus
    connections and corresponding clean-up.
    """

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


fe_list: list[weakref.ReferenceType[FeatureExtractor[Any]]] = []
FeatureType = TypeVar("FeatureType")


class FeatureExtractor(Perception, Generic[FeatureType], ABC):
    def __init__(self) -> None:
        super().__init__()
        fe_list.append(weakref.ref(self))

    def do_perception(self, e: PerceptionEvent) -> None:
        f = self.get_feature(e)
        if f is None:
            return

        self.pb_conn.send(f)

    def settled(self) -> None:
        self.pb_conn.send(Settled())

    @abstractmethod
    def get_feature(self, e: PerceptionEvent) -> NewFeature | None: ...

    @classmethod
    def list(cls) -> list[str]:
        ret: list[str] = []
        for fe_ref in fe_list:
            fe = fe_ref()
            if fe is None:
                continue
            ret.append(f"{fe.name}:{fe.type}")

        return ret


class HashingNoneFeature(Exception):
    pass
