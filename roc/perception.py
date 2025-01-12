"""The Perception system breaks down the environment into features that can be
re-assembled as concepts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar
from weakref import ReferenceType, WeakValueDictionary, ref

import numpy as np
import numpy.typing as npt

from .component import Component
from .event import Event, EventBus
from .graphdb import Edge, EdgeConnectionsList, Node
from .location import XLoc, YLoc

FeatureType = TypeVar("FeatureType")


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


class Direction(str, Enum):
    up = "UP"
    down = "DOWN"
    left = "LEFT"
    right = "RIGHT"
    up_right = "UP_RIGHT"
    up_left = "UP_LEFT"
    down_right = "DOWN_RIGHT"
    down_left = "DOWN_LEFT"

    def __str__(self) -> str:
        return self.value


class Detail(Edge):
    allowed_connections: EdgeConnectionsList = [("FeatureGroup", "FeatureNode")]


class FeatureNode(Node):
    def __hash__(self) -> int:
        # XXX: this is dangerous because ID changes when a node is saved
        # should be okay for this use case though
        return self.id

    def __str__(self) -> str:
        return f"""{self.__class__.__name__}({",".join(self.attr_strs)})"""

    @property
    @abstractmethod
    def attr_strs(self) -> list[str]: ...


cache_registry: dict[str, WeakValueDictionary[int, Node]] = defaultdict(WeakValueDictionary)
FeatureNodeType = TypeVar("FeatureNodeType", bound=FeatureNode)


@dataclass(kw_only=True)
class Feature(ABC, Generic[FeatureNodeType]):
    feature_name: str
    origin_id: tuple[str, str]

    def to_nodes(self) -> FeatureNodeType:
        # check local cache
        cache = cache_registry[self.feature_name]

        h = self.node_hash()
        if h in cache:
            return cache[h]  # type: ignore

        # if cache miss, find node in database
        n = self._dbfetch_nodes()
        if n is None:
            # if node doesn't exist, create it
            n = self._create_nodes()
            # n.labels.add("Feature")
            # n.labels.add(self.feature_name)

        cache[h] = n
        return n

    @abstractmethod
    def get_points(self) -> set[tuple[XLoc, YLoc]]: ...

    @abstractmethod
    def _create_nodes(self) -> FeatureNodeType: ...

    @abstractmethod
    def _dbfetch_nodes(self) -> FeatureNodeType | None: ...

    @abstractmethod
    def node_hash(self) -> int: ...


@dataclass(kw_only=True)
class AreaFeature(Feature[FeatureNodeType]):
    type: int
    points: set[tuple[XLoc, YLoc]]
    size: int

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        return self.points

    def node_hash(self) -> int:
        return hash((self.type, self.size))


@dataclass(kw_only=True)
class PointFeature(Feature[FeatureNodeType]):
    type: int
    point: tuple[XLoc, YLoc]

    def get_points(self) -> set[tuple[XLoc, YLoc]]:
        return {self.point}

    def node_hash(self) -> int:
        return self.type


PerceptionData = VisionData | Settled | Feature[Any]
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


fe_list: list[ReferenceType[FeatureExtractor[Any]]] = []


class FeatureExtractor(Perception, Generic[FeatureType], ABC):
    def __init__(self) -> None:
        super().__init__()
        fe_list.append(ref(self))

    def do_perception(self, e: PerceptionEvent) -> None:
        f = self.get_feature(e)
        if f is None:
            return

        self.pb_conn.send(f)

    def settled(self) -> None:
        self.pb_conn.send(Settled())

    @abstractmethod
    def get_feature(self, e: PerceptionEvent) -> Feature[Any] | None: ...

    @classmethod
    def list(cls) -> list[str]:
        ret: list[str] = []
        for fe_ref in fe_list:
            fe = fe_ref()
            if fe is None:
                continue
            ret.append(str(fe.id))

        return ret


class HashingNoneFeature(Exception):
    pass
