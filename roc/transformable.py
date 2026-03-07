from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .graphdb import Node

if TYPE_CHECKING:
    from .sequencer import Frame


class Transformable(ABC):
    @abstractmethod
    def same_transform_type(self, other: Any) -> bool:
        """Indicates if two things are the same. e.g. Object(type="foo") and Object(type="bar")
        are the same Python class but different instance types
        """

    @abstractmethod
    def compatible_transform(self, t: Transform) -> bool:
        """Indicates if the specified transform can be applied to this transformable type"""

    @abstractmethod
    def create_transform(self, other: Any) -> Transform | None:
        """Generates a transform for two of the same types for how to turn the
        values of one into the values of the other. Returns None if the
        transform values are the same.
        """

    @abstractmethod
    def apply_transform(self, t: Transform) -> Node:
        """Applys a transform to a thing to convert it to the new thing"""


class Transform(Node):
    @property
    def src_frame(self) -> Frame:
        edges = self.dst_edges.select(type="Change")
        assert len(edges) == 1
        n = edges[0].src
        assert "Frame" in n.labels
        return n  # type: ignore

    @property
    def dst_frame(self) -> Frame:
        edges = self.src_edges.select(type="Change")
        assert len(edges) == 1
        n = edges[0].dst
        assert "Frame" in n.labels
        return n  # type: ignore
