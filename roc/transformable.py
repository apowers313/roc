from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .graphdb import Node


class Transformable(ABC):
    @abstractmethod
    def same_transform_type(self, other: Any) -> bool:
        """Says if two things are the same. e.g. Object(type="foo") and Object(type="bar")
        are the same Python class but different instance types
        """

    # def same_transform_value(self, other: Any) -> bool:
    #     """Says if there is any difference between two things. e.g.
    #     Object(type="foo", val=1) and Object(type="foo", val=2) are the same
    #     transform type but have different values.
    #     """

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
    pass
