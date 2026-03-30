"""Agent internal state (HP, energy, hunger, etc.) with normalization and graph persistence."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from ..framework.component import Component
from ..framework.config import Config, ConfigIntrinsicType
from ..framework.event import Event, EventBus
from ..db.graphdb import Node
from .temporal.transformable import Transform, Transformable

# intrinsic_op_list


class IntrinsicOp[IntrinsicType](ABC):
    """Base class for intrinsic operations that validate and normalize raw game values."""

    intrinsic_type = str()

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name

    def __init_subclass__(cls) -> None:
        if cls.intrinsic_type == IntrinsicOp.intrinsic_type:
            raise TypeError("must set intrinsic_type in subclass of IntrinsicOp")

        if cls.intrinsic_type in intrinsic_op_registry:
            raise TypeError(f"'{intrinsic_op_registry}' is already registered as an intrinsic type")

        intrinsic_op_registry[cls.intrinsic_type] = cls

    @abstractmethod
    def validate(self, val: IntrinsicType) -> bool:
        """Returns True if the value is within the expected range."""
        ...

    @abstractmethod
    def normalize(self, val: IntrinsicType, *, raw_intrinsics: dict[str, Any]) -> float:
        """Converts a raw intrinsic value to a normalized float between 0 and 1."""
        ...


intrinsic_op_registry: dict[str, type[IntrinsicOp[Any]]] = {}


class IntrinsicIntOp(IntrinsicOp[int]):
    """Normalizes integer intrinsics by mapping a min-max range to 0-1."""

    intrinsic_type = "int"

    def __init__(self, name: str, config: tuple[int, int]) -> None:
        super().__init__(name)
        min_val = config[0]
        max_val = config[1]
        self.min = min_val
        self.max = max_val
        self.range = abs(min_val) + abs(max_val)

    def validate(self, val: int) -> bool:
        """Returns True if val is within [min, max]."""
        if (val < self.min) or (val > self.max):
            return False

        return True

    def normalize(self, val: int, **kwargs: Any) -> float:
        """Normalizes val to a 0-1 range based on the configured min and max."""
        return (val + abs(self.min)) / self.range


class IntrinsicPercentOp(IntrinsicOp[int]):
    """Normalizes intrinsics as a percentage of another intrinsic (e.g. hp/hpmax)."""

    intrinsic_type = "percent"

    def __init__(self, name: str, config: str) -> None:
        super().__init__(name)
        self.base = config

    def validate(self, val: int) -> bool:
        """Returns True if val is a positive integer."""
        return isinstance(val, int) and val > 0

    def normalize(self, val: int, *, raw_intrinsics: dict[str, Any]) -> float:
        """Normalizes val as a fraction of the base intrinsic."""
        return float(val / raw_intrinsics[self.base])


class IntrinsicMapOp(IntrinsicOp[int]):
    """Normalizes intrinsics via a lookup table mapping raw values to floats."""

    intrinsic_type = "map"

    def __init__(self, name: str, config: dict[int, float]) -> None:
        super().__init__(name)
        self.map = config

    def validate(self, val: int) -> bool:
        """Returns True if val exists in the lookup map."""
        return val in self.map

    def normalize(self, val: int, **kwargs: Any) -> float:
        """Returns the mapped float for the given raw value."""
        return self.map[val]


class IntrinsicBoolOp(IntrinsicOp[bool]):
    """Normalizes boolean intrinsics to 1.0 (True) or 0.0 (False)."""

    intrinsic_type = "bool"

    def validate(self, val: bool) -> bool:
        """Always returns True since any boolean is valid."""
        return True

    def normalize(self, val: bool, **kwargs: Any) -> float:
        """Returns 1.0 for True, 0.0 for False."""
        if val:
            return 1.0

        return 0.0


class IntrinsicNode(Node, Transformable):
    """A graph node representing one intrinsic value with its raw and normalized forms."""

    name: str
    raw_value: Any
    normalized_value: float

    def __str__(self) -> str:
        return f"IntrinsicNode('{self.name}', {self.raw_value}({self.normalized_value}))"

    def same_transform_type(self, other: Transformable) -> bool:
        """Returns True if other is an IntrinsicNode with the same name."""
        return isinstance(other, IntrinsicNode) and other.name == self.name

    def compatible_transform(self, t: Transform) -> bool:
        """Returns True if the transform is an IntrinsicTransform."""
        return isinstance(t, IntrinsicTransform)

    def create_transform(self, previous: Any) -> Transform | None:
        """Creates an IntrinsicTransform representing the change from the previous value, or None if unchanged."""
        if math.isclose(self.normalized_value, previous.normalized_value):
            return None

        return IntrinsicTransform(
            name=self.name,
            normalized_change=self.normalized_value - previous.normalized_value,
        )

    def apply_transform(self, t: Transform) -> IntrinsicNode:
        """Creates a new IntrinsicNode by applying the normalized change from the transform."""
        assert isinstance(t, IntrinsicTransform)
        new_val = self.normalized_value + t.normalized_change
        return IntrinsicNode(name=self.name, raw_value=None, normalized_value=new_val)


class IntrinsicTransform(Transform):
    """A transform representing the change in a single intrinsic's normalized value."""

    name: str
    normalized_change: float

    def __str__(self) -> str:
        return f"IntrinsicTransform('{self.name}', {self.normalized_change})"


class IntrinsicData:
    """Holds raw intrinsic values and their normalized forms for one game step."""

    def __init__(self, received_intrinsics: dict[str, Any]) -> None:
        self.intrinsics = received_intrinsics
        normalized_intrinsics: dict[str, float] = {}

        for spec in Intrinsic.intrinsic_spec.values():
            name = spec.name
            if name in received_intrinsics:
                spec.validate(received_intrinsics[name])
                normalized_intrinsics[name] = spec.normalize(
                    received_intrinsics[name],
                    raw_intrinsics=received_intrinsics,
                )

        self.normalized_intrinsics = normalized_intrinsics

    def __repr__(self) -> str:
        ret = ""
        for k in self.intrinsics.keys():
            ret += f"{k}: {self.intrinsics[k]}\n"

        return ret

    def to_nodes(self) -> list[IntrinsicNode]:
        """Converts the normalized intrinsics into a list of IntrinsicNode graph nodes."""
        ret: list[IntrinsicNode] = []

        node_intrinsics = [
            k
            for k, v in self.normalized_intrinsics.items()
            if self.normalized_intrinsics[k] != math.nan
        ]

        for key in node_intrinsics:
            n = IntrinsicNode(
                name=key,
                raw_value=self.intrinsics[key],
                normalized_value=(
                    self.normalized_intrinsics[key]
                    if key in self.normalized_intrinsics
                    else math.nan
                ),
            )
            ret.append(n)

        return ret


IntrinsicEvent = Event[IntrinsicData]


class Intrinsic(Component):
    """Component that receives raw game stats and publishes normalized intrinsic values."""

    name: str = "intrinsic"
    type: str = "intrinsic"
    auto: bool = True
    bus = EventBus[IntrinsicData]("intrinsic")

    def __new__(cls, *args: Any, **kwargs: Any) -> Intrinsic:
        settings = Config.get()
        cls.intrinsic_spec = _config_intrinsics(settings.intrinsics)

        return super().__new__(cls, *args, **kwargs)

    def __init__(self) -> None:
        super().__init__()
        self.int_conn = self.connect_bus(Intrinsic.bus)


def _config_intrinsics(intrinsics_desc: list[ConfigIntrinsicType]) -> dict[str, IntrinsicOp[Any]]:
    """Builds a dict of IntrinsicOp instances from the config intrinsic definitions."""
    ret: dict[str, IntrinsicOp[Any]] = {}

    for known_intrinsic in intrinsics_desc:
        name = known_intrinsic.name
        intrinsic_type = known_intrinsic.type
        config = known_intrinsic.model_dump(exclude={"name", "type"})

        cls = intrinsic_op_registry[intrinsic_type]

        ret[name] = cls(name, **config)

    return ret
