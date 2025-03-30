from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .component import Component
from .config import Config, ConfigIntrinsicType
from .event import Event, EventBus
from .graphdb import Node
from .transformable import Transform, Transformable

# intrinsic_op_list

IntrinsicType = TypeVar("IntrinsicType")


class IntrinsicOp(ABC, Generic[IntrinsicType]):
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
    def validate(self, val: IntrinsicType) -> bool: ...

    @abstractmethod
    def normalize(self, val: IntrinsicType, *, raw_intrinsics: dict[str, Any]) -> float: ...


intrinsic_op_registry: dict[str, type[IntrinsicOp[Any]]] = {}


class IntrinsicIntOp(IntrinsicOp[int]):
    intrinsic_type = "int"

    def __init__(self, name: str, config: tuple[int, int]) -> None:
        super().__init__(name)
        min = config[0]
        max = config[1]
        self.min = min
        self.max = max
        self.range = abs(min) + abs(max)

    def validate(self, val: int) -> bool:
        if (val < self.min) or (val > self.max):
            return False

        return True

    def normalize(self, val: int, **kwargs: Any) -> float:
        return (val + abs(self.min)) / self.range


class IntrinsicPercentOp(IntrinsicOp[int]):
    intrinsic_type = "percent"

    def __init__(self, name: str, config: str) -> None:
        super().__init__(name)
        self.base = config

    def validate(self, val: int) -> bool:
        return isinstance(val, int) and val > 0

    def normalize(self, val: int, raw_intrinsics: dict[str, Any]) -> float:
        return float(val / raw_intrinsics[self.base])


class IntrinsicMapOp(IntrinsicOp[int]):
    intrinsic_type = "map"

    def __init__(self, name: str, config: dict[int, float]) -> None:
        super().__init__(name)
        self.map = config

    def validate(self, val: int) -> bool:
        return val in self.map

    def normalize(self, val: int, **kwargs: Any) -> float:
        return self.map[val]


class IntrinsicBoolOp(IntrinsicOp[bool]):
    intrinsic_type = "bool"

    def validate(self, val: bool) -> bool:
        return True

    def normalize(self, val: bool, **kwargs: Any) -> float:
        if val:
            return 1.0

        return 0.0


class IntrinsicNode(Node, Transformable):
    name: str
    raw_value: Any
    normalized_value: float

    def same_transform_type(self, other: Transformable) -> bool:
        return isinstance(other, IntrinsicNode) and other.name == self.name

    def compatible_transform(self, t: Transform) -> bool:
        return isinstance(t, IntrinsicTransform)

    def create_transform(self, other: Any) -> Transform | None:
        if math.isclose(self.normalized_value, other.normalized_value):
            return None

        # TODO: create transform for raw values using IntrinsicOps?
        return IntrinsicTransform(normalized_change=other.normalized_value - self.normalized_value)
        # normalized_change = 0.4 - 0.5 = -0.1

    def apply_transform(self, t: Transform) -> IntrinsicNode:
        assert isinstance(t, IntrinsicTransform)
        new_val = self.normalized_value + t.normalized_change
        # new_val = 0.5 + -0.1 = 0.4
        return IntrinsicNode(name=self.name, raw_value=None, normalized_value=new_val)


class IntrinsicTransform(Transform):
    normalized_change: float


class IntrinsicData:
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
                normalized_value=self.normalized_intrinsics[key]
                if key in self.normalized_intrinsics
                else math.nan,
            )
            ret.append(n)

        return ret


IntrinsicEvent = Event[IntrinsicData]


class Intrinsic(Component):
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
    ret: dict[str, IntrinsicOp[Any]] = {}

    for known_intrinsic in intrinsics_desc:
        name = known_intrinsic.name
        type = known_intrinsic.type
        config = known_intrinsic.model_dump(exclude={"name", "type"})

        cls = intrinsic_op_registry[type]

        ret[name] = cls(name, **config)

    return ret
