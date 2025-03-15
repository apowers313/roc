from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar

from .component import Component, register_component
from .config import Config
from .event import Event, EventBus
from .graphdb import Node

# intrinsic_op_list

IntrinsicType = TypeVar("IntrinsicType")


class IntrinsicOp(ABC, Generic[IntrinsicType]):
    intrinsic_type = "unknown"

    def __init__(self, name: str) -> None:
        self.name = name

    def __init_subclass__(cls) -> None:
        if cls.intrinsic_type == "unknown":
            raise TypeError("must set intrinsic_type in subclass of IntrinsicOp")

        if cls.intrinsic_type in intrinsic_op_registry:
            raise TypeError(f"'{intrinsic_op_registry}' is already registered as an intrinsic type")

        intrinsic_op_registry[cls.intrinsic_type] = cls

    @staticmethod
    def convert_args(args: Iterable[str]) -> Iterable[Any]:
        return args

    @abstractmethod
    def validate(self, val: IntrinsicType) -> bool: ...

    @abstractmethod
    def normalize(self, val: IntrinsicType) -> float: ...

    # register

    # deregister

    # clear


intrinsic_op_registry: dict[str, type[IntrinsicOp[Any]]] = {}


class IntrinsicIntOp(IntrinsicOp[int]):
    intrinsic_type = "int"

    def __init__(self, name: str, min: int, max: int) -> None:
        super().__init__(name)
        self.min = min
        self.max = max
        self.range = abs(min) + abs(max)

    @staticmethod
    def convert_args(args: Iterable[Any]) -> list[int]:
        ret = []
        for arg in args:
            ret.append(int(arg))
        return ret

    def validate(self, val: int) -> bool:
        if (val < self.min) or (val > self.max):
            return False

        return True

    def normalize(self, val: int) -> float:
        return (val + abs(self.min)) / self.range


class IntrinsicBoolOp(IntrinsicOp[bool]):
    intrinsic_type = "bool"

    def validate(self, val: bool) -> bool:
        return True

    def normalize(self, val: bool) -> float:
        if val:
            return 1.0

        return 0.0


class IntrinsicNode(Node):
    name: str
    raw_value: Any
    normalized_value: float


class IntrinsicData:
    def __init__(self, received_intrinsics: dict[str, Any]) -> None:
        self.intrinsics = received_intrinsics
        normalized_intrinsics: dict[str, float] = {}

        for spec in Intrinsic.intrinsic_spec.values():
            name = spec.name
            if name in received_intrinsics:
                spec.validate(received_intrinsics[name])
                normalized_intrinsics[name] = spec.normalize(received_intrinsics[name])

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


@register_component("intrinsic", "intrinsic", auto=True)
class Intrinsic(Component):
    bus = EventBus[IntrinsicData]("intrinsic")

    def __new__(cls, *args: Any, **kwargs: Any) -> Intrinsic:
        settings = Config.get()
        cls.intrinsic_spec = config_intrinsics(settings.intrinsics)

        return super().__new__(cls, *args, **kwargs)

    def __init__(self) -> None:
        super().__init__()
        self.int_conn = self.connect_bus(Intrinsic.bus)


def config_intrinsics(intrinsics_desc: Iterable[tuple[str, str]]) -> dict[str, IntrinsicOp[Any]]:
    ret: dict[str, IntrinsicOp[Any]] = {}

    for known_intrinsic in intrinsics_desc:
        name, desc = known_intrinsic
        args = desc.split(":")
        intrinsic_type = args.pop(0)

        cls = intrinsic_op_registry[intrinsic_type]

        ret[name] = cls(name, *cls.convert_args(args))

    return ret
