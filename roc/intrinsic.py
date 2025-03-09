from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar

from .component import Component, register_component
from .config import Config
from .event import Event, EventBus

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


class IntrinsicData:
    def __init__(self, intrinsics: dict[str, Any]) -> None:
        self.intrinsics = intrinsics

    def __repr__(self) -> str:
        ret = ""
        for k in self.intrinsics.keys():
            ret += f"{k}: {self.intrinsics[k]}\n"

        return ret


IntrinsicEvent = Event[IntrinsicData]


@register_component("intrinsic", "intrinsic", auto=True)
class Intrinsic(Component):
    bus = EventBus[IntrinsicData]("intrinsic")

    def __init__(self) -> None:
        super().__init__()
        self.int_conn = self.connect_bus(Intrinsic.bus)
        self.int_conn.listen(self.do_intrinsic)

        settings = Config.get()
        self.intrinsic_spec = config_intrinsics(settings.intrinsics)

    def event_filter(self, e: Event[Any]) -> bool:
        return isinstance(e.data, IntrinsicData)

    def do_intrinsic(self, e: IntrinsicEvent) -> None:
        normalized_intrinsics: dict[str, float] = {}

        for spec in self.intrinsic_spec.values():
            name = spec.name
            received_intrinsics = e.data.intrinsics
            if name in received_intrinsics:
                spec.validate(received_intrinsics[name])
                normalized_intrinsics[name] = spec.normalize(received_intrinsics[name])

        print("normalized_intrinsics", normalized_intrinsics)
        # TODO: emit normalized_intrinsics


def config_intrinsics(intrinsics_desc: Iterable[tuple[str, str]]) -> dict[str, IntrinsicOp[Any]]:
    ret: dict[str, IntrinsicOp[Any]] = {}

    for known_intrinsic in intrinsics_desc:
        name, desc = known_intrinsic
        args = desc.split(":")
        intrinsic_type = args.pop(0)

        cls = intrinsic_op_registry[intrinsic_type]

        ret[name] = cls(name, *cls.convert_args(args))

    return ret
