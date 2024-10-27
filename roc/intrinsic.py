from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .component import Component, register_component
from .event import Event, EventBus

# intrinsic_op_list

IntrinsicType = TypeVar("IntrinsicType")


class IntrinsicOp(ABC, Generic[IntrinsicType]):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def validate(self, val: IntrinsicType) -> bool: ...

    @abstractmethod
    def normalize(self, val: IntrinsicType) -> float: ...

    # register

    # deregister

    # clear


class IntrinsicIntOp(IntrinsicOp[int]):
    def __init__(self, name: str, min: int, max: int) -> None:
        super().__init__(name)
        self.min = min
        self.max = max
        self.range = abs(min) + abs(max)

    def validate(self, val: int) -> bool:
        if (val < self.min) or (val > self.max):
            return False

        return True

    def normalize(self, val: int) -> float:
        return (val + abs(self.min)) / self.range


class IntrinsicBoolOp(IntrinsicOp[bool]):
    def validate(self, val: bool) -> bool:
        return True

    def normalize(self, val: bool) -> float:
        if val:
            return 1.0

        return 0.0


IntrinsicData = dict[str, Any]
IntrinsicEvent = Event[IntrinsicData]


@register_component("intrinsic", "intrinsic", auto=True)
class Intrinsic(Component):
    bus = EventBus[IntrinsicData]("intrinsic")

    def __init__(self) -> None:
        super().__init__()
        self.int_conn = self.connect_bus(Intrinsic.bus)
        self.int_conn.listen(self.do_intrinsic)

    def event_filter(self, e: IntrinsicEvent) -> bool:
        print("Intrinsic event filter")
        return True

    def do_intrinsic(self, e: IntrinsicEvent) -> None:
        return None
