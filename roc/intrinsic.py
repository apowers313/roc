from typing import Callable

from pydantic import BaseModel

from .component import Component, register_component
from .event import Event


class IntrinsicData(BaseModel):
    pass


IntrinsicEvent = Event[IntrinsicData]


@register_component("intrinsic", "intrinsic")
class Intrinsic(Component):
    pass


IntrinsicFn = Callable[[IntrinsicEvent], None]
intrinsic_registry: dict[str, IntrinsicFn] = {}


class register_intrinsic:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, fn: IntrinsicFn) -> IntrinsicFn:
        if self.name in intrinsic_registry:
            raise ValueError(f"Registering duplicate intrinsic '{self.name}'")

        intrinsic_registry[self.name] = fn

        return fn


@register_intrinsic("no-op")
def noop_intrinsic(e: IntrinsicEvent) -> None:
    pass
