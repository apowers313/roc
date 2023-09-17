"""This module defines the Component base class, which is instantiated for
nearly every part of the system. It implements interfaces for communications,
initialization, shutdown, etc."""

from __future__ import annotations

# import traceback
from abc import ABC
from typing import TYPE_CHECKING, Any, TypeVar, cast

from typing_extensions import Self

from .config import Config
from .logger import logger

if TYPE_CHECKING:
    from .event import BusConnection, Event, EventBus

loaded_components: dict[str, Component] = {}
component_count = 0

T = TypeVar("T")


class Component(ABC):
    """An abstract component class for building pieces of ROC that will talk to each other."""

    name: str = "<name unassigned>"
    type: str = "<type unassigned>"

    def __init__(self) -> None:
        global component_count
        component_count = component_count + 1
        self.bus_conns: dict[str, BusConnection[Any]] = {}
        # print("\n\n++ incrementing component count:", self.name, self.type, self)
        # traceback.print_stack()

    def __del__(self) -> None:
        global component_count
        component_count = component_count - 1
        # print("\n\n-- decrementing component count", self.name, self.type, self)

    def connect_bus(self, bus: EventBus[T]) -> BusConnection[T]:
        if bus.name in self.bus_conns:
            raise ValueError(
                f"Component '{self.name}' attempting duplicate connection to bus '{bus.name}'"
            )

        conn = bus.connect(self)
        self.bus_conns[bus.name] = conn
        return conn

    def event_filter(self, e: Event[Any]) -> bool:
        return e.src is not self

    def shutdown(self) -> None:
        for conn in self.bus_conns:
            for obs in self.bus_conns[conn].attached_bus.subject.observers:
                obs.on_completed()

    @staticmethod
    def init() -> None:
        settings = Config.get()
        component_list = default_components
        component_list = component_list.union(settings.perception_components)

        # TODO: shutdown previously loaded components

        for reg_str in component_list:
            logger.trace(f"Loading component: {reg_str} ...")
            (name, type) = reg_str.split(":")
            loaded_components[reg_str] = Component.get(name, type)

    @classmethod
    def get(cls, name: str, type: str, *args: Any, **kwargs: Any) -> Self:
        reg_str = _component_registry_key(name, type)
        return cast(Self, component_registry[reg_str](*args, **kwargs))

    @staticmethod
    def get_component_count() -> int:
        global component_count
        return component_count

    @staticmethod
    def deregister(name: str, type: str) -> None:
        reg_str = _component_registry_key(name, type)
        del component_registry[reg_str]

    @staticmethod
    def reset() -> None:
        # shutdown all components
        global loaded_components
        for name in loaded_components:
            logger.trace(f"Shutting down component: {name}.")
            c = loaded_components[name]
            c.shutdown()

        loaded_components.clear()


WrappedComponentBase = TypeVar("WrappedComponentBase", bound=Component)
component_registry: dict[str, type[Component]] = {}
default_components: set[str] = set()


def _component_registry_key(name: str, type: str) -> str:
    return f"{name}:{type}"


class register_component:
    def __init__(self, name: str, type: str, *, auto: bool = False) -> None:
        self.name = name
        self.type = type
        self.auto = auto

    def __call__(self, cls: type[Component]) -> type[Component]:
        global register_component
        global component_registry

        reg_str = _component_registry_key(self.name, self.type)
        if reg_str in component_registry:
            raise ValueError(f"Registering duplicate component name: '{self.name}'")

        if self.auto:
            global default_components
            default_components.add(reg_str)

        component_registry[reg_str] = cls
        cls.name = self.name
        cls.type = self.type

        return cls
