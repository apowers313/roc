"""This module defines the Component base class, which is instantiated for
nearly every part of the system. It implements interfaces for communications,
initialization, shutdown, etc.
"""

from __future__ import annotations

# import traceback
from abc import ABC
from typing import TYPE_CHECKING, Any, TypeVar, cast
from weakref import WeakSet

from typing_extensions import Self

from .config import Config
from .logger import logger

if TYPE_CHECKING:
    from .event import BusConnection, Event, EventBus

loaded_components: dict[str, Component] = {}
component_set: WeakSet[Component] = WeakSet()

T = TypeVar("T")


class Component(ABC):
    """An abstract component class for building pieces of ROC that will talk to each other."""

    name: str = "<name unassigned>"
    type: str = "<type unassigned>"

    def __init__(self) -> None:
        global component_set
        component_set.add(self)
        self.bus_conns: dict[str, BusConnection[Any]] = {}
        logger.trace(f"++ incrementing component count: {self.name}:{self.type} {self}")
        # traceback.print_stack()

    def __del__(self) -> None:
        global component_set
        component_set.add(self)
        logger.trace(f"-- decrementing component count: {self.name}:{self.type} {self}")

    def connect_bus(self, bus: EventBus[T]) -> BusConnection[T]:
        """Create a new bus connection for the component, storing the result for
        later shutdown.

        Args:
            bus (EventBus[T]): The event bus to attach to

        Raises:
            ValueError: if the bus has already been connected to by this component

        Returns:
            BusConnection[T]: The bus connection for listening or sending events
        """
        if bus.name in self.bus_conns:
            raise ValueError(
                f"Component '{self.name}' attempting duplicate connection to bus '{bus.name}'"
            )

        conn = bus.connect(self)
        self.bus_conns[bus.name] = conn
        return conn

    def event_filter(self, e: Event[Any]) -> bool:
        """A filter for any incoming events. By default it filters out events
        sent by itself, but it is especially useful for creating new filters in
        sub-classes.

        Args:
            e (Event[Any]): The event to be evaluated

        Returns:
            bool: True if the event should be sent, False if it should be dropped
        """
        return e.src is not self

    def shutdown(self) -> None:
        """De-initializes the component, removing any bus connections and any
        other clean-up that needs to be performed
        """
        logger.debug(f"Component {self.name}:{self.type} shutting down.")

        for conn in self.bus_conns:
            for obs in self.bus_conns[conn].attached_bus.subject.observers:
                obs.on_completed()

    @staticmethod
    def init() -> None:
        """Loads all components registered as `auto` and perception components
        in the `perception_components` config field.
        """
        settings = Config.get()
        component_list = default_components
        component_list = component_list.union(settings.perception_components)
        logger.debug(f"Component.init: default components: {component_list}")

        # TODO: shutdown previously loaded components

        for reg_str in component_list:
            logger.trace(f"Loading component: {reg_str} ...")
            (name, type) = reg_str.split(":")
            loaded_components[reg_str] = Component.get(name, type)

    @classmethod
    def get(cls, name: str, type: str, *args: Any, **kwargs: Any) -> Self:
        """Retreives a component with the specified name from the registry and
        creates a new version of it with the specified args. Used by
        `Config.init` and for testing.

        Args:
            name (str): The name of the component to get, as specified during
                its registration
            type (str): The type of the component to get, as specified during
                its registration
            args (Any): Fixed position arguments to pass to the Component
                constructor
            kwargs (Any): Keyword args to pass to the Component constructor

        Returns:
            Self: the component that was created, casted as the calling class.
            (e.g. `Perception.get(...)` will return a Perception component and
            `Action.get(...)` will return an Action component)
        """
        reg_str = _component_registry_key(name, type)
        return cast(Self, component_registry[reg_str](*args, **kwargs))

    @staticmethod
    def get_component_count() -> int:
        """Returns the number of currently created Components. The number goes
        up on __init__ and down on __del__. Primarily used for testing to ensure
        Components are being shutdown appropriately.

        Returns:
            int: The number of currently active Component instances
        """
        # global component_count
        # return component_count
        global component_set
        return len(component_set)

    @staticmethod
    def deregister(name: str, type: str) -> None:
        """Removes a component from the Component registry. Primarlly used for testing.

        Args:
            name (str): The name of the Component to deregister
            type (str): The type of the Component to deregister
        """
        reg_str = _component_registry_key(name, type)
        del component_registry[reg_str]

    @staticmethod
    def reset() -> None:
        """Shuts down all components"""
        # shutdown all components
        global loaded_components
        for name in loaded_components:
            logger.trace(f"Shutting down component: {name}.")
            c = loaded_components[name]
            c.shutdown()

        loaded_components.clear()

        global component_set
        for c in component_set:
            c.shutdown()


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

        logger.trace(f"Registering component: {self.name}:{self.type} (auto={self.auto})")

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
