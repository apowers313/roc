from __future__ import annotations

from abc import ABC
from typing import Callable, Generic, TypeVar

import reactivex as rx
from loguru import logger
from reactivex import Observable
from reactivex import operators as op
from rich.pretty import pretty_repr

from .component import Component

EventData = TypeVar("EventData")


class Event(ABC, Generic[EventData]):
    """An abstract event class for sending messages between Components over an EventBus

    Args:
        ABC (ABC): Abstract base class
        Generic (EventData): The data to be carried by the event
    """

    def __init__(self, data: EventData, src: Component, bus: EventBus[EventData]):
        """The initializer for the Event

        Args:
            data (EventData): The data for this event
            src (Component): The Component sending the event
            bus (EventBus): The EventBus that the event is being sent over
        """
        self.data = data
        self.src = src
        self.bus = bus

    def __repr__(self) -> str:
        data_str = pretty_repr(
            self.data,
            # max_depth=4, # Maximum depth of nested data structure
            max_length=5,  # Maximum length of containers before abbreviating
            max_string=60,  # Maximum length of string before truncating
            expand_all=False,  # Expand all containers regardless of available width
            max_width=120,
        )
        if "\n" in data_str:
            data_str = "\n" + data_str
        return f"[EVENT: {self.src.name} >>> {self.bus.name}]: {data_str}"


class BusConnection(Generic[EventData]):
    """A connection between an EventBus and a Component, used to send Events

    Args:
        Generic (EventData): The data type that will be sent over this connection
    """

    def __init__(self, bus: EventBus[EventData], component: Component):
        self.attached_bus = bus
        self.attached_component = component
        self.subject: rx.Subject[Event[EventData]] = self.attached_bus.subject

    def send(self, data: EventData) -> None:
        """Send data over the EventBus. Internally, the data is converted to an Event
        with the relevant data (such as the source Component).

        Args:
            data (EventData): The data type of the data to be sent
        """
        e = Event[EventData](data, self.attached_component, self.attached_bus)
        logger.trace(f">>> Sending {e}")
        self.attached_bus.subject.on_next(e)

    def listen(
        self,
        listener: Callable[[Event[EventData]], None],
        *,
        filter: Callable[[Event[EventData]], bool] | None = None,
    ) -> None:
        obs: Observable[Event[EventData]] = self.subject
        if filter is not None:
            obs = obs.pipe(op.filter(filter))
        obs.subscribe(listener)

    def close(self) -> None:
        logger.trace(
            f"Closing connection {self.attached_component.name} -> {self.attached_bus.name}"
        )
        self.subject.on_completed()


eventbus_names: set[str] = set()


class EventBus(Generic[EventData]):
    """A communication channel for sending events between Components

    Args:
        Generic (EventData): The data type that is allowed to be sent over the bus
    """

    def __init__(self, name: str) -> None:
        if name in eventbus_names:
            raise Exception(f"Duplicate EventBus name: {name}")
        self.name = name
        eventbus_names.add(name)
        self.subject = rx.Subject[Event[EventData]]()

    def connect(self, component: Component) -> BusConnection[EventData]:
        """Creates a connection between an EventBus and a Component for sending Events

        Args:
            component (Component): The Component to connect to the bus

        Returns:
            BusConnection[EventData]: A new connection that can be used to send data
        """
        return BusConnection[EventData](self, component)

    @staticmethod
    def clear_names() -> None:
        """Clears all EventBusses that have been registered, mostly used for testing."""
        eventbus_names.clear()


# bus_registry: dict[str, EventBus[Any]] = {}
# T = TypeVar("T")
# def create_bus(name: str, data_type: T) -> EventBus[T]:
#     bus = EventBus[T](name)
#     bus_registry[name] = bus
#     return bus
