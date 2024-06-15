"""This module defines all the communications and eventing interfaces for the system."""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Generic, TypeVar

import reactivex as rx
from loguru import logger
from reactivex import Observable
from reactivex import operators as op
from reactivex.abc.disposable import DisposableBase as Disposable
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


EventFilter = Callable[[Event[EventData]], bool]
EventListener = Callable[[Event[EventData]], None]


class BusConnection(Generic[EventData]):
    """A connection between an EventBus and a Component, used to send Events

    Args:
        Generic (EventData): The data type that will be sent over this connection
    """

    def __init__(self, bus: EventBus[EventData], component: Component):
        self.attached_bus = bus
        self.attached_component = component
        self.subject: rx.Subject[Event[EventData]] = self.attached_bus.subject
        self.subscribers: list[Disposable] = []

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
        listener: EventListener[EventData],
        *,
        filter: EventFilter[EventData] | None = None,
    ) -> None:
        pipe_args: list[Callable[[Any], Observable[Event[EventData]]]] = [
            # op.filter(lambda e: e.src is not self.attached_component),
            op.filter(self.attached_component.event_filter),
        ]
        if filter is not None:
            pipe_args.append(op.filter(filter))

        sub = self.subject.pipe(*pipe_args).subscribe(listener)
        self.subscribers.append(sub)

    def close(self) -> None:
        logger.trace(
            f"Closing connection {self.attached_component.name} -> {self.attached_bus.name}"
        )

        for sub in self.subscribers:
            sub.dispose()


eventbus_names: set[str] = set()


class EventBus(Generic[EventData]):
    """A communication channel for sending events between Components

    Args:
        Generic (EventData): The data type that is allowed to be sent over the bus
    """

    name: str
    """The name of the bus. Used to ensure uniqueness."""
    subject: rx.Subject[Event[EventData]]
    """The RxPy Subject that the bus uses to communicate."""

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
