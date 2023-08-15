from __future__ import annotations

from typing import Generic, TypeVar

from abc import ABC

import reactivex as rx
from loguru import logger

from roc.component import Component

EventData = TypeVar("EventData")


class Event(ABC, Generic[EventData]):
    """An abstract event class for sending messages between Components over an EventBus

    Args:
        ABC (ABC): Abstract base class
        Generic (EventData): The data to be carried by the event
    """

    def __init__(self, data: EventData, src: Component):
        """The initializer for the Event

        Args:
            data (EventData): The data for this event
            src (Component): The Component sending the event
        """
        self.data = data
        self.src = src


class BusConnection(Generic[EventData]):
    """A connection between an EventBus and a Component, used to send Events

    Args:
        Generic (EventData): The data type that will be sent over this connection
    """

    def __init__(self, bus: EventBus[EventData], component: Component):
        self.attached_bus = bus
        self.attached_component = component
        pass

    def send(self, data: EventData) -> None:
        """Send data over the EventBus. Internally, the data is converted to an Event
        with the relevant data (such as the source Component).

        Args:
            data (EventData): The data type of the data to be sent
        """
        e = Event[EventData](data, self.attached_component)
        logger.trace("Sending event:", e)
        self.attached_bus.subject.on_next(e)


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
