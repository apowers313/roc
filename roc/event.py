"""This module defines all the communications and eventing interfaces for the system."""

from __future__ import annotations

import multiprocessing
from abc import ABC
from collections import deque
from typing import Any, Callable

import reactivex as rx
from loguru import logger
from reactivex import Observable
from reactivex import operators as op
from reactivex.abc.disposable import DisposableBase as Disposable
from reactivex.scheduler import ThreadPoolScheduler
from rich.pretty import pretty_repr

from .component import Component, ComponentId
from .reporting.observability import Observability

event_counter = Observability.meter.create_counter(
    "roc.event", unit="event", description="total number of events"
)

thread_count = multiprocessing.cpu_count() * 2
pool_scheduler = ThreadPoolScheduler(thread_count)


class Event[EventData](ABC):
    """An abstract event class for sending messages between Components over an EventBus.

    Generic over EventData, the type of data carried by the event.
    """

    #: Per-step event counts by bus name, reset after each emission.
    _step_counts: dict[str, int] = {}

    def __init__(self, data: EventData, src_id: ComponentId, bus: EventBus[EventData]):
        """The initializer for the Event

        Args:
            data (EventData): The data for this event
            src_id (ComponentId): The name and type of the Component sending the event
            bus (EventBus): The EventBus that the event is being sent over
        """
        event_counter.add(1, attributes={"source": src_id, "bus": bus.name})
        Event._step_counts[bus.name] = Event._step_counts.get(bus.name, 0) + 1
        self.data = data
        self.src_id = src_id
        self.bus = bus

    @staticmethod
    def get_step_counts() -> dict[str, int]:
        """Return per-bus event counts since last reset and clear them."""
        counts = dict(Event._step_counts)
        Event._step_counts.clear()
        return counts

    def __repr__(self) -> str:
        data_str = pretty_repr(
            self.data,
            max_length=5,  # Maximum length of containers before abbreviating
            max_string=60,  # Maximum length of string before truncating
            expand_all=False,  # Expand all containers regardless of available width
            max_width=120,
        )
        if "\n" in data_str:
            data_str = "\n" + data_str
        return f"[EVENT: {self.src_id} >>> {self.bus.name}]: {data_str}"


type EventFilter[EventData] = Callable[[Event[EventData]], bool]
type EventListener[EventData] = Callable[[Event[EventData]], None]


class BusConnection[EventData]:
    """A connection between an EventBus and a Component, used to send Events.

    Generic over EventData, the type of data sent over this connection.
    """

    def __init__(self, bus: EventBus[EventData], component: Component):
        logger.debug(f"{component.name}:{component.type} attaching to bus {bus.name}")
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
        e = Event[EventData](data, self.attached_component.id, self.attached_bus)
        logger.trace(lambda: f">>> Sending {e}")
        self.attached_bus.subject.on_next(e)

    def listen(
        self,
        listener: EventListener[EventData],
        *,
        filter: EventFilter[EventData] | None = None,
    ) -> None:
        """Subscribes a listener to events on this bus connection.

        Args:
            listener: Callback invoked for each matching event.
            filter: Optional additional filter beyond the component's event_filter.
        """
        pipe_args: list[Callable[[Any], Observable[Event[EventData]]]] = [
            op.filter(self.attached_component.event_filter),
        ]
        if filter is not None:
            pipe_args.append(op.filter(filter))

        sub = self.subject.pipe(*pipe_args).subscribe(listener, scheduler=pool_scheduler)
        self.subscribers.append(sub)

    def close(self) -> None:
        """Disposes all subscriptions on this bus connection."""
        logger.debug(
            f"Closing connection from component {self.attached_component.id}  -> {self.attached_bus.name} bus"
        )

        for sub in self.subscribers:
            sub.dispose()


eventbus_names: set[str] = set()


class EventBus[EventData]:
    """A communication channel for sending events between Components.

    Generic over EventData, the type of data allowed to be sent over the bus.
    """

    name: str
    """The name of the bus. Used to ensure uniqueness."""
    subject: rx.Subject[Event[EventData]]
    """The RxPy Subject that the bus uses to communicate."""

    def __init__(self, name: str, cache_depth: int = 0) -> None:
        if name in eventbus_names:
            raise ValueError(f"Duplicate EventBus name: {name}")
        self.name = name
        eventbus_names.add(name)
        self.subject = rx.Subject[Event[EventData]]()
        self.cache_depth = cache_depth
        self.cache: deque[Event[EventData]] | None = None

        if cache_depth > 0:
            self.cache = deque(maxlen=cache_depth)
            self.subject.subscribe(lambda e: self.cache.append(e))  # type: ignore

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
