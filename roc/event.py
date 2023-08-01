from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

import reactivex as rx
from loguru import logger

from roc.component import Component

EventData = TypeVar("EventData")


class Event(ABC, Generic[EventData]):
    def __init__(self, data: EventData, src: Component):
        self.data = data
        self.src = src


class BusConnection(Generic[EventData]):
    def __init__(self, bus: EventBus[EventData], component: Component):
        self.attached_bus = bus
        self.attached_component = component
        pass

    def send(self, data: EventData) -> None:
        e = Event[EventData](data, self.attached_component)
        logger.trace("Sending event:", e)
        self.attached_bus.subject.on_next(e)


eventbus_names: set[str] = set()


class EventBus(Generic[EventData]):
    def __init__(self, name: str) -> None:
        if name in eventbus_names:
            raise Exception(f"Duplicate EventBus name: {name}")
        self.name = name
        eventbus_names.add(name)
        self.subject = rx.Subject[Event[EventData]]()

    def connect(self, component: Component) -> BusConnection[EventData]:
        return BusConnection[EventData](self, component)

    @staticmethod
    def clear_names() -> None:
        eventbus_names.clear()
