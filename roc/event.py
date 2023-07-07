from __future__ import annotations

from typing import Any, Generic, TypeVar

from abc import ABC, abstractmethod

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


class EventBus(Generic[EventData]):
    def __init__(self):
        self.subject = rx.Subject[Event[EventData]]()

    def connect(self, component: Component) -> BusConnection[EventData]:
        return BusConnection[EventData](self, component)
