from roc.component import Component
from roc.event import Event, EventBus


class PerceptionData:
    pass


class PerceptionEvent(Event[PerceptionData]):
    pass


perception_bus = EventBus[PerceptionData]("perception")


class PerceptionComponent(Component):
    def __init__(self) -> None:
        self.pb_conn = perception_bus.connect(self)