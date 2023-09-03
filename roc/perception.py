from pydantic import BaseModel

from .component import Component
from .event import Event, EventBus
from .logger import logger


# TODO: vision input
# TODO: sound input
# TODO: other input
class VisionData(BaseModel):
    spectrum: tuple[tuple[tuple[int | str, ...], ...], ...]
    # spectrum: tuple[int | str, ...]


PerceptionData = VisionData

PerceptionEvent = Event[PerceptionData]

perception_bus = EventBus[PerceptionData]("perception")


class Perception(Component):
    def __init__(self) -> None:
        self.pb_conn = perception_bus.connect(self)
        self.pb_conn.subject.subscribe(self.do_perception)

    def do_perception(self, e: PerceptionEvent) -> None:
        lambda e: logger.info(f"Perception got {e}")
