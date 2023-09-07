from pydantic import BaseModel

from .component import Component, register_component
from .event import Event, EventBus
from .logger import logger


# TODO: vision input
# TODO: sound input
# TODO: other input
class VisionData(BaseModel):
    # spectrum: tuple[tuple[tuple[int | str, ...], ...], ...]
    screen: tuple[tuple[int | str, ...], ...]
    # spectrum: tuple[int | str, ...]


class DeltaData(BaseModel):
    wtf: int


PerceptionData = VisionData | DeltaData

PerceptionEvent = Event[PerceptionData]

perception_bus = EventBus[PerceptionData]("perception")


class Perception(Component):
    def __init__(self) -> None:
        super().__init__()
        self.pb_conn = self.connect_bus(perception_bus)
        self.pb_conn.listen(self.do_perception)

    def do_perception(self, e: PerceptionEvent) -> None:
        lambda e: logger.info(f"Perception got {e}")


@register_component("delta", "perception")
class Delta(Perception):
    def event_filter(self, e: PerceptionEvent) -> bool:
        print(f"DOING EVENT FILTER: {e}", isinstance(e.data, VisionData))
        return isinstance(e.data, VisionData)

    def do_perception(self, e: PerceptionEvent) -> None:
        logger.debug(f"got perception event {e}")
        self.pb_conn.send(DeltaData(wtf=42))
