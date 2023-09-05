from pydantic import BaseModel

from .component import Component, register_component
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
        super().__init__()
        self.pb_conn = perception_bus.connect(self)
        self.pb_conn.subject.subscribe(self.do_perception)

    def do_perception(self, e: PerceptionEvent) -> None:
        lambda e: logger.info(f"Perception got {e}")

    def shutdown(self) -> None:
        super().shutdown()
        self.pb_conn.subject.on_completed()


@register_component("delta", "perception")
class Delta(Perception):
    def do_perception(self, e: PerceptionEvent) -> None:
        pass
