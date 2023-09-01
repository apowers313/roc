from pydantic import BaseModel

from .component import Component
from .event import EventBus


# TODO: vision input
# TODO: sound input
# TODO: other input
class VisionData(BaseModel):
    spectrum: tuple[tuple[tuple[int | str, ...], ...], ...]
    # spectrum: tuple[int | str, ...]


PerceptionData = VisionData

perception_bus = EventBus[PerceptionData]("perception")


class PerceptionComponent(Component):
    def __init__(self) -> None:
        self.pb_conn = perception_bus.connect(self)
