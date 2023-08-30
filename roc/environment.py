from pydantic import BaseModel

from .event import EventBus

# TODO: vision input
# TODO: sound input
# TODO: other input
# class EnvData(BaseModel):
#     pass


class VisionData(BaseModel):
    spectrum: tuple[tuple[tuple[int | str, ...], ...], ...]
    # spectrum: tuple[int | str, ...]


EnvData = VisionData

environment_bus = EventBus[EnvData]("environment")
