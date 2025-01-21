from typing import Any

from pydantic import Field

from roc.action import Action, TakeAction
from roc.component import Component, register_component
from roc.event import Event
from roc.graphdb import Edge, EdgeConnectionsList, Node
from roc.object import FeatureGroup, ObjectResolver

tick = 0


def get_next_tick() -> int:
    tick += 1
    return tick


class Frame(Node):
    tick: int = Field(default_factory=get_next_tick)


class FrameAttributes(Edge):
    allowed_connections: EdgeConnectionsList = [
        ("Frame", "FeatureGroup"),
        ("Frame", "TakeAction"),
    ]


class NextFrame(Edge):
    allowed_connections: EdgeConnectionsList = [("Frame", "Frame")]


@register_component("sequencer", "sequencer", auto=True)
class Sequencer(Component):
    def __init__(self) -> None:
        super().__init__()
        self.obj_res_conn = self.connect_bus(ObjectResolver.bus)
        self.obj_res_conn.listen(self.event_collector)
        self.action_conn = self.connect_bus(Action.bus)
        self.action_conn.listen(self.event_collector)
        self.last_frame: Frame | None = None
        # TODO: listen to intrinsics bus

    def event_filter(self, e: Event[Any]) -> bool:
        return isinstance(e, FeatureGroup) or isinstance(e, TakeAction)

    def event_collector(self, e: Event[Any]) -> None:
        this_frame = Frame()
        if self.last_frame is not None:
            NextFrame.connect(self.last_frame, this_frame)

        self.last_frame = this_frame
