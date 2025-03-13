from typing import Any

from pydantic import Field

from roc.action import Action, TakeAction
from roc.component import Component, register_component
from roc.event import Event
from roc.graphdb import Edge, EdgeConnectionsList, Node
from roc.intrinsic import Intrinsic, IntrinsicData
from roc.object import Object, ObjectResolver

tick = 0


def get_next_tick() -> int:
    global tick

    tick += 1

    return tick


class Frame(Node):
    tick: int = Field(default_factory=get_next_tick)


class FrameAttributes(Edge):
    allowed_connections: EdgeConnectionsList = [
        ("Frame", "Object"),
        ("Frame", "TakeAction"),
        ("Frame", "IntrinsicNode"),
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
        self.intrinsic_conn = self.connect_bus(Intrinsic.bus)
        self.intrinsic_conn.listen(self.event_collector)
        self.last_frame: Frame | None = None
        self.current_frame: Frame = Frame()

    def event_filter(self, e: Event[Any]) -> bool:
        return (
            isinstance(e.data, Object)
            or isinstance(e.data, TakeAction)
            or isinstance(e.data, IntrinsicData)
        )

    def event_collector(self, e: Event[Any]) -> None:
        # start new frame on action
        if isinstance(e.data, TakeAction):
            if self.last_frame is not None:
                NextFrame.connect(self.last_frame, self.current_frame)

            self.last_frame = self.current_frame
            self.current_frame = Frame()
            FrameAttributes.connect(self.current_frame, e.data)
        elif isinstance(e.data, Object):
            FrameAttributes.connect(self.current_frame, e.data)
        elif isinstance(e.data, IntrinsicData):
            for intrinsic_node in e.data.to_nodes():
                FrameAttributes.connect(self.current_frame, intrinsic_node)
