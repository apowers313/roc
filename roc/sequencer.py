from typing import Any

from pydantic import Field

from roc.action import Action, TakeAction
from roc.component import Component, register_component
from roc.event import Event, EventBus
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
        ("TakeAction", "Frame"),
        ("Frame", "IntrinsicNode"),
    ]


class NextFrame(Edge):
    allowed_connections: EdgeConnectionsList = [("Frame", "Frame")]


@register_component("sequencer", "sequencer", auto=True)
class Sequencer(Component):
    bus = EventBus[Frame]("sequencer")

    def __init__(self) -> None:
        super().__init__()
        self.sequencer_conn = self.connect_bus(Sequencer.bus)
        self.obj_res_conn = self.connect_bus(ObjectResolver.bus)
        self.obj_res_conn.listen(self.handle_object_resolution_event)
        self.action_conn = self.connect_bus(Action.bus)
        self.action_conn.listen(self.handle_action_event)
        self.intrinsic_conn = self.connect_bus(Intrinsic.bus)
        self.intrinsic_conn.listen(self.handle_intrinsic_event)
        self.last_frame: Frame | None = None
        self.current_frame: Frame = Frame()

    def event_filter(self, e: Event[Any]) -> bool:
        return (
            isinstance(e.data, Object)
            or isinstance(e.data, TakeAction)
            or isinstance(e.data, IntrinsicData)
        )

    def handle_object_resolution_event(self, e: Event[Any]) -> None:
        FrameAttributes.connect(self.current_frame, e.data)

    def handle_intrinsic_event(self, e: Event[Any]) -> None:
        for intrinsic_node in e.data.to_nodes():
            FrameAttributes.connect(self.current_frame, intrinsic_node)

    def handle_action_event(self, e: Event[Any]) -> None:
        # start new frame on action
        if isinstance(e.data, TakeAction):
            # connect action data to the old frame
            FrameAttributes.connect(self.current_frame, e.data)

            # create a new frame and connect the previous frame to the current frame
            self.last_frame = self.current_frame
            self.current_frame = Frame()
            NextFrame.connect(self.last_frame, self.current_frame)

            # connect action data to the new frame
            FrameAttributes.connect(e.data, self.current_frame)

            # emit the frame on the bus
            self.sequencer_conn.send(self.last_frame)
