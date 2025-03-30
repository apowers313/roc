from __future__ import annotations

from typing import Any

from pydantic import Field

from roc.action import Action, TakeAction
from roc.component import Component
from roc.event import Event, EventBus
from roc.graphdb import Edge, EdgeConnectionsList, Node
from roc.intrinsic import Intrinsic, IntrinsicData
from roc.object import Object, ObjectResolver, ResolvedObject
from roc.transformable import Transform, Transformable

tick = 0
PREDICTED_FRAME_TICK = -1


def get_next_tick() -> int:
    global tick

    tick += 1

    return tick


class Frame(Node):
    tick: int = Field(default_factory=get_next_tick)

    @property
    def transforms(self) -> list[Transform]:
        ret: list[Transform] = []

        changes = self.src_edges.select(type="Change")
        if len(changes) == 0:
            return ret
        assert len(changes) == 1
        transform_node = changes[0].dst

        for n in self.successors:
            if isinstance(n, Transform):
                change_edges = n.src_edges.select(type="Change")
                for e in change_edges:
                    if isinstance(e.dst, Transform):
                        ret.append(e.dst)

        return ret

    @property
    def transformable(self) -> list[Transformable]:
        ret: list[Transformable] = []

        for n in self.successors:
            if isinstance(n, Transformable):
                ret.append(n)

        return ret

    @staticmethod
    def merge_transforms(src: Frame, mod: Frame) -> Frame:
        ret = Frame(tick=PREDICTED_FRAME_TICK)

        for st in src.transformable:
            for mt in mod.transforms:
                if st.compatible_transform(mt):
                    t = st.apply_transform(mt)
                    FrameAttribute.connect(ret, t)

        return ret

    @property
    def objects(self) -> list[Object]:
        ret: list[Object] = []

        for e in self.src_edges:
            if isinstance(e.dst, Object):
                ret.append(e.dst)

        return ret


class FrameAttribute(Edge):
    allowed_connections: EdgeConnectionsList = [
        ("Frame", "FeatureGroup"),
        ("Frame", "TakeAction"),
        ("TakeAction", "Frame"),
        ("Frame", "IntrinsicNode"),
    ]


class NextFrame(Edge):
    allowed_connections: EdgeConnectionsList = [("Frame", "Frame")]


class Sequencer(Component):
    name: str = "sequencer"
    type: str = "sequencer"
    auto: bool = True
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
            isinstance(e.data, ResolvedObject)
            or isinstance(e.data, TakeAction)
            or isinstance(e.data, IntrinsicData)
        )

    def handle_object_resolution_event(self, e: Event[ResolvedObject]) -> None:
        FrameAttribute.connect(self.current_frame, e.data.feature_group)

    def handle_intrinsic_event(self, e: Event[IntrinsicData]) -> None:
        for intrinsic_node in e.data.to_nodes():
            FrameAttribute.connect(self.current_frame, intrinsic_node)

    def handle_action_event(self, e: Event[Any]) -> None:
        # start new frame on action
        if isinstance(e.data, TakeAction):
            # connect action data to the old frame
            FrameAttribute.connect(self.current_frame, e.data)

            # create a new frame and connect the previous frame to the current frame
            self.last_frame = self.current_frame
            self.current_frame = Frame()
            NextFrame.connect(self.last_frame, self.current_frame)

            # connect action data to the new frame
            FrameAttribute.connect(e.data, self.current_frame)

            # emit the frame on the bus
            self.sequencer_conn.send(self.last_frame)
