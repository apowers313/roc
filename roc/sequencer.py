"""Assembles game state snapshots (Frames) from perception, intrinsic, and action events."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from roc.action import Action, TakeAction
from roc.component import Component
from roc.event import Event, EventBus
from roc.graphdb import Edge, EdgeConnectionsList, Node
from roc.intrinsic import Intrinsic, IntrinsicData
from roc.object import FeatureGroup, Object, ObjectResolver, ResolvedObject
from roc.transformable import Transform, Transformable

tick = 0
PREDICTED_FRAME_TICK = -1


def get_next_tick() -> int:
    """Returns the next sequential tick number for frame ordering."""
    global tick

    tick += 1

    return tick


class Frame(Node):
    """A snapshot of the game state at one timestep, containing objects, intrinsics, and actions."""

    tick: int = Field(default_factory=get_next_tick)

    @property
    def transforms(self) -> list[Transform]:
        """All transforms associated with this frame's change node."""
        changes = self.src_edges.select(type="Change")
        if len(changes) == 0:
            return []
        assert len(changes) == 1

        return [
            e.dst
            for n in self.successors
            if isinstance(n, Transform)
            for e in n.src_edges.select(type="Change")
            if isinstance(e.dst, Transform)
        ]

    @property
    def transformable(self) -> list[Transformable]:
        """All transformable attributes attached to this frame."""
        return [n for n in self.successors if isinstance(n, Transformable)]

    @staticmethod
    def merge_transforms(src: Frame, mod: Frame) -> Frame:
        """Creates a predicted frame by applying transforms from mod to src's transformables."""
        ret = Frame(tick=PREDICTED_FRAME_TICK)

        for st in src.transformable:
            for mt in mod.transforms:
                if st.compatible_transform(mt):
                    t = st.apply_transform(mt)
                    FrameAttribute.connect(ret, t)

        return ret

    @property
    def objects(self) -> list[Object]:
        """All Objects referenced by this frame's feature groups."""
        ret: list[Object] = []

        for e in self.src_edges:
            if isinstance(e.dst, FeatureGroup):
                feature_group = e.dst
                for fg_edge in feature_group.src_edges:
                    if isinstance(fg_edge.dst, Object):
                        ret.append(fg_edge.dst)

        return ret


class FrameAttribute(Edge):
    """An edge connecting a Frame to its constituent data (features, actions, intrinsics)."""

    allowed_connections: EdgeConnectionsList = [
        ("Frame", "FeatureGroup"),
        ("Frame", "TakeAction"),
        ("TakeAction", "Frame"),
        ("Frame", "IntrinsicNode"),
    ]


class NextFrame(Edge):
    """An edge linking consecutive Frames in temporal order."""

    allowed_connections: EdgeConnectionsList = [("Frame", "Frame")]


class Sequencer(Component):
    """Component that assembles Frames from object resolution, intrinsic, and action events."""

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
        """Accept ResolvedObject, TakeAction, and IntrinsicData events."""
        return (
            isinstance(e.data, ResolvedObject)
            or isinstance(e.data, TakeAction)
            or isinstance(e.data, IntrinsicData)
        )

    def handle_object_resolution_event(self, e: Event[ResolvedObject]) -> None:
        """Attaches a resolved object's feature group to the current frame."""
        FrameAttribute.connect(self.current_frame, e.data.feature_group)

    def handle_intrinsic_event(self, e: Event[IntrinsicData]) -> None:
        """Attaches intrinsic nodes to the current frame."""
        for intrinsic_node in e.data.to_nodes():
            FrameAttribute.connect(self.current_frame, intrinsic_node)

    def handle_action_event(self, e: Event[Any]) -> None:
        """On TakeAction, closes the current frame and starts a new one."""
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
