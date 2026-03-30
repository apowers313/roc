"""Assembles game state snapshots (Frames) from perception, intrinsic, and action events."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from roc.pipeline.action import Action, TakeAction
from roc.framework.component import Component
from roc.framework.event import Event, EventBus
from roc.db.graphdb import Edge, EdgeConnectionsList, Node
from roc.pipeline.intrinsic import Intrinsic, IntrinsicData
from roc.pipeline.object.object import (
    FeatureGroup,
    Features,
    Object,
    ObjectResolver,
    ResolvedObject,
    ResolutionContext,
)
from roc.pipeline.object.object_instance import (
    FrameFeatures,
    ObjectInstance,
    ObservedAs,
    RelationshipGroup,
    Relationships,
    SituatedObjectInstance,
)
from roc.perception.base import FeatureKind
from roc.pipeline.temporal.transformable import Transform, Transformable

tick = 0
PREDICTED_FRAME_TICK = -1


def get_next_tick() -> int:
    """Returns the next sequential tick number for frame ordering."""
    global tick

    tick += 1

    return tick


def _collect_legacy_objects(frame: Frame, ret: list[Object], seen: set[int]) -> None:
    """Collect Objects via legacy Frame -> FeatureGroup -> Object path."""
    for e in frame.src_edges:
        if not isinstance(e.dst, FeatureGroup):
            continue
        for fg_edge in e.dst.src_edges:
            if isinstance(fg_edge.dst, Object) and fg_edge.dst.id not in seen:
                seen.add(fg_edge.dst.id)
                ret.append(fg_edge.dst)


def _collect_instance_objects(frame: Frame, ret: list[Object], seen: set[int]) -> None:
    """Collect Objects via new Frame -> ObjectInstance -> Object path."""
    for e in frame.src_edges:
        if not isinstance(e, SituatedObjectInstance):
            continue
        for oi_edge in e.dst.src_edges:
            if isinstance(oi_edge, ObservedAs) and oi_edge.dst.id not in seen:
                seen.add(oi_edge.dst.id)
                ret.append(oi_edge.dst)  # type: ignore[arg-type]


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
        """All Objects referenced by this frame.

        Traverses two paths:
        - Legacy: Frame -> FeatureGroup (any edge) -> Object (via Features)
        - New: Frame -> ObjectInstance (via SituatedObjectInstance) -> Object (via ObservedAs)
        """
        ret: list[Object] = []
        seen: set[int] = set()
        _collect_legacy_objects(self, ret, seen)
        _collect_instance_objects(self, ret, seen)
        return ret


class FrameAttribute(Edge):
    """An edge connecting a Frame to its constituent data (features, actions, intrinsics)."""

    allowed_connections: EdgeConnectionsList = [
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
        """Creates an ObjectInstance and attaches it to the current frame.

        Splits features by FeatureKind: PHYSICAL features stay in a FeatureGroup,
        RELATIONAL features go into a RelationshipGroup. Both are linked to the
        ObjectInstance.

        Graph connections created:
        - Frame -[FrameFeatures]-> FeatureGroup (physical features)
        - Frame -[SituatedObjectInstance]-> ObjectInstance
        - ObjectInstance -[ObservedAs]-> Object
        - ObjectInstance -[Features]-> FeatureGroup (physical)
        - ObjectInstance -[Relationships]-> RelationshipGroup (relational)
        """
        rd = e.data

        # Split features by kind
        physical_nodes = [
            fn for fn in rd.feature_group.feature_nodes if fn.kind == FeatureKind.PHYSICAL
        ]
        relational_nodes = [
            fn for fn in rd.feature_group.feature_nodes if fn.kind == FeatureKind.RELATIONAL
        ]

        # Create split groups
        phys_fg = FeatureGroup.from_nodes(physical_nodes)
        rg = RelationshipGroup.from_nodes(relational_nodes) if relational_nodes else None

        ctx = ResolutionContext(x=rd.x, y=rd.y, tick=tick)
        oi = ObjectInstance.from_resolution(rd.object, phys_fg, ctx, rg=rg)

        # Attach FeatureGroup to frame
        FrameFeatures.connect(self.current_frame, phys_fg)
        # Link ObjectInstance into frame
        SituatedObjectInstance.connect(self.current_frame, oi)
        # Link ObjectInstance to persistent Object
        ObservedAs.connect(oi, rd.object)
        # Link ObjectInstance to its physical FeatureGroup
        Features.connect(oi, phys_fg)
        # Link ObjectInstance to relational features
        if rg is not None:
            Relationships.connect(oi, rg)

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
