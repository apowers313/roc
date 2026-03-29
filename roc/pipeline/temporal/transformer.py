"""Detects changes between consecutive frames by comparing transformable attributes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from .component import Component
from .event import Event, EventBus
from .graphdb import Edge, EdgeConnectionsList
from .sequencer import Frame, Sequencer
from .transformable import Transform, Transformable


@dataclass
class TransformResult:
    """Carries the result of comparing two consecutive frames."""

    transform: Transform


class Change(Edge):
    """An edge connecting Frames to their Transforms."""

    allowed_connections: EdgeConnectionsList = [
        ("Transform", "Transform"),
        ("Frame", "Transform"),
        ("Transform", "Frame"),
    ]


class Transformer(Component):
    """Component that detects changes between consecutive frames."""

    name: str = "transformer"
    type: str = "transformer"
    auto: bool = True
    bus = EventBus[TransformResult]("transformer")

    def __init__(self) -> None:
        super().__init__()
        self.transformer_conn = self.connect_bus(Transformer.bus)
        self.sequencer_conn = self.connect_bus(Sequencer.bus)
        self.sequencer_conn.listen(self.do_transformer)

    def event_filter(self, e: Event[Any]) -> bool:
        """Only process Frame events."""
        return isinstance(e.data, Frame)

    def do_transformer(self, e: Event[Frame]) -> None:
        """Compares current and previous frames, emitting transforms for any changes."""
        current_frame = e.data

        previous_frame = _get_previous_frame(current_frame)
        if previous_frame is None:
            return

        ret = _compute_transforms(current_frame, previous_frame)
        Change.connect(previous_frame, ret)
        Change.connect(ret, current_frame)

        self.transformer_conn.send(TransformResult(transform=ret))


def _get_previous_frame(current_frame: Frame) -> Frame | None:
    """Return the previous frame linked via NextFrame, or None."""
    previous_frames = current_frame.dst_edges.select(type="NextFrame")
    if len(previous_frames) < 1:
        return None
    assert len(previous_frames) == 1
    return cast(Frame, previous_frames[0].src)


def _select_transformable_edges(frame: Frame) -> list[Any]:
    """Select edges pointing to Transformable nodes from a frame.

    Checks both FrameAttribute edges (IntrinsicNodes) and SituatedObjectInstance
    edges (ObjectInstances).
    """
    return frame.src_edges.select(
        filter_fn=lambda e: (  # type: ignore
            e.type in ("FrameAttribute", "SituatedObjectInstance")
            and isinstance(e.dst, Transformable)
        )
    )


def _get_ambiguous_uuids(frame: Frame) -> set[int]:
    """Return Object uuids that have multiple ObjectInstances in a frame."""
    from .object_instance import ObjectInstance

    edges = _select_transformable_edges(frame)
    uuid_counts: Counter[int] = Counter()
    for e in edges:
        if isinstance(e.dst, ObjectInstance):
            uuid_counts[e.dst.object_uuid] += 1
    return {uuid for uuid, count in uuid_counts.items() if count > 1}


def _connect_transform_result(transform_node: Any, ret: Transform) -> None:
    """Connect a created transform to the result and link ObjectTransforms to their parent Object."""
    from .object import Object
    from .object_transform import ObjectHistory, ObjectTransform

    Change.connect(ret, transform_node)
    if isinstance(transform_node, ObjectTransform):
        obj = Object.find_one(f"src.uuid = {transform_node.object_uuid}")
        if obj is not None:
            ObjectHistory.connect(obj, transform_node)


def _compute_node_transforms(cn: Transformable, previous_edges: list[Any], ret: Transform) -> None:
    """Compare one current-frame node against all previous-frame nodes and record changes."""
    for pe in previous_edges:
        if cn.same_transform_type(pe.dst):
            t = cn.create_transform(pe.dst)
            if t is not None:
                _connect_transform_result(t, ret)


def _compute_transforms(current_frame: Frame, previous_frame: Frame) -> Transform:
    """Compute all transforms between two consecutive frames."""
    from .object_instance import ObjectInstance

    current_edges = _select_transformable_edges(current_frame)
    previous_edges = _select_transformable_edges(previous_frame)

    # Ambiguity detection: skip transforms for Objects with multiple instances per frame
    ambiguous_current = _get_ambiguous_uuids(current_frame)
    ambiguous_previous = _get_ambiguous_uuids(previous_frame)
    ambiguous = ambiguous_current | ambiguous_previous

    ret = Transform()
    for ce in current_edges:
        cn = ce.dst
        assert isinstance(cn, Transformable)

        # Skip ambiguous ObjectInstances
        if isinstance(cn, ObjectInstance) and cn.object_uuid in ambiguous:
            continue

        _compute_node_transforms(cn, previous_edges, ret)
    return ret
