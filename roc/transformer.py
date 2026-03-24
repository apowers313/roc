"""Detects changes between consecutive frames by comparing transformable attributes."""

from __future__ import annotations

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
    """Select edges pointing to Transformable nodes from a frame."""
    return frame.src_edges.select(
        filter_fn=lambda e: e.type == "FrameAttribute" and isinstance(e.dst, Transformable)  # type: ignore
    )


def _compute_transforms(current_frame: Frame, previous_frame: Frame) -> Transform:
    """Compute all transforms between two consecutive frames."""
    current_edges = _select_transformable_edges(current_frame)
    previous_edges = _select_transformable_edges(previous_frame)

    ret = Transform()
    for ce in current_edges:
        cn = ce.dst
        assert isinstance(cn, Transformable)
        for pe in previous_edges:
            if cn.same_transform_type(pe.dst):
                t = cn.create_transform(pe.dst)
                if t is not None:
                    Change.connect(ret, t)
    return ret
