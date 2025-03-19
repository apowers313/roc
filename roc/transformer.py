from __future__ import annotations

from typing import Any

from .component import Component
from .event import Event, EventBus
from .graphdb import Edge, EdgeConnectionsList
from .sequencer import Frame, Sequencer
from .transformable import Transform, Transformable

TransformerEvent = Event[Transform]


class Change(Edge):
    allowed_connections: EdgeConnectionsList = [
        ("Transform", "Transform"),
        ("Frame", "Transform"),
        ("Transform", "Frame"),
    ]


class Transformer(Component):
    name: str = "transformer"
    type: str = "transformer"
    auto: bool = True
    bus = EventBus[Transform]("transformer")

    def __init__(self) -> None:
        super().__init__()
        self.transformer_conn = self.connect_bus(Transformer.bus)
        self.sequencer_conn = self.connect_bus(Sequencer.bus)
        self.sequencer_conn.listen(self.do_transformer)

    def event_filter(self, e: Event[Any]) -> bool:
        return isinstance(e.data, Frame)

    def do_transformer(self, e: Event[Frame]) -> None:
        current_frame = e.data

        # get previous frame
        previous_frames = current_frame.dst_edges.select(type="NextFrame")
        if len(previous_frames) < 1:
            return
        assert len(previous_frames) == 1
        previous_frame = previous_frames[0].src

        # get all transformable attributes
        current_edges = current_frame.src_edges.select(
            filter_fn=lambda e: e.type == "FrameAttributes" and isinstance(e.dst, Transformable)  # type: ignore
        )
        previous_edges = previous_frame.src_edges.select(
            filter_fn=lambda e: e.type == "FrameAttributes" and isinstance(e.dst, Transformable)  # type: ignore
        )

        # find all changes between previous frame and this frame
        ret = Transform()
        for ce in current_edges:
            cn = ce.dst
            assert isinstance(cn, Transformable)

            for pe in previous_edges:
                pn = pe.dst

                if cn.same_transform_type(pn):
                    t = cn.create_transform(pn)
                    if t is not None:
                        Change.connect(ret, t)

        Change.connect(previous_frame, ret)
        Change.connect(ret, current_frame)

        self.transformer_conn.send(ret)
