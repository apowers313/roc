from __future__ import annotations

from collections import defaultdict
from typing import Collection, NewType
from uuid import uuid4

from flexihumanhash import FlexiHumanHash
from pydantic import Field

from .attention import Attention, AttentionEvent
from .component import Component, register_component
from .event import EventBus
from .graphdb import Node, NodeId, register_node
from .location import XLoc, YLoc
from .perception import FeatureNode

ObjectId = NewType("ObjectId", int)


@register_node("Object")
class Object(Node):
    uuid: ObjectId = Field(default_factory=lambda: ObjectId(uuid4().int))
    annotations: list[str] = Field(default_factory=list)
    resolve_count: int = Field(default=0)

    @property
    def features(self) -> Collection[Node]:
        return [e.dst for e in self.src_edges if e.type == "Feature"]

    def __str__(self) -> str:
        fhh = FlexiHumanHash(
            "{{firstname|lower}}-{{lastname|lower}}-is-the-{{adj}}-{{noun}}-{{hex(6)}}"
        )
        h = fhh.hash(self.uuid)
        ret = f"Object({h})"
        for f in self.features:
            ret += f"\n\t{f}"

        return ret

    @staticmethod
    def with_features(features: Collection[FeatureNode]) -> Object:
        o = Object()
        for f in features:
            Node.connect(o, f, "Feature")

        return o


class CandidateObjects:
    def __init__(self, feature_nodes: Collection[FeatureNode]) -> None:
        # TODO: this currently only uses features, not context, for resolution
        # the other objects in the current context should influence resolution
        strength_idx: dict[NodeId, float] = defaultdict(float)

        def node_to_obj_ids(n: FeatureNode) -> None:
            for e in n.dst_edges:
                if "Object" in e.src.labels:
                    # XXX: this is silly, just a placeholder until we have some
                    # weighted features
                    strength_idx[e.src.id] += 1

        for n in feature_nodes:
            node_to_obj_ids(n)

        self.strength_idx = strength_idx
        self.order: list[NodeId] = sorted(
            self.strength_idx,
            key=lambda k: self.strength_idx[k],
            reverse=True,
        )

    def __getitem__(self, idx: int) -> tuple[Object, float]:
        n = self.order[idx]
        return (Object.get(n), self.strength_idx[n])

    def __len__(self) -> int:
        return len(self.order)


@register_component("resolver", "object", auto=True)
class ObjectResolver(Component):
    bus = EventBus[Object]("object")

    def __init__(self) -> None:
        super().__init__()
        self.att_conn = self.connect_bus(Attention.bus)
        self.att_conn.listen(self.do_object_resolution)
        self.obj_res_conn = self.connect_bus(ObjectResolver.bus)

    def event_filter(self, e: AttentionEvent) -> bool:
        return e.src_id.name == "vision" and e.src_id.type == "attention"

    def do_object_resolution(self, e: AttentionEvent) -> None:
        focus_point = e.data.focus_points.iloc[0]
        x = XLoc(int(focus_point["x"]))
        y = YLoc(int(focus_point["y"]))
        features = e.data.saliency_map.get_val(x, y)
        feature_nodes: set[FeatureNode] = {f.to_nodes() for f in features}
        objs = CandidateObjects(feature_nodes)

        o: Object | None = None
        if len(objs) > 0:
            o, strength = objs[0]
            o.resolve_count += 1

        if o is None or strength < 2:
            o = Object.with_features(feature_nodes)

        self.obj_res_conn.send(o)
