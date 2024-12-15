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
            "{{adj}}-{{noun}}-named-{{firstname|lower}}-{{lastname|lower}}-{{hex(6)}}"
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

    @staticmethod
    def distance(obj: Node, features: Collection[FeatureNode]) -> float:
        assert isinstance(obj, Object)
        # TODO: allowed_attrs is physical attributes, not really great but
        # NetHack doesn't give us much feature-space to work with. in the future
        # we may want to come back and use motion or other features for object recognition
        allowed_attrs = {"Single", "Color", "Shape"}
        features_strs: set[str] = {str(f) for f in features if f.labels & allowed_attrs}
        obj_features: set[str] = {
            str(e.dst)
            for e in obj.src_edges
            if e.type == "Feature" and e.dst.labels & allowed_attrs
        }
        return float(len(features_strs ^ obj_features))


class CandidateObjects:
    def __init__(self, feature_nodes: Collection[FeatureNode]) -> None:
        # TODO: this currently only uses features, not context, for resolution
        # the other objects in the current context should influence resolution
        distance_idx: dict[NodeId, float] = defaultdict(float)

        def node_to_obj_ids(n: FeatureNode) -> None:
            for e in n.dst_edges:
                if "Object" in e.src.labels:
                    # XXX: this is silly, just a placeholder until we have some
                    # weighted features
                    distance_idx[e.src.id] += Object.distance(e.src, feature_nodes)

        for n in feature_nodes:
            node_to_obj_ids(n)

        self.distance_idx = distance_idx
        self.order: list[NodeId] = sorted(self.distance_idx, key=lambda k: self.distance_idx[k])

    def __getitem__(self, idx: int) -> tuple[Object, float]:
        n = self.order[idx]
        return (Object.get(n), self.distance_idx[n])

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
            o, dist = objs[0]
            o.resolve_count += 1

        # TODO: "> 1" as a cutoff for matching is pretty arbitrary
        # should it be a % of features?
        # or the cutoff for matching be determined by how well the prediction is works?
        if o is None or dist > 1:
            o = Object.with_features(feature_nodes)

        self.obj_res_conn.send(o)
