from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Collection, NewType, cast

from flexihumanhash import FlexiHumanHash
from pydantic import Field

from .attention import Attention, AttentionEvent
from .component import Component, register_component
from .event import EventBus
from .graphdb import Edge, EdgeConnectionsList, Node, NodeId
from .location import XLoc, YLoc
from .perception import Detail, FeatureNode
from .perception import Feature as PerceptionFeature
from .reporting.observability import Observability

ObjectId = NewType("ObjectId", int)


class Features(Edge):
    allowed_connections: EdgeConnectionsList = [("Object", "FeatureGroup")]


class Object(Node):
    # XXX: this was originally a UUIDv4, but Memgraph can't store Integers that
    # large right now
    uuid: ObjectId = Field(default_factory=lambda: ObjectId(random.randint(0, 2**63)))
    annotations: list[str] = Field(default_factory=list)
    resolve_count: int = Field(default=0)

    @property
    def features(self) -> list[FeatureNode]:
        feature_groups = [e.dst for e in self.src_edges if e.type == "Features"]
        feature_nodes: list[FeatureNode] = []
        for fg in feature_groups:
            assert isinstance(fg, FeatureGroup)
            feature_nodes += fg.feature_nodes

        return feature_nodes

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
    def with_features(fg: FeatureGroup) -> Object:
        o = Object()
        Features.connect(o, fg)

        return o

    @staticmethod
    def distance(obj: Object, features: Collection[FeatureNode]) -> float:
        assert isinstance(obj, Object)
        # TODO: allowed_attrs is physical attributes, not really great but
        # NetHack doesn't give us much feature-space to work with. in the future
        # we may want to come back and use motion or other features for object recognition
        allowed_attrs = {"SingleNode", "ColorNode", "ShapeNode"}  # TODO: line? flood?
        features_strs: set[str] = {str(f) for f in features if f.labels & allowed_attrs}
        obj_features: set[str] = {
            str(f) for f in obj.features if isinstance(f, FeatureNode) and f.labels & allowed_attrs
        }
        return float(len(features_strs ^ obj_features))


class FeatureGroup(Node):
    @staticmethod
    def with_features(features: Collection[PerceptionFeature[Any]]) -> FeatureGroup:
        feature_nodes: set[FeatureNode] = {f.to_nodes() for f in features}

        return FeatureGroup.from_nodes(feature_nodes)

    @staticmethod
    def from_nodes(feature_nodes: Collection[FeatureNode]) -> FeatureGroup:
        fg = FeatureGroup()
        for f in feature_nodes:
            Detail.connect(fg, f)

        return fg

    @property
    def feature_nodes(self) -> list[FeatureNode]:
        return [cast(FeatureNode, e.dst) for e in self.src_edges if e.type == "Detail"]


class CandidateObjects:
    @Observability.tracer.start_as_current_span("create_candidate_object")
    def __init__(self, feature_nodes: Collection[FeatureNode]) -> None:
        # TODO: this currently only uses features, not context, for resolution
        # the other objects in the current context should influence resolution
        distance_idx: dict[NodeId, float] = defaultdict(float)

        # TODO: getting all objects for the set of features is going to be a
        # huge explosion of objects... need to come back to this an make a
        # smarter selection algorithm
        feature_groups = [
            fg for n in feature_nodes for fg in n.predecessors.select(labels={"FeatureGroup"})
        ]
        objs = [obj for fg in feature_groups for obj in fg.predecessors.select(labels={"Object"})]
        for obj in objs:
            assert isinstance(obj, Object)
            distance_idx[obj.id] += Object.distance(obj, feature_nodes)

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
        self.candidate_object_counter = Observability.meter.create_counter(
            "roc.candidate_objects",
            unit="object",
            description="total number of objects scanned for object recognition",
        )
        self.resolved_object_counter = Observability.meter.create_counter(
            "roc.objects_resolved",
            unit="object",
            description="total number of objects resolved",
        )

    def event_filter(self, e: AttentionEvent) -> bool:
        return e.src_id.name == "vision" and e.src_id.type == "attention"

    @Observability.tracer.start_as_current_span("do_object_resolution")
    def do_object_resolution(self, e: AttentionEvent) -> None:
        # TODO: instead of just taking the first focus_point (highest saliency
        # strength) we probably want to adjust the strength for known objects /
        # novel objects
        focus_point = e.data.focus_points.iloc[0]
        x = XLoc(int(focus_point["x"]))
        y = YLoc(int(focus_point["y"]))
        features = e.data.saliency_map.get_val(x, y)
        fg = FeatureGroup.with_features(features)
        # Argument 1 to "with_features" of "FeatureGroup" has incompatible type "list[Feature[Any]]"; expected "FeatureGroup"
        objs = CandidateObjects(fg.feature_nodes)
        self.candidate_object_counter.add(len(objs))

        o: Object | None = None
        if len(objs) > 0:
            o, dist = objs[0]
            self.resolved_object_counter.add(1, attributes={"new": False})
            o.resolve_count += 1

        # TODO: "> 1" as a cutoff for matching is pretty arbitrary
        # should it be a % of features?
        # or the cutoff for matching be determined by how well the prediction is works?
        if o is None or dist > 1:
            self.resolved_object_counter.add(1, attributes={"new": True})
            o = Object.with_features(fg)

        self.obj_res_conn.send(o)
