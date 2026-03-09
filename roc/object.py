"""Object identification and resolution from visual features."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Collection, NewType, cast

from cachetools import LRUCache
from flexihumanhash import FlexiHumanHash
from pydantic import Field

from .attention import Attention, AttentionEvent
from .component import Component
from .event import EventBus
from .expmod import ExpMod
from .graphdb import Edge, EdgeConnectionsList, Node, NodeId
from .location import XLoc, YLoc
from .perception import Detail, FeatureNode
from .perception import VisualFeature as PerceptionFeature
from .reporting.observability import Observability

if TYPE_CHECKING:
    from .sequencer import Frame

ObjectId = NewType("ObjectId", int)


class Features(Edge):
    """An edge connecting an Object to its FeatureGroups."""

    allowed_connections: EdgeConnectionsList = [("Object", "FeatureGroup")]


class Object(Node):
    """A persistent entity identified by matching feature groups across frames."""

    # XXX: this was originally a UUIDv4, but Memgraph can't store Integers that
    # large right now
    uuid: ObjectId = Field(default_factory=lambda: ObjectId(random.randint(0, 2**63)))
    annotations: list[str] = Field(default_factory=list)
    resolve_count: int = Field(default=0)

    @property
    def feature_groups(self) -> list[FeatureGroup]:
        """All feature groups associated with this object."""
        feature_groups: list[FeatureGroup] = []

        for e in self.src_edges:
            if e.type == "Features":
                assert isinstance(e.dst, FeatureGroup)
                feature_groups.append(e.dst)

        return feature_groups

    @property
    def features(self) -> list[FeatureNode]:
        """All feature nodes across all feature groups of this object."""
        feature_nodes: list[FeatureNode] = []

        for fg in self.feature_groups:
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
        """Creates a new Object and connects it to the given FeatureGroup."""
        o = Object()
        Features.connect(o, fg)

        return o

    @property
    def frames(self) -> list[Frame]:
        """All frames that reference this object."""
        ret: list[Frame] = []

        for e in self.dst_edges:
            if isinstance(e.src, Frame):
                ret.append(e.src)

        return ret


class FeatureGroup(Node):
    """A collection of feature nodes that together describe one observation."""

    @staticmethod
    def with_features(features: Collection[PerceptionFeature[Any]]) -> FeatureGroup:
        """Creates a FeatureGroup from perception-level features, converting them to nodes."""
        feature_nodes: set[FeatureNode] = {f.to_nodes() for f in features}

        return FeatureGroup.from_nodes(feature_nodes)

    @staticmethod
    def from_nodes(feature_nodes: Collection[FeatureNode]) -> FeatureGroup:
        """Creates a FeatureGroup from existing feature nodes."""
        fg = FeatureGroup()
        for f in feature_nodes:
            Detail.connect(fg, f)

        return fg

    @property
    def feature_nodes(self) -> list[FeatureNode]:
        """The feature nodes connected to this group via Detail edges."""
        return [cast(FeatureNode, e.dst) for e in self.src_edges if e.type == "Detail"]


class ObjectResolutionExpMod(ExpMod):
    """Base class for object resolution experiment modules.

    Subclasses implement ``resolve()`` to match a set of observed feature nodes to an
    existing Object, or return None to indicate a new Object should be created.
    """

    modtype = "object-resolution"

    def resolve(
        self,
        feature_nodes: Collection[FeatureNode],
        feature_group: FeatureGroup,
    ) -> Object | None:
        """Match feature nodes to an existing Object or return None to create a new one.

        Args:
            feature_nodes: The feature nodes from the current observation.
            feature_group: The FeatureGroup node for the current observation.

        Returns:
            The matched Object, or None if no match was found.
        """
        raise NotImplementedError


class SymmetricDifferenceResolution(ObjectResolutionExpMod):
    """Matches objects using symmetric set difference of physical features.

    Walks the graph backwards from feature nodes to find candidate Objects, computes
    the symmetric difference between physical feature sets (SingleNode, ColorNode,
    ShapeNode), and matches if the best candidate has distance <= 1.
    """

    name = "symmetric-difference"

    candidate_object_counter = Observability.meter.create_counter(
        "roc.candidate_objects",
        unit="object",
        description="total number of candidate objects scanned during resolution",
    )

    def resolve(
        self,
        feature_nodes: Collection[FeatureNode],
        feature_group: FeatureGroup,
    ) -> Object | None:
        """Match feature nodes to an existing Object using symmetric set difference."""
        candidates = self._find_candidates(feature_nodes)
        self.candidate_object_counter.add(len(candidates))
        if not candidates:
            return None

        best_obj, best_dist = candidates[0]
        if best_dist <= 1:
            return best_obj
        return None

    @Observability.tracer.start_as_current_span("find_candidate_objects")
    def _find_candidates(
        self, feature_nodes: Collection[FeatureNode]
    ) -> list[tuple[Object, float]]:
        """Find and rank candidate Objects by feature distance.

        Args:
            feature_nodes: The feature nodes to match against.

        Returns:
            List of (Object, distance) tuples sorted by ascending distance.
        """
        distance_idx: dict[NodeId, float] = defaultdict(float)

        feature_groups = [
            fg for n in feature_nodes for fg in n.predecessors.select(labels={"FeatureGroup"})
        ]
        objs = [obj for fg in feature_groups for obj in fg.predecessors.select(labels={"Object"})]
        for obj in objs:
            assert isinstance(obj, Object)
            distance_idx[obj.id] += self._distance(obj, feature_nodes)

        order = sorted(distance_idx, key=lambda k: distance_idx[k])
        return [(Object.get(n), distance_idx[n]) for n in order]

    @staticmethod
    def _distance(obj: Object, features: Collection[FeatureNode]) -> float:
        """Compute symmetric set difference between an Object's features and new features.

        Only considers physical attributes: SingleNode, ColorNode, ShapeNode.

        Args:
            obj: The candidate Object.
            features: The new observation's feature nodes.

        Returns:
            The count of features present in one set but not the other.
        """
        allowed_attrs = {"SingleNode", "ColorNode", "ShapeNode"}
        features_strs: set[str] = {str(f) for f in features if f.labels & allowed_attrs}
        obj_features: set[str] = {
            str(f) for f in obj.features if isinstance(f, FeatureNode) and f.labels & allowed_attrs
        }
        return float(len(features_strs ^ obj_features))


@dataclass
class ResolvedObject:
    """The result of object resolution: the matched or new object with its features and location."""

    object: Object
    feature_group: FeatureGroup
    x: XLoc
    y: YLoc


ObjectData = ResolvedObject


class ObjectResolver(Component):
    """Component that matches visual features to existing objects or creates new ones."""

    name: str = "resolver"
    type: str = "object"
    auto: bool = True
    bus = EventBus[ObjectData]("object")

    def __init__(self) -> None:
        super().__init__()
        self.att_conn = self.connect_bus(Attention.bus)
        self.att_conn.listen(self.do_object_resolution)
        self.obj_res_conn = self.connect_bus(ObjectResolver.bus)
        self.resolved_object_counter = Observability.meter.create_counter(
            "roc.objects_resolved",
            unit="object",
            description="total number of objects resolved",
        )

    def event_filter(self, e: AttentionEvent) -> bool:
        """Only process events from the vision attention component."""
        return e.src_id.name == "vision" and e.src_id.type == "attention"

    @Observability.tracer.start_as_current_span("do_object_resolution")
    def do_object_resolution(self, e: AttentionEvent) -> None:
        """Resolves the highest-saliency focus point to an existing or new object."""
        # TODO: instead of just taking the first focus_point (highest saliency
        # strength) we probably want to adjust the strength for known objects /
        # novel objects
        focus_point = e.data.focus_points.iloc[0]
        x = XLoc(int(focus_point["x"]))
        y = YLoc(int(focus_point["y"]))
        features = e.data.saliency_map.get_val(x, y)
        fg = FeatureGroup.with_features(features)

        resolution = ObjectResolutionExpMod.get(default="symmetric-difference")
        o = resolution.resolve(fg.feature_nodes, fg)

        if o is not None:
            self.resolved_object_counter.add(1, attributes={"new": False})
            o.resolve_count += 1
        else:
            self.resolved_object_counter.add(1, attributes={"new": True})
            o = Object.with_features(fg)

        self.obj_res_conn.send(ResolvedObject(object=o, feature_group=fg, x=x, y=y))


class ObjectCache(LRUCache[tuple[XLoc, YLoc], ResolvedObject]):
    """LRU cache mapping screen locations to their most recently resolved objects."""
