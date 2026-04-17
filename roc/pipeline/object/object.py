"""Object identification and resolution from visual features."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from time import time_ns
from typing import TYPE_CHECKING, Any, Collection, Iterable, NewType, cast

from opentelemetry import trace as otel_trace


from cachetools import LRUCache
from loguru import logger
from flexihumanhash import FlexiHumanHash
from opentelemetry._logs import SeverityNumber
from opentelemetry._logs import LogRecord
from pydantic import Field

from ..attention.attention import Attention, AttentionEvent, VisionAttentionData
from ...framework.component import Component
from ...framework.event import EventBus
from ...framework.expmod import ExpMod
from ...db.graphdb import Edge, EdgeConnectionsList, Node, NodeId
from ...perception.location import XLoc, YLoc
from ...perception.base import Detail, FeatureKind, FeatureNode
from ...perception.base import VisualFeature as PerceptionFeature
from ...reporting.observability import Observability

_otel_logger = Observability.get_logger("roc.resolution")

# OTel attribute/metric name constants (S1192: extract duplicated string literals).
_METRIC_RESOLUTION_DECISION = "roc.resolution.decision"
_OTEL_ATTR_EVENT_NAME = "event.name"

if TYPE_CHECKING:
    from ..temporal.sequencer import Frame

ObjectId = NewType("ObjectId", int)

# Reverse index: FeatureNode ID -> set of Object IDs that contain that feature.
# Updated when new Objects are created, avoids expensive graph walks in _find_candidates.
_feature_to_objects: dict[NodeId, set[NodeId]] = defaultdict(set)


class Features(Edge):
    """An edge connecting an Object to its FeatureGroups."""

    allowed_connections: EdgeConnectionsList = [
        ("Object", "FeatureGroup"),
        ("ObjectInstance", "FeatureGroup"),
    ]


class Object(Node):
    """A persistent entity identified by matching feature groups across frames."""

    # XXX: this was originally a UUIDv4, but Memgraph can't store Integers that
    # large right now
    uuid: ObjectId = Field(default_factory=lambda: ObjectId(random.randint(0, 2**63)))
    annotations: list[str] = Field(default_factory=list)
    resolve_count: int = Field(default=0)
    last_x: XLoc | None = Field(default=None)
    last_y: YLoc | None = Field(default=None)
    last_tick: int = Field(default=0)

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

        for fn in fg.feature_nodes:
            _feature_to_objects[fn.id].add(o.id)

        return o

    @property
    def frames(self) -> list[Frame]:
        """All frames that reference this object."""
        from ..temporal.sequencer import Frame as _Frame

        ret: list[Frame] = []

        for e in self.dst_edges:
            if isinstance(e.src, _Frame):
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


@dataclass
class ResolutionContext:
    """Per-observation metadata passed to object resolution.

    Provides spatial and temporal context that resolution algorithms can use
    for priors or filtering. Algorithms that don't need this context (e.g.,
    SymmetricDifferenceResolution) can ignore it.
    """

    x: XLoc
    y: YLoc
    tick: int


@dataclass
class ResolutionLogData:
    """Data collected during resolution for structured logging."""

    feature_strs: list[str]
    context: ResolutionContext
    observed_attrs: dict[str, Any] | None = None
    log_posteriors: dict[NodeId | str, float] | None = None


@dataclass
class SpatialNodeAttrs:
    """Visual attributes from a spatial feature node (FloodNode or LineNode)."""

    shape: int
    glyph_type: int
    color: int
    type_name: str


def _apply_spatial_node_attrs(
    result: dict[str, Any],
    attrs: SpatialNodeAttrs,
    *,
    override_type: bool = False,
) -> None:
    """Apply char/glyph/color/type from a spatial node (FloodNode or LineNode).

    Args:
        result: The visual-attrs dict being built.
        attrs: The spatial node attributes to apply.
        override_type: If True, always set "type"; otherwise only set if absent.
    """
    result.setdefault("char", chr(attrs.shape))
    result.setdefault("glyph", attrs.glyph_type)
    result.setdefault("color", _COLOR_NAMES.get(attrs.color, str(attrs.color)))
    if override_type:
        result["type"] = attrs.type_name
    else:
        result.setdefault("type", attrs.type_name)


def _extract_visual_attrs_from_nodes(
    nodes: Iterable[FeatureNode],
) -> dict[str, Any]:
    """Extract visual attributes (char, color, glyph, type) from feature nodes."""
    from ...perception.feature_extractors.color import ColorNode
    from ...perception.feature_extractors.flood import FloodNode
    from ...perception.feature_extractors.line import LineNode
    from ...perception.feature_extractors.shape import ShapeNode
    from ...perception.feature_extractors.single import SingleNode

    result: dict[str, Any] = {}
    for f in nodes:
        if isinstance(f, ShapeNode):
            result["char"] = chr(f.type)
        elif isinstance(f, ColorNode) and f.attr_strs:
            result["color"] = f.attr_strs[0]
        elif isinstance(f, SingleNode):
            result["glyph"] = f.type
            result.setdefault("type", "single")
        elif isinstance(f, FloodNode):
            _apply_spatial_node_attrs(
                result, SpatialNodeAttrs(f.shape, f.type, f.color, "flood"), override_type=True
            )
        elif isinstance(f, LineNode):
            _apply_spatial_node_attrs(result, SpatialNodeAttrs(f.shape, f.type, f.color, "line"))
    return result


def _extract_visual_attrs(obj: Object) -> dict[str, Any]:
    """Extract glyph char, color name, and glyph ID from an Object's features."""
    return _extract_visual_attrs_from_nodes(obj.features)


# Standard terminal color codes matching ColorNode.attr_strs output.
_COLOR_NAMES: dict[int, str] = {
    0: "BLACK",
    1: "RED",
    2: "GREEN",
    3: "BROWN",
    4: "BLUE",
    5: "MAGENTA",
    6: "CYAN",
    7: "GREY",
    8: "NO COLOR",
    9: "ORANGE",
    10: "BRIGHT GREEN",
    11: "YELLOW",
    12: "BRIGHT BLUE",
    13: "BRIGHT MAGENTA",
    14: "BRIGHT CYAN",
    15: "WHITE",
    16: "MAX",
}


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
        context: ResolutionContext,
    ) -> Object | None:
        """Match feature nodes to an existing Object or return None to create a new one.

        Args:
            feature_nodes: The feature nodes from the current observation.
            feature_group: The FeatureGroup node for the current observation.
            context: Spatial and temporal context for the observation.

        Returns:
            The matched Object, or None if no match was found.
        """
        raise NotImplementedError


# Concrete ObjectResolutionExpMod implementations live under
# roc/expmods/object_resolution/ (symmetric_difference.py,
# dirichlet_categorical.py). They import the shared helpers and
# abstract base from this module.


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
    spatial_distance_histogram = Observability.meter.create_histogram(
        "roc.resolution.spatial_distance",
        unit="cells",
        description="manhattan distance between observation and matched object",
    )
    temporal_gap_histogram = Observability.meter.create_histogram(
        "roc.resolution.temporal_gap",
        unit="ticks",
        description="ticks since matched object was last seen",
    )

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
        """Only process events from the vision attention component. Skip AttentionSettled."""
        from ..attention.attention import AttentionSettled

        if isinstance(e.data, AttentionSettled):
            return False
        return e.src_id.name == "vision" and e.src_id.type == "attention"

    def _record_existing_match(self, o: Object, x: XLoc, y: YLoc, current_tick: int) -> None:
        """Record metrics for an existing object match."""
        self.resolved_object_counter.add(1, attributes={"new": False})
        o.resolve_count += 1
        logger.debug("object resolved: matched existing id={}", o.uuid)
        if o.last_x is not None and o.last_y is not None:
            dist = abs(int(x) - int(o.last_x)) + abs(int(y) - int(o.last_y))
            self.spatial_distance_histogram.record(dist)
        if o.last_tick > 0:
            gap = current_tick - o.last_tick
            self.temporal_gap_histogram.record(gap)

    def _handle_new_object(
        self,
        fg: FeatureGroup,
        x: XLoc,
        y: YLoc,
        resolution: ObjectResolutionExpMod,
    ) -> Object:
        """Create a new object and emit related telemetry."""
        self.resolved_object_counter.add(1, attributes={"new": True})
        o = Object.with_features(fg)
        logger.info(
            "object resolved: NEW object id={} at ({},{}) features={}",
            o.uuid,
            x,
            y,
            [str(f) for f in fg.feature_nodes],
        )
        if hasattr(resolution, "initialize_alphas"):
            physical_nodes = [f for f in fg.feature_nodes if f.kind == FeatureKind.PHYSICAL]
            feature_strs = [str(f) for f in physical_nodes]
            resolution.initialize_alphas(o.id, feature_strs)

        from roc.reporting.state import State

        current_res = State.get_states().resolution.val
        if isinstance(current_res, dict):
            current_res["new_object_id"] = o.id

        self._emit_new_object_event(o)
        return o

    @staticmethod
    def _emit_new_object_event(o: Object) -> None:
        """Emit an OTel event linking this step to the new object ID."""
        span_context = otel_trace.get_current_span().get_span_context()
        _otel_logger.emit(
            LogRecord(
                timestamp=time_ns(),
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=json.dumps({"new_object_id": o.id}, default=str),
                attributes={_OTEL_ATTR_EVENT_NAME: "roc.resolution.new_object_id"},
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
            )
        )

    @Observability.tracer.start_as_current_span("do_object_resolution")
    def do_object_resolution(self, e: AttentionEvent) -> None:
        """Resolves the highest-saliency focus point to an existing or new object."""
        assert isinstance(e.data, VisionAttentionData)
        # TODO: instead of just taking the first focus_point (highest saliency
        # strength) we probably want to adjust the strength for known objects /
        # novel objects
        focus_point = e.data.focus_points.iloc[0]
        x = XLoc(int(focus_point["x"]))
        y = YLoc(int(focus_point["y"]))
        features = e.data.saliency_map.get_val(x, y)
        fg = FeatureGroup.with_features(features)

        from roc.framework.clock import Clock

        current_tick = Clock.get()
        ctx = ResolutionContext(x=x, y=y, tick=current_tick)
        resolution = ObjectResolutionExpMod.get(default="symmetric-difference")
        o = resolution.resolve(fg.feature_nodes, fg, ctx)

        if o is not None:
            self._record_existing_match(o, x, y, current_tick)
            Features.connect(o, fg)
        else:
            o = self._handle_new_object(fg, x, y, resolution)

        o.last_x = XLoc(int(x))
        o.last_y = YLoc(int(y))
        o.last_tick = current_tick

        self.obj_res_conn.send(ResolvedObject(object=o, feature_group=fg, x=x, y=y))


class ObjectCache(LRUCache[tuple[XLoc, YLoc], ResolvedObject]):
    """LRU cache mapping screen locations to their most recently resolved objects."""
