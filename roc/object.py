"""Object identification and resolution from visual features."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from time import time_ns
from typing import TYPE_CHECKING, Any, Collection, NewType, cast

from opentelemetry import trace as otel_trace

from scipy.special import logsumexp

from cachetools import LRUCache
from loguru import logger
from flexihumanhash import FlexiHumanHash
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LogRecord
from pydantic import Field

from .attention import Attention, AttentionEvent
from .component import Component
from .event import EventBus
from .expmod import ExpMod
from .graphdb import Edge, EdgeConnectionsList, Node, NodeId
from .location import XLoc, YLoc
from .perception import Detail, FeatureKind, FeatureNode
from .perception import VisualFeature as PerceptionFeature
from .reporting.observability import Observability

_otel_logger = Observability.get_logger("roc.resolution")

if TYPE_CHECKING:
    from .sequencer import Frame

ObjectId = NewType("ObjectId", int)

# Reverse index: FeatureNode ID -> set of Object IDs that contain that feature.
# Updated when new Objects are created, avoids expensive graph walks in _find_candidates.
_feature_to_objects: dict[NodeId, set[NodeId]] = defaultdict(set)


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


def _extract_visual_attrs(obj: Object) -> dict[str, Any]:
    """Extract glyph char, color name, and glyph ID from an Object's features."""
    from .feature_extractors.color import ColorNode
    from .feature_extractors.shape import ShapeNode
    from .feature_extractors.single import SingleNode

    result: dict[str, Any] = {}
    for f in obj.features:
        if isinstance(f, ShapeNode):
            result["char"] = chr(f.type)
        elif isinstance(f, ColorNode) and f.attr_strs:
            result["color"] = f.attr_strs[0]
        elif isinstance(f, SingleNode):
            result["glyph"] = f.type
    return result


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
    candidates_histogram = Observability.meter.create_histogram(
        "roc.resolution.candidates",
        unit="count",
        description="number of candidate objects per resolution",
    )
    decision_counter = Observability.meter.create_counter(
        "roc.resolution.decision",
        unit="resolution",
        description="resolution outcome: match or new_object",
    )

    def resolve(
        self,
        feature_nodes: Collection[FeatureNode],
        feature_group: FeatureGroup,
        context: ResolutionContext,
    ) -> Object | None:
        """Match feature nodes to an existing Object using symmetric set difference."""
        candidates = self._find_candidates(feature_nodes)
        self.candidate_object_counter.add(len(candidates))
        self.candidates_histogram.record(len(candidates))
        feature_strs = [str(f) for f in feature_nodes]

        if not candidates:
            self.decision_counter.add(1, attributes={"outcome": "new_object"})
            self._log_decision("new_object", None, feature_strs, candidates, context)
            return None

        best_obj, best_dist = candidates[0]
        if best_dist <= 1:
            self.decision_counter.add(1, attributes={"outcome": "match"})
            self._log_decision(
                "match", best_obj, feature_strs, candidates, context, best_distance=best_dist
            )
            return best_obj
        self.decision_counter.add(1, attributes={"outcome": "new_object"})
        self._log_decision(
            "new_object", None, feature_strs, candidates, context, best_distance=best_dist
        )
        return None

    def _log_decision(
        self,
        outcome: str,
        matched_obj: Object | None,
        feature_strs: list[str],
        candidates: list[tuple[Object, float]],
        context: ResolutionContext,
        *,
        best_distance: float | None = None,
    ) -> None:
        """Emit an OTel log record describing this resolution decision."""
        record: dict[str, Any] = {
            "event": "resolution_decision",
            "algorithm": self.name,
            "outcome": outcome,
            "tick": context.tick,
            "x": int(context.x),
            "y": int(context.y),
            "features": feature_strs,
            "num_candidates": len(candidates),
            "matched_object_id": matched_obj.id if matched_obj is not None else None,
        }
        if best_distance is not None:
            record["best_distance"] = best_distance
        if matched_obj is not None:
            record["matched_attrs"] = _extract_visual_attrs(matched_obj)
        if candidates:
            record["candidate_distances"] = [
                (str(obj.id), round(dist, 2)) for obj, dist in candidates[:5]
            ]
            record["candidate_details"] = [
                {"id": str(obj.id), "distance": round(dist, 2), **_extract_visual_attrs(obj)}
                for obj, dist in candidates[:5]
            ]

        span_context = otel_trace.get_current_span().get_span_context()
        _otel_logger.emit(
            LogRecord(
                timestamp=time_ns(),
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=json.dumps(record, default=str),
                attributes={"event.name": "roc.resolution.decision"},
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
            )
        )

        # Update live state for dashboard
        from roc.reporting.state import State

        State.get_states().resolution.set(record)

    @Observability.tracer.start_as_current_span("find_candidate_objects")
    def _find_candidates(
        self, feature_nodes: Collection[FeatureNode]
    ) -> list[tuple[Object, float]]:
        """Find and rank candidate Objects by feature distance.

        Uses the reverse index (_feature_to_objects) to find candidates in O(f)
        instead of walking the graph via predecessors.select().

        Args:
            feature_nodes: The feature nodes to match against.

        Returns:
            List of (Object, distance) tuples sorted by ascending distance.
        """
        distance_idx: dict[NodeId, float] = {}

        for fn in feature_nodes:
            for obj_id in _feature_to_objects.get(fn.id, ()):
                if obj_id not in distance_idx:
                    obj = Object.get(obj_id)
                    assert isinstance(obj, Object)
                    distance_idx[obj_id] = self._distance(obj, feature_nodes)

        order = sorted(distance_idx, key=lambda k: distance_idx[k])
        return [(Object.get(n), distance_idx[n]) for n in order]

    @staticmethod
    def _distance(obj: Object, features: Collection[FeatureNode]) -> float:
        """Compute symmetric set difference between an Object's physical features and new features.

        Only considers features where FeatureNode.physical is True (appearance-based
        features like shape, color, lines, floods) and ignores event-based features
        (deltas, motion).

        Args:
            obj: The candidate Object.
            features: The new observation's feature nodes.

        Returns:
            The count of features present in one set but not the other.
        """
        features_strs: set[str] = {str(f) for f in features if f.kind == FeatureKind.PHYSICAL}
        obj_features: set[str] = {
            str(f)
            for f in obj.features
            if isinstance(f, FeatureNode) and f.kind == FeatureKind.PHYSICAL
        }
        if not features_strs and not obj_features:
            return float("inf")
        return float(len(features_strs ^ obj_features))


class DirichletCategoricalResolution(ObjectResolutionExpMod):
    """Bayesian object resolution using Dirichlet-Categorical model.

    Computes posterior probabilities over object identities using
    spatial/temporal priors and Dirichlet posterior predictive likelihoods.
    """

    name = "dirichlet-categorical"

    # Configurable parameters
    prior_alpha: float = 1.0
    spatial_scale: float = 3.0
    temporal_scale: float = 50.0
    confidence_threshold: float = 0.5
    excluded_feature_labels: set[str] = set()

    # Telemetry
    posterior_max_histogram = Observability.meter.create_histogram(
        "roc.dirichlet.posterior_max",
        unit="probability",
        description="maximum posterior probability across candidates",
    )
    posterior_margin_histogram = Observability.meter.create_histogram(
        "roc.dirichlet.posterior_margin",
        unit="probability",
        description="margin between best and second-best posterior",
    )
    new_object_posterior_histogram = Observability.meter.create_histogram(
        "roc.dirichlet.new_object_posterior",
        unit="probability",
        description="posterior probability assigned to new-object hypothesis",
    )
    alpha_sum_histogram = Observability.meter.create_histogram(
        "roc.dirichlet.alpha_sum",
        unit="count",
        description="sum of alpha vector for matched object",
    )
    dirichlet_decision_counter = Observability.meter.create_counter(
        "roc.dirichlet.decision",
        unit="resolution",
        description="resolution outcome: match, new_object, or low_confidence",
    )

    def __init__(self) -> None:
        super().__init__()
        self._alphas: dict[NodeId, dict[str, float]] = {}
        self._global_vocab: set[str] = set()

    def resolve(
        self,
        feature_nodes: Collection[FeatureNode],
        feature_group: FeatureGroup,
        context: ResolutionContext,
    ) -> Object | None:
        """Full Bayesian resolution pipeline."""
        # Step 1: Find candidates
        candidates = self._find_candidates(feature_nodes)

        # Filter features by exclusion set
        active_features = self._filter_features(feature_nodes)
        feature_strs = [str(f) for f in active_features]

        # Update global vocabulary
        self._global_vocab.update(feature_strs)

        if not candidates:
            self.dirichlet_decision_counter.add(1, attributes={"outcome": "new_object"})
            self._log_decision(
                "new_object", None, feature_strs, [], {}, context, reason="no_candidates"
            )
            return None

        # Step 2: Compute priors (candidates + "new" hypothesis)
        log_priors = self._compute_priors(candidates, context)

        # Step 3: Compute likelihoods
        log_likelihoods = self._compute_likelihoods(candidates, feature_strs)

        # Step 4: Compute posteriors
        log_posteriors = self._compute_posteriors(log_priors, log_likelihoods)

        # Step 5: Decision
        result = self._decide(candidates, log_posteriors)

        # Step 6: Update alphas if matched
        if result is not None:
            self._update_alphas(result.id, feature_strs)

        # Log the decision
        self._log_decision(
            "match" if result is not None else "new_object",
            result,
            feature_strs,
            candidates,
            log_posteriors,
            context,
        )

        return result

    def _find_candidates(self, feature_nodes: Collection[FeatureNode]) -> list[Object]:
        """Find candidate Objects using reverse index.

        Uses the reverse index (_feature_to_objects) to find candidates in O(f)
        instead of walking the graph via predecessors.select().
        """
        seen: set[NodeId] = set()
        candidates: list[Object] = []

        for fn in feature_nodes:
            for obj_id in _feature_to_objects.get(fn.id, ()):
                if obj_id not in seen:
                    seen.add(obj_id)
                    obj = Object.get(obj_id)
                    assert isinstance(obj, Object)
                    candidates.append(obj)

        return candidates

    def _filter_features(self, feature_nodes: Collection[FeatureNode]) -> list[FeatureNode]:
        """Keep only PHYSICAL features, then remove any in excluded_feature_labels."""
        result = [f for f in feature_nodes if f.kind == FeatureKind.PHYSICAL]
        if self.excluded_feature_labels:
            result = [f for f in result if not (f.labels & self.excluded_feature_labels)]
        return result

    def _spatial_weight(self, obj: Object, context: ResolutionContext) -> float:
        """Exponential decay weight based on manhattan distance."""
        if obj.last_x is None or obj.last_y is None:
            return 1.0
        dist = abs(int(context.x) - obj.last_x) + abs(int(context.y) - obj.last_y)
        return math.exp(-dist / self.spatial_scale)

    def _temporal_weight(self, obj: Object, context: ResolutionContext) -> float:
        """Exponential decay weight based on tick gap."""
        if obj.last_tick == 0:
            return 1.0
        gap = context.tick - obj.last_tick
        if gap <= 0:
            return 1.0
        return math.exp(-gap / self.temporal_scale)

    @Observability.tracer.start_as_current_span("compute_priors")
    def _compute_priors(
        self, candidates: list[Object], context: ResolutionContext
    ) -> dict[NodeId | str, float]:
        """Uniform priors over candidates and the new-object hypothesis.

        All candidates and the "new" hypothesis get equal prior weight.
        Resolution is driven entirely by the Dirichlet feature likelihoods.

        Returns dict mapping object_id -> log_prior, plus "new" -> log_prior.
        """
        unnormalized: dict[NodeId | str, float] = {}

        for obj in candidates:
            unnormalized[obj.id] = 1.0

        unnormalized["new"] = 1.0

        # Normalize to probabilities
        total = sum(unnormalized.values())
        return {k: math.log(v / total) for k, v in unnormalized.items()}

    def _log_likelihood_for_object(self, obj_id: NodeId, feature_strs: list[str]) -> float:
        """Dirichlet posterior predictive log-likelihood for an existing object."""
        alphas = self._alphas.get(obj_id, {})
        alpha_sum = sum(alphas.values())
        if alpha_sum == 0:
            # No alpha information -- treat as new
            return self._log_likelihood_new(feature_strs)

        # Add prior_alpha for any unseen features in vocab
        vocab_size = len(self._global_vocab)
        total_alpha = alpha_sum + self.prior_alpha * max(0, vocab_size - len(alphas))

        log_ll = 0.0
        for f in feature_strs:
            alpha_j = alphas.get(f, self.prior_alpha)
            log_ll += math.log(alpha_j / total_alpha)

        return log_ll

    def _log_likelihood_new(self, feature_strs: list[str]) -> float:
        """Log-likelihood under the new-object (uniform) model."""
        vocab_size = max(len(self._global_vocab), 1)
        prob = self.prior_alpha / (vocab_size * self.prior_alpha)
        return len(feature_strs) * math.log(prob)

    @Observability.tracer.start_as_current_span("compute_likelihoods")
    def _compute_likelihoods(
        self, candidates: list[Object], feature_strs: list[str]
    ) -> dict[NodeId | str, float]:
        """Dirichlet posterior predictive likelihoods in log-space."""
        log_lls: dict[NodeId | str, float] = {}

        for obj in candidates:
            log_lls[obj.id] = self._log_likelihood_for_object(obj.id, feature_strs)

        log_lls["new"] = self._log_likelihood_new(feature_strs)
        return log_lls

    @Observability.tracer.start_as_current_span("compute_posteriors")
    def _compute_posteriors(
        self,
        log_priors: dict[NodeId | str, float],
        log_likelihoods: dict[NodeId | str, float],
    ) -> dict[NodeId | str, float]:
        """Bayes rule: log_posterior = log_prior + log_likelihood - log_Z."""
        keys = list(log_priors.keys())
        log_joints = [log_priors[k] + log_likelihoods[k] for k in keys]

        log_z = float(logsumexp(log_joints))

        return {k: lj - log_z for k, lj in zip(keys, log_joints)}

    def _decide(
        self,
        candidates: list[Object],
        log_posteriors: dict[NodeId | str, float],
    ) -> Object | None:
        """MAP decision with confidence threshold."""
        # Find best hypothesis
        best_key = max(log_posteriors, key=lambda k: log_posteriors[k])
        best_log_p = log_posteriors[best_key]
        best_p = math.exp(best_log_p)

        # Record telemetry
        self.posterior_max_histogram.record(best_p)
        new_p = math.exp(log_posteriors.get("new", float("-inf")))
        self.new_object_posterior_histogram.record(new_p)

        # Compute margin
        sorted_probs = sorted(log_posteriors.values(), reverse=True)
        margin = (
            math.exp(sorted_probs[0]) - math.exp(sorted_probs[1])
            if len(sorted_probs) > 1
            else best_p
        )
        self.posterior_margin_histogram.record(margin)

        if best_key == "new" or best_p < self.confidence_threshold:
            outcome = "low_confidence" if best_key != "new" else "new_object"
            self.dirichlet_decision_counter.add(1, attributes={"outcome": outcome})
            return None

        # Find the matched object
        for obj in candidates:
            if obj.id == best_key:
                self.dirichlet_decision_counter.add(1, attributes={"outcome": "match"})
                alpha_sum = sum(self._alphas.get(obj.id, {}).values())
                self.alpha_sum_histogram.record(alpha_sum)
                return obj

        return None  # pragma: no cover

    def _update_alphas(self, object_id: NodeId, feature_strs: list[str]) -> None:
        """Increment alpha counts for matched object."""
        if object_id not in self._alphas:
            self._alphas[object_id] = {}
        for f in feature_strs:
            self._alphas[object_id][f] = self._alphas[object_id].get(f, self.prior_alpha) + 1.0

    def initialize_alphas(self, object_id: NodeId, feature_strs: list[str]) -> None:
        """Initialize alpha vector for a newly created object."""
        self._alphas[object_id] = {}
        for f in feature_strs:
            self._alphas[object_id][f] = self.prior_alpha + 1.0
        self._global_vocab.update(feature_strs)

    def _log_decision(
        self,
        outcome: str,
        matched_obj: Object | None,
        feature_strs: list[str],
        candidates: list[Object],
        log_posteriors: dict[NodeId | str, float],
        context: ResolutionContext,
        *,
        reason: str = "",
    ) -> None:
        """Emit an OTel log record describing this resolution decision."""
        # Build posteriors summary sorted by probability
        posteriors_summary: list[tuple[str, float]] = []
        for k, lp in sorted(log_posteriors.items(), key=lambda x: x[1], reverse=True):
            posteriors_summary.append((str(k), round(math.exp(lp), 6)))

        record: dict[str, Any] = {
            "event": "resolution_decision",
            "algorithm": self.name,
            "outcome": outcome,
            "reason": reason,
            "tick": context.tick,
            "x": int(context.x),
            "y": int(context.y),
            "features": feature_strs,
            "num_candidates": len(candidates),
            "posteriors": posteriors_summary,
            "matched_object_id": matched_obj.id if matched_obj is not None else None,
            "vocab_size": len(self._global_vocab),
            "total_objects_tracked": len(self._alphas),
        }

        if matched_obj is not None:
            record["matched_attrs"] = _extract_visual_attrs(matched_obj)
            alphas = self._alphas.get(matched_obj.id, {})
            record["matched_alpha_sum"] = round(sum(alphas.values()), 1)
            record["matched_alpha_count"] = len(alphas)

        if candidates:
            # Build lookup from posteriors for candidate details
            candidate_details = []
            for obj in candidates[:5]:
                lp = log_posteriors.get(obj.id, float("-inf"))
                candidate_details.append({
                    "id": str(obj.id),
                    "probability": round(math.exp(lp), 6),
                    **_extract_visual_attrs(obj),
                })
            record["candidate_details"] = candidate_details

        span_context = otel_trace.get_current_span().get_span_context()
        _otel_logger.emit(
            LogRecord(
                timestamp=time_ns(),
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=json.dumps(record, default=str),
                attributes={"event.name": "roc.resolution.decision"},
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
            )
        )

        # Update live state for dashboard
        from roc.reporting.state import State

        State.get_states().resolution.set(record)


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

        from .sequencer import tick as current_tick

        ctx = ResolutionContext(x=x, y=y, tick=current_tick)
        resolution = ObjectResolutionExpMod.get(default="symmetric-difference")
        o = resolution.resolve(fg.feature_nodes, fg, ctx)

        if o is not None:
            self.resolved_object_counter.add(1, attributes={"new": False})
            o.resolve_count += 1
            logger.debug("object resolved: matched existing id={}", o.uuid)
            if o.last_x is not None and o.last_y is not None:
                dist = abs(int(x) - int(o.last_x)) + abs(int(y) - int(o.last_y))
                self.spatial_distance_histogram.record(dist)
            if o.last_tick > 0:
                gap = current_tick - o.last_tick
                self.temporal_gap_histogram.record(gap)
        else:
            self.resolved_object_counter.add(1, attributes={"new": True})
            o = Object.with_features(fg)
            logger.info(
                "object resolved: NEW object id={} at ({},{}) features={}",
                o.uuid,
                x,
                y,
                [str(f) for f in fg.feature_nodes],
            )
            # Initialize alphas for the new object in the ExpMod
            if hasattr(resolution, "initialize_alphas"):
                feature_strs = [str(f) for f in fg.feature_nodes]
                resolution.initialize_alphas(o.id, feature_strs)

            # Update the resolution state with the new object's ID so the
            # dashboard can track all objects by their node ID.
            from roc.reporting.state import State

            current_res = State.get_states().resolution.val
            if isinstance(current_res, dict):
                current_res["new_object_id"] = o.id

            # Emit a lightweight OTel event linking this step to the new object ID.
            # The resolution _log_decision fires before the object exists, so this
            # supplements it with the node ID for historical queries.
            span_context = otel_trace.get_current_span().get_span_context()
            _otel_logger.emit(
                LogRecord(
                    timestamp=time_ns(),
                    severity_number=SeverityNumber.INFO,
                    severity_text="INFO",
                    body=json.dumps({"new_object_id": o.id}, default=str),
                    attributes={"event.name": "roc.resolution.new_object_id"},
                    trace_id=span_context.trace_id,
                    span_id=span_context.span_id,
                    trace_flags=span_context.trace_flags,
                )
            )

        o.last_x = XLoc(int(x))
        o.last_y = YLoc(int(y))
        o.last_tick = current_tick

        self.obj_res_conn.send(ResolvedObject(object=o, feature_group=fg, x=x, y=y))


class ObjectCache(LRUCache[tuple[XLoc, YLoc], ResolvedObject]):
    """LRU cache mapping screen locations to their most recently resolved objects."""
