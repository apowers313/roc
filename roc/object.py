"""Object identification and resolution from visual features."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Collection, NewType, cast

from scipy.special import logsumexp

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
        if not candidates:
            self.decision_counter.add(1, attributes={"outcome": "new_object"})
            return None

        best_obj, best_dist = candidates[0]
        if best_dist <= 1:
            self.decision_counter.add(1, attributes={"outcome": "match"})
            return best_obj
        self.decision_counter.add(1, attributes={"outcome": "new_object"})
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

        return result

    def _find_candidates(self, feature_nodes: Collection[FeatureNode]) -> list[Object]:
        """Graph walk: FeatureNode -> FeatureGroup -> Object."""
        seen: set[NodeId] = set()
        candidates: list[Object] = []

        feature_groups = [
            fg for n in feature_nodes for fg in n.predecessors.select(labels={"FeatureGroup"})
        ]
        objs = [obj for fg in feature_groups for obj in fg.predecessors.select(labels={"Object"})]
        for obj in objs:
            assert isinstance(obj, Object)
            if obj.id not in seen:
                seen.add(obj.id)
                candidates.append(obj)

        return candidates

    def _filter_features(self, feature_nodes: Collection[FeatureNode]) -> list[FeatureNode]:
        """Remove features whose labels intersect excluded_feature_labels."""
        if not self.excluded_feature_labels:
            return list(feature_nodes)
        return [f for f in feature_nodes if not (f.labels & self.excluded_feature_labels)]

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
        """Spatial and temporal exponential decay priors.

        Computes unnormalized prior weights for each candidate (spatial * temporal
        decay) and the "new object" hypothesis (weight 1.0 -- Option C from design,
        where the new-object hypothesis competes purely on likelihood). All weights
        are then normalized so they sum to 1.

        Returns dict mapping object_id -> log_prior, plus "new" -> log_prior.
        """
        unnormalized: dict[NodeId | str, float] = {}

        for obj in candidates:
            spatial_w = self._spatial_weight(obj, context)
            temporal_w = self._temporal_weight(obj, context)
            unnormalized[obj.id] = spatial_w * temporal_w + 1e-300

        # New object prior weight: 1.0 (Option C -- likelihood-driven).
        # The new-object hypothesis has no spatial/temporal context, so it gets
        # a uniform weight. It competes with existing objects purely on likelihood.
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
            if o.last_x is not None and o.last_y is not None:
                dist = abs(int(x) - int(o.last_x)) + abs(int(y) - int(o.last_y))
                self.spatial_distance_histogram.record(dist)
            if o.last_tick > 0:
                gap = current_tick - o.last_tick
                self.temporal_gap_histogram.record(gap)
        else:
            self.resolved_object_counter.add(1, attributes={"new": True})
            o = Object.with_features(fg)
            # Initialize alphas for the new object in the ExpMod
            if hasattr(resolution, "initialize_alphas"):
                feature_strs = [str(f) for f in fg.feature_nodes]
                resolution.initialize_alphas(o.id, feature_strs)

        o.last_x = XLoc(int(x))
        o.last_y = YLoc(int(y))
        o.last_tick = current_tick

        self.obj_res_conn.send(ResolvedObject(object=o, feature_group=fg, x=x, y=y))


class ObjectCache(LRUCache[tuple[XLoc, YLoc], ResolvedObject]):
    """LRU cache mapping screen locations to their most recently resolved objects."""
