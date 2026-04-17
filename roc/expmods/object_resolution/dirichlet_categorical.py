"""Dirichlet-Categorical Bayesian object resolution ExpMod."""

from __future__ import annotations

import json
import math
from time import time_ns
from typing import Any, Collection, cast

from opentelemetry import trace as otel_trace
from opentelemetry._logs import LogRecord, SeverityNumber
from loguru import logger
from scipy.special import logsumexp

from roc.db.graphdb import NodeId
from roc.framework.expmod import ExpModConfig
from roc.perception.base import FeatureKind, FeatureNode
from roc.pipeline.object.object import (
    FeatureGroup,
    Object,
    ObjectResolutionExpMod,
    ResolutionContext,
    ResolutionLogData,
    _METRIC_RESOLUTION_DECISION,
    _OTEL_ATTR_EVENT_NAME,
    _extract_visual_attrs,
    _extract_visual_attrs_from_nodes,
    _feature_to_objects,
    _otel_logger,
)
from roc.reporting.observability import Observability


class DirichletCategoricalConfig(ExpModConfig):
    """Configuration for Dirichlet-Categorical object resolution.

    NOTE: with only 3 physical features per cell, 2-of-3 overlap is a strong enough
    signal that posteriors top out around 0.65, so ``confidence_threshold=0.7`` means
    nothing ever matches. The algorithm is currently not suitable for production;
    tune the threshold and/or the prior when adding additional physical features.
    """

    prior_alpha: float = 1.0
    spatial_scale: float = 3.0
    temporal_scale: float = 50.0
    confidence_threshold: float = 0.7
    excluded_feature_labels: set[str] = set()


class DirichletCategoricalResolution(ObjectResolutionExpMod):
    """Bayesian object resolution using a Dirichlet-Categorical model.

    Computes posterior probabilities over object identities using spatial/temporal
    priors and Dirichlet posterior predictive likelihoods.
    """

    name = "dirichlet-categorical"
    config_schema = DirichletCategoricalConfig

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

    def _cfg(self) -> DirichletCategoricalConfig:
        assert isinstance(self.config, DirichletCategoricalConfig)
        return self.config

    @property
    def prior_alpha(self) -> float:
        return self._cfg().prior_alpha

    @prior_alpha.setter
    def prior_alpha(self, value: float) -> None:
        self._cfg().prior_alpha = value

    @property
    def spatial_scale(self) -> float:
        return self._cfg().spatial_scale

    @spatial_scale.setter
    def spatial_scale(self, value: float) -> None:
        self._cfg().spatial_scale = value

    @property
    def temporal_scale(self) -> float:
        return self._cfg().temporal_scale

    @temporal_scale.setter
    def temporal_scale(self, value: float) -> None:
        self._cfg().temporal_scale = value

    @property
    def confidence_threshold(self) -> float:
        return self._cfg().confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._cfg().confidence_threshold = value

    @property
    def excluded_feature_labels(self) -> set[str]:
        return self._cfg().excluded_feature_labels

    @excluded_feature_labels.setter
    def excluded_feature_labels(self, value: set[str]) -> None:
        self._cfg().excluded_feature_labels = value

    def resolve(
        self,
        feature_nodes: Collection[FeatureNode],
        feature_group: FeatureGroup,
        context: ResolutionContext,
    ) -> Object | None:
        """Full Bayesian resolution pipeline."""
        candidates = self._find_candidates(feature_nodes)

        active_features = self._filter_features(feature_nodes)
        feature_strs = [str(f) for f in active_features]

        logger.debug(
            "[dirichlet] tick={} pos=({},{}) | all_features={} physical={} "
            "feature_strs={} | candidates={} vocab={} tracked_objects={}",
            context.tick,
            context.x,
            context.y,
            len(list(feature_nodes)),
            len(active_features),
            feature_strs,
            len(candidates),
            len(self._global_vocab),
            len(self._alphas),
        )

        self._global_vocab.update(feature_strs)

        observed_attrs = _extract_visual_attrs_from_nodes(feature_nodes)
        log_data = ResolutionLogData(feature_strs, context, observed_attrs)

        if not candidates:
            self.dirichlet_decision_counter.add(1, attributes={"outcome": "new_object"})
            log_data.log_posteriors = {}
            self._log_decision("new_object", None, [], log_data, reason="no_candidates")
            logger.info(
                "[dirichlet] tick={} -> NEW (no candidates) | features={}",
                context.tick,
                feature_strs,
            )
            return None

        log_priors = self._compute_priors(candidates, context)
        log_likelihoods = self._compute_likelihoods(candidates, feature_strs)
        log_posteriors = self._compute_posteriors(log_priors, log_likelihoods)
        self._log_posterior_breakdown(log_posteriors, log_likelihoods, log_priors)

        result = self._decide(candidates, log_posteriors)
        self._log_resolution_outcome(result, candidates, feature_strs, log_posteriors, context)

        log_data.log_posteriors = log_posteriors
        self._log_decision(
            "match" if result is not None else "new_object",
            result,
            candidates,
            log_data,
        )

        return result

    def _log_posterior_breakdown(
        self,
        log_posteriors: dict[NodeId | str, float],
        log_likelihoods: dict[NodeId | str, float],
        log_priors: dict[NodeId | str, float],
    ) -> None:
        for k in sorted(log_posteriors, key=lambda k: log_posteriors[k], reverse=True):
            label = "new" if k == "new" else f"obj:{k}"
            lp = log_posteriors[k]
            ll = log_likelihoods.get(k, float("-inf"))
            prior = log_priors.get(k, float("-inf"))
            alphas = self._alphas.get(cast(NodeId, k), {}) if k != "new" else {}
            alpha_sum = round(sum(alphas.values()), 1) if alphas else 0
            logger.debug(
                "[dirichlet]   {} => posterior={:.4f} (p={:.4f}) "
                "ll={:.4f} prior={:.4f} alpha_sum={}",
                label,
                lp,
                math.exp(lp),
                ll,
                prior,
                alpha_sum,
            )

    def _log_resolution_outcome(
        self,
        result: Object | None,
        candidates: list[Object],
        feature_strs: list[str],
        log_posteriors: dict[NodeId | str, float],
        context: ResolutionContext,
    ) -> None:
        if result is not None:
            self._update_alphas(result.id, feature_strs)
            logger.info(
                "[dirichlet] tick={} -> MATCH obj:{} (p={:.4f}) | features={}",
                context.tick,
                result.id,
                math.exp(log_posteriors[result.id]),
                feature_strs,
            )
        else:
            best_key = max(log_posteriors, key=lambda k: log_posteriors[k])
            best_p = math.exp(log_posteriors[best_key])
            logger.info(
                "[dirichlet] tick={} -> NEW (best={} p={:.4f} thresh={}) "
                "| candidates={} features={}",
                context.tick,
                "new" if best_key == "new" else f"obj:{best_key}",
                best_p,
                self.confidence_threshold,
                len(candidates),
                feature_strs,
            )

    def _find_candidates(self, feature_nodes: Collection[FeatureNode]) -> list[Object]:
        """Find candidates using the reverse index, PHYSICAL features only."""
        seen: set[NodeId] = set()
        candidates: list[Object] = []

        for fn in feature_nodes:
            if fn.kind != FeatureKind.PHYSICAL:
                continue
            obj_ids = _feature_to_objects.get(fn.id, ())
            self._log_index_lookup(fn, obj_ids)
            for obj_id in obj_ids:
                if obj_id not in seen:
                    seen.add(obj_id)
                    obj = Object.get(obj_id)
                    assert isinstance(obj, Object)
                    candidates.append(obj)

        return candidates

    def _log_index_lookup(self, fn: FeatureNode, obj_ids: set[NodeId] | tuple[()]) -> None:
        if obj_ids:
            logger.debug(
                "[dirichlet] reverse-index HIT: fn.id={} ({}) -> {} objects",
                fn.id,
                str(fn),
                len(obj_ids),
            )
        else:
            logger.debug(
                "[dirichlet] reverse-index MISS: fn.id={} ({}) not in index "
                "(index has {} feature entries)",
                fn.id,
                str(fn),
                len(_feature_to_objects),
            )

    def _filter_features(self, feature_nodes: Collection[FeatureNode]) -> list[FeatureNode]:
        result = [f for f in feature_nodes if f.kind == FeatureKind.PHYSICAL]
        if self.excluded_feature_labels:
            result = [f for f in result if not (f.labels & self.excluded_feature_labels)]
        return result

    def _spatial_weight(self, obj: Object, context: ResolutionContext) -> float:
        if obj.last_x is None or obj.last_y is None:
            return 1.0
        dist = abs(int(context.x) - obj.last_x) + abs(int(context.y) - obj.last_y)
        return math.exp(-dist / self.spatial_scale)

    def _temporal_weight(self, obj: Object, context: ResolutionContext) -> float:
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
        unnormalized: dict[NodeId | str, float] = {}

        for obj in candidates:
            unnormalized[obj.id] = 1.0

        unnormalized["new"] = 1.0

        total = sum(unnormalized.values())
        return {k: math.log(v / total) for k, v in unnormalized.items()}

    def _log_likelihood_for_object(self, obj_id: NodeId, feature_strs: list[str]) -> float:
        alphas = self._alphas.get(obj_id, {})
        alpha_sum = sum(alphas.values())
        if alpha_sum == 0:
            logger.debug("[dirichlet] ll obj:{} alpha_sum=0, treating as new", obj_id)
            return self._log_likelihood_new(feature_strs)

        vocab_size = len(self._global_vocab)
        total_alpha = alpha_sum + self.prior_alpha * max(0, vocab_size - len(alphas))

        log_ll = 0.0
        for f in feature_strs:
            alpha_j = alphas.get(f, self.prior_alpha)
            contrib = math.log(alpha_j / total_alpha)
            log_ll += contrib
            logger.debug(
                "[dirichlet]   ll obj:{} feature='{}' alpha_j={} total_alpha={:.1f} "
                "contrib={:.4f} (in_alphas={})",
                obj_id,
                f,
                alpha_j,
                total_alpha,
                contrib,
                f in alphas,
            )

        logger.debug(
            "[dirichlet] ll obj:{} => {:.4f} | alpha_sum={:.1f} vocab={} "
            "obj_alpha_keys={} observed={}",
            obj_id,
            log_ll,
            alpha_sum,
            vocab_size,
            len(alphas),
            len(feature_strs),
        )
        return log_ll

    def _log_likelihood_new(self, feature_strs: list[str]) -> float:
        vocab_size = max(len(self._global_vocab), 1)
        prob = self.prior_alpha / (vocab_size * self.prior_alpha)
        log_ll = len(feature_strs) * math.log(prob)
        logger.debug(
            "[dirichlet] ll new => {:.4f} | vocab={} n_features={} per_feature={:.4f}",
            log_ll,
            vocab_size,
            len(feature_strs),
            math.log(prob),
        )
        return log_ll

    @Observability.tracer.start_as_current_span("compute_likelihoods")
    def _compute_likelihoods(
        self, candidates: list[Object], feature_strs: list[str]
    ) -> dict[NodeId | str, float]:
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
        keys = list(log_priors.keys())
        log_joints = [log_priors[k] + log_likelihoods[k] for k in keys]
        log_z = float(logsumexp(log_joints))
        return {k: lj - log_z for k, lj in zip(keys, log_joints)}

    def _decide(
        self,
        candidates: list[Object],
        log_posteriors: dict[NodeId | str, float],
    ) -> Object | None:
        best_key = max(log_posteriors, key=lambda k: log_posteriors[k])
        best_log_p = log_posteriors[best_key]
        best_p = math.exp(best_log_p)

        self.posterior_max_histogram.record(best_p)
        new_p = math.exp(log_posteriors.get("new", float("-inf")))
        self.new_object_posterior_histogram.record(new_p)

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

        for obj in candidates:
            if obj.id == best_key:
                self.dirichlet_decision_counter.add(1, attributes={"outcome": "match"})
                alpha_sum = sum(self._alphas.get(obj.id, {}).values())
                self.alpha_sum_histogram.record(alpha_sum)
                return obj

        return None  # pragma: no cover

    def _update_alphas(self, object_id: NodeId, feature_strs: list[str]) -> None:
        if object_id not in self._alphas:
            self._alphas[object_id] = {}
        for f in feature_strs:
            self._alphas[object_id][f] = self._alphas[object_id].get(f, self.prior_alpha) + 1.0

    def initialize_alphas(self, object_id: NodeId, feature_strs: list[str]) -> None:
        self._alphas[object_id] = {}
        for f in feature_strs:
            self._alphas[object_id][f] = self.prior_alpha + 1.0
        self._global_vocab.update(feature_strs)
        logger.debug(
            "[dirichlet] initialized alphas for obj:{} with {} features: {}",
            object_id,
            len(feature_strs),
            feature_strs,
        )

    def _log_decision(
        self,
        outcome: str,
        matched_obj: Object | None,
        candidates: list[Object],
        log_data: ResolutionLogData,
        *,
        reason: str = "",
    ) -> None:
        log_posteriors = log_data.log_posteriors or {}
        posteriors_summary: list[tuple[str, float]] = []
        for k, lp in sorted(log_posteriors.items(), key=lambda x: x[1], reverse=True):
            posteriors_summary.append((str(k), round(math.exp(lp), 6)))

        record: dict[str, Any] = {
            "event": "resolution_decision",
            "algorithm": self.name,
            "outcome": outcome,
            "reason": reason,
            "tick": log_data.context.tick,
            "x": int(log_data.context.x),
            "y": int(log_data.context.y),
            "features": log_data.feature_strs,
            "num_candidates": len(candidates),
            "posteriors": posteriors_summary,
            "matched_object_id": matched_obj.id if matched_obj is not None else None,
            "vocab_size": len(self._global_vocab),
            "total_objects_tracked": len(self._alphas),
        }
        if log_data.observed_attrs:
            record["observed_attrs"] = log_data.observed_attrs

        if matched_obj is not None:
            record["matched_attrs"] = _extract_visual_attrs(matched_obj)
            alphas = self._alphas.get(matched_obj.id, {})
            record["matched_alpha_sum"] = round(sum(alphas.values()), 1)
            record["matched_alpha_count"] = len(alphas)

        if candidates:
            candidate_details = []
            for obj in candidates[:5]:
                lp = log_posteriors.get(obj.id, float("-inf"))
                candidate_details.append(
                    {
                        "id": str(obj.id),
                        "probability": round(math.exp(lp), 6),
                        **_extract_visual_attrs(obj),
                    }
                )
            record["candidate_details"] = candidate_details

        span_context = otel_trace.get_current_span().get_span_context()
        _otel_logger.emit(
            LogRecord(
                timestamp=time_ns(),
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=json.dumps(record, default=str),
                attributes={_OTEL_ATTR_EVENT_NAME: _METRIC_RESOLUTION_DECISION},
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
            )
        )

        from roc.reporting.state import State

        State.get_states().resolution.set(record)
        State.get_states().resolution_cycles.append(record)
