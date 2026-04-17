"""Symmetric-difference object resolution ExpMod."""

from __future__ import annotations

import json
from time import time_ns
from typing import Any, Collection

from opentelemetry import trace as otel_trace
from opentelemetry._logs import LogRecord, SeverityNumber

from roc.db.graphdb import NodeId
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


class SymmetricDifferenceResolution(ObjectResolutionExpMod):
    """Matches objects using symmetric set difference of physical features.

    Walks the reverse feature index to find candidate Objects, computes the symmetric
    difference between physical feature sets (SingleNode, ColorNode, ShapeNode), and
    matches if the best candidate has distance <= 1.
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
        _METRIC_RESOLUTION_DECISION,
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
        observed_attrs = _extract_visual_attrs_from_nodes(feature_nodes)
        log_data = ResolutionLogData(feature_strs, context, observed_attrs)

        if not candidates:
            self.decision_counter.add(1, attributes={"outcome": "new_object"})
            self._log_decision("new_object", None, candidates, log_data)
            return None

        best_obj, best_dist = candidates[0]
        if best_dist <= 1:
            self.decision_counter.add(1, attributes={"outcome": "match"})
            self._log_decision("match", best_obj, candidates, log_data, best_distance=best_dist)
            return best_obj
        self.decision_counter.add(1, attributes={"outcome": "new_object"})
        self._log_decision("new_object", None, candidates, log_data, best_distance=best_dist)
        return None

    def _log_decision(
        self,
        outcome: str,
        matched_obj: Object | None,
        candidates: list[tuple[Object, float]],
        log_data: ResolutionLogData,
        *,
        best_distance: float | None = None,
    ) -> None:
        """Emit an OTel log record describing this resolution decision."""
        record: dict[str, Any] = {
            "event": "resolution_decision",
            "algorithm": self.name,
            "outcome": outcome,
            "tick": log_data.context.tick,
            "x": int(log_data.context.x),
            "y": int(log_data.context.y),
            "features": log_data.feature_strs,
            "num_candidates": len(candidates),
            "matched_object_id": matched_obj.id if matched_obj is not None else None,
        }
        if log_data.observed_attrs:
            record["observed_attrs"] = log_data.observed_attrs
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
                attributes={_OTEL_ATTR_EVENT_NAME: _METRIC_RESOLUTION_DECISION},
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
            )
        )

        from roc.reporting.state import State

        State.get_states().resolution.set(record)
        State.get_states().resolution_cycles.append(record)

    @Observability.tracer.start_as_current_span("find_candidate_objects")
    def _find_candidates(
        self, feature_nodes: Collection[FeatureNode]
    ) -> list[tuple[Object, float]]:
        """Find and rank candidate Objects by feature distance."""
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
        """Symmetric set difference of PHYSICAL features."""
        features_strs: set[str] = {str(f) for f in features if f.kind == FeatureKind.PHYSICAL}
        obj_features: set[str] = {
            str(f)
            for f in obj.features
            if isinstance(f, FeatureNode) and f.kind == FeatureKind.PHYSICAL
        }
        if not features_strs and not obj_features:
            return float("inf")
        return float(len(features_strs ^ obj_features))
