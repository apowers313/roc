"""Unified metrics abstraction for OTel emission.

RocMetrics provides static methods that route numeric metrics to the
OTel pipeline (Prometheus/Grafana).
"""

from __future__ import annotations

from typing import Any

from .observability import Observability

# Cache for OTel instruments to avoid re-creating them
_histograms: dict[str, Any] = {}
_counters: dict[str, Any] = {}


class RocMetrics:
    """Central dispatch for numeric metrics to OTel."""

    @staticmethod
    def _get_histogram(name: str, description: str = "") -> Any:
        """Get or create an OTel histogram by name."""
        if name not in _histograms:
            _histograms[name] = Observability.meter.create_histogram(name, description=description)
        return _histograms[name]

    @staticmethod
    def _get_counter(name: str, description: str = "") -> Any:
        """Get or create an OTel counter by name."""
        if name not in _counters:
            _counters[name] = Observability.meter.create_counter(name, description=description)
        return _counters[name]

    @staticmethod
    def record_histogram(
        name: str,
        value: float,
        attributes: dict[str, Any] | None = None,
        *,
        description: str = "",
    ) -> None:
        """Record a histogram value to OTel."""
        histogram = RocMetrics._get_histogram(name, description)
        histogram.record(value, attributes=attributes)

    @staticmethod
    def increment_counter(
        name: str,
        amount: int = 1,
        attributes: dict[str, Any] | None = None,
        *,
        description: str = "",
    ) -> None:
        """Increment a counter in OTel."""
        counter = RocMetrics._get_counter(name, description)
        counter.add(amount, attributes=attributes)

    @staticmethod
    def log_step(data: dict[str, Any]) -> None:
        """Log a dict of metrics (no-op, kept for API compatibility)."""
        pass
