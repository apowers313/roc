"""Unified metrics abstraction for dual-emission to OTel and W&B.

RocMetrics provides static methods that route numeric metrics to both
the existing OTel pipeline and WandbReporter. This eliminates duplicate
instrumentation while keeping both backends in sync.
"""

from __future__ import annotations

from typing import Any

from .observability import Observability
from .wandb_reporter import WandbReporter

# Cache for OTel instruments to avoid re-creating them
_histograms: dict[str, Any] = {}
_counters: dict[str, Any] = {}


class RocMetrics:
    """Central dispatch for numeric metrics to OTel + W&B."""

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
        """Record a histogram value to both OTel and W&B.

        Args:
            name: Metric name (e.g. "roc.saliency_attenuation.peak_count").
            value: The value to record.
            attributes: Optional OTel attributes.
            description: Optional description for instrument creation.
        """
        histogram = RocMetrics._get_histogram(name, description)
        histogram.record(value, attributes=attributes)

        # Also send to W&B as a per-step metric
        WandbReporter.log_step({name: value})

    @staticmethod
    def increment_counter(
        name: str,
        amount: int = 1,
        attributes: dict[str, Any] | None = None,
        *,
        description: str = "",
    ) -> None:
        """Increment a counter in both OTel and W&B.

        Args:
            name: Metric name.
            amount: Amount to increment by.
            attributes: Optional OTel attributes.
            description: Optional description for instrument creation.
        """
        counter = RocMetrics._get_counter(name, description)
        counter.add(amount, attributes=attributes)

        # Also send to W&B as a per-step metric
        WandbReporter.log_step({name: amount})

    @staticmethod
    def log_step(data: dict[str, Any]) -> None:
        """Log a dict of metrics to W&B only (not OTel).

        Use this for metrics that don't map to OTel instruments,
        like composite game-state snapshots.

        Args:
            data: Dictionary of metric names to values.
        """
        WandbReporter.log_step(data)

    @staticmethod
    def log_media(key: str, html: str) -> None:
        """Log HTML media content to W&B only.

        Args:
            key: The media key (e.g. "screen", "saliency_map").
            html: HTML string to log.
        """
        WandbReporter.log_media(key, html)

    @staticmethod
    def flush_step() -> None:
        """Flush all buffered step data as a single W&B log call.

        Call once per tick after all ``log_step()`` and ``log_media()``
        calls for that tick are complete.
        """
        WandbReporter.flush_step()
