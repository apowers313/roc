"""Resource utilization gauges and counters for leak detection.

Long-running server processes accumulate threads, connections, and memory
through caches that never evict. These observable gauges surface the sizes
of the top offenders to Grafana/Prometheus so regressions are visible
before they cause production incidents. Counters track DuckDB connection
opens vs closes so a leak is detectable as the difference between the two
rates.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING

from opentelemetry.metrics import CallbackOptions, Observation

from roc.framework.logger import logger
from roc.reporting.observability import Observability

if TYPE_CHECKING:
    from roc.reporting.run_registry import RunRegistry


_disable_for_pytest = "pytest" in sys.modules and "PYTEST_VERSION" in os.environ

_registered = False
_sweeper_thread: threading.Thread | None = None
_sweeper_stop = threading.Event()

SWEEPER_INTERVAL_SECONDS = 60.0
SWEEPER_IDLE_TTL_SECONDS = 300.0


def _get_registry_safe() -> RunRegistry | None:
    """Return the singleton RunRegistry if initialized, else None.

    Imported lazily to avoid a circular import with ``api_server``.
    """
    try:
        from roc.reporting.api_server import _get_registry

        return _get_registry()
    except Exception:
        return None


def _count_run_entries(_: CallbackOptions) -> Iterable[Observation]:
    reg = _get_registry_safe()
    if reg is None:
        return []
    with reg._lock:
        total = len(reg._entries)
        with_store = sum(1 for e in reg._entries.values() if e.store is not None)
        growing = sum(1 for e in reg._entries.values() if e.range.tail_growing)
    return [
        Observation(total, {"state": "total"}),
        Observation(with_store, {"state": "open"}),
        Observation(growing, {"state": "growing"}),
    ]


def _count_graph_service_cache(_: CallbackOptions) -> Iterable[Observation]:
    try:
        from roc.reporting import api_server

        svc = getattr(api_server, "_graph_service", None)
    except Exception:
        return []
    if svc is None:
        return [Observation(0)]
    return [Observation(len(svc._cache))]


def _count_graphdb_cache(_: CallbackOptions) -> Iterable[Observation]:
    try:
        from roc.db import graphdb
    except Exception:
        return []
    node_cache = graphdb.node_cache
    edge_cache = graphdb.edge_cache
    return [
        Observation(len(node_cache) if node_cache is not None else 0, {"kind": "node"}),
        Observation(len(edge_cache) if edge_cache is not None else 0, {"kind": "edge"}),
    ]


def _count_feature_to_objects(_: CallbackOptions) -> Iterable[Observation]:
    try:
        from roc.pipeline.object.object import _feature_to_objects
    except Exception:
        return []
    keys = len(_feature_to_objects)
    pairs = sum(len(v) for v in _feature_to_objects.values())
    return [
        Observation(keys, {"stat": "keys"}),
        Observation(pairs, {"stat": "pairs"}),
    ]


def register_resource_gauges() -> None:
    """Register observable gauges for long-lived caches. Idempotent."""
    global _registered
    if _registered:
        return
    meter = Observability.meter
    meter.create_observable_gauge(
        "roc.registry.run_entries",
        callbacks=[_count_run_entries],
        description="RunRegistry cached entries; state=total|open|growing",
    )
    meter.create_observable_gauge(
        "roc.graph_service.cached_graphs",
        callbacks=[_count_graph_service_cache],
        description="NetworkX DiGraphs cached in GraphService (grows without bound)",
    )
    meter.create_observable_gauge(
        "roc.graphdb.cache_size",
        callbacks=[_count_graphdb_cache],
        description="GraphDB Node/Edge LRU cache occupancy",
    )
    meter.create_observable_gauge(
        "roc.pipeline.feature_to_objects",
        callbacks=[_count_feature_to_objects],
        description="Object-resolver reverse index; grows across games",
    )
    _registered = True


def _sweeper_loop() -> None:
    """Background thread: periodically close idle DuckLake stores.

    Runs every ``SWEEPER_INTERVAL_SECONDS``. For each cached run entry that
    owns its store, is not receiving writer pushes, and has been idle for
    more than ``SWEEPER_IDLE_TTL_SECONDS``, close the DuckDB connection and
    null the store. The next ``get()`` lazy-reopens without losing
    subscribers or summary metadata.
    """
    logger.info(
        "Starting DuckLake idle sweeper (interval={}s, ttl={}s)",
        SWEEPER_INTERVAL_SECONDS,
        SWEEPER_IDLE_TTL_SECONDS,
    )
    while not _sweeper_stop.wait(SWEEPER_INTERVAL_SECONDS):
        reg = _get_registry_safe()
        if reg is None:
            continue
        try:
            closed = reg.sweep_idle(idle_ttl=SWEEPER_IDLE_TTL_SECONDS)
            if closed:
                logger.info("Swept {} idle DuckLake stores", closed)
        except Exception:
            logger.opt(exception=True).warning("DuckLake sweeper iteration failed")


def start_idle_sweeper() -> None:
    """Start the background sweeper thread. Idempotent and safe under pytest.

    Skipped under pytest so test suites don't leak background threads.
    """
    global _sweeper_thread
    if _disable_for_pytest:
        return
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_stop.clear()
    _sweeper_thread = threading.Thread(
        target=_sweeper_loop,
        name="ducklake-sweeper",
        daemon=True,
    )
    _sweeper_thread.start()


def stop_idle_sweeper(timeout: float = 5.0) -> None:
    """Signal the sweeper to stop and wait for it to exit. For tests."""
    global _sweeper_thread
    _sweeper_stop.set()
    if _sweeper_thread is not None:
        _sweeper_thread.join(timeout=timeout)
        _sweeper_thread = None
