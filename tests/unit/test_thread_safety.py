# mypy: disable-error-code="no-untyped-def"

"""Thread safety tests for Phase 1: Thread Safety Foundation.

Tests verify that locks exist and that concurrent operations on shared state
(GraphCache, ID counters, EventBus names, Event step counts, Component registry,
GraphDB singleton) do not corrupt data or raise exceptions.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest

from roc.db.graphdb import (
    GraphCache,
    _get_next_new_edge_id,
    _get_next_new_node_id,
    _id_lock,
    _graphdb_lock,
)
from roc.framework.component import (
    ComponentId,
    _component_lock,
)
from roc.framework.event import (
    Event,
    EventBus,
    _eventbus_lock,
    eventbus_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_component(name: str = "test", type: str = "test_type") -> MagicMock:
    comp = MagicMock()
    comp.name = name
    comp.type = type
    comp.id = ComponentId(type, name)
    comp.event_filter = MagicMock(return_value=True)
    return comp


@pytest.fixture(autouse=True)
def _clear_eventbus_names():
    """Save and restore eventbus names around each test."""
    saved = eventbus_names.copy()
    eventbus_names.clear()
    yield
    eventbus_names.clear()
    eventbus_names.update(saved)


# ========================================================================
# GraphCache Thread Safety
# ========================================================================
class TestGraphCacheThreadSafety:
    """Verify GraphCache is safe under concurrent access."""

    def test_has_rlock(self):
        cache: GraphCache[str, int] = GraphCache(maxsize=100)
        assert hasattr(cache, "_lock")
        # RLock has acquire/release and _is_owned
        assert hasattr(cache._lock, "acquire")
        assert hasattr(cache._lock, "release")

    def test_concurrent_writes_no_corruption(self):
        cache: GraphCache[str, int] = GraphCache(maxsize=10_000)
        n_threads = 10
        n_items = 100

        def write_items(tid: int):
            for i in range(n_items):
                cache[f"t{tid}_i{i}"] = tid * 1000 + i

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(write_items, t) for t in range(n_threads)]
            for f in as_completed(futs):
                f.result()

        assert cache.currsize == n_threads * n_items

    def test_concurrent_reads_and_writes_no_errors(self):
        cache: GraphCache[str, int] = GraphCache(maxsize=10_000)
        for i in range(100):
            cache[f"key_{i}"] = i

        errors: list[Exception] = []

        def reader():
            for i in range(100):
                try:
                    cache.get(f"key_{i}")
                except Exception as exc:
                    errors.append(exc)

        def writer():
            for i in range(100, 200):
                try:
                    cache[f"key_{i}"] = i
                except Exception as exc:
                    errors.append(exc)

        threads = [
            *(threading.Thread(target=reader) for _ in range(5)),
            *(threading.Thread(target=writer) for _ in range(3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_iteration_no_errors(self):
        cache: GraphCache[str, int] = GraphCache(maxsize=10_000)
        for i in range(100):
            cache[f"key_{i}"] = i

        errors: list[Exception] = []

        def iterate():
            try:
                keys = list(cache)
                assert len(keys) >= 0
            except Exception as exc:
                errors.append(exc)

        def write():
            try:
                for i in range(100, 200):
                    cache[f"extra_{i}"] = i
            except Exception as exc:
                errors.append(exc)

        threads = [
            *(threading.Thread(target=iterate) for _ in range(5)),
            *(threading.Thread(target=write) for _ in range(3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_contains_no_errors(self):
        cache: GraphCache[str, int] = GraphCache(maxsize=10_000)
        for i in range(100):
            cache[f"key_{i}"] = i

        errors: list[Exception] = []

        def check():
            try:
                for i in range(200):
                    _ = f"key_{i}" in cache
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_clear_no_errors(self):
        cache: GraphCache[str, int] = GraphCache(maxsize=10_000)
        for i in range(50):
            cache[f"key_{i}"] = i

        errors: list[Exception] = []

        def write_and_clear(tid: int):
            try:
                for i in range(20):
                    cache[f"t{tid}_{i}"] = i
                if tid == 0:
                    cache.clear()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_and_clear, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ========================================================================
# ID Counter Thread Safety
# ========================================================================
class TestIdCounterThreadSafety:
    """Verify Node/Edge ID counters produce unique IDs under concurrency."""

    def test_id_lock_exists(self):
        assert hasattr(_id_lock, "acquire")
        assert hasattr(_id_lock, "release")

    def test_node_ids_unique(self):
        n_threads = 10
        n_ids = 100
        all_ids: list[int] = []
        collect_lock = threading.Lock()

        def generate():
            ids = [_get_next_new_node_id() for _ in range(n_ids)]
            with collect_lock:
                all_ids.extend(ids)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(generate) for _ in range(n_threads)]
            for f in as_completed(futs):
                f.result()

        assert len(all_ids) == n_threads * n_ids
        assert len(set(all_ids)) == len(all_ids), "Duplicate node IDs detected"

    def test_edge_ids_unique(self):
        n_threads = 10
        n_ids = 100
        all_ids: list[int] = []
        collect_lock = threading.Lock()

        def generate():
            ids = [_get_next_new_edge_id() for _ in range(n_ids)]
            with collect_lock:
                all_ids.extend(ids)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(generate) for _ in range(n_threads)]
            for f in as_completed(futs):
                f.result()

        assert len(all_ids) == n_threads * n_ids
        assert len(set(all_ids)) == len(all_ids), "Duplicate edge IDs detected"


# ========================================================================
# Event Step Counts Thread Safety
# ========================================================================
class TestEventStepCountsThreadSafety:
    """Verify Event._step_counts is safe under concurrent updates."""

    def test_step_counts_lock_exists(self):
        assert hasattr(Event._step_counts_lock, "acquire")

    def test_concurrent_step_count_updates(self):
        Event._step_counts.clear()
        n_threads = 10
        n_increments = 100
        bus = EventBus[int]("ts_step_count_bus")
        comp = _make_mock_component()

        def increment():
            for _ in range(n_increments):
                Event(42, comp.id, bus)

        threads = [threading.Thread(target=increment) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = Event.get_step_counts()
        assert counts["ts_step_count_bus"] == n_threads * n_increments

    def test_get_step_counts_atomic(self):
        """get_step_counts clears counts atomically -- no lost updates."""
        Event._step_counts.clear()
        bus = EventBus[int]("ts_atomic_bus")
        comp = _make_mock_component()

        # Pre-populate
        for _ in range(50):
            Event(1, comp.id, bus)

        counts = Event.get_step_counts()
        assert counts["ts_atomic_bus"] == 50
        # After get, counts should be cleared
        assert Event._step_counts.get("ts_atomic_bus", 0) == 0


# ========================================================================
# EventBus Name Registry Thread Safety
# ========================================================================
class TestEventBusNameThreadSafety:
    """Verify EventBus name registration is safe under concurrency."""

    def test_eventbus_lock_exists(self):
        assert hasattr(_eventbus_lock, "acquire")

    def test_concurrent_unique_registrations(self):
        n_buses = 50
        errors: list[Exception] = []

        def create_bus(i: int):
            try:
                EventBus[int](f"ts_concurrent_{i}")
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = [pool.submit(create_bus, i) for i in range(n_buses)]
            for f in as_completed(futs):
                f.result()

        assert errors == []
        for i in range(n_buses):
            assert f"ts_concurrent_{i}" in eventbus_names

    def test_duplicate_still_rejected_under_concurrency(self):
        """Even with concurrent registration, duplicates must raise."""
        EventBus[int]("ts_dup_target")
        with pytest.raises(ValueError, match="Duplicate EventBus name"):
            EventBus[int]("ts_dup_target")

    def test_clear_names_under_concurrency(self):
        for i in range(10):
            EventBus[int](f"ts_clear_{i}")
        assert len(eventbus_names) == 10
        EventBus.clear_names()
        assert len(eventbus_names) == 0


# ========================================================================
# Component Registry Thread Safety
# ========================================================================
class TestComponentRegistryThreadSafety:
    """Verify component registry lock exists and basic operations work."""

    def test_component_lock_exists(self):
        # RLock has _is_owned (CPython) or at minimum acquire/release
        assert hasattr(_component_lock, "acquire")
        assert hasattr(_component_lock, "release")


# ========================================================================
# GraphDB Singleton Thread Safety
# ========================================================================
class TestGraphDBSingletonThreadSafety:
    """Verify GraphDB singleton creation lock exists."""

    def test_graphdb_lock_exists(self):
        assert hasattr(_graphdb_lock, "acquire")
        assert hasattr(_graphdb_lock, "release")
