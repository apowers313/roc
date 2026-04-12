"""Unit tests for the RunReader unified read facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_reader import RunReader
from roc.reporting.run_registry import RunRegistry
from roc.reporting.run_store import RunStore, StepData
from roc.reporting.step_cache import StepCache


def _seed_run(data_dir: Path, name: str, *, steps: int) -> None:
    """Create a real DuckLake run with ``steps`` rows in the screens table."""
    run_dir = data_dir / name
    store = DuckLakeStore(run_dir, read_only=False)
    try:
        records = [
            {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
            for s in range(1, steps + 1)
        ]
        store.insert("screens", records)
    finally:
        store.close()


def _make_reader(tmp_path: Path, cache: StepCache | None = None) -> RunReader:
    if cache is None:
        cache = StepCache(capacity=100)
    return RunReader(RunRegistry(tmp_path), cache)


class TestRunReaderGetStep:
    def test_get_step_returns_ok_envelope_for_valid_step(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=5)
        reader = _make_reader(tmp_path)
        resp = reader.get_step("r1", 3)
        assert resp.status == "ok"
        assert resp.data is not None
        assert resp.data["step"] == 3
        assert resp.range is not None
        assert resp.range.max == 5

    def test_get_step_returns_run_not_found_for_unknown_run(self, tmp_path: Path) -> None:
        reader = _make_reader(tmp_path)
        resp = reader.get_step("nope", 1)
        assert resp.status == "run_not_found"
        assert resp.data is None

    def test_get_step_returns_out_of_range(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=5)
        reader = _make_reader(tmp_path)
        resp = reader.get_step("r1", 99)
        assert resp.status == "out_of_range"
        assert resp.data is None
        assert resp.range is not None
        assert resp.range.max == 5

    def test_get_step_below_min_returns_out_of_range(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=5)
        reader = _make_reader(tmp_path)
        resp = reader.get_step("r1", 0)
        assert resp.status == "out_of_range"


class TestRunReaderCacheBehavior:
    def test_get_step_cache_hit_skips_store(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=5)
        cache = StepCache(capacity=10)
        # Pre-populate cache with sentinel data
        sentinel = StepData(step=1, game_number=42)
        cache.put("r1", 1, sentinel)
        reader = _make_reader(tmp_path, cache=cache)
        resp = reader.get_step("r1", 1)
        assert resp.status == "ok"
        assert resp.data is not None
        # Cache hit should yield the sentinel game_number=42, not 1
        assert resp.data["game_number"] == 42

    def test_get_step_populates_cache_on_miss(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=2)
        cache = StepCache(capacity=10)
        reader = _make_reader(tmp_path, cache=cache)
        assert cache.get("r1", 1) is None
        reader.get_step("r1", 1)
        assert cache.get("r1", 1) is not None


class TestRunReaderErrorHandling:
    def test_get_step_returns_error_envelope_on_store_exception(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=5)
        cache = StepCache(capacity=10)
        registry = RunRegistry(tmp_path)
        # Force the registry entry to load
        registry.get("r1")
        # Replace the run_store with a mock that raises
        entry = registry._entries["r1"]
        original_run_store = entry.run_store

        class BoomError(Exception):
            pass

        mock_store = MagicMock(spec=RunStore)
        mock_store.get_step_data.side_effect = BoomError("kaboom")
        entry.run_store = mock_store

        reader = RunReader(registry, cache)
        resp = reader.get_step("r1", 2)
        assert resp.status == "error"
        assert resp.error is not None
        assert "BoomError" in resp.error
        # Restore for clean teardown
        entry.run_store = original_run_store


class TestRunReaderGetStepRange:
    def test_get_step_range_returns_range_for_known_run(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=7)
        reader = _make_reader(tmp_path)
        rng = reader.get_step_range("r1")
        assert rng.min == 1
        assert rng.max == 7
        assert rng.tail_growing is False

    def test_get_step_range_raises_file_not_found_for_unknown_run(self, tmp_path: Path) -> None:
        reader = _make_reader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.get_step_range("nope")


class TestRunReaderListRuns:
    def test_list_runs_returns_summaries(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-a", steps=20)
        _seed_run(tmp_path, "run-b", steps=20)
        reader = _make_reader(tmp_path)
        summaries = reader.list_runs(include_all=True)
        names = {s.name for s in summaries}
        assert "run-a" in names
        assert "run-b" in names

    def test_list_runs_empty_dir(self, tmp_path: Path) -> None:
        reader = _make_reader(tmp_path)
        assert reader.list_runs() == []


class TestRunReaderGetHistory:
    def test_get_history_unknown_run_raises(self, tmp_path: Path) -> None:
        reader = _make_reader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.get_history("nope", "graph")

    def test_get_history_returns_list(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=3)
        reader = _make_reader(tmp_path)
        # Even with no events table, should return an empty list
        history = reader.get_history("r1", "graph")
        assert isinstance(history, list)


class TestRunReaderSubscribe:
    def test_subscribe_unknown_run_returns_noop(self, tmp_path: Path) -> None:
        reader = _make_reader(tmp_path)
        unsub = reader.subscribe("nope", lambda step: None)
        unsub()  # must not raise

    def test_subscribe_known_run_invokes_callback(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r1", steps=3)
        registry = RunRegistry(tmp_path)
        reader = RunReader(registry, StepCache())
        received: list[int] = []
        unsub = reader.subscribe("r1", lambda step: received.append(step))
        registry.notify_subscribers("r1", 5)
        assert received == [5]
        unsub()
        registry.notify_subscribers("r1", 6)
        assert received == [5]
