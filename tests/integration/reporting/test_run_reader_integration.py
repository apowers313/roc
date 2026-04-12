# mypy: disable-error-code="no-untyped-def"

"""Integration tests for the unified-run RunReader.

These exercise the real DuckLake stack: open a writer, push steps,
close, and verify ``RunReader`` can read them back through a fresh
``RunRegistry``. Also covers the cache-hit path during an active
write to confirm Phase 0's pivot (writer's store is the read store).
"""

from __future__ import annotations

from pathlib import Path

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_reader import RunReader
from roc.reporting.run_registry import RunRegistry
from roc.reporting.run_store import StepData
from roc.reporting.step_cache import StepCache


def _seed_run_via_writer(data_dir: Path, name: str, *, steps: int) -> None:
    """Create a run by opening a writer, inserting screen rows, and closing."""
    run_dir = data_dir / name
    store = DuckLakeStore(run_dir, read_only=False)
    try:
        records = [
            {
                "step": s,
                "game_number": 1,
                "timestamp": s * 1000,
                "body": "{}",
            }
            for s in range(1, steps + 1)
        ]
        store.insert("screens", records)
    finally:
        store.close()


class TestInProcessReadAfterFlush:
    def test_read_after_writer_flush(self, tmp_path: Path) -> None:
        """Push 5 steps via a writer, close, verify reader sees all 5."""
        _seed_run_via_writer(tmp_path, "flush-run", steps=5)
        registry = RunRegistry(tmp_path)
        reader = RunReader(registry, StepCache(capacity=100))
        rng = reader.get_step_range("flush-run")
        assert rng.min == 1
        assert rng.max == 5
        assert rng.tail_growing is False
        # Spot-check a step lookup
        resp = reader.get_step("flush-run", 3)
        assert resp.status == "ok"
        assert resp.data is not None


class TestCacheHitDuringActiveWrite:
    def test_cache_hit_during_active_write(self, tmp_path: Path) -> None:
        """Write a step into the cache, then read it via RunReader.

        The read should hit the cache and never invoke the underlying
        store. This is the Phase 0 pivot path: for active runs we never
        open a separate read_only=True store; the cache absorbs hot reads.
        """
        # Seed a small run on disk so RunRegistry has a valid entry.
        _seed_run_via_writer(tmp_path, "cache-run", steps=3)
        registry = RunRegistry(tmp_path)
        cache = StepCache(capacity=100)
        reader = RunReader(registry, cache)

        # Pre-populate the cache with a sentinel value for step 2.
        sentinel = StepData(step=2, game_number=99)
        cache.put("cache-run", 2, sentinel)

        resp = reader.get_step("cache-run", 2)
        assert resp.status == "ok"
        assert resp.data is not None
        # If the cache hit short-circuited, we'll see game_number=99,
        # not the seeded value of 1.
        assert resp.data["game_number"] == 99


class TestRunReaderListIntegration:
    def test_list_runs_includes_seeded_runs(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "list-a", steps=20)
        _seed_run_via_writer(tmp_path, "list-b", steps=20)
        registry = RunRegistry(tmp_path)
        reader = RunReader(registry, StepCache(capacity=100))
        summaries = reader.list_runs(include_all=True)
        names = {s.name for s in summaries}
        assert "list-a" in names
        assert "list-b" in names
