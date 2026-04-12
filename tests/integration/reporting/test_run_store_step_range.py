# mypy: disable-error-code="no-untyped-def"

"""Step range regression tests (BUG-H2).

The dashboard's per-game step range was wrong for closed runs:
``GET /api/runs/<run>/step-range?game=N`` always returned the global
``(min, max)`` regardless of ``N``. The bug was in
``RunReader.get_step_range`` -- for closed runs it returned the
cached ``entry.range`` (populated once at load via
``RunStore.step_range(None)``) instead of routing per-game queries
through ``entry.run_store.step_range(game)``.

These tests exercise the real DuckLake stack: open a writer, push
multi-game data, close, and verify ``RunReader.get_step_range`` returns
the correct per-game range. They also cover the closed-run reload path
and a single-game baseline.
"""

from __future__ import annotations

from pathlib import Path

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_reader import RunReader
from roc.reporting.run_registry import RunRegistry
from roc.reporting.run_store import RunStore
from roc.reporting.step_cache import StepCache


def _seed_multi_game_run(
    data_dir: Path,
    name: str,
    *,
    game_specs: list[tuple[int, int, int]],
) -> None:
    """Seed a run with rows in multiple games.

    ``game_specs`` is a list of ``(game_number, first_step, last_step)``
    tuples. Each game's rows go into the screens table.
    """
    run_dir = data_dir / name
    store = DuckLakeStore(run_dir, read_only=False)
    try:
        records = []
        for game_number, first_step, last_step in game_specs:
            for s in range(first_step, last_step + 1):
                records.append(
                    {
                        "step": s,
                        "game_number": game_number,
                        "timestamp": s * 1000,
                        "body": "{}",
                    }
                )
        store.insert("screens", records)
    finally:
        store.close()


def _make_reader(tmp_path: Path) -> RunReader:
    return RunReader(RunRegistry(tmp_path), StepCache(capacity=100))


class TestStepRangePerGame:
    """``RunStore.step_range`` already supports per-game filtering.

    These tests pin the underlying behavior so we know the bug is
    purely in the reader's pass-through, not in the SQL.
    """

    def test_run_store_step_range_returns_per_game_range(self, tmp_path: Path) -> None:
        _seed_multi_game_run(
            tmp_path,
            "multi-run",
            game_specs=[(1, 1, 250), (2, 251, 256)],
        )
        store = DuckLakeStore(tmp_path / "multi-run", read_only=True)
        try:
            rs = RunStore(store)
            assert rs.step_range(None) == (1, 256)
            assert rs.step_range(1) == (1, 250)
            assert rs.step_range(2) == (251, 256)
        finally:
            store.close()


class TestRunReaderGetStepRangeMultiGame:
    """``RunReader.get_step_range`` must respect the ``game`` parameter
    for closed runs. Bug-H2: it used to ignore game and return the
    cached global range from ``entry.range``.
    """

    def test_get_step_range_per_game_for_closed_run(self, tmp_path: Path) -> None:
        """Multi-game closed run: each game must report its own range."""
        _seed_multi_game_run(
            tmp_path,
            "multi-run",
            game_specs=[(1, 1, 250), (2, 251, 256)],
        )
        reader = _make_reader(tmp_path)

        rng_all = reader.get_step_range("multi-run")
        assert rng_all.min == 1
        assert rng_all.max == 256
        assert rng_all.tail_growing is False

        rng_g1 = reader.get_step_range("multi-run", game=1)
        assert rng_g1.min == 1
        assert rng_g1.max == 250
        assert rng_g1.tail_growing is False

        rng_g2 = reader.get_step_range("multi-run", game=2)
        assert rng_g2.min == 251
        assert rng_g2.max == 256
        assert rng_g2.tail_growing is False

    def test_get_step_range_single_game_baseline(self, tmp_path: Path) -> None:
        """Single-game closed run: ``game=1`` and ``game=None`` agree."""
        _seed_multi_game_run(
            tmp_path,
            "single-run",
            game_specs=[(1, 1, 100)],
        )
        reader = _make_reader(tmp_path)

        rng_all = reader.get_step_range("single-run")
        rng_g1 = reader.get_step_range("single-run", game=1)
        assert rng_all == rng_g1
        assert rng_g1.min == 1
        assert rng_g1.max == 100

    def test_get_step_range_game_with_no_rows_returns_zero(self, tmp_path: Path) -> None:
        """Querying a game that has no rows returns ``(0, 0)``."""
        _seed_multi_game_run(
            tmp_path,
            "g1-only",
            game_specs=[(1, 1, 50)],
        )
        reader = _make_reader(tmp_path)

        rng_g2 = reader.get_step_range("g1-only", game=2)
        assert rng_g2.min == 0
        assert rng_g2.max == 0


class TestStepRangeAfterWriterFlush:
    """Closed-run reload path: a run created via the real exporter
    background thread must be readable through ``RunRegistry._load``
    after the writer detaches.
    """

    def test_step_range_after_writer_close_is_persisted(self, tmp_path: Path) -> None:
        """Direct insert + close: a fresh registry must see all steps."""
        _seed_multi_game_run(
            tmp_path,
            "flush-run",
            game_specs=[(1, 1, 10)],
        )
        # New registry process: simulates a server restart between
        # writer close and the next read.
        reader = _make_reader(tmp_path)
        rng = reader.get_step_range("flush-run")
        assert rng.min == 1
        assert rng.max == 10
        assert rng.tail_growing is False
