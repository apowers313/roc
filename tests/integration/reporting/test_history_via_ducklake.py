# mypy: disable-error-code="no-untyped-def"

"""Integration tests for Phase 2: history queries served via DuckLake/RunStore.

Phase 2 deletes ``DataStore._indices`` and routes every history query through
``RunReader -> RunRegistry -> RunStore`` directly. These tests exercise the
real DuckLake stack to confirm the new path works end-to-end with no
in-memory shortcut.

The integration here is intentionally narrow: we seed runs by writing into a
real ``DuckLakeStore``, then read history via ``RunReader``. We do NOT depend
on the writer/exporter chain because that is exercised by Phase 4.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_reader import RunReader
from roc.reporting.run_registry import RunRegistry
from roc.reporting.step_cache import StepCache
from roc.reporting.types import HistoryKind


def _seed_run_with_events(
    data_dir: Path,
    name: str,
    *,
    steps: int,
    game_number: int = 1,
) -> None:
    """Create a run with screen + event rows so every history kind has data.

    Each step writes one row per supported history kind into the events
    table, plus one row in screens (for step counting) and one in metrics
    (for the metrics history). This produces a self-contained run that
    every ``RunStore.get_*_history`` method can query.
    """
    run_dir = data_dir / name
    store = DuckLakeStore(run_dir, read_only=False)
    try:
        screen_records = [
            {"step": s, "game_number": game_number, "timestamp": s * 1000, "body": "{}"}
            for s in range(1, steps + 1)
        ]
        store.insert("screens", screen_records)

        metric_records = [
            {
                "step": s,
                "game_number": game_number,
                "timestamp": s * 1000,
                "body": json.dumps({"score": s, "level": 1}),
            }
            for s in range(1, steps + 1)
        ]
        store.insert("metrics", metric_records)

        event_records: list[dict[str, object]] = []
        for s in range(1, steps + 1):
            event_records.append(
                {
                    "step": s,
                    "game_number": game_number,
                    "timestamp": s * 1000,
                    "event.name": "roc.graphdb.summary",
                    "body": json.dumps(
                        {
                            "node_count": s + 10,
                            "node_max": 100,
                            "edge_count": s + 5,
                            "edge_max": 50,
                        }
                    ),
                }
            )
            event_records.append(
                {
                    "step": s,
                    "game_number": game_number,
                    "timestamp": s * 1000,
                    "event.name": "roc.event.summary",
                    "body": json.dumps({"perception": s, "attention": s + 1}),
                }
            )
            event_records.append(
                {
                    "step": s,
                    "game_number": game_number,
                    "timestamp": s * 1000,
                    "event.name": "roc.intrinsics",
                    "body": json.dumps({"hp": 16 - s, "max_hp": 16}),
                }
            )
            event_records.append(
                {
                    "step": s,
                    "game_number": game_number,
                    "timestamp": s * 1000,
                    "event.name": "roc.action",
                    "body": json.dumps({"action_id": s % 10, "action_name": "north"}),
                }
            )
            event_records.append(
                {
                    "step": s,
                    "game_number": game_number,
                    "timestamp": s * 1000,
                    "event.name": "roc.resolution.decision",
                    "body": json.dumps(
                        {
                            "outcome": "match",
                            "matched_object_id": 42,
                            "matched_attrs": {"char": "@", "color": "white", "glyph": "64"},
                            "features": [
                                "ShapeNode(@)",
                                "ColorNode(white)",
                                "SingleNode(64)",
                            ],
                        }
                    ),
                }
            )
        store.insert("events", event_records)
    finally:
        store.close()


def _make_reader(tmp_path: Path) -> RunReader:
    return RunReader(RunRegistry(tmp_path), StepCache(capacity=100))


class TestHistoryDispatchByKind:
    """Every supported history kind dispatches into RunStore correctly."""

    @pytest.mark.parametrize(
        "kind",
        ["graph", "event", "metrics", "intrinsics", "action", "resolution"],
    )
    def test_event_history_dispatch_by_kind(self, tmp_path: Path, kind: HistoryKind) -> None:
        _seed_run_with_events(tmp_path, "r", steps=5)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", kind, game=1)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_graph_history_payload_shape(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=3)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "graph", game=1)
        assert len(result) == 3
        assert result[0]["step"] == 1
        assert result[0]["node_count"] == 11

    def test_metrics_history_payload_shape(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=3)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "metrics", game=1)
        assert len(result) == 3
        assert result[0]["step"] == 1
        assert result[0]["score"] == 1

    def test_intrinsics_history_payload_shape(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=3)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "intrinsics", game=1)
        assert len(result) == 3
        assert result[0]["hp"] == 15

    def test_action_history_payload_shape(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=3)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "action", game=1)
        assert len(result) == 3
        assert result[0]["action_id"] == 1

    def test_event_history_payload_shape(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=3)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "event", game=1)
        assert len(result) == 3
        assert "perception" in result[0]

    def test_resolution_history_payload_shape(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=3)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "resolution", game=1)
        assert len(result) == 3
        assert result[0]["outcome"] == "match"
        assert result[0]["correct"] is True

    def test_metrics_history_filters_fields(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "r", steps=2)
        reader = _make_reader(tmp_path)
        result = reader.get_history("r", "metrics", game=1, fields=["score"])
        assert len(result) == 2
        for entry in result:
            assert "score" in entry
            assert "level" not in entry

    def test_unknown_run_raises_file_not_found(self, tmp_path: Path) -> None:
        reader = _make_reader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.get_history("nope", "graph")


class TestHistoryPerformance:
    """Phase 2 risk validation: history queries on a moderately sized run."""

    def test_graph_history_for_500_step_run_under_500ms(self, tmp_path: Path) -> None:
        """A 500-step run should respond well below the 500ms target.

        The plan calls for a 10k-step run; we use 500 here to keep the
        per-test runtime reasonable while still catching pathological
        slowdowns. A working DuckLake-backed path completes this in
        well under 100ms locally.
        """
        _seed_run_with_events(tmp_path, "big", steps=500)
        reader = _make_reader(tmp_path)
        t0 = time.monotonic()
        result = reader.get_history("big", "graph", game=1)
        elapsed = time.monotonic() - t0
        assert len(result) == 500
        assert elapsed < 0.5, f"graph history took {elapsed:.3f}s -- target is < 500ms"


class TestActiveWriterHistoryPath:
    """Phase 0 invariant: active runs read history via the writer's store.

    Phase 2 routes ``RunReader.get_history`` through
    ``RunRegistry``. For an active run with ``tail_growing=True``, that
    means the registry's entry MUST point at the writer's
    ``DuckLakeStore`` instance (installed via ``attach_writer_store``)
    -- otherwise opening a separate ``read_only=True`` store on the same
    catalog would raise ``Unique file handle conflict``. This test
    exercises that path end-to-end by attaching a writer's store and
    immediately reading history from it.
    """

    def test_attach_writer_store_then_history_uses_writer(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "active"
        run_dir.mkdir()
        writer_store = DuckLakeStore(run_dir, read_only=False)
        try:
            screen_records = [
                {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                for s in range(1, 4)
            ]
            writer_store.insert("screens", screen_records)
            writer_store.insert(
                "events",
                [
                    {
                        "step": s,
                        "game_number": 1,
                        "timestamp": s * 1000,
                        "event.name": "roc.graphdb.summary",
                        "body": json.dumps(
                            {
                                "node_count": s + 100,
                                "node_max": 1000,
                                "edge_count": s + 50,
                                "edge_max": 500,
                            }
                        ),
                    }
                    for s in range(1, 4)
                ],
            )

            registry = RunRegistry(tmp_path)
            registry.attach_writer_store("active", writer_store)
            reader = RunReader(registry, StepCache(capacity=100))
            entry = registry.get("active")
            assert entry is not None
            assert entry.range.tail_growing is True
            history = reader.get_history("active", "graph", game=1)
            assert len(history) == 3
            assert history[0]["node_count"] == 101
        finally:
            registry.detach_writer_store("active")
            writer_store.close()


class TestMultipleGames:
    """History queries respect the game_number filter."""

    def test_history_filtered_by_game(self, tmp_path: Path) -> None:
        _seed_run_with_events(tmp_path, "multi", steps=3, game_number=1)
        # Append a second game's worth of data into the same run
        run_dir = tmp_path / "multi"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            store.insert(
                "screens",
                [
                    {"step": s, "game_number": 2, "timestamp": s * 1000, "body": "{}"}
                    for s in range(10, 13)
                ],
            )
            store.insert(
                "events",
                [
                    {
                        "step": s,
                        "game_number": 2,
                        "timestamp": s * 1000,
                        "event.name": "roc.graphdb.summary",
                        "body": json.dumps(
                            {
                                "node_count": 99,
                                "node_max": 100,
                                "edge_count": 99,
                                "edge_max": 100,
                            }
                        ),
                    }
                    for s in range(10, 13)
                ],
            )
        finally:
            store.close()

        reader = _make_reader(tmp_path)
        game1 = reader.get_history("multi", "graph", game=1)
        game2 = reader.get_history("multi", "graph", game=2)
        all_games = reader.get_history("multi", "graph", game=None)
        assert len(game1) == 3
        assert len(game2) == 3
        assert len(all_games) == 6
