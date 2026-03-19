# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/run_store.py."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from helpers.otel import make_log_record

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.parquet_exporter import ParquetExporter
from roc.reporting.run_store import RunStore, StepData


@pytest.fixture()
def populated_store(tmp_path: Path) -> DuckLakeStore:
    """Create a DuckLakeStore with known test data using ParquetExporter."""
    store = DuckLakeStore(tmp_path)
    exporter = ParquetExporter(store=store, background=False)

    # Game 1: steps 1-10
    exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
    for i in range(10):
        screen_body = json.dumps({"chars": [[65 + i]], "fg": [["#fff"]], "bg": [["#000"]]})
        exporter.export([make_log_record(event_name="roc.screen", body=screen_body)])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.saliency",
                    body=json.dumps({"chars": [[0]], "fg": [["#f00"]], "bg": [["#000"]]}),
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.features",
                    body=f"Delta: {i}",
                    attributes={"feature.count": i + 1},
                )
            ]
        )
        exporter.export([make_log_record(body=f"loguru message {i}")])

    # Game 2: steps 11-15
    exporter.export([make_log_record(event_name="roc.game_start", body="game 2")])
    for i in range(5):
        screen_body = json.dumps({"chars": [[75 + i]], "fg": [["#fff"]], "bg": [["#000"]]})
        exporter.export([make_log_record(event_name="roc.screen", body=screen_body)])

    exporter.shutdown()
    return store


class TestStepCount:
    def test_step_count_returns_max_step(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        assert store.step_count() == 15

    def test_step_count_filtered_by_game(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        assert store.step_count(game_number=1) == 10
        assert store.step_count(game_number=2) == 5


class TestGetStep:
    def test_get_step_returns_matching_rows(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        df = store.get_step(step=5, table="events")
        assert all(df["step"] == 5)

    def test_get_step_returns_empty_for_missing(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        df = store.get_step(step=999, table="screens")
        assert len(df) == 0

    def test_get_step_from_screens(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        df = store.get_step(step=1, table="screens")
        assert len(df) == 1
        assert df.iloc[0]["step"] == 1


class TestListGames:
    def test_list_games_returns_summary(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        games = store.list_games()
        assert len(games) == 2
        assert list(games["game_number"]) == [1, 2]
        assert "steps" in games.columns
        assert "start_ts" in games.columns
        assert "end_ts" in games.columns


class TestListRuns:
    def test_list_runs_finds_directories(self, tmp_path: Path):
        # Create two valid run dirs
        for name in ["run-1", "run-2"]:
            run_dir = tmp_path / name
            dl_store = DuckLakeStore(run_dir)
            exporter = ParquetExporter(store=dl_store, background=False)
            exporter.export([make_log_record(event_name="roc.screen", body="x")])
            exporter.shutdown()

        runs = RunStore.list_runs(tmp_path)
        assert len(runs) == 2
        assert "run-1" in runs
        assert "run-2" in runs

    def test_list_runs_ignores_incomplete(self, tmp_path: Path):
        # Valid run
        valid = tmp_path / "valid-run"
        dl_store = DuckLakeStore(valid)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body="x")])
        exporter.shutdown()

        # Incomplete run (no catalog.duckdb)
        incomplete = tmp_path / "incomplete-run"
        incomplete.mkdir()

        runs = RunStore.list_runs(tmp_path)
        assert runs == ["valid-run"]


class TestGetStepData:
    def test_get_step_data_assembles_all_sources(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        sd = store.get_step_data(1)
        assert isinstance(sd, StepData)
        assert sd.step == 1
        assert sd.game_number == 1
        assert sd.screen is not None
        assert sd.saliency is not None
        assert sd.logs is not None

    def test_get_step_data_handles_missing_tables(self, tmp_path: Path):
        # Create a run with only screens
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.shutdown()

        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.screen is not None
        assert sd.saliency is None
        assert sd.logs is None


class TestConcurrentAccess:
    """Regression: DuckDB connections are not thread-safe.

    Before the fix, concurrent requests from FastAPI's thread pool would
    share a RunStore's DuckDB connection without synchronization, causing
    corrupted query results (e.g. game_number=0 and screen=None for a run
    that actually has data).
    """

    def test_concurrent_get_step_data_returns_correct_results(
        self,
        populated_store: DuckLakeStore,
    ):
        """Multiple threads reading different steps must each get correct data."""
        store = RunStore(populated_store)
        errors: list[str] = []

        def read_step(step: int) -> None:
            sd = store.get_step_data(step)
            if sd.step != step:
                errors.append(f"step {step}: got sd.step={sd.step}")
            if sd.game_number == 0:
                errors.append(f"step {step}: game_number was 0 (corrupted)")
            if step <= 10 and sd.screen is None:
                errors.append(f"step {step}: screen was None (corrupted)")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_step, s) for s in range(1, 16)]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        assert errors == [], f"Concurrent access errors: {errors}"

    def test_concurrent_step_range_returns_consistent_results(
        self,
        populated_store: DuckLakeStore,
    ):
        """Multiple threads calling step_range must all get the same answer."""
        store = RunStore(populated_store)
        results: list[tuple[int, int]] = []
        lock = threading.Lock()

        def read_range() -> None:
            r = store.step_range()
            with lock:
                results.append(r)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_range) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        assert all(r == (1, 15) for r in results), f"Inconsistent ranges: {set(results)}"

    def test_concurrent_read_write(self, tmp_path: Path):
        """Reads and writes on the same DuckLakeStore must not crash."""
        dl_store = DuckLakeStore(tmp_path)
        # Seed initial data
        for i in range(1, 51):
            dl_store.insert("screens", [{"step": i, "game_number": 1, "body": "x"}])

        reader = RunStore(dl_store)
        errors: list[str] = []

        def write_loop() -> None:
            for i in range(51, 101):
                dl_store.insert("screens", [{"step": i, "game_number": 2, "body": "x"}])

        def read_loop() -> None:
            for i in range(1, 51):
                sd = reader.get_step_data(i)
                if sd.game_number == 0:
                    errors.append(f"step {i}: corrupted")

        w = threading.Thread(target=write_loop)
        r = threading.Thread(target=read_loop)
        w.start()
        r.start()
        w.join()
        r.join()

        assert errors == [], f"Concurrent read/write errors: {errors}"


class TestEdgeCases:
    def test_step_count_no_tables(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        assert store.step_count() == 0

    def test_step_range_no_tables(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        assert store.step_range() == (0, 0)

    def test_list_games_no_tables(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        games = store.list_games()
        assert len(games) == 0

    def test_list_runs_nonexistent_dir(self, tmp_path: Path):
        runs = RunStore.list_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_get_step_missing_table(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        df = store.get_step(1, "nonexistent")
        assert len(df) == 0

    def test_parse_body_non_json(self, populated_store: DuckLakeStore):
        """StepData handles non-JSON body strings gracefully."""
        store = RunStore(populated_store)
        sd = store.get_step_data(1)
        assert sd.logs is not None

    def test_step_count_stable(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        assert store.step_count() == 15


class TestGraphHistory:
    def test_returns_empty_when_no_events_table(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        assert store.get_graph_history() == []

    def test_returns_graph_summary_entries(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        for i in range(1, 4):
            exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
            exporter.export(
                [
                    make_log_record(
                        event_name="roc.graphdb.summary",
                        body=json.dumps(
                            {
                                "node_count": i * 10,
                                "node_max": 100,
                                "edge_count": i * 20,
                                "edge_max": 200,
                            }
                        ),
                    )
                ]
            )
        exporter.shutdown()
        store = RunStore(dl_store)
        history = store.get_graph_history()
        assert len(history) == 3
        assert history[0]["node_count"] == 10
        assert history[2]["edge_count"] == 60
        assert all("step" in h for h in history)

    def test_filters_by_game(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        # Game 1
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.graphdb.summary",
                    body=json.dumps(
                        {"node_count": 5, "node_max": 50, "edge_count": 10, "edge_max": 100}
                    ),
                )
            ]
        )
        # Game 2
        exporter.export([make_log_record(event_name="roc.game_start", body="game 2")])
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.graphdb.summary",
                    body=json.dumps(
                        {"node_count": 20, "node_max": 50, "edge_count": 40, "edge_max": 100}
                    ),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        history = store.get_graph_history(game_number=1)
        assert len(history) == 1
        assert history[0]["node_count"] == 5


class TestEventHistory:
    def test_returns_empty_when_no_events_table(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        assert store.get_event_history() == []

    def test_returns_event_summary_entries(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        for i in range(1, 4):
            exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
            exporter.export(
                [
                    make_log_record(
                        event_name="roc.event.summary",
                        body=json.dumps({"roc.perception": i * 2, "roc.attention": i}),
                    )
                ]
            )
        exporter.shutdown()
        store = RunStore(dl_store)
        history = store.get_event_history()
        assert len(history) == 3
        assert history[0]["roc.perception"] == 2
        assert history[2]["roc.attention"] == 3
        assert all("step" in h for h in history)


class TestMetricsHistory:
    def test_returns_empty_when_no_metrics_table(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        store = RunStore(dl_store)
        assert store.get_metrics_history() == []

    def test_returns_all_fields_when_no_filter(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        for i in range(1, 4):
            exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
            exporter.export(
                [
                    make_log_record(
                        event_name="roc.game_metrics",
                        body=json.dumps({"hp": 10 + i, "score": i * 100, "depth": 1}),
                    )
                ]
            )
        exporter.shutdown()
        store = RunStore(dl_store)
        history = store.get_metrics_history()
        assert len(history) == 3
        assert "hp" in history[0]
        assert "score" in history[0]
        assert "depth" in history[0]
        assert history[0]["hp"] == 11

    def test_returns_filtered_fields(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.game_metrics",
                    body=json.dumps({"hp": 15, "score": 200, "depth": 2}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        history = store.get_metrics_history(fields=["hp", "score"])
        assert len(history) == 1
        assert "hp" in history[0]
        assert "score" in history[0]
        assert "depth" not in history[0]

    def test_filters_by_game(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [make_log_record(event_name="roc.game_metrics", body=json.dumps({"hp": 10}))]
        )
        exporter.export([make_log_record(event_name="roc.game_start", body="game 2")])
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [make_log_record(event_name="roc.game_metrics", body=json.dumps({"hp": 20}))]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        history = store.get_metrics_history(game_number=2)
        assert len(history) == 1
        assert history[0]["hp"] == 20


class TestGetStepDataEvents:
    """Test get_step_data event parsing branches for various event types."""

    def test_includes_object_info(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.object",
                    body=json.dumps({"id": 1, "type": "monster"}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.object_info is not None
        assert len(sd.object_info) == 1
        assert sd.object_info[0]["type"] == "monster"

    def test_includes_focus_points(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.focus_points",
                    body=json.dumps({"x": 5, "y": 10}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.focus_points is not None
        assert sd.focus_points[0]["x"] == 5

    def test_includes_attenuation(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.saliency_attenuation",
                    body=json.dumps(
                        {"base": 0.5, "decay": 0.1, "saliency_grid": [], "focus_points": []}
                    ),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.attenuation is not None
        assert sd.attenuation["base"] == 0.5
        # saliency_grid and focus_points should be filtered out
        assert "saliency_grid" not in sd.attenuation

    def test_includes_resolution_decision(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.resolution.decision",
                    body=json.dumps({"outcome": "match", "matched_object_id": 42}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.resolution_metrics is not None
        assert sd.resolution_metrics["outcome"] == "match"

    def test_includes_graph_summary(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.graphdb.summary",
                    body=json.dumps({"node_count": 10, "edge_count": 20}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.graph_summary is not None
        assert sd.graph_summary["node_count"] == 10

    def test_includes_event_summary(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.event.summary",
                    body=json.dumps({"roc.perception": 5}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.event_summary is not None
        assert len(sd.event_summary) == 1
        assert sd.event_summary[0]["roc.perception"] == 5


class TestGetStepDataNewEvents:
    """Regression tests for new pipeline event parsing (task 11)."""

    def test_includes_intrinsics(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.intrinsics",
                    body=json.dumps({"raw": {"hp": 14}, "normalized": {"hp": 0.5}}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.intrinsics is not None
        assert sd.intrinsics["raw"]["hp"] == 14
        assert sd.intrinsics["normalized"]["hp"] == 0.5

    def test_includes_significance(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.significance",
                    body=json.dumps({"significance": 10.5}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.significance is not None
        assert sd.significance == 10.5

    def test_includes_action(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.action",
                    body=json.dumps({"action_id": 4, "action_name": "NE"}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.action_taken is not None
        assert sd.action_taken["action_id"] == 4
        assert sd.action_taken["action_name"] == "NE"

    def test_includes_transform_summary(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.transform_summary",
                    body=json.dumps({"count": 2, "changes": ["a", "b"]}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.transform_summary is not None
        assert sd.transform_summary["count"] == 2
        assert len(sd.transform_summary["changes"]) == 2

    def test_includes_prediction(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.prediction",
                    body=json.dumps({"made": True}),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.prediction is not None
        assert sd.prediction["made"] is True

    def test_includes_message(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export([make_log_record(event_name="roc.message", body="It's a wall.")])
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.message is not None
        assert sd.message == "It's a wall."

    def test_includes_inventory(self, tmp_path: Path):
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        inv = [{"letter": "a", "item": "a sword", "glyph": 1234}]
        exporter.export([make_log_record(event_name="roc.inventory", body=json.dumps(inv))])
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.inventory is not None
        assert len(sd.inventory) == 1
        assert sd.inventory[0]["letter"] == "a"

    def test_attenuation_includes_history_and_focus_points(self, tmp_path: Path):
        """Regression: attenuation filter was excluding history and focus_points."""
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.saliency_attenuation",
                    body=json.dumps(
                        {
                            "flavor": "linear-decline",
                            "peak_count": 5,
                            "saliency_grid": [[1, 2]],
                            "focus_points": [{"x": 10, "y": 20}],
                            "history": [{"x": 1, "y": 2, "tick": 100}],
                        }
                    ),
                )
            ]
        )
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.attenuation is not None
        # saliency_grid should still be filtered out (too large)
        assert "saliency_grid" not in sd.attenuation
        # But history and focus_points should now be included
        assert "history" in sd.attenuation
        assert "focus_points" in sd.attenuation
        assert sd.attenuation["history"][0]["tick"] == 100

    def test_missing_events_leave_fields_none(self, tmp_path: Path):
        """All new fields default to None when no events present."""
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.shutdown()
        store = RunStore(dl_store)
        sd = store.get_step_data(1)
        assert sd.intrinsics is None
        assert sd.significance is None
        assert sd.action_taken is None
        assert sd.transform_summary is None
        assert sd.prediction is None
        assert sd.message is None
        assert sd.inventory is None


class TestStepRange:
    def test_step_range_for_game(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        min_step, max_step = store.step_range(game_number=1)
        assert min_step == 1
        assert max_step == 10

    def test_step_range_overall(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        min_step, max_step = store.step_range()
        assert min_step == 1
        assert max_step == 15

    def test_step_range_game_2(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        min_step, max_step = store.step_range(game_number=2)
        assert min_step == 11
        assert max_step == 15


# ---------------------------------------------------------------------------
# Performance regression tests
# ---------------------------------------------------------------------------

_STEP_TIME_LIMIT = 0.300  # seconds


class TestPerformanceHistorical:
    """Performance: DuckLake catalog step access must be < 300ms."""

    def test_get_step_data_under_limit(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        # Warm up
        store.get_step_data(1)

        times = []
        for step in range(2, 12):
            t0 = time.perf_counter()
            sd = store.get_step_data(step)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            assert sd.step == step

        avg = sum(times) / len(times)
        worst = max(times)
        assert worst < _STEP_TIME_LIMIT, (
            f"Worst step time {worst:.3f}s exceeds {_STEP_TIME_LIMIT}s (avg={avg:.3f}s)"
        )

    def test_step_range_under_limit(self, populated_store: DuckLakeStore):
        store = RunStore(populated_store)
        # Warm up
        store.step_range()

        t0 = time.perf_counter()
        for _ in range(10):
            store.step_range()
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < _STEP_TIME_LIMIT, (
            f"step_range avg {elapsed:.3f}s exceeds {_STEP_TIME_LIMIT}s"
        )


class TestPerformanceLive:
    """Performance: live StepBuffer access must be < 300ms."""

    def test_step_buffer_get_step_under_limit(self):
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=10_000)
        for i in range(1, 1001):
            buf.push(StepData(step=i, game_number=1))

        # Warm up
        buf.get_step(500)

        times = []
        for step in range(100, 1000, 100):
            t0 = time.perf_counter()
            sd = buf.get_step(step)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            assert sd is not None
            assert sd.step == step

        worst = max(times)
        assert worst < _STEP_TIME_LIMIT, (
            f"StepBuffer worst step time {worst:.3f}s exceeds {_STEP_TIME_LIMIT}s"
        )
