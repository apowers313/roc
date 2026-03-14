# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/run_store.py."""

import json
from pathlib import Path

import pytest
from helpers.otel import make_log_record

from roc.reporting.parquet_exporter import ParquetExporter
from roc.reporting.run_store import RunStore, StepData


@pytest.fixture()
def populated_run_dir(tmp_path: Path) -> Path:
    """Create a run directory with known test data using ParquetExporter."""
    exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)

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
    return tmp_path


class TestStepCount:
    def test_step_count_returns_max_step(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        assert store.step_count() == 15

    def test_step_count_filtered_by_game(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        assert store.step_count(game_number=1) == 10
        assert store.step_count(game_number=2) == 5


class TestGetStep:
    def test_get_step_returns_matching_rows(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        df = store.get_step(step=5, table="events")
        assert all(df["step"] == 5)

    def test_get_step_returns_empty_for_missing(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        df = store.get_step(step=999, table="screens")
        assert len(df) == 0

    def test_get_step_from_screens(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        df = store.get_step(step=1, table="screens")
        assert len(df) == 1
        assert df.iloc[0]["step"] == 1


class TestListGames:
    def test_list_games_returns_summary(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        games = store.list_games()
        assert len(games) == 2
        assert list(games["game_number"]) == [1, 2]
        assert "steps" in games.columns
        assert "start_ts" in games.columns
        assert "end_ts" in games.columns


class TestListRuns:
    def test_list_runs_finds_directories(self, tmp_path: Path):
        # Create two valid run dirs
        run1 = tmp_path / "run-1"
        run1.mkdir()
        exporter1 = ParquetExporter(run_dir=run1, flush_interval=1)
        exporter1.export([make_log_record(event_name="roc.screen", body="x")])
        exporter1.shutdown()

        run2 = tmp_path / "run-2"
        run2.mkdir()
        exporter2 = ParquetExporter(run_dir=run2, flush_interval=1)
        exporter2.export([make_log_record(event_name="roc.screen", body="x")])
        exporter2.shutdown()

        runs = RunStore.list_runs(tmp_path)
        assert len(runs) == 2
        assert "run-1" in runs
        assert "run-2" in runs

    def test_list_runs_ignores_incomplete(self, tmp_path: Path):
        # Valid run
        valid = tmp_path / "valid-run"
        valid.mkdir()
        exporter = ParquetExporter(run_dir=valid, flush_interval=1)
        exporter.export([make_log_record(event_name="roc.screen", body="x")])
        exporter.shutdown()

        # Incomplete run (no screens.parquet)
        incomplete = tmp_path / "incomplete-run"
        incomplete.mkdir()
        (incomplete / "events.parquet").touch()

        runs = RunStore.list_runs(tmp_path)
        assert runs == ["valid-run"]


class TestGetStepData:
    def test_get_step_data_assembles_all_sources(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        sd = store.get_step_data(1)
        assert isinstance(sd, StepData)
        assert sd.step == 1
        assert sd.game_number == 1
        assert sd.screen is not None
        assert sd.saliency is not None
        assert sd.logs is not None

    def test_get_step_data_handles_missing_tables(self, tmp_path: Path):
        # Create a run with only screens
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        exporter.export([make_log_record(event_name="roc.screen", body='{"chars":[]}')])
        exporter.shutdown()

        store = RunStore(tmp_path)
        sd = store.get_step_data(1)
        assert sd.screen is not None
        assert sd.saliency is None
        assert sd.logs is None


class TestEdgeCases:
    def test_step_count_no_screens_file(self, tmp_path: Path):
        store = RunStore(tmp_path)
        assert store.step_count() == 0

    def test_step_range_no_screens_file(self, tmp_path: Path):
        store = RunStore(tmp_path)
        assert store.step_range() == (0, 0)

    def test_list_games_no_screens_file(self, tmp_path: Path):
        store = RunStore(tmp_path)
        games = store.list_games()
        assert len(games) == 0

    def test_list_runs_nonexistent_dir(self, tmp_path: Path):
        runs = RunStore.list_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_get_step_missing_table_file(self, tmp_path: Path):
        store = RunStore(tmp_path)
        df = store.get_step(1, "nonexistent")
        assert len(df) == 0

    def test_parse_body_non_json(self, populated_run_dir: Path):
        """StepData handles non-JSON body strings gracefully."""
        store = RunStore(populated_run_dir)
        # Logs have non-JSON bodies -- check that they're populated
        sd = store.get_step_data(1)
        assert sd.logs is not None

    def test_reload_is_noop(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        store.reload()  # should not raise
        assert store.step_count() == 15


class TestStepRange:
    def test_step_range_for_game(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        min_step, max_step = store.step_range(game_number=1)
        assert min_step == 1
        assert max_step == 10

    def test_step_range_overall(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        min_step, max_step = store.step_range()
        assert min_step == 1
        assert max_step == 15

    def test_step_range_game_2(self, populated_run_dir: Path):
        store = RunStore(populated_run_dir)
        min_step, max_step = store.step_range(game_number=2)
        assert min_step == 11
        assert max_step == 15
