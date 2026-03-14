# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/parquet_exporter.py."""

import json
from pathlib import Path

import pandas as pd
from helpers.otel import make_log_record

from roc.reporting.parquet_exporter import ParquetExporter


class TestParquetExporterRouting:
    def test_export_creates_run_directory(self, tmp_path: Path):
        run_dir = tmp_path / "test-run"
        exporter = ParquetExporter(run_dir=run_dir, flush_interval=1)
        record = make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')
        exporter.export([record])
        assert run_dir.exists()

    def test_export_routes_screen_to_screens_parquet(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        record = make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')
        exporter.export([record])
        assert (tmp_path / "screens.parquet").exists()
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert len(df) == 1
        assert df.iloc[0]["step"] == 1

    def test_export_routes_saliency_to_saliency_parquet(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        # Need a screen event first to set step counter
        screen = make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')
        sal = make_log_record(
            event_name="roc.attention.saliency", body='{"chars":[],"fg":[],"bg":[]}'
        )
        exporter.export([screen, sal])
        assert (tmp_path / "saliency.parquet").exists()
        df = pd.read_parquet(tmp_path / "saliency.parquet")
        assert len(df) == 1

    def test_export_routes_named_events_to_events_parquet(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        screen = make_log_record(event_name="roc.screen", body="x")
        record = make_log_record(event_name="roc.attention.features", body="Delta: 12")
        exporter.export([screen, record])
        assert (tmp_path / "events.parquet").exists()
        df = pd.read_parquet(tmp_path / "events.parquet")
        assert len(df) == 1

    def test_export_routes_unnamed_to_logs_parquet(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        screen = make_log_record(event_name="roc.screen", body="x")
        record = make_log_record(body="some loguru message")
        exporter.export([screen, record])
        assert (tmp_path / "logs.parquet").exists()
        df = pd.read_parquet(tmp_path / "logs.parquet")
        assert len(df) == 1


class TestStepAndGameCounters:
    def test_step_counter_increments_on_screen_event(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        for _ in range(3):
            exporter.export(
                [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
            )
        exporter.shutdown()
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert list(df["step"]) == [1, 2, 3]

    def test_game_counter_increments_on_game_start(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        exporter.export([make_log_record(event_name="roc.game_start", body="game 2")])
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        exporter.shutdown()
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert list(df["game_number"]) == [1, 2]

    def test_parquet_has_step_and_game_columns(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        for table_name in ["screens"]:
            df = pd.read_parquet(tmp_path / f"{table_name}.parquet")
            assert "step" in df.columns
            assert "game_number" in df.columns


class TestFlushBehavior:
    def test_flush_interval_triggers_write(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=2)
        # First screen -- no flush yet (steps_since_flush=1, interval=2)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        assert not (tmp_path / "screens.parquet").exists()
        # Second screen -- flush triggers (steps_since_flush=2, interval=2)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        assert (tmp_path / "screens.parquet").exists()

    def test_shutdown_flushes_remaining(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        assert not (tmp_path / "screens.parquet").exists()
        exporter.shutdown()
        assert (tmp_path / "screens.parquet").exists()
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert len(df) == 1

    def test_append_mode(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert len(df) == 2
        assert list(df["step"]) == [1, 2]


class TestRecordConversion:
    def test_record_to_dict_preserves_attributes(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        record = make_log_record(
            event_name="roc.screen",
            body="test",
            attributes={"custom.key": "custom_value"},
        )
        exporter.export([record])
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert df.iloc[0]["custom.key"] == "custom_value"

    def test_record_to_dict_preserves_body(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        body_data = json.dumps({"chars": [[65, 66]], "fg": [["#fff"]], "bg": [["#000"]]})
        record = make_log_record(event_name="roc.screen", body=body_data)
        exporter.export([record])
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert df.iloc[0]["body"] == body_data

    def test_record_to_dict_preserves_timestamp(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        ts = 1700000000000000000  # fixed nanosecond timestamp
        record = make_log_record(event_name="roc.screen", body="test", timestamp=ts)
        exporter.export([record])
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert df.iloc[0]["timestamp"] == ts

    def test_record_to_dict_handles_none_body(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        record = make_log_record(event_name="roc.screen", body=None)
        exporter.export([record])
        df = pd.read_parquet(tmp_path / "screens.parquet")
        assert df.iloc[0]["body"] is None


class TestEdgeCases:
    def test_force_flush(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        assert not (tmp_path / "screens.parquet").exists()
        result = exporter.force_flush()
        assert result is True
        assert (tmp_path / "screens.parquet").exists()

    def test_flush_empty_buffers_is_noop(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        exporter._flush_all()
        # No files created
        assert not any(tmp_path.iterdir())

    def test_export_failure_returns_failure(self, tmp_path: Path):
        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
        # Pass something that isn't a LogRecord to trigger an exception
        result = exporter.export([object()])
        from opentelemetry.sdk._logs.export import LogExportResult

        assert result == LogExportResult.FAILURE

    def test_record_with_no_attributes(self, tmp_path: Path):
        from opentelemetry.sdk._logs import LogRecord as SDKLogRecord

        exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)
        record = SDKLogRecord(body="bare record", timestamp=1700000000000000000)
        # This should route to logs (no event.name)
        exporter.export([record])
        exporter.shutdown()
        assert (tmp_path / "logs.parquet").exists()
