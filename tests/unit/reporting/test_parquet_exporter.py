# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/parquet_exporter.py."""

import json
import time
from pathlib import Path

import pandas as pd
from helpers.otel import make_log_record

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.parquet_exporter import ParquetExporter


def _read_table(store: DuckLakeStore, table: str) -> pd.DataFrame:
    """Read all rows from a DuckLake table."""
    if not store.has_table(table):
        return pd.DataFrame()
    return store.execute(f'SELECT * FROM lake."{table}" ORDER BY step').fetchdf()  # nosec B608


def _has_table(store: DuckLakeStore, table: str) -> bool:
    """Check if a DuckLake table has any data."""
    return store.has_table(table)


class TestParquetExporterRouting:
    def test_export_creates_run_directory(self, tmp_path: Path):
        run_dir = tmp_path / "test-run"
        store = DuckLakeStore(run_dir)
        exporter = ParquetExporter(store=store, background=False)
        record = make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')
        exporter.export([record])
        exporter.force_flush()
        assert run_dir.exists()

    def test_export_routes_screen_to_screens(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        record = make_log_record(
            event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}', tick=1
        )
        exporter.export([record])
        exporter.force_flush()
        assert _has_table(store, "screens")
        df = _read_table(store, "screens")
        assert len(df) == 1
        assert df.iloc[0]["step"] == 1

    def test_export_routes_saliency(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        screen = make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')
        sal = make_log_record(
            event_name="roc.attention.saliency", body='{"chars":[],"fg":[],"bg":[]}'
        )
        exporter.export([screen, sal])
        exporter.force_flush()
        assert _has_table(store, "saliency")
        df = _read_table(store, "saliency")
        assert len(df) == 1

    def test_export_routes_named_events(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        screen = make_log_record(event_name="roc.screen", body="x")
        record = make_log_record(event_name="roc.attention.features", body="Delta: 12")
        exporter.export([screen, record])
        exporter.force_flush()
        assert _has_table(store, "events")
        df = _read_table(store, "events")
        assert len(df) == 1

    def test_export_routes_unnamed_to_logs(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        screen = make_log_record(event_name="roc.screen", body="x")
        record = make_log_record(body="some loguru message")
        exporter.export([screen, record])
        exporter.force_flush()
        assert _has_table(store, "logs")
        df = _read_table(store, "logs")
        assert len(df) == 1


class TestStepAndGameCounters:
    def test_step_matches_tick_attribute(self, tmp_path: Path):
        """Step column should echo the ``tick`` attribute stamped at emit time."""
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        for t in (1, 2, 3):
            exporter.export(
                [
                    make_log_record(
                        event_name="roc.screen",
                        body='{"chars":[],"fg":[],"bg":[]}',
                        tick=t,
                    )
                ]
            )
        exporter.shutdown()
        df = _read_table(store, "screens")
        assert list(df["step"]) == [1, 2, 3]

    def test_game_counter_increments_on_game_start(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1", tick=1)])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=1,
                )
            ]
        )
        exporter.export([make_log_record(event_name="roc.game_start", body="game 2", tick=2)])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=2,
                )
            ]
        )
        exporter.shutdown()
        df = _read_table(store, "screens")
        assert list(df["game_number"]) == [1, 2]

    def test_has_step_and_game_columns(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=1,
                )
            ]
        )
        exporter.force_flush()
        df = _read_table(store, "screens")
        assert "step" in df.columns
        assert "game_number" in df.columns

    def test_step_falls_back_to_clock_when_no_tick_attribute(self, tmp_path: Path):
        """Records without a tick attribute (e.g. loguru passthroughs) use Clock.get()."""
        from roc.framework.clock import Clock

        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        Clock.set(7)
        try:
            # tick=None produces a record without a tick attribute, so the
            # exporter falls back to reading the clock.
            exporter.export([make_log_record(body="a loguru-style log line", tick=None)])
        finally:
            Clock.reset()
        exporter.force_flush()
        df = _read_table(store, "logs")
        assert len(df) == 1
        assert df.iloc[0]["step"] == 7


class TestFlushBehavior:
    def test_background_flush_writes_data(self, tmp_path: Path):
        """Background thread should flush data without explicit call."""
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        time.sleep(0.3)
        assert _has_table(store, "screens")
        df = _read_table(store, "screens")
        assert len(df) == 1
        exporter.shutdown()

    def test_sync_mode_writes_immediately(self, tmp_path: Path):
        """With background=False, export() writes directly to DuckLake."""
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        # Data visible immediately -- no flush needed
        assert _has_table(store, "screens")
        df = _read_table(store, "screens")
        assert len(df) == 1

    def test_shutdown_drains_remaining(self, tmp_path: Path):
        """Shutdown writes any queued records from the background thread."""
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=True)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        exporter.shutdown()
        assert _has_table(store, "screens")
        df = _read_table(store, "screens")
        assert len(df) == 1

    def test_multiple_flushes_accumulate(self, tmp_path: Path):
        """Multiple flushes should accumulate data."""
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=1,
                )
            ]
        )
        exporter.force_flush()
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=2,
                )
            ]
        )
        exporter.force_flush()
        df = _read_table(store, "screens")
        assert len(df) == 2
        assert list(df["step"]) == [1, 2]


class TestRecordConversion:
    def test_record_to_dict_preserves_attributes(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        record = make_log_record(
            event_name="roc.screen",
            body="test",
            attributes={"custom.key": "custom_value"},
        )
        exporter.export([record])
        exporter.force_flush()
        df = _read_table(store, "screens")
        assert df.iloc[0]["custom.key"] == "custom_value"

    def test_record_to_dict_preserves_body(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        body_data = json.dumps({"chars": [[65, 66]], "fg": [["#fff"]], "bg": [["#000"]]})
        record = make_log_record(event_name="roc.screen", body=body_data)
        exporter.export([record])
        exporter.force_flush()
        df = _read_table(store, "screens")
        assert df.iloc[0]["body"] == body_data

    def test_record_to_dict_preserves_timestamp(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        ts = 1700000000000000000
        record = make_log_record(event_name="roc.screen", body="test", timestamp=ts)
        exporter.export([record])
        exporter.force_flush()
        df = _read_table(store, "screens")
        assert df.iloc[0]["timestamp"] == ts

    def test_record_to_dict_handles_none_body(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        record = make_log_record(event_name="roc.screen", body=None)
        exporter.export([record])
        exporter.force_flush()
        df = _read_table(store, "screens")
        assert df.iloc[0]["body"] is None


class TestEdgeCases:
    def test_force_flush(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=True)
        exporter.export(
            [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
        )
        result = exporter.force_flush()
        assert result is True
        assert _has_table(store, "screens")

    def test_drain_empty_queue_is_noop(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter._drain()  # should not raise

    def test_export_failure_returns_failure(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        result = exporter.export([object()])
        from opentelemetry.sdk._logs.export import LogRecordExportResult

        assert result == LogRecordExportResult.FAILURE

    def test_record_with_no_attributes(self, tmp_path: Path):
        from opentelemetry._logs import LogRecord as SDKLogRecord

        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        record = SDKLogRecord(body="bare record", timestamp=1700000000000000000)
        exporter.export([record])
        exporter.shutdown()
        assert _has_table(store, "logs")


class TestThreadSafety:
    def test_periodic_flush_writes_partial_data(self, tmp_path: Path):
        """Background thread should flush data before shutdown."""
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=True)
        for _ in range(5):
            exporter.export(
                [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
            )
        time.sleep(0.3)
        assert _has_table(store, "screens")
        df = _read_table(store, "screens")
        assert len(df) == 5
        exporter.shutdown()

    def test_game_boundary_increments_game_number(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
        for _ in range(3):
            exporter.export(
                [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
            )
        exporter.export([make_log_record(event_name="roc.game_start", body="game 2")])
        for _ in range(2):
            exporter.export(
                [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
            )
        exporter.shutdown()
        df = _read_table(store, "screens")
        assert list(df["game_number"]) == [1, 1, 1, 2, 2]

    def test_flush_during_game_preserves_game_number(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1", tick=1)])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=1,
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=2,
                )
            ]
        )
        exporter.force_flush()
        exporter.export(
            [
                make_log_record(
                    event_name="roc.screen",
                    body='{"chars":[],"fg":[],"bg":[]}',
                    tick=3,
                )
            ]
        )
        exporter.shutdown()
        df = _read_table(store, "screens")
        assert list(df["game_number"]) == [1, 1, 1]
        assert list(df["step"]) == [1, 2, 3]

    def test_thread_safe_export(self, tmp_path: Path):
        """Concurrent exports should not lose data."""
        import threading

        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])

        def write_records():
            for _ in range(10):
                exporter.export(
                    [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
                )

        threads = [threading.Thread(target=write_records) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        exporter.shutdown()
        df = _read_table(store, "screens")
        assert len(df) == 30

    def test_shutdown_stops_background_thread(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=True)
        assert exporter._flush_thread is not None
        assert exporter._flush_thread.is_alive()
        exporter.shutdown()
        assert not exporter._flush_thread.is_alive()

    def test_no_background_thread_when_disabled(self, tmp_path: Path):
        store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=store, background=False)
        assert exporter._flush_thread is None
