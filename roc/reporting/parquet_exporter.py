"""OTel log exporter that writes records to Parquet files partitioned by event type."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from opentelemetry.sdk._logs.export import LogExporter, LogExportResult


class ParquetExporter(LogExporter):
    """Write OTel log records to Parquet files, partitioned by event type.

    Records are buffered in memory and flushed to disk either when the number
    of steps since the last flush reaches ``flush_interval``, or on shutdown.

    Buffer routing rules:
        - ``roc.screen`` -> ``screens.parquet``
        - ``roc.attention.saliency`` -> ``saliency.parquet``
        - other named events -> ``events.parquet``
        - unnamed (loguru) -> ``logs.parquet``
    """

    def __init__(self, run_dir: Path, flush_interval: int = 100) -> None:
        self.run_dir = run_dir
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._step_counter = 0
        self._game_counter = 0
        self._flush_interval = flush_interval
        self._steps_since_flush = 0

    def export(self, batch: Sequence[Any]) -> LogExportResult:
        """Route log records to named buffers and flush when interval is reached."""
        try:
            for log_data in batch:
                # Support both raw LogRecord and LogData wrapper
                record = getattr(log_data, "log_record", log_data)
                attrs = dict(record.attributes) if record.attributes else {}
                event_name = attrs.get("event.name")

                # Detect game boundary
                if event_name == "roc.game_start":
                    self._game_counter += 1

                # Increment step on each screen event
                if event_name == "roc.screen":
                    self._step_counter += 1
                    self._steps_since_flush += 1

                entry = self._record_to_dict(record, attrs)
                entry["step"] = self._step_counter
                entry["game_number"] = self._game_counter

                # Route to appropriate buffer
                if event_name == "roc.screen":
                    self._buffers["screens"].append(entry)
                elif event_name == "roc.attention.saliency":
                    self._buffers["saliency"].append(entry)
                elif event_name == "roc.game_metrics":
                    self._buffers["metrics"].append(entry)
                elif event_name is not None:
                    self._buffers["events"].append(entry)
                else:
                    self._buffers["logs"].append(entry)

            # Periodic flush
            if self._steps_since_flush >= self._flush_interval:
                self._flush_all()
                self._steps_since_flush = 0

            return LogExportResult.SUCCESS
        except Exception:
            return LogExportResult.FAILURE

    def shutdown(self) -> None:
        """Flush all remaining buffered data."""
        self._flush_all()

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """Flush all buffers immediately."""
        self._flush_all()
        return True

    def _flush_all(self) -> None:
        """Write each buffer to its Parquet file, appending if the file exists."""
        if not any(self._buffers.values()):
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name, records in self._buffers.items():
            if not records:
                continue
            table = pa.Table.from_pylist(records)
            path = self.run_dir / f"{name}.parquet"
            if path.exists():
                existing = pq.read_table(path)
                # Unify schemas before concatenating (handles missing columns)
                table = pa.concat_tables([existing, table], promote_options="default")
            pq.write_table(table, path, compression="snappy")
        self._buffers.clear()

    @staticmethod
    def _record_to_dict(record: Any, attrs: dict[str, Any]) -> dict[str, Any]:
        """Convert an OTel LogRecord to a flat dictionary."""
        entry: dict[str, Any] = {
            "timestamp": record.timestamp,
            "severity_number": record.severity_number.value if record.severity_number else None,
            "severity_text": record.severity_text,
            "body": str(record.body) if record.body is not None else None,
            "trace_id": str(record.trace_id) if record.trace_id else None,
            "span_id": str(record.span_id) if record.span_id else None,
        }
        # Flatten attributes into top-level columns
        for key, value in attrs.items():
            entry[key] = value
        return entry
