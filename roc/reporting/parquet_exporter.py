"""OTel log exporter that writes records to DuckLake via background thread."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk._logs.export import LogExporter, LogExportResult

from roc.reporting.ducklake_store import DuckLakeStore

_GAME_METRICS_EVENT = "roc.game_metrics"


class ParquetExporter(LogExporter):
    """Route OTel log records to DuckLake tables.

    ``export()`` converts records to dicts, routes them by event name,
    and queues them for a background thread that does the DuckLake INSERT.
    The game loop never blocks on database writes.

    Pass ``background=False`` (used by tests) to write synchronously
    in ``export()`` instead of queuing.

    Routing rules:
        - ``roc.screen`` -> ``screens``
        - ``roc.attention.saliency`` -> ``saliency``
        - ``roc.game_metrics`` -> ``metrics``
        - other named events -> ``events``
        - unnamed (loguru) -> ``logs``
    """

    def __init__(
        self,
        store: DuckLakeStore,
        *,
        background: bool = True,
        checkpoint_interval: int = 200,
    ) -> None:
        self._store = store
        self.run_dir = store.run_dir
        self._step_counter = 0
        self._step_incremented = False
        self._game_counter = 0

        # Queue: list of (table_name, record_dict) tuples
        self._queue: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._data_ready = threading.Event()
        self._shutdown_event = threading.Event()
        self._background = background
        self._checkpoint_interval = checkpoint_interval
        self._last_checkpoint_step = 0

        self._flush_thread: threading.Thread | None = None
        if background:
            self._flush_thread = threading.Thread(
                target=self._background_loop,
                daemon=True,
                name="ducklake-writer",
            )
            self._flush_thread.start()

    def _background_loop(self) -> None:
        """Drain queue and INSERT into DuckLake, repeat until shutdown."""
        while not self._shutdown_event.is_set():
            self._data_ready.wait()
            if self._shutdown_event.is_set():
                break
            self._data_ready.clear()
            self._drain()
            # Periodically run CHECKPOINT to flush inlined data to
            # parquet and merge small files for faster historical reads.
            if self._step_counter - self._last_checkpoint_step >= self._checkpoint_interval:
                self._last_checkpoint_step = self._step_counter
                try:
                    self._store.checkpoint()
                except Exception:
                    pass  # checkpoint errors must not break the game loop

    def _drain(self) -> None:
        """Move queued records to DuckLake. Safe to call from any thread."""
        with self._lock:
            if not self._queue:
                return
            batch = self._queue
            self._queue = []

        # Group by table
        by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for table, record in batch:
            by_table[table].append(record)

        for table, records in by_table.items():
            self._store.insert(table, records)

    def export(self, batch: Sequence[Any]) -> LogExportResult:
        """Route log records to the queue (or write directly if no background thread)."""
        try:
            with self._lock:
                for log_data in batch:
                    self._enqueue_record(log_data)

            if self._background:
                self._data_ready.set()
            else:
                self._drain()

            return LogExportResult.SUCCESS
        except Exception:
            return LogExportResult.FAILURE

    def _enqueue_record(self, log_data: Any) -> None:
        """Process a single log record: update counters and queue for writing."""
        record = getattr(log_data, "log_record", log_data)
        attrs = dict(record.attributes) if record.attributes else {}
        event_name = attrs.get("event.name")

        self._update_counters(event_name)

        entry = self._record_to_dict(record, attrs)
        entry["step"] = self._step_counter
        entry["game_number"] = self._game_counter
        table = self._route(event_name)
        self._queue.append((table, entry))

    def _update_counters(self, event_name: str | None) -> None:
        """Update game and step counters based on event type."""
        if event_name == "roc.game_start":
            self._game_counter += 1
        if event_name == "roc.screen":
            self._step_counter += 1
            self._step_incremented = True
        elif event_name == _GAME_METRICS_EVENT and not self._step_incremented:
            self._step_counter += 1
        elif event_name == _GAME_METRICS_EVENT:
            self._step_incremented = False

    def shutdown(self) -> None:
        """Stop background thread and write any remaining queued records."""
        self._shutdown_event.set()
        self._data_ready.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
        self._drain()
        # Final checkpoint: flush inlined data to parquet so historical
        # reads can find all data.
        try:
            self._store.checkpoint()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """Write all queued records to DuckLake immediately."""
        self._drain()
        return True

    @staticmethod
    def _route(event_name: str | None) -> str:
        """Map an event name to a DuckLake table name."""
        if event_name == "roc.screen":
            return "screens"
        if event_name == "roc.attention.saliency":
            return "saliency"
        if event_name == _GAME_METRICS_EVENT:
            return "metrics"
        if event_name is not None:
            return "events"
        return "logs"

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
        for key, value in attrs.items():
            entry[key] = value
        return entry
