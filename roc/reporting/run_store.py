"""Query layer for ROC run data stored as DuckLake-managed Parquet files."""

from __future__ import annotations

import glob as globmod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from roc.reporting.ducklake_store import DuckLakeStore


@dataclass
class StepData:
    """All data for a single step, assembled from multiple tables."""

    step: int
    game_number: int
    timestamp: int | None = None
    screen: dict[str, Any] | None = None
    saliency: dict[str, Any] | None = None
    features: list[dict[str, Any]] | None = None
    object_info: list[dict[str, Any]] | None = None
    focus_points: list[dict[str, Any]] | None = None
    attenuation: dict[str, Any] | None = None
    resolution_metrics: dict[str, Any] | None = None
    graph_summary: dict[str, Any] | None = None
    event_summary: list[dict[str, Any]] | None = None
    game_metrics: dict[str, Any] | None = None
    logs: list[dict[str, Any]] | None = None


class RunStore:
    """Read-only query layer over DuckLake-managed Parquet files.

    Uses a standalone DuckDB connection to read Parquet files directly
    from the ``data/main/<table>/`` directories.  Does not open the
    DuckLake SQLite catalog, so it never conflicts with the writer.

    Can also accept a ``DuckLakeStore`` for in-process use (tests).
    """

    _TABLES = ("screens", "saliency", "events", "logs", "metrics")

    def __init__(self, store_or_dir: DuckLakeStore | Path) -> None:
        if isinstance(store_or_dir, DuckLakeStore):
            # In-process mode (tests): query through the shared store
            self._store: DuckLakeStore | None = store_or_dir
            self._conn: duckdb.DuckDBPyConnection | None = None
            self.run_dir = store_or_dir.run_dir
        else:
            # Reader mode (dashboard): direct Parquet reads, no catalog
            self._store = None
            self._conn = duckdb.connect()
            self.run_dir = store_or_dir
        self._last_max_step: int = 0
        self._refresh_max_step()

    def _table_glob(self, table: str) -> str:
        """Glob pattern for a table's Parquet files."""
        return str(self.run_dir / "data" / "main" / table / "*.parquet")

    def _has_table(self, table: str) -> bool:
        """Check if any Parquet files exist for a table."""
        if self._store is not None:
            return self._store.has_table(table)
        return bool(globmod.glob(self._table_glob(table)))

    def _query(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute a query via the appropriate connection."""
        if self._store is not None:
            return self._store.execute(sql, params)
        if params:
            return self._conn.execute(sql, params)  # type: ignore[union-attr]
        return self._conn.execute(sql)  # type: ignore[union-attr]

    def _read_sql(self, table: str) -> str:
        """SQL fragment to read a table's data."""
        if self._store is not None:
            return f'lake."{table}"'
        return f"read_parquet('{self._table_glob(table)}', union_by_name=true)"

    def get_step(self, step: int, table: str) -> pd.DataFrame:
        """Return all rows for a given step from the specified table."""
        if not self._has_table(table):
            return pd.DataFrame()
        try:
            src = self._read_sql(table)
            return self._query(f"SELECT * FROM {src} WHERE step = ?", [step]).fetchdf()
        except (duckdb.CatalogException, duckdb.InvalidInputException):
            return pd.DataFrame()

    def _step_table(self) -> str | None:
        """Return the best available table for step queries.

        Prefers ``screens``, falls back to ``metrics`` so that historical
        review works even when ``emit_state_screen=False``.
        """
        for table in ("screens", "metrics"):
            if self._has_table(table):
                return table
        return None

    def step_count(self, game_number: int | None = None) -> int:
        """Return the total number of steps, optionally filtered by game."""
        table = self._step_table()
        if table is None:
            return 0
        src = self._read_sql(table)
        if game_number is not None:
            result = self._query(
                f"SELECT COUNT(*) FROM {src} WHERE game_number = ?",
                [game_number],
            ).fetchone()
        else:
            result = self._query(f"SELECT MAX(step) FROM {src}").fetchone()
        return result[0] if result and result[0] is not None else 0

    def step_range(self, game_number: int | None = None) -> tuple[int, int]:
        """Return the (min, max) step for a game or the whole run."""
        table = self._step_table()
        if table is None:
            return (0, 0)
        src = self._read_sql(table)
        if game_number is not None:
            result = self._query(
                f"SELECT MIN(step), MAX(step) FROM {src} WHERE game_number = ?",
                [game_number],
            ).fetchone()
        else:
            result = self._query(f"SELECT MIN(step), MAX(step) FROM {src}").fetchone()
        if result and result[0] is not None:
            return (result[0], result[1])
        return (0, 0)

    def list_games(self) -> pd.DataFrame:
        """Return a summary DataFrame of all games in the run."""
        table = self._step_table()
        if table is None:
            return pd.DataFrame(columns=["game_number", "steps", "start_ts", "end_ts"])
        src = self._read_sql(table)
        return self._query(
            f"""
            SELECT
                game_number,
                COUNT(*) AS steps,
                MIN(timestamp) AS start_ts,
                MAX(timestamp) AS end_ts
            FROM {src}
            GROUP BY game_number
            ORDER BY game_number
            """,
        ).fetchdf()

    @staticmethod
    def list_runs(data_dir: Path) -> list[str]:
        """Scan a data directory for valid run directories."""
        runs: list[str] = []
        if not data_dir.exists():
            return runs
        for child in sorted(data_dir.iterdir()):
            if child.is_dir() and DuckLakeStore.is_valid_run(child):
                runs.append(child.name)
        return runs

    def get_step_data(self, step: int) -> StepData:
        """Assemble all available data for a single step."""
        screen_df = self.get_step(step, "screens")
        screen = None
        game_number = 0
        timestamp = None
        if len(screen_df) > 0:
            row = screen_df.iloc[0]
            screen = _parse_body(row.get("body"))
            game_number = int(row["game_number"])
            timestamp = row.get("timestamp")

        sal_df = self.get_step(step, "saliency")
        saliency = None
        if len(sal_df) > 0:
            saliency = _parse_body(sal_df.iloc[0].get("body"))

        events_df = self.get_step(step, "events")
        features = None
        object_info = None
        focus_points = None
        attenuation = None
        resolution_metrics = None
        graph_summary = None
        event_summary: list[dict[str, Any]] | None = None
        if len(events_df) > 0:
            event_list: list[dict[str, Any]] = []
            for _, row in events_df.iterrows():
                row_dict = cast(dict[str, Any], dict(row))
                event_name = row_dict.get("event.name", "")
                body = _parse_body(row_dict.get("body"))

                if event_name == "roc.attention.features":
                    if features is None:
                        features = []
                    features.append(body if body is not None else {"raw": row_dict.get("body", "")})
                elif event_name == "roc.attention.object":
                    if object_info is None:
                        object_info = []
                    object_info.append(
                        body if body is not None else {"raw": row_dict.get("body", "")}
                    )
                elif event_name == "roc.attention.focus_points":
                    if focus_points is None:
                        focus_points = []
                    focus_points.append(
                        body if body is not None else {"raw": row_dict.get("body", "")}
                    )
                elif event_name == "roc.saliency_attenuation":
                    if body is not None:
                        attenuation = {
                            k: v
                            for k, v in body.items()
                            if k not in ("saliency_grid", "focus_points", "history")
                        }
                    else:
                        attenuation = body
                elif event_name == "roc.resolution.decision":
                    resolution_metrics = body
                elif event_name == "roc.graphdb.summary":
                    graph_summary = body
                elif event_name == "roc.event.summary":
                    event_list.append(
                        body if body is not None else {"raw": row_dict.get("body", "")}
                    )
                else:
                    event_list.append(row_dict)

            if event_list:
                event_summary = event_list

        game_metrics = None
        metrics_df = self.get_step(step, "metrics")
        if len(metrics_df) > 0:
            game_metrics = _parse_body(metrics_df.iloc[0].get("body"))

        logs_df = self.get_step(step, "logs")
        logs = None
        if len(logs_df) > 0:
            logs = cast(list[dict[str, Any]], logs_df.to_dict("records"))

        return StepData(
            step=step,
            game_number=game_number,
            timestamp=timestamp,
            screen=screen,
            saliency=saliency,
            features=features,
            object_info=object_info,
            focus_points=focus_points,
            attenuation=attenuation,
            resolution_metrics=resolution_metrics,
            graph_summary=graph_summary,
            event_summary=event_summary,
            game_metrics=game_metrics,
            logs=logs,
        )

    def _refresh_max_step(self) -> None:
        """Update the cached max step value."""
        try:
            if not self._has_table("screens"):
                return
            src = self._read_sql("screens")
            result = self._query(f"SELECT MAX(step) FROM {src}").fetchone()
            self._last_max_step = result[0] if result and result[0] is not None else 0
        except Exception:
            pass


def _parse_body(body: str | None) -> dict[str, Any] | None:
    """Try to parse a body string as JSON; return None on failure."""
    if body is None:
        return None
    try:
        import json

        result: dict[str, Any] = json.loads(body)
        return result
    except (json.JSONDecodeError, TypeError):
        return {"raw": body}
