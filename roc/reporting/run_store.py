"""Query layer for ROC run data stored in DuckLake catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    """Read-only query layer over a DuckLake catalog.

    All queries go through the DuckLakeStore's thread-safe ``query_df``
    and ``query_one`` methods, which hold the connection lock for the
    full execute+fetch cycle.  This is safe for concurrent use from
    FastAPI's thread pool and for sharing with a live game writer.
    """

    _TABLES = ("screens", "saliency", "events", "logs", "metrics")

    def __init__(self, store: DuckLakeStore) -> None:
        self._store = store
        self.run_dir = store.run_dir
        self._alias = store._alias

    def _read_sql(self, table: str) -> str:
        """SQL fragment to read a table's data."""
        return f'{self._alias}."{table}"'

    def _has_table(self, table: str) -> bool:
        """Check if a table exists in the catalog."""
        return self._store.has_table(table)

    def get_step(self, step: int, table: str) -> pd.DataFrame:
        """Return all rows for a given step from the specified table."""
        if not self._has_table(table):
            return pd.DataFrame()
        try:
            src = self._read_sql(table)
            return self._store.query_df(f"SELECT * FROM {src} WHERE step = ?", [step])
        except Exception:
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
            result = self._store.query_one(
                f"SELECT COUNT(*) FROM {src} WHERE game_number = ?",
                [game_number],
            )
        else:
            result = self._store.query_one(f"SELECT MAX(step) FROM {src}")
        return result[0] if result and result[0] is not None else 0

    def step_range(self, game_number: int | None = None) -> tuple[int, int]:
        """Return the (min, max) step for a game or the whole run."""
        table = self._step_table()
        if table is None:
            return (0, 0)
        src = self._read_sql(table)
        if game_number is not None:
            result = self._store.query_one(
                f"SELECT MIN(step), MAX(step) FROM {src} WHERE game_number = ?",
                [game_number],
            )
        else:
            result = self._store.query_one(f"SELECT MIN(step), MAX(step) FROM {src}")
        if result and result[0] is not None:
            return (result[0], result[1])
        return (0, 0)

    def list_games(self) -> pd.DataFrame:
        """Return a summary DataFrame of all games in the run."""
        table = self._step_table()
        if table is None:
            return pd.DataFrame(columns=["game_number", "steps", "start_ts", "end_ts"])
        src = self._read_sql(table)
        return self._store.query_df(
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
        )

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

    def get_metrics_history(
        self,
        game_number: int | None = None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return game_metrics for all steps in a game (or the whole run).

        Each entry has ``step`` plus the parsed metric fields.  Returns an
        empty list when no metrics table exists.
        """
        if not self._has_table("metrics"):
            return []
        src = self._read_sql("metrics")
        if game_number is not None:
            df = self._store.query_df(
                f"SELECT step, body FROM {src} WHERE game_number = ? ORDER BY step",
                [game_number],
            )
        else:
            df = self._store.query_df(f"SELECT step, body FROM {src} ORDER BY step")

        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            body = _parse_body(row.get("body"))
            if body is None:
                continue
            entry: dict[str, Any] = {"step": int(row["step"])}
            if fields:
                for f in fields:
                    if f in body:
                        entry[f] = body[f]
            else:
                entry.update(body)
            results.append(entry)
        return results

    def get_graph_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return graph_summary for all steps in a game (or the whole run).

        Each entry has ``step`` plus ``node_count``, ``node_max``,
        ``edge_count``, ``edge_max``.  Returns an empty list when the
        events table does not exist.

        Uses DuckDB JSON extraction to avoid Python-side parsing.
        """
        if not self._has_table("events"):
            return []
        src = self._read_sql("events")
        where = "\"event.name\" = 'roc.graphdb.summary'"
        params: list[Any] = []
        if game_number is not None:
            where += " AND game_number = ?"
            params.append(game_number)
        df = self._store.query_df(
            f"""SELECT step,
                       CAST(body::JSON->>'node_count' AS INTEGER) AS node_count,
                       CAST(body::JSON->>'node_max' AS INTEGER) AS node_max,
                       CAST(body::JSON->>'edge_count' AS INTEGER) AS edge_count,
                       CAST(body::JSON->>'edge_max' AS INTEGER) AS edge_max
                FROM {src}
                WHERE {where}
                ORDER BY step""",
            params or None,
        )
        return cast(list[dict[str, Any]], df.to_dict("records"))

    def get_event_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return event_summary for all steps in a game (or the whole run).

        Each entry has ``step`` plus per-bus event counts.  Returns an
        empty list when the events table does not exist.
        """
        if not self._has_table("events"):
            return []
        src = self._read_sql("events")
        where = "\"event.name\" = 'roc.event.summary'"
        params: list[Any] = []
        if game_number is not None:
            where += " AND game_number = ?"
            params.append(game_number)
        df = self._store.query_df(
            f"SELECT step, body FROM {src} WHERE {where} ORDER BY step",
            params or None,
        )
        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            body = _parse_body(row.get("body"))
            if body is None:
                continue
            entry: dict[str, Any] = {"step": int(row["step"])}
            entry.update(body)
            results.append(entry)
        return results

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

        logs = None
        logs_df = self.get_step(step, "logs")
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
