"""DuckDB-based query layer over Parquet files produced by ParquetExporter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd


@dataclass
class StepData:
    """All data for a single step, assembled from multiple Parquet sources."""

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
    """Query Parquet files for a single run via DuckDB.

    Each instance connects to an in-memory DuckDB database and queries
    Parquet files on disk directly. DuckDB reads fresh data on each query,
    so no explicit reload is needed.
    """

    #: Parquet files that map to named tables.
    _TABLES = ("screens", "saliency", "events", "logs", "metrics")

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._conn = duckdb.connect(":memory:")

    def get_step(self, step: int, table: str) -> pd.DataFrame:
        """Return all rows for a given step from the specified table.

        Args:
            step: The step number to query.
            table: The Parquet file name (without extension).

        Returns:
            DataFrame with matching rows, or an empty DataFrame if the
            file does not exist or the step is not found.
        """
        path = self.run_dir / f"{table}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return self._conn.execute(
            "SELECT * FROM read_parquet(?) WHERE step = ?",
            [str(path), step],
        ).fetchdf()

    def step_count(self, game_number: int | None = None) -> int:
        """Return the total number of steps, optionally filtered by game.

        Args:
            game_number: If provided, count only steps for this game.

        Returns:
            The number of distinct steps.
        """
        path = self.run_dir / "screens.parquet"
        if not path.exists():
            return 0
        if game_number is not None:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM read_parquet(?) WHERE game_number = ?",
                [str(path), game_number],
            ).fetchone()
        else:
            result = self._conn.execute(
                "SELECT MAX(step) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()
        return result[0] if result and result[0] is not None else 0

    def step_range(self, game_number: int | None = None) -> tuple[int, int]:
        """Return the (min, max) step for a game or the whole run.

        Args:
            game_number: If provided, scope to this game.

        Returns:
            Tuple of (min_step, max_step).
        """
        path = self.run_dir / "screens.parquet"
        if not path.exists():
            return (0, 0)
        if game_number is not None:
            result = self._conn.execute(
                "SELECT MIN(step), MAX(step) FROM read_parquet(?) WHERE game_number = ?",
                [str(path), game_number],
            ).fetchone()
        else:
            result = self._conn.execute(
                "SELECT MIN(step), MAX(step) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()
        if result and result[0] is not None:
            return (result[0], result[1])
        return (0, 0)

    def list_games(self) -> pd.DataFrame:
        """Return a summary DataFrame of all games in the run.

        Columns: game_number, steps, start_ts, end_ts.
        """
        path = self.run_dir / "screens.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["game_number", "steps", "start_ts", "end_ts"])
        return self._conn.execute(
            """
            SELECT
                game_number,
                COUNT(*) AS steps,
                MIN(timestamp) AS start_ts,
                MAX(timestamp) AS end_ts
            FROM read_parquet(?)
            GROUP BY game_number
            ORDER BY game_number
            """,
            [str(path)],
        ).fetchdf()

    @staticmethod
    def list_runs(data_dir: Path) -> list[str]:
        """Scan a data directory for valid run directories.

        A directory is considered a valid run if it contains ``screens.parquet``.

        Args:
            data_dir: The parent directory containing run directories.

        Returns:
            Sorted list of run directory names.
        """
        runs: list[str] = []
        if not data_dir.exists():
            return runs
        for child in sorted(data_dir.iterdir()):
            if child.is_dir() and (child / "screens.parquet").exists():
                runs.append(child.name)
        return runs

    def get_step_data(self, step: int) -> StepData:
        """Assemble all available data for a single step.

        Args:
            step: The step number to query.

        Returns:
            A StepData instance with fields populated from available Parquet files.
        """
        # Get screen data
        screen_df = self.get_step(step, "screens")
        screen = None
        game_number = 0
        timestamp = None
        if len(screen_df) > 0:
            row = screen_df.iloc[0]
            screen = _parse_body(row.get("body"))
            game_number = int(row["game_number"])
            timestamp = row.get("timestamp")

        # Get saliency data
        sal_df = self.get_step(step, "saliency")
        saliency = None
        if len(sal_df) > 0:
            saliency = _parse_body(sal_df.iloc[0].get("body"))

        # Get events for this step -- route by event.name to specific fields
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
                    # Strip large nested fields (saliency_grid is already in
                    # saliency.parquet; focus_points/history are verbose lists)
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

        # Get game metrics for this step
        game_metrics = None
        metrics_df = self.get_step(step, "metrics")
        if len(metrics_df) > 0:
            game_metrics = _parse_body(metrics_df.iloc[0].get("body"))

        # Get logs for this step
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

    def reload(self) -> None:
        """No-op -- DuckDB reads fresh data on each query."""


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
