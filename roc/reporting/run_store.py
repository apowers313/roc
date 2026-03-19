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
    intrinsics: dict[str, Any] | None = None
    significance: float | None = None
    action_taken: dict[str, Any] | None = None
    transform_summary: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    message: str | None = None
    inventory: list[dict[str, Any]] | None = None


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

    def get_intrinsics_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return intrinsics for all steps in a game (or the whole run).

        Each entry has ``step`` plus raw and normalized intrinsic values.
        Returns an empty list when no events table exists.
        """
        if not self._has_table("events"):
            return []
        src = self._read_sql("events")
        where = "\"event.name\" = 'roc.intrinsics'"
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

    def get_action_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return action taken for all steps in a game (or the whole run).

        Each entry has ``step`` plus ``action_id`` and optionally ``action_name``.
        Returns an empty list when no events table exists.
        """
        if not self._has_table("events"):
            return []
        src = self._read_sql("events")
        where = "\"event.name\" = 'roc.action'"
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

    def get_resolution_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return resolution accuracy history for all steps in a game.

        Each entry has ``step``, ``outcome`` (match/new_object/low_confidence),
        and ``correct`` (bool | None) indicating whether observed non-relational
        features match the stored object's features.
        """
        if not self._has_table("events"):
            return []
        src = self._read_sql("events")
        where = "\"event.name\" = 'roc.resolution.decision'"
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
            outcome = body.get("outcome", "unknown")
            entry: dict[str, Any] = {"step": int(row["step"]), "outcome": outcome}

            # Compute correctness for matches by comparing non-relational attrs
            if outcome == "match":
                matched_attrs = body.get("matched_attrs")
                features = body.get("features", [])
                if matched_attrs and isinstance(features, list):
                    # Parse observed attrs from feature strings
                    obs_shape = obs_color = obs_glyph = None
                    for f in features:
                        s = str(f)
                        if s.startswith("ShapeNode(") and s.endswith(")"):
                            obs_shape = s[10:-1]
                        elif s.startswith("ColorNode(") and s.endswith(")"):
                            obs_color = s[10:-1]
                        elif s.startswith("SingleNode(") and s.endswith(")"):
                            obs_glyph = s[11:-1]
                    # Compare with matched object's stored attrs
                    m_shape = matched_attrs.get("char")
                    m_color = matched_attrs.get("color")
                    m_glyph = str(matched_attrs.get("glyph", ""))
                    correct = True
                    if obs_shape and m_shape and obs_shape != m_shape:
                        correct = False
                    if obs_color and m_color and obs_color != m_color:
                        correct = False
                    if obs_glyph and m_glyph and obs_glyph != m_glyph:
                        correct = False
                    entry["correct"] = correct
                else:
                    entry["correct"] = None  # unknown -- no matched_attrs available
            results.append(entry)
        return results

    def get_all_objects(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of all resolved objects with their attributes.

        Built from resolution events: new_object events create entries (with
        new_object_id if available), match events increment match counts.
        """
        if not self._has_table("events"):
            return []
        src = self._read_sql("events")
        where = "\"event.name\" = 'roc.resolution.decision'"
        params: list[Any] = []
        if game_number is not None:
            where += " AND game_number = ?"
            params.append(game_number)
        df = self._store.query_df(
            f"SELECT step, body FROM {src} WHERE {where} ORDER BY step",
            params or None,
        )

        def _parse_attrs(features: list[Any]) -> tuple[str | None, str | None, str | None]:
            shape = color = glyph = None
            for f in features:
                s = str(f)
                if s.startswith("ShapeNode(") and s.endswith(")"):
                    shape = s[10:-1]
                elif s.startswith("ColorNode(") and s.endswith(")"):
                    color = s[10:-1]
                elif s.startswith("SingleNode(") and s.endswith(")"):
                    glyph = s[11:-1]
            return shape, color, glyph

        # Pass 1: collect new_object entries keyed by step
        objects: dict[str, dict[str, Any]] = {}
        step_to_key: dict[int, str] = {}
        match_events: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            body = _parse_body(row.get("body"))
            if body is None:
                continue
            step = int(row["step"])
            outcome = body.get("outcome")
            features = body.get("features", [])
            shape, color, glyph = _parse_attrs(features)

            if outcome == "new_object":
                key = f"new@{step}"
                step_to_key[step] = key
                objects[key] = {
                    "shape": shape,
                    "glyph": glyph,
                    "color": color,
                    "node_id": None,
                    "step_added": step,
                    "match_count": 0,
                }
            elif outcome == "match":
                match_events.append({
                    "matched_id": body.get("matched_object_id"),
                    "matched_attrs": body.get("matched_attrs", {}),
                    "shape": shape, "glyph": glyph, "color": color,
                })

        # Pass 2: read new_object_id events to link node IDs
        id_df = self._store.query_df(
            f"SELECT step, body FROM {src} "
            f"WHERE \"event.name\" = 'roc.resolution.new_object_id'"
            + (" AND game_number = ?" if game_number is not None else "")
            + " ORDER BY step",
            [game_number] if game_number is not None else None,
        )
        for _, row in id_df.iterrows():
            body = _parse_body(row.get("body"))
            if body is None:
                continue
            step = int(row["step"])
            nid = body.get("new_object_id")
            if nid is None:
                continue
            mid = str(nid)
            old_key = step_to_key.get(step)
            if old_key and old_key in objects:
                obj = objects.pop(old_key)
                obj["node_id"] = mid
                objects[mid] = obj

        # Pass 3: process match events (now that objects are keyed by node ID)
        for m in match_events:
            matched_id = m["matched_id"]
            if matched_id is None:
                continue
            mid = str(matched_id)
            if mid in objects:
                objects[mid]["match_count"] += 1
            else:
                ma = m["matched_attrs"]
                objects[mid] = {
                    "shape": ma.get("char", m["shape"]),
                    "glyph": str(ma["glyph"]) if ma.get("glyph") else m["glyph"],
                    "color": ma.get("color", m["color"]),
                    "node_id": mid,
                    "step_added": None,
                    "match_count": 1,
                }

        return list(objects.values())

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
        intrinsics = None
        significance = None
        action_taken = None
        transform_summary = None
        prediction = None
        message = None
        inventory = None
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
                        attenuation = {k: v for k, v in body.items() if k not in ("saliency_grid",)}
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
                elif event_name == "roc.intrinsics":
                    intrinsics = body
                elif event_name == "roc.significance":
                    if body is not None and "significance" in body:
                        significance = float(body["significance"])
                elif event_name == "roc.action":
                    action_taken = body
                elif event_name == "roc.transform_summary":
                    transform_summary = body
                elif event_name == "roc.prediction":
                    prediction = body
                elif event_name == "roc.message":
                    raw_body = row_dict.get("body", "")
                    message = str(raw_body).strip() if raw_body else None
                elif event_name == "roc.inventory":
                    raw_inv = row_dict.get("body", "")
                    if raw_inv:
                        try:
                            import json as _json

                            parsed_inv = _json.loads(raw_inv)
                            if isinstance(parsed_inv, list):
                                inventory = parsed_inv
                        except (ValueError, TypeError):
                            pass
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
            intrinsics=intrinsics,
            significance=significance,
            action_taken=action_taken,
            transform_summary=transform_summary,
            prediction=prediction,
            message=message,
            inventory=inventory,
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
