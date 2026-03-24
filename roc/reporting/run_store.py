"""Query layer for ROC run data stored in DuckLake catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from roc.reporting.ducklake_store import DuckLakeStore

_STEP_BODY_COLUMNS = "step, body"
_EVENT_SUMMARY_NAME = "roc.event.summary"


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
    sequence_summary: dict[str, Any] | None = None
    transform_summary: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    message: str | None = None
    phonemes: list[dict[str, Any]] | None = None
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

    def _query_table(
        self,
        table: str,
        columns: str,
        game_number: int | None,
        extra_where: str = "",
    ) -> pd.DataFrame:
        """Query a table with optional game_number filter and extra WHERE clause."""
        src = self._read_sql(table)
        where_parts: list[str] = []
        params: list[Any] = []
        if extra_where:
            where_parts.append(extra_where)
        if game_number is not None:
            where_parts.append("game_number = ?")
            params.append(game_number)
        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        return self._store.query_df(
            f"SELECT {columns} FROM {src}{where_sql} ORDER BY step",
            params or None,
        )

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
        df = self._query_table("metrics", _STEP_BODY_COLUMNS, game_number)

        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            body = _parse_body(row.get("body"))
            if body is None:
                continue
            entry = _build_entry_from_body(int(row["step"]), body, fields)
            results.append(entry)
        return results

    def _event_body_history(
        self,
        event_name: str,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return parsed body dicts for an event type across all steps.

        Common implementation for history methods that read events by name,
        parse the JSON body, and return ``{step: ..., **body}`` dicts.
        """
        if not self._has_table("events"):
            return []
        df = self._query_table(
            "events",
            _STEP_BODY_COLUMNS,
            game_number,
            extra_where=f"\"event.name\" = '{event_name}'",
        )
        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            body = _parse_body(row.get("body"))
            if body is None:
                continue
            entry = _build_entry_from_body(int(row["step"]), body)
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
        columns = (
            "step,"
            " CAST(body::JSON->>'node_count' AS INTEGER) AS node_count,"
            " CAST(body::JSON->>'node_max' AS INTEGER) AS node_max,"
            " CAST(body::JSON->>'edge_count' AS INTEGER) AS edge_count,"
            " CAST(body::JSON->>'edge_max' AS INTEGER) AS edge_max"
        )
        df = self._query_table(
            "events",
            columns,
            game_number,
            extra_where="\"event.name\" = 'roc.graphdb.summary'",
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
        return self._event_body_history(_EVENT_SUMMARY_NAME, game_number)

    def get_intrinsics_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return intrinsics for all steps in a game (or the whole run).

        Each entry has ``step`` plus raw and normalized intrinsic values.
        Returns an empty list when no events table exists.
        """
        return self._event_body_history("roc.intrinsics", game_number)

    def get_action_history(
        self,
        game_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return action taken for all steps in a game (or the whole run).

        Each entry has ``step`` plus ``action_id`` and optionally ``action_name``.
        Returns an empty list when no events table exists.
        """
        return self._event_body_history("roc.action", game_number)

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
        df = self._query_table(
            "events",
            _STEP_BODY_COLUMNS,
            game_number,
            extra_where="\"event.name\" = 'roc.resolution.decision'",
        )
        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            entry = _build_resolution_entry(row)
            if entry is not None:
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
        df = self._query_table(
            "events",
            _STEP_BODY_COLUMNS,
            game_number,
            extra_where="\"event.name\" = 'roc.resolution.decision'",
        )

        objects, step_to_key, match_events = _collect_resolution_decisions(df)

        id_df = self._query_table(
            "events",
            _STEP_BODY_COLUMNS,
            game_number,
            extra_where="\"event.name\" = 'roc.resolution.new_object_id'",
        )
        _link_object_node_ids(id_df, objects, step_to_key)

        _apply_match_events(match_events, objects)

        return list(objects.values())

    def get_step_data(self, step: int) -> StepData:
        """Assemble all available data for a single step."""
        # Query all tables in one lock acquisition to avoid per-query
        # DuckLake catalog overhead (~100ms each x 5 tables = 500ms).
        dfs = self._store.query_step_batch([step])[step]
        return self._assemble_step(step, dfs)

    def get_steps_data(self, steps: list[int]) -> dict[int, StepData]:
        """Assemble data for multiple steps in a single batched query.

        Much faster than calling ``get_step_data`` in a loop because the
        underlying DuckLake queries use ``step IN (...)`` -- each table is
        scanned once for all requested steps.
        """
        all_dfs = self._store.query_step_batch(steps)
        results: dict[int, StepData] = {}
        for step in steps:
            dfs = all_dfs.get(step, {})
            results[step] = self._assemble_step(step, dfs)
        return results

    def _assemble_step(self, step: int, dfs: dict[str, pd.DataFrame]) -> StepData:
        """Build a StepData from pre-fetched DataFrames for each table."""
        screen, game_number, timestamp = _extract_screen(dfs)
        saliency = _extract_first_body(dfs, "saliency")
        parsed_events = _parse_events(dfs.get("events", pd.DataFrame()))
        game_metrics = _extract_first_body(dfs, "metrics")
        logs = _extract_logs(dfs)

        return StepData(
            step=step,
            game_number=game_number,
            timestamp=timestamp,
            screen=screen,
            saliency=saliency,
            features=parsed_events.features,
            object_info=parsed_events.object_info,
            focus_points=parsed_events.focus_points,
            attenuation=parsed_events.attenuation,
            resolution_metrics=parsed_events.resolution_metrics,
            graph_summary=parsed_events.graph_summary,
            event_summary=parsed_events.event_summary,
            game_metrics=game_metrics,
            logs=logs,
            intrinsics=parsed_events.intrinsics,
            significance=parsed_events.significance,
            action_taken=parsed_events.action_taken,
            sequence_summary=parsed_events.sequence_summary,
            transform_summary=parsed_events.transform_summary,
            prediction=parsed_events.prediction,
            message=parsed_events.message,
            phonemes=parsed_events.phonemes,
            inventory=parsed_events.inventory,
        )


@dataclass
class _ParsedEvents:
    """Accumulated results from parsing event rows for a single step."""

    features: list[dict[str, Any]] | None = None
    object_info: list[dict[str, Any]] | None = None
    focus_points: list[dict[str, Any]] | None = None
    attenuation: dict[str, Any] | None = None
    resolution_metrics: dict[str, Any] | None = None
    graph_summary: dict[str, Any] | None = None
    event_summary: list[dict[str, Any]] | None = None
    intrinsics: dict[str, Any] | None = None
    significance: float | None = None
    action_taken: dict[str, Any] | None = None
    sequence_summary: dict[str, Any] | None = None
    transform_summary: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    message: str | None = None
    phonemes: list[dict[str, Any]] | None = None
    inventory: list[dict[str, Any]] | None = None


def _extract_screen(
    dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any] | None, int, int | None]:
    """Extract screen data, game_number, and timestamp from screen DataFrame."""
    screen_df = dfs.get("screens", pd.DataFrame())
    if len(screen_df) == 0:
        return None, 0, None
    row = screen_df.iloc[0]
    screen = _parse_body(row.get("body"))
    return screen, int(row["game_number"]), row.get("timestamp")


def _extract_first_body(
    dfs: dict[str, pd.DataFrame],
    table: str,
) -> dict[str, Any] | None:
    """Extract parsed body from the first row of a DataFrame, or None."""
    df = dfs.get(table, pd.DataFrame())
    if len(df) == 0:
        return None
    return _parse_body(df.iloc[0].get("body"))


def _extract_logs(dfs: dict[str, pd.DataFrame]) -> list[dict[str, Any]] | None:
    """Extract log records from the logs DataFrame."""
    logs_df = dfs.get("logs", pd.DataFrame())
    if len(logs_df) == 0:
        return None
    return cast(list[dict[str, Any]], logs_df.to_dict("records"))


def _body_or_raw(
    body: dict[str, Any] | None,
    row_dict: dict[str, Any],
) -> dict[str, Any]:
    """Return parsed body or a fallback raw dict."""
    return body if body is not None else {"raw": row_dict.get("body", "")}


def _parse_json_list(raw_body: str | None) -> list[dict[str, Any]] | None:
    """Try to parse a raw body string as a JSON list; return None on failure."""
    if not raw_body:
        return None
    try:
        import json as _json

        parsed = _json.loads(raw_body)
        if isinstance(parsed, list):
            return cast(list[dict[str, Any]], parsed)
    except (ValueError, TypeError):
        pass
    return None


def _append_to_list(
    target: list[dict[str, Any]] | None,
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append an item to a list, creating it if None."""
    if target is None:
        target = []
    target.append(item)
    return target


def _handle_event_row(
    result: _ParsedEvents,
    row_dict: dict[str, Any],
    event_name: str,
    body: dict[str, Any] | None,
    event_list: list[dict[str, Any]],
) -> None:
    """Dispatch a single event row into the appropriate _ParsedEvents field."""
    handler = _EVENT_HANDLERS.get(event_name)
    if handler is not None:
        handler(result, row_dict, body)
    else:
        event_list.append(row_dict)


def _handle_features(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    result.features = _append_to_list(result.features, _body_or_raw(body, row_dict))


def _handle_object(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    result.object_info = _append_to_list(result.object_info, _body_or_raw(body, row_dict))


def _handle_focus_points(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    result.focus_points = _append_to_list(result.focus_points, _body_or_raw(body, row_dict))


def _handle_attenuation(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    if body is not None:
        result.attenuation = {k: v for k, v in body.items() if k != "saliency_grid"}
    else:
        result.attenuation = body


def _handle_simple_body(attr: str) -> Any:
    """Create a handler that assigns parsed body directly to a _ParsedEvents attribute."""

    def handler(
        result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
    ) -> None:
        setattr(result, attr, body)

    return handler


def _handle_significance(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    if body is not None and "significance" in body:
        result.significance = float(body["significance"])


def _handle_event_summary(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    # event_summary items are accumulated in the event_list passed to _handle_event_row,
    # but roc.event.summary uses body_or_raw and goes to the same list.
    # We handle it specially in _parse_events below.
    pass


def _handle_message(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    raw_body = row_dict.get("body", "")
    result.message = str(raw_body).strip() if raw_body else None


def _handle_phonemes(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    result.phonemes = _parse_json_list(row_dict.get("body", ""))


def _handle_inventory(
    result: _ParsedEvents, row_dict: dict[str, Any], body: dict[str, Any] | None
) -> None:
    result.inventory = _parse_json_list(row_dict.get("body", ""))


# Map event names to handler functions. This replaces the if/elif chain.
_EventHandler = Any  # Callable[[_ParsedEvents, dict, dict | None], None]
_EVENT_HANDLERS: dict[str, _EventHandler] = {
    "roc.attention.features": _handle_features,
    "roc.attention.object": _handle_object,
    "roc.attention.focus_points": _handle_focus_points,
    "roc.saliency_attenuation": _handle_attenuation,
    "roc.resolution.decision": _handle_simple_body("resolution_metrics"),
    "roc.graphdb.summary": _handle_simple_body("graph_summary"),
    _EVENT_SUMMARY_NAME: _handle_event_summary,
    "roc.intrinsics": _handle_simple_body("intrinsics"),
    "roc.significance": _handle_significance,
    "roc.action": _handle_simple_body("action_taken"),
    "roc.sequence_summary": _handle_simple_body("sequence_summary"),
    "roc.transform_summary": _handle_simple_body("transform_summary"),
    "roc.prediction": _handle_simple_body("prediction"),
    "roc.message": _handle_message,
    "roc.phonemes": _handle_phonemes,
    "roc.inventory": _handle_inventory,
}


def _parse_events(events_df: pd.DataFrame) -> _ParsedEvents:
    """Parse all event rows into a _ParsedEvents accumulator."""
    result = _ParsedEvents()
    if len(events_df) == 0:
        return result

    event_list: list[dict[str, Any]] = []
    for _, row in events_df.iterrows():
        row_dict = cast(dict[str, Any], dict(row))
        event_name = row_dict.get("event.name", "")
        body = _parse_body(row_dict.get("body"))

        if event_name == _EVENT_SUMMARY_NAME:
            event_list.append(_body_or_raw(body, row_dict))
        else:
            _handle_event_row(result, row_dict, event_name, body, event_list)

    if event_list:
        result.event_summary = event_list
    return result


def _extract_observed_attrs(
    body: dict[str, Any],
    shape: str | None,
    color: str | None,
    glyph: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Merge observed_attrs from body into shape/color/glyph, return (shape, color, glyph, type)."""
    oa = body.get("observed_attrs", {})
    if not oa:
        return shape, color, glyph, None
    shape = shape or oa.get("char")
    color = color or oa.get("color")
    glyph = glyph or (str(oa["glyph"]) if oa.get("glyph") is not None else None)
    return shape, color, glyph, oa.get("type")


def _collect_resolution_decisions(
    df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[int, str], list[dict[str, Any]]]:
    """Pass 1: collect new_object entries and match events from resolution decisions."""
    objects: dict[str, dict[str, Any]] = {}
    step_to_key: dict[int, str] = {}
    match_events: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        body = _parse_body(row.get("body"))
        if body is None:
            continue
        step = int(row["step"])
        outcome = body.get("outcome")
        shape, color, glyph = parse_feature_attrs(body.get("features", []))
        shape, color, glyph, obj_type = _extract_observed_attrs(body, shape, color, glyph)

        if outcome == "new_object":
            key = f"new@{step}"
            step_to_key[step] = key
            objects[key] = {
                "shape": shape,
                "glyph": glyph,
                "color": color,
                "type": obj_type,
                "node_id": None,
                "step_added": step,
                "match_count": 0,
            }
        elif outcome == "match":
            match_events.append(
                {
                    "matched_id": body.get("matched_object_id"),
                    "matched_attrs": body.get("matched_attrs", {}),
                    "shape": shape,
                    "glyph": glyph,
                    "color": color,
                    "type": obj_type,
                }
            )

    return objects, step_to_key, match_events


def _link_object_node_ids(
    id_df: pd.DataFrame,
    objects: dict[str, dict[str, Any]],
    step_to_key: dict[int, str],
) -> None:
    """Pass 2: link new_object_id events to objects, re-keying by node ID."""
    for _, row in id_df.iterrows():
        body = _parse_body(row.get("body"))
        if body is None:
            continue
        nid = body.get("new_object_id")
        if nid is None:
            continue
        step = int(row["step"])
        mid = str(nid)
        old_key = step_to_key.get(step)
        if old_key and old_key in objects:
            obj = objects.pop(old_key)
            obj["node_id"] = mid
            objects[mid] = obj


def _apply_match_events(
    match_events: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
) -> None:
    """Pass 3: process match events, incrementing counts or creating new entries."""
    for m in match_events:
        matched_id = m["matched_id"]
        if matched_id is None:
            continue
        mid = str(matched_id)
        if mid in objects:
            _update_existing_object(objects[mid], m)
        else:
            objects[mid] = _create_object_from_match(mid, m)


def _update_existing_object(obj: dict[str, Any], match: dict[str, Any]) -> None:
    """Increment match count and fill in missing attributes from a match event."""
    obj["match_count"] += 1
    for attr in ("shape", "glyph", "color", "type"):
        if match.get(attr) and not obj.get(attr):
            obj[attr] = match[attr]


def _create_object_from_match(mid: str, match: dict[str, Any]) -> dict[str, Any]:
    """Create a new object entry from a match event when the object was not yet tracked."""
    ma = match["matched_attrs"]
    return {
        "shape": ma.get("char", match["shape"]),
        "glyph": str(ma["glyph"]) if ma.get("glyph") is not None else match["glyph"],
        "color": ma.get("color", match["color"]),
        "type": match.get("type"),
        "node_id": mid,
        "step_added": None,
        "match_count": 1,
    }


def _build_resolution_entry(row: Any) -> dict[str, Any] | None:
    """Build a resolution history entry from a DataFrame row."""
    body = _parse_body(row.get("body"))
    if body is None:
        return None
    outcome = body.get("outcome", "unknown")
    entry: dict[str, Any] = {"step": int(row["step"]), "outcome": outcome}
    if outcome == "match":
        entry["correct"] = compute_match_correctness(body)
    return entry


def _build_entry_from_body(
    step: int,
    body: dict[str, Any],
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build a step entry dict from a parsed body, optionally filtering to specific fields."""
    entry: dict[str, Any] = {"step": step}
    if fields:
        for f in fields:
            if f in body:
                entry[f] = body[f]
    else:
        entry.update(body)
    return entry


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


def _parse_composite_feature(s: str, prefix_len: int) -> tuple[str, str, str] | None:
    """Extract shape, color, glyph from a composite feature string (FloodNode/LineNode).

    Format: ``PrefixNode(glyph_id,size,color_int,shape_char)``.
    Returns ``(shape_char, color_name, glyph_id_str)`` or ``None`` on parse failure.
    """
    content = s[prefix_len:-1]  # strip prefix and trailing ")"
    # Last char is always the shape character, preceded by a comma separator
    if len(content) < 6 or content[-2] != ",":
        return None
    shape_char = content[-1]
    parts = content[:-2].split(",")
    if len(parts) != 3:
        return None
    glyph_str = parts[0]
    try:
        color_idx = int(parts[2])
    except ValueError:
        return None
    color_name = _COLOR_NAMES.get(color_idx)
    return shape_char, color_name or str(color_idx), glyph_str


# Standard terminal color codes used by ColorNode.attr_strs (NetHack color.h).
_COLOR_NAMES: dict[int, str] = {
    0: "BLACK",
    1: "RED",
    2: "GREEN",
    3: "BROWN",
    4: "BLUE",
    5: "MAGENTA",
    6: "CYAN",
    7: "GREY",
    8: "NO COLOR",
    9: "ORANGE",
    10: "BRIGHT GREEN",
    11: "YELLOW",
    12: "BRIGHT BLUE",
    13: "BRIGHT MAGENTA",
    14: "BRIGHT CYAN",
    15: "WHITE",
    16: "MAX",
}


def _parse_single_feature(s: str) -> tuple[str | None, str | None, str | None]:
    """Parse a single feature string into (shape, color, glyph) components.

    Returns a tuple where each element is set only if this feature provides it.
    """
    if not s.endswith(")"):
        return None, None, None
    if s.startswith("ShapeNode("):
        return s[10:-1], None, None
    if s.startswith("ColorNode("):
        return None, s[10:-1], None
    if s.startswith("SingleNode("):
        return None, None, s[11:-1]
    # Composite features embed all three attributes
    prefix_len = _COMPOSITE_PREFIXES.get(s.split("(", 1)[0] + "(")
    if prefix_len is not None:
        parsed = _parse_composite_feature(s, prefix_len)
        if parsed is not None:
            return parsed[0], parsed[1], parsed[2]
    return None, None, None


# Prefix -> length for composite node types that embed shape/color/glyph.
_COMPOSITE_PREFIXES: dict[str, int] = {
    "FloodNode(": 10,
    "LineNode(": 9,
}


def parse_feature_attrs(features: list[Any]) -> tuple[str | None, str | None, str | None]:
    """Parse shape, color, glyph from feature strings.

    Used by both RunStore and DataStore for resolution event processing.
    Recognises ShapeNode, ColorNode, SingleNode directly, and falls back to
    composite features (FloodNode, LineNode) which embed shape/color/glyph.
    """
    shape = color = glyph = None
    for f in features:
        f_shape, f_color, f_glyph = _parse_single_feature(str(f))
        shape = shape or f_shape
        color = color or f_color
        glyph = glyph or f_glyph
    return shape, color, glyph


def compute_match_correctness(body: dict[str, Any]) -> bool | None:
    """Determine whether a 'match' resolution decision was correct.

    Compares observed features with the matched object's stored attributes.
    Returns True/False for correctness, or None if comparison data is missing.
    """
    matched_attrs = body.get("matched_attrs")
    features = body.get("features", [])
    if not matched_attrs or not isinstance(features, list):
        return None
    obs_shape, obs_color, obs_glyph = parse_feature_attrs(features)
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
    return correct
