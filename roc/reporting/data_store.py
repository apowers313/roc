"""Unified data access layer for the ROC debug dashboard.

Provides a single DataStore class that:
- Receives live step pushes and incrementally builds per-game history indices
- Serves history queries from in-memory indices for live runs (O(1) lookup)
- Delegates to DuckLake/RunStore for historical runs (unchanged path)
- Absorbs the module-level globals previously scattered across api_server.py
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from roc.framework.logger import logger
from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.graph_service import GraphService
from roc.reporting.run_store import (
    RunStore,
    StepData,
    compute_match_correctness,
    parse_feature_attrs,
)
from roc.reporting.step_buffer import StepBuffer


class RunSummary(BaseModel):
    """Summary metadata for a single run."""

    name: str
    games: int
    steps: int


def _extract_event_attrs(ev: dict[str, Any]) -> dict[str, Any]:
    """Extract shape/color/glyph/type from a resolution event."""
    features = ev.get("features", [])
    shape, color, glyph = parse_feature_attrs(features)

    oa = ev.get("observed_attrs", {})
    if oa:
        shape = shape or oa.get("char")
        color = color or oa.get("color")
        glyph = glyph or (str(oa["glyph"]) if oa.get("glyph") is not None else None)

    return {"shape": shape, "color": color, "glyph": glyph, "type": oa.get("type")}


def _collect_new_object(
    ev: dict[str, Any],
    attrs: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    step_to_key: dict[int, str],
) -> None:
    """Register a new_object event in the objects dict."""
    step = ev["step"]
    nid = ev.get("new_object_id")
    if nid is not None:
        mid = str(nid)
        objects[mid] = {**attrs, "node_id": mid, "step_added": step, "match_count": 0}
        return
    key = f"new@{step}"
    step_to_key[step] = key
    objects[key] = {**attrs, "node_id": None, "step_added": step, "match_count": 0}


def _apply_match_event(m: dict[str, Any], objects: dict[str, dict[str, Any]]) -> None:
    """Apply a match event to the objects dict, updating or creating the entry."""
    matched_id = m["matched_id"]
    if matched_id is None:
        return
    mid = str(matched_id)
    if mid in objects:
        objects[mid]["match_count"] += 1
        for attr in ("shape", "glyph", "color", "type"):
            if m.get(attr) and not objects[mid].get(attr):
                objects[mid][attr] = m[attr]
        return
    ma = m["matched_attrs"]
    objects[mid] = {
        "shape": ma.get("char", m["shape"]),
        "glyph": str(ma["glyph"]) if ma.get("glyph") is not None else m["glyph"],
        "color": ma.get("color", m["color"]),
        "type": m.get("type"),
        "node_id": mid,
        "step_added": None,
        "match_count": 1,
    }


@dataclass
class _GameIndex:
    """Incremental history accumulator for a single game.

    Each field is appended to on every step push.  Memory cost is
    ~5.6 MB for 100K steps x 6 indices (pointers to existing dicts).
    """

    graph_history: list[dict[str, Any]] = field(default_factory=list)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    intrinsics_history: list[dict[str, Any]] = field(default_factory=list)
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    resolution_events: list[dict[str, Any]] = field(default_factory=list)


def _wrap_legacy_cycles(data: StepData) -> None:
    """Wrap old single-field data into multi-cycle list format for backward compat.

    Historical runs have saliency/resolution_metrics as single values.
    New runs have saliency_cycles/resolution_cycles as lists.
    This wraps old format into new format at read time so the frontend always
    sees the list format.
    """
    if data.saliency_cycles is None and data.saliency is not None:
        data.saliency_cycles = [{"saliency": data.saliency, "attenuation": {}}]
    if data.resolution_cycles is None and data.resolution_metrics is not None:
        data.resolution_cycles = [data.resolution_metrics]


class DataStore:
    """Single point of access for all dashboard data.

    Receives live step pushes and incrementally builds per-game history
    indices.  Serves history queries from in-memory indices for live runs.
    Delegates to DuckLake/RunStore for historical runs.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._live_run_name: str | None = None
        self._live_buffer: StepBuffer | None = None
        self._live_store: DuckLakeStore | None = None
        self._indices: dict[int, _GameIndex] = {}
        self._lock = threading.Lock()
        self._run_stores: dict[str, RunStore] = {}
        self._store_lock = threading.Lock()
        self._run_summary_cache: dict[str, RunSummary] = {}
        self._action_maps: dict[str, list[dict[str, Any]]] = {}

    @property
    def data_dir(self) -> Path:
        """The root data directory containing run subdirectories."""
        return self._data_dir

    @property
    def live_run_name(self) -> str | None:
        """Name of the currently live run, or None."""
        return self._live_run_name

    @property
    def live_buffer(self) -> StepBuffer | None:
        """The StepBuffer for the live session, or None."""
        return self._live_buffer

    # -------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------

    def set_live_session(
        self,
        run_name: str,
        buffer: StepBuffer,
        store: DuckLakeStore | None = None,
    ) -> None:
        """Configure a live session with the given buffer.

        Args:
            run_name: Directory name of the live run.
            buffer: StepBuffer that receives live pushes.
            store: Optional DuckLakeStore for in-process mode (shared
                   with the game writer for log supplementation).
        """
        self._live_run_name = run_name
        self._live_buffer = buffer
        self._live_store = store
        with self._lock:
            self._indices.clear()
        with self._store_lock:
            self._run_stores.pop(run_name, None)
        buffer.add_listener(self._on_step_pushed)
        logger.info("DataStore: live session for run {}", run_name)

    def clear_live_session(self) -> None:
        """Stop listening for new live data but keep the buffer readable.

        The buffer data stays accessible for historical queries until a
        new game starts (which calls ``set_live_session`` to replace it).
        This prevents cycle data from being lost when the game ends.

        IMPORTANT: this method is called around the same time that
        ``Observability.reset()`` closes the in-process ``DuckLakeStore``
        that was registered here via ``set_live_session(store=...)``. Any
        cached ``RunStore`` wrapping that store is about to point at a
        dead connection, so we evict the cache entry and drop our own
        reference. Future queries for this run will create a fresh
        read-only ``DuckLakeStore`` from the run directory on disk.
        Without this eviction, later queries for any run (not just the
        ended one) can hit a stale closed-connection error during
        iteration in list_runs / populate_run_summaries.
        """
        if self._live_buffer is not None:
            self._live_buffer.remove_listener(self._on_step_pushed)
        with self._store_lock:
            if self._live_run_name is not None:
                self._run_stores.pop(self._live_run_name, None)
        self._live_store = None
        logger.debug("DataStore: live session paused (buffer retained)")

    # -------------------------------------------------------------------
    # Live step push and indexing
    # -------------------------------------------------------------------

    def push_live_step(self, step_data: StepData) -> None:
        """Push a step to the live buffer."""
        if self._live_buffer is not None:
            self._live_buffer.push(step_data)

    def _on_step_pushed(self) -> None:
        """StepBuffer listener: index the latest step."""
        if self._live_buffer is None:
            return
        step_data = self._live_buffer.get_latest()
        if step_data is None:
            return
        self._index_step(step_data)

    def _index_step(self, step_data: StepData) -> None:
        """Incrementally add a step's data to the per-game history indices."""
        game = step_data.game_number
        step = step_data.step

        with self._lock:
            if game not in self._indices:
                self._indices[game] = _GameIndex()
            idx = self._indices[game]

        self._append_if_present(idx.graph_history, step, step_data.graph_summary)
        self._index_event_summary(idx, step, step_data.event_summary)
        self._append_if_present(idx.intrinsics_history, step, step_data.intrinsics)
        self._append_if_present(idx.metrics_history, step, step_data.game_metrics)
        self._append_if_present(idx.action_history, step, step_data.action_taken)
        self._append_if_present(idx.resolution_events, step, step_data.resolution_metrics)

    @staticmethod
    def _append_if_present(
        history: list[dict[str, Any]], step: int, data: dict[str, Any] | None
    ) -> None:
        """Append a step entry to a history list if data is not None."""
        if data is not None:
            history.append({"step": step, **data})

    @staticmethod
    def _index_event_summary(
        idx: _GameIndex, step: int, event_summary: list[dict[str, Any]] | None
    ) -> None:
        """Index event summary entries for a step."""
        if event_summary is None:
            return
        for entry in event_summary:
            if isinstance(entry, dict):
                idx.event_history.append({"step": step, **entry})

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def is_live(self, run_name: str) -> bool:
        """Check if the given run is the currently active live session."""
        return run_name == self._live_run_name and self._live_buffer is not None

    def _is_live(self, run_name: str) -> bool:
        return self.is_live(run_name)

    def _get_live_history(self, game: int | None, field_name: str) -> list[dict[str, Any]]:
        """Get accumulated history from live indices."""
        with self._lock:
            if game is not None:
                idx = self._indices.get(game)
                return list(getattr(idx, field_name)) if idx else []
            result: list[dict[str, Any]] = []
            for idx in self._indices.values():
                result.extend(getattr(idx, field_name))
            return result

    def _get_run_store(self, run_name: str) -> RunStore:
        """Get or create a RunStore for a historical or live run."""
        with self._store_lock:
            if run_name not in self._run_stores:
                if run_name == self._live_run_name and self._live_store is not None:
                    self._run_stores[run_name] = RunStore(self._live_store)
                else:
                    run_dir = self._data_dir / run_name
                    if not run_dir.is_dir():
                        raise FileNotFoundError(f"Run not found: {run_name}")
                    safe_alias = "r_" + run_name.replace("-", "_").replace(".", "_")
                    dl_store = DuckLakeStore(run_dir, read_only=True, alias=safe_alias)
                    self._run_stores[run_name] = RunStore(dl_store)
            return self._run_stores[run_name]

    def _get_run_summary(self, name: str) -> RunSummary:
        """Return a RunSummary, from cache if available."""
        is_live = name == self._live_run_name
        if not is_live and name in self._run_summary_cache:
            return self._run_summary_cache[name]
        if is_live and self._live_buffer is not None and len(self._live_buffer) > 0:
            counts = self._live_buffer.steps_per_game()
            return RunSummary(
                name=name,
                games=len(counts),
                steps=sum(counts.values()),
            )
        if is_live:
            try:
                store = self._get_run_store(name)
                games_df = store.list_games()
                games = len(games_df)
                steps = int(games_df["steps"].sum()) if games > 0 else store.step_count()
                return RunSummary(name=name, games=games, steps=steps)
            except Exception:
                return RunSummary(name=name, games=0, steps=0)
        return RunSummary(name=name, games=0, steps=0)

    def populate_run_summaries(self) -> None:
        """Background thread target: scan all runs and cache game/step counts.

        Processes newest runs first (most likely to be requested).
        Uses temporary DuckDB connections closed immediately after each
        query to avoid holding locks that block request-serving threads.
        """
        names = RunStore.list_runs(self._data_dir)
        names.reverse()
        for name in names:
            if name in self._run_summary_cache or name == self._live_run_name:
                continue
            self._populate_single_run_summary(name)
            time.sleep(0.05)

    def _populate_single_run_summary(self, name: str) -> None:
        """Compute and cache the summary for a single historical run."""
        run_dir = self._data_dir / name
        if not run_dir.is_dir():
            return
        try:
            safe_alias = "sum_" + name.replace("-", "_").replace(".", "_")
            dl_store = DuckLakeStore(run_dir, read_only=True, alias=safe_alias)
            try:
                store = RunStore(dl_store)
                games_df = store.list_games()
                games = len(games_df)
                steps = int(games_df["steps"].sum()) if games > 0 else store.step_count()
                self._run_summary_cache[name] = RunSummary(name=name, games=games, steps=steps)
            finally:
                dl_store.close()
        except Exception:
            self._run_summary_cache[name] = RunSummary(name=name, games=0, steps=0)

    # -------------------------------------------------------------------
    # Public query methods
    # -------------------------------------------------------------------

    def list_runs(self, min_steps: int = 10) -> list[RunSummary]:
        """List available runs with metadata.

        ``min_steps`` filters out short-lived / crashed runs (default 10).
        """
        names = RunStore.list_runs(self._data_dir)
        names.reverse()
        if self._live_run_name and self._live_run_name not in names:
            names.insert(0, self._live_run_name)
        results: list[RunSummary] = []
        for name in names:
            summary = self._get_run_summary(name)
            if summary.steps >= min_steps:
                results.append(summary)
        return results

    def list_games(self, run_name: str) -> list[dict[str, Any]]:
        """List games in a run."""
        if self._is_live(run_name):
            assert self._live_buffer is not None
            counts = self._live_buffer.steps_per_game()
            return [
                {"game_number": g, "steps": counts.get(g, 0), "start_ts": None, "end_ts": None}
                for g in sorted(counts.keys())
            ]
        store = self._get_run_store(run_name)
        df = store.list_games()
        return [
            {
                "game_number": int(row["game_number"]),
                "steps": int(row["steps"]),
                "start_ts": int(row["start_ts"]) if row.get("start_ts") is not None else None,
                "end_ts": int(row["end_ts"]) if row.get("end_ts") is not None else None,
            }
            for _, row in df.iterrows()
        ]

    def get_step_range(self, run_name: str, game: int | None = None) -> tuple[int, int]:
        """Get min/max step for a run or game."""
        if self._is_live(run_name):
            assert self._live_buffer is not None
            if len(self._live_buffer) > 0:
                if game is not None:
                    return self._live_buffer.step_range_for_game(game)
                return (self._live_buffer.min_step, self._live_buffer.max_step)
            return (0, 0)
        store = self._get_run_store(run_name)
        return store.step_range(game)

    def get_step_data(self, run_name: str, step: int) -> StepData:
        """Get all data for a specific step.

        For the live run, tries StepBuffer first (in-memory, instant).
        Falls back to RunStore for evicted steps.  Supplements missing
        logs from DuckLake when the buffer hit has no log data.
        """
        if run_name == self._live_run_name and self._live_buffer is not None:
            buf_data = self._live_buffer.get_step(step)
            if buf_data is not None:
                self._supplement_logs(buf_data, run_name, step)
                return buf_data
        store = self._get_run_store(run_name)
        data = store.get_step_data(step)
        _wrap_legacy_cycles(data)
        return data

    @staticmethod
    def _wrap_legacy_step(data: StepData) -> None:
        """Wrap old single-field data into multi-cycle list format for backward compat."""
        _wrap_legacy_cycles(data)

    def _supplement_logs(self, buf_data: StepData, run_name: str, step: int) -> None:
        """Fill in missing logs from DuckLake when the buffer has no log data."""
        if buf_data.logs is not None or self._live_store is None:
            return
        try:
            store = self._get_run_store(run_name)
            logs_df = store.get_step(step, "logs")
            if len(logs_df) > 0:
                from typing import cast

                buf_data.logs = cast(list[dict[str, Any]], logs_df.to_dict("records"))
        except Exception:
            pass

    def get_steps_batch(self, run_name: str, steps: list[int]) -> dict[int, StepData]:
        """Get data for multiple steps in a single request."""
        if run_name != self._live_run_name or self._live_buffer is None:
            store = self._get_run_store(run_name)
            return store.get_steps_data(steps)
        result: dict[int, StepData] = {}
        for step in steps:
            try:
                result[step] = self.get_step_data(run_name, step)
            except Exception:
                pass
        return result

    def get_live_step(self, step: int) -> StepData | None:
        """Get step data from the live buffer."""
        if self._live_buffer is None:
            return None
        return self._live_buffer.get_step(step)

    def get_live_status(self) -> dict[str, Any]:
        """Return live session status from the buffer."""
        if self._live_buffer is not None and len(self._live_buffer) > 0:
            latest = self._live_buffer.get_latest()
            return {
                "active": True,
                "run_name": self._live_run_name,
                "step": latest.step if latest else 0,
                "game_number": latest.game_number if latest else 0,
                "step_min": self._live_buffer.min_step,
                "step_max": self._live_buffer.max_step,
                "game_numbers": self._live_buffer.game_numbers,
            }
        return {
            "active": False,
            "run_name": None,
            "step": 0,
            "game_number": 0,
            "step_min": 0,
            "step_max": 0,
            "game_numbers": [],
        }

    # -------------------------------------------------------------------
    # History queries
    # -------------------------------------------------------------------

    def get_graph_history(self, run_name: str, game: int | None = None) -> list[dict[str, Any]]:
        """Get graph_summary for all steps in a game."""
        if self._is_live(run_name):
            return self._get_live_history(game, "graph_history")
        return self._get_run_store(run_name).get_graph_history(game)

    def get_event_history(self, run_name: str, game: int | None = None) -> list[dict[str, Any]]:
        """Get event_summary for all steps in a game."""
        if self._is_live(run_name):
            return self._get_live_history(game, "event_history")
        return self._get_run_store(run_name).get_event_history(game)

    def get_intrinsics_history(
        self, run_name: str, game: int | None = None
    ) -> list[dict[str, Any]]:
        """Get intrinsics for all steps in a game."""
        if self._is_live(run_name):
            return self._get_live_history(game, "intrinsics_history")
        return self._get_run_store(run_name).get_intrinsics_history(game)

    def get_metrics_history(
        self,
        run_name: str,
        game: int | None = None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get game_metrics for all steps in a game."""
        if self._is_live(run_name):
            raw = self._get_live_history(game, "metrics_history")
            if fields is None:
                return raw
            return [{"step": e["step"], **{f: e[f] for f in fields if f in e}} for e in raw]
        return self._get_run_store(run_name).get_metrics_history(game, fields)

    def get_action_history(self, run_name: str, game: int | None = None) -> list[dict[str, Any]]:
        """Get action_taken for all steps in a game."""
        if self._is_live(run_name):
            return self._get_live_history(game, "action_history")
        return self._get_run_store(run_name).get_action_history(game)

    def get_resolution_history(
        self, run_name: str, game: int | None = None
    ) -> list[dict[str, Any]]:
        """Get resolution accuracy history for all steps in a game."""
        if self._is_live(run_name):
            return self._live_resolution_history(game)
        return self._get_run_store(run_name).get_resolution_history(game)

    def get_all_objects(self, run_name: str, game: int | None = None) -> list[dict[str, Any]]:
        """Get all resolved objects with attributes, match counts, and creation step."""
        if self._is_live(run_name):
            return self._live_all_objects(game)
        return self._get_run_store(run_name).get_all_objects(game)

    def set_action_map(self, run_name: str, action_map: list[dict[str, Any]]) -> None:
        """Store the action map for a run and persist to disk."""
        self._action_maps[run_name] = action_map
        action_map_path = self._data_dir / run_name / "action_map.json"
        try:
            action_map_path.parent.mkdir(parents=True, exist_ok=True)
            action_map_path.write_text(json.dumps(action_map))
        except Exception:
            logger.opt(exception=True).warning("Failed to persist action_map.json")

    def get_action_map(self, run_name: str) -> list[dict[str, Any]] | None:
        """Get the full action map for a run (from memory or disk)."""
        cached = self._action_maps.get(run_name)
        if cached is not None:
            return cached
        action_map_path = self._data_dir / run_name / "action_map.json"
        if action_map_path.exists():
            try:
                data: list[dict[str, Any]] = json.loads(action_map_path.read_text())
                self._action_maps[run_name] = data
                return data
            except Exception:
                return None
        return None

    def get_schema(self, run_name: str) -> dict[str, Any] | None:
        """Get the graph database schema for a run.

        Reads from the schema.json file saved in the run directory.
        Returns None if no schema is available.
        """
        schema_path = self._data_dir / run_name / "schema.json"
        if schema_path.exists():
            try:
                result: dict[str, Any] = json.loads(schema_path.read_text())
                return result
            except Exception:
                return None
        return None

    # -------------------------------------------------------------------
    # Graph service
    # -------------------------------------------------------------------

    def get_graph_service(self) -> GraphService:
        """Return a GraphService backed by this DataStore's data directory."""
        return GraphService(self._data_dir)

    def get_run_dir(self, run_name: str) -> Path:
        """Return the directory path for a run."""
        return self._data_dir / run_name

    # -------------------------------------------------------------------
    # Live resolution/objects computation
    # -------------------------------------------------------------------

    def _live_resolution_history(self, game: int | None) -> list[dict[str, Any]]:
        """Build resolution history from accumulated resolution_events."""
        events = self._get_live_history(game, "resolution_events")
        results: list[dict[str, Any]] = []
        for ev in events:
            outcome = ev.get("outcome", "unknown")
            entry: dict[str, Any] = {"step": ev["step"], "outcome": outcome}
            if outcome == "match":
                entry["correct"] = compute_match_correctness(ev)
            results.append(entry)
        return results

    def _live_all_objects(self, game: int | None) -> list[dict[str, Any]]:
        """Build all-objects list from accumulated resolution_events.

        Uses the same 3-pass algorithm as RunStore.get_all_objects, but
        reads from in-memory indices.  new_object_id is available directly
        in resolution_metrics (set by object.py before StepData assembly),
        so no separate DuckDB query is needed.
        """
        events = self._get_live_history(game, "resolution_events")

        objects: dict[str, dict[str, Any]] = {}
        step_to_key: dict[int, str] = {}
        match_events: list[dict[str, Any]] = []

        # Pass 1: collect new_object and match entries
        for ev in events:
            attrs = _extract_event_attrs(ev)
            outcome = ev.get("outcome")
            if outcome == "new_object":
                _collect_new_object(ev, attrs, objects, step_to_key)
            elif outcome == "match":
                match_events.append(
                    {
                        "matched_id": ev.get("matched_object_id"),
                        "matched_attrs": ev.get("matched_attrs", {}),
                        **attrs,
                    }
                )

        # Pass 2: (skipped -- new_object_id handled in Pass 1 for live data)

        # Pass 3: process match events
        for m in match_events:
            _apply_match_event(m, objects)

        return list(objects.values())
