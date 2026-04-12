"""FastAPI + Socket.io server for the React debug dashboard.

Provides a clean separation of concerns:
- REST endpoints serve step data via ``RunReader`` (cache-first, then DuckLake)
- Socket.io pushes ``step_added`` invalidation events to connected browsers
- In production, serves the React static build from dashboard-ui/dist/
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from roc.framework.logger import logger
from roc.reporting.graph_api import graph_router
from roc.reporting.run_reader import RunReader
from roc.reporting.run_registry import RunRegistry
from roc.reporting.run_store import StepData
from roc.reporting.run_writer import RunWriter
from roc.reporting.step_cache import StepCache
from roc.reporting.types import RunSummary

# ---------------------------------------------------------------------------
# Socket.io server
# ---------------------------------------------------------------------------

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ROC Debug Dashboard API")
app.include_router(graph_router)

_NOT_INITIALIZED = "Dashboard not initialized"
_NOT_INITIALIZED_RESPONSE: dict[int | str, dict[str, Any]] = {
    503: {"description": _NOT_INITIALIZED}
}

_server_ready = threading.Event()


@app.on_event("startup")
async def _capture_event_loop() -> None:
    """Capture the asyncio event loop and signal that the server is ready.

    Called by uvicorn after the event loop is fully initialized.
    The threading.Event unblocks start_dashboard() so the game loop
    doesn't push to the writer before the async infrastructure can
    handle cross-thread coroutine submissions.
    """
    import asyncio

    global _sio_loop
    _sio_loop = asyncio.get_running_loop()
    _server_ready.set()


# Process-wide singletons that every step/range/history endpoint reads
# through. ``RunReader`` -> ``RunRegistry`` -> ``StepCache`` ->
# ``DuckLakeStore`` is the only public read path. ``RunWriter`` is the
# only public write path: one writer per active run, indexed by run
# name. The writer flips ``StepRange.tail_growing`` to ``True`` for the
# run via ``RunRegistry.attach_writer_store`` on init and back to
# ``False`` on close. See design/unified-run-architecture.md.
_step_cache: StepCache = StepCache(capacity=5000)
_run_registry: RunRegistry | None = None
_run_reader: RunReader | None = None
_writers: dict[str, RunWriter] = {}
_active_writer: RunWriter | None = None


def _get_registry() -> RunRegistry | None:
    """Return the singleton RunRegistry, constructing it lazily if needed."""
    return _run_registry


def _get_reader() -> RunReader | None:
    """Return the singleton RunReader, constructing it lazily if needed."""
    global _run_reader
    if _run_reader is not None:
        return _run_reader
    if _run_registry is None:
        return None
    _run_reader = RunReader(_run_registry, _step_cache)
    return _run_reader


def init_data_dir(data_dir: Path) -> None:
    """Initialize the registry against ``data_dir``.

    Called by CLI entry points (``server_cli.py``, ``dashboard_cli.py``)
    and ``start_dashboard()`` to wire the singleton ``RunRegistry`` to
    the on-disk run directory. Idempotent: a second call with the same
    directory is a no-op.
    """
    global _run_registry, _run_reader
    if _run_registry is not None and _run_registry.data_dir == data_dir:
        return
    _run_registry = RunRegistry(data_dir)
    _run_reader = None  # rebuilt lazily against the new registry


def push_step_from_game(step_data: StepData) -> None:
    """Push a step from the game thread to the active writer.

    Called from ``roc/game/gymnasium.py:_push_dashboard_data``. Routes
    through the active ``RunWriter`` (set by ``_start_live_session``)
    which:

    1. Mirrors the step into ``_step_cache`` (cache-first reads).
    2. Advances the registry's ``max`` step.
    3. Notifies subscribers (Socket.io ``step_added`` invalidation).

    No-op when there is no active writer (e.g., test runs without a
    live dashboard).
    """
    writer = _active_writer
    if writer is None:
        return
    try:
        writer.push_step(step_data)
    except Exception as exc:
        logger.warning("push_step_from_game failed: {}", exc)


@app.exception_handler(FileNotFoundError)
async def _handle_file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Convert FileNotFoundError (from RunReader) to a 404 response."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class GameSummary(BaseModel):
    game_number: int
    steps: int
    start_ts: int | None = None
    end_ts: int | None = None


class StepRange(BaseModel):
    min: int
    max: int
    tail_growing: bool = False


class Bookmark(BaseModel):
    step: int
    game: int
    annotation: str
    created: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable values.

    Handles:
    - numpy scalars (int*, float*, bool_) and ndarrays
    - pandas NA / NaT (returned by DuckDB for nullable columns) -- mapped to None
    - float NaN -- mapped to None so the response is valid JSON

    Without the pandas-NA branch, ``json.dumps`` raises ``TypeError:
    Object of type NAType is not JSON serializable`` and the entire
    /step/{n} endpoint returns 500. The dashboard renders that as
    "no data" with no visible error -- one of the recurring "missing
    data, errors not visible in the UI" failure modes.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        # JSON has no representation for NaN/+-Inf -- collapse to None.
        if not np.isfinite(f):
            return None
        return f
    if isinstance(obj, np.ndarray):
        return [_convert_numpy(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        # Catch raw Python floats that may also be NaN.
        import math

        if not math.isfinite(obj):
            return None
        return obj
    # Pandas NA / NaT (used by nullable Int/Float columns from DuckDB).
    # We import lazily so reporting modules that don't read DuckDB
    # don't pull pandas in just for this check.
    try:
        import pandas as pd

        if obj is pd.NA:
            return None
        if isinstance(obj, type(pd.NaT)) and pd.isna(obj):
            return None
        # pd.isna is broadcast-friendly: must guard against arrays.
        if not isinstance(obj, (str, bytes)) and pd.isna(obj):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    return obj


@app.get("/api/runs")
def list_runs(
    min_steps: int = 10,
    include_all: bool = False,
) -> list[RunSummary]:
    """List available runs with metadata.

    Every run is returned with a ``status`` field
    (``ok``/``empty``/``short``/``corrupt``). By default only
    ``status=ok`` runs are returned. Pass ``include_all=true`` to
    receive every run -- this is what the "Show all runs" toggle in
    the dashboard uses, and it's the recommended way to debug a
    "where did my run go?" report.

    ``min_steps`` defines the minimum step count for a run to count
    as ``ok`` instead of ``short`` (default 10).
    """
    reader = _get_reader()
    if reader is None:
        return []
    return reader.list_runs(min_steps=min_steps, include_all=include_all)


@app.get("/api/runs/{run_name}/games", responses=_NOT_INITIALIZED_RESPONSE)
def list_games(run_name: str) -> list[GameSummary]:
    """List games in a run.

    Phase 1 bug-fix migration (BUG-C1): reads through ``RunReader`` so the
    call shares the registry's single ``DuckLakeStore`` instead of opening
    a second read-only connection that DuckDB rejects with ``BinderException:
    Unique file handle conflict``.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    games = reader.list_games(run_name)
    return [GameSummary(**g) for g in games]


@app.get(
    "/api/runs/{run_name}/step/{step}",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_step(
    run_name: str,
    step: int,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get all data for a specific step.

    Returns the step data on success. On a known failure mode the response
    body is the typed ``StepResponse`` envelope so the dashboard can render
    the right message instead of silently dropping the step:

    - Unknown run -> 404 with ``{"detail": ...}``.
    - Out of range / not emitted -> 404 with the envelope.
    - Backend error -> 500 with the envelope.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    t0 = time.monotonic()
    resp = reader.get_step(run_name, step)
    t1 = time.monotonic()
    if resp.status == "ok":
        data = _convert_numpy(resp.data)
        t2 = time.monotonic()
        logger.debug(
            "GET step {} fetch={:.1f}ms serialize={:.1f}ms total={:.1f}ms",
            step,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t2 - t0) * 1000,
        )
        return JSONResponse(content=data)
    if resp.status == "run_not_found":
        raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
    if resp.status in ("out_of_range", "not_emitted"):
        return JSONResponse(
            status_code=404,
            content=_convert_numpy(resp.model_dump()),
        )
    # status == "error"
    return JSONResponse(
        status_code=500,
        content=_convert_numpy(resp.model_dump()),
    )


@app.get("/api/runs/{run_name}/steps", responses=_NOT_INITIALIZED_RESPONSE)
def get_steps_batch(
    run_name: str,
    steps: Annotated[str, Query(description="Comma-separated step numbers")],
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get data for multiple steps in a single request.

    Returns a dict mapping step number (as string key) to StepData.
    Steps that fail to load are silently skipped.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    t0 = time.monotonic()
    step_nums = [int(s.strip()) for s in steps.split(",") if s.strip()]
    try:
        batch = reader.get_steps_batch(run_name, step_nums)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        batch = {}
    result: dict[str, Any] = {
        str(s): _convert_numpy(dataclasses.asdict(sd)) for s, sd in batch.items()
    }
    t1 = time.monotonic()
    logger.debug(
        "GET steps batch ({} steps) total={:.1f}ms",
        len(result),
        (t1 - t0) * 1000,
    )
    return JSONResponse(content=result)


@app.get("/api/runs/{run_name}/step-range", responses=_NOT_INITIALIZED_RESPONSE)
def get_step_range(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> StepRange:
    """Get min/max step for a run or game.

    Phase 3: ``tail_growing`` is the only liveness signal at the API
    boundary. ``True`` means a ``RunWriter`` is currently attached to
    the run via the registry; ``False`` means the run is closed (or
    has not been opened by any writer in this process). Sourced from
    ``RunRegistry`` via ``RunReader.get_step_range``.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    rng = reader.get_step_range(run_name, game)
    return StepRange(min=rng.min, max=rng.max, tail_growing=rng.tail_growing)


@app.get(
    "/api/runs/{run_name}/metrics-history",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_metrics_history(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
    fields: Annotated[str | None, Query(description="Comma-separated field names")] = None,
) -> JSONResponse:
    """Get game_metrics for all steps in a game (for charting trends)."""
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None
    data = reader.get_history(run_name, "metrics", game, fields=field_list)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/graph-history",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_graph_history(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get graph_summary for all steps in a game (for charting cache utilization)."""
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = reader.get_history(run_name, "graph", game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/event-history",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_event_history(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get event_summary for all steps in a game (for charting event activity)."""
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = reader.get_history(run_name, "event", game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/intrinsics-history",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_intrinsics_history(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get intrinsics for all steps in a game (for charting trends)."""
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = reader.get_history(run_name, "intrinsics", game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/action-history",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_action_history(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get action taken for all steps in a game (for charting action distribution)."""
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = reader.get_history(run_name, "action", game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/resolution-history",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_resolution_history(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get resolution accuracy history for charting correct/incorrect/new counts."""
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = reader.get_history(run_name, "resolution", game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/all-objects",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_all_objects(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get all resolved objects with attributes, match counts, and creation step.

    Phase 1 bug-fix migration (BUG-C1): reads through ``RunReader``.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = reader.get_all_objects(run_name, game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/schema",
    responses={
        404: {"description": "Resource not found"},
        503: {"description": "Dashboard not initialized"},
    },
)
def get_schema(run_name: str) -> JSONResponse:
    """Get the graph database schema for a run.

    Phase 1 bug-fix migration (BUG-C1): reads through ``RunReader``. The
    schema is read directly from ``schema.json`` -- no DuckLake involved --
    but routing through the reader keeps the single read facade consistent.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    schema = reader.get_schema(run_name)
    if schema is None:
        raise HTTPException(status_code=404, detail="No schema found for this run")
    return JSONResponse(content=schema)


@app.get("/api/runs/{run_name}/action-map", responses=_NOT_INITIALIZED_RESPONSE)
def get_action_map(run_name: str) -> JSONResponse:
    """Get the full action-id-to-name mapping for a run.

    Phase 1 bug-fix migration (BUG-C1): reads through ``RunReader``. The
    action map is read directly from ``action_map.json`` -- no DuckLake
    involved.
    """
    reader = _get_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    action_map = reader.get_action_map(run_name)
    if action_map is None:
        return JSONResponse(content=[])
    return JSONResponse(content=action_map)


def _collect_object_states(obj: Any) -> list[dict[str, Any]]:
    """Collect per-tick observation dicts from ObjectInstance nodes attached to an Object."""
    from roc.pipeline.object.object_instance import ObjectInstance, ObservedAs

    states: list[dict[str, Any]] = []
    for e in obj.dst_edges:
        if isinstance(e, ObservedAs) and isinstance(e.src, ObjectInstance):
            oi = e.src
            states.append(
                {
                    "tick": oi.tick,
                    "x": oi.x,
                    "y": oi.y,
                    "glyph_type": oi.glyph_type,
                    "color_type": oi.color_type,
                    "shape_type": oi.shape_type,
                    "flood_size": oi.flood_size,
                    "line_size": oi.line_size,
                    "distance": oi.distance,
                    "motion_direction": oi.motion_direction,
                    "delta_old": oi.delta_old,
                    "delta_new": oi.delta_new,
                }
            )
    states.sort(key=lambda s: s["tick"])
    return states


def _prop_node_to_change_dict(prop_node: Any) -> dict[str, Any] | None:
    """Convert a property transform node to a change dict, or None if unnamed."""
    prop_name = getattr(prop_node, "property_name", None)
    if prop_name is None:
        return None
    return {
        "property": prop_name,
        "type": getattr(prop_node, "change_type", None),
        "delta": getattr(prop_node, "delta", None),
        "old_value": getattr(prop_node, "old_value", None),
        "new_value": getattr(prop_node, "new_value", None),
    }


def _collect_object_transforms(obj: Any) -> list[dict[str, Any]]:
    """Collect transform dicts from ObjectTransform nodes attached to an Object."""
    from roc.pipeline.object.object_transform import ObjectHistory, ObjectTransform

    transforms: list[dict[str, Any]] = []
    for e in obj.src_edges:
        if not (isinstance(e, ObjectHistory) and isinstance(e.dst, ObjectTransform)):
            continue
        ot = e.dst
        t_dict: dict[str, Any] = {
            "num_discrete_changes": ot.num_discrete_changes,
            "num_continuous_changes": ot.num_continuous_changes,
            "changes": [],
        }
        for de in ot.src_edges:
            change = _prop_node_to_change_dict(de.dst)
            if change is not None:
                t_dict["changes"].append(change)
        transforms.append(t_dict)
    return transforms


@app.get(
    "/api/runs/{run_name}/object/{object_id}/history",
    responses={503: {"description": _NOT_INITIALIZED}, 404: {"description": "Object not found"}},
)
def get_object_history(run_name: str, object_id: int) -> JSONResponse:
    """Get the full observation history and transforms for an object.

    Returns states (per-tick observations) and transforms (property changes between
    consecutive observations).
    """
    if _run_registry is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)

    try:
        from roc.db.graphdb import NodeId
        from roc.pipeline.object.object import Object

        obj = Object.load(NodeId(object_id))
    except Exception:
        raise HTTPException(status_code=404, detail="Object not found")

    states = _collect_object_states(obj)
    transforms = _collect_object_transforms(obj)
    # Object UUIDs are 63-bit ints; emit as string so JS clients don't
    # truncate them when parsing JSON. See graph_service.to_cytoscape for
    # the same conversion on graph endpoints.
    from roc.reporting.graph_service import object_human_name

    info: dict[str, Any] = {
        "uuid": str(obj.uuid),
        "human_name": object_human_name(obj.uuid),
        "resolve_count": obj.resolve_count,
    }

    return JSONResponse(
        content=_convert_numpy(
            {
                "states": states,
                "transforms": transforms,
                "info": info,
            }
        )
    )


@app.get("/api/runs/{run_name}/bookmarks")
def get_bookmarks(run_name: str) -> list[Bookmark]:
    """Get bookmarks for a run."""
    if _run_registry is None:
        return []
    bookmarks_file = _run_registry.data_dir / run_name / "bookmarks.json"
    if not bookmarks_file.exists():
        return []
    try:
        data = json.loads(bookmarks_file.read_text())
        return [Bookmark(**b) for b in data]
    except Exception:
        return []


@app.post("/api/runs/{run_name}/bookmarks", responses=_NOT_INITIALIZED_RESPONSE)
def save_bookmarks(run_name: str, bookmarks: list[Bookmark]) -> dict[str, str]:
    """Save bookmarks for a run."""
    if _run_registry is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    bookmarks_file = _run_registry.data_dir / run_name / "bookmarks.json"
    bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
    bookmarks_file.write_text(json.dumps([b.model_dump() for b in bookmarks], indent=2))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Game lifecycle endpoints
# ---------------------------------------------------------------------------

# Single source of truth for the active ``GameManager`` instance.
#
# Historically this was a bare module global (``_game_manager``) that
# ``server_cli.py`` wrote to directly via ``srv._game_manager = mgr``.
# That "attribute hop" made the dependency invisible at call sites and
# broke the "one UI server, one API server" invariant from the moment
# two things tried to own it at once. All access now flows through
# ``set_game_manager``/``get_game_manager`` so the single-ownership
# contract is enforced in one place and the read path is discoverable.
_game_manager: Any = None


def set_game_manager(mgr: Any) -> None:
    """Install the GameManager singleton.

    Called once at server startup from ``server_cli.py``. Raises on a
    second install unless the caller first clears the slot with
    ``set_game_manager(None)`` (used by test fixtures). This guards the
    "one UI server, one API server" invariant -- silent overwrites
    were how TC-GAME-004-class drifts crept in.
    """
    global _game_manager
    if mgr is not None and _game_manager is not None and _game_manager is not mgr:
        raise RuntimeError(
            "GameManager already installed; clear it first with set_game_manager(None)",
        )
    _game_manager = mgr


def get_game_manager() -> Any:
    """Return the active GameManager, or ``None`` if none is installed."""
    return _game_manager


class GameStatus(BaseModel):
    state: str
    run_name: str | None = None
    exit_code: int | None = None
    error: str | None = None


@app.get("/api/game/status")
def game_status() -> GameStatus:
    """Get current game state."""
    mgr = get_game_manager()
    if mgr is None:
        return GameStatus(state="idle")
    status = mgr.get_status()
    return GameStatus(
        state=status["state"],
        run_name=status.get("run_name"),
        exit_code=status.get("exit_code"),
        error=status.get("error"),
    )


@app.post(
    "/api/game/start",
    responses={
        409: {"description": "Game already running or already stopped"},
        503: {"description": "Dashboard not initialized"},
    },
)
def game_start(num_games: Annotated[int, Query()] = 5) -> dict[str, str]:
    """Start a new game."""
    mgr = get_game_manager()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Game manager not initialized")
    try:
        result = mgr.start_game(num_games=num_games)
        return {"status": result}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/api/game/stop",
    responses={
        409: {"description": "Game already running or already stopped"},
        503: {"description": "Dashboard not initialized"},
    },
)
def game_stop() -> dict[str, str]:
    """Stop the running game."""
    mgr = get_game_manager()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Game manager not initialized")
    try:
        mgr.stop_game()
        return {"status": "stopping"}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ---------------------------------------------------------------------------
# Socket.io events
# ---------------------------------------------------------------------------


@sio.event
async def connect(sid: str, environ: dict[str, Any]) -> None:
    """Client connected."""
    logger.debug("Dashboard client connected: {}", sid)


@sio.event
async def disconnect(sid: str) -> None:
    """Client disconnected -- drop any per-sid run subscriptions."""
    logger.debug("Dashboard client disconnected: {}", sid)
    # Phase 4: each Socket.io client may have one active run subscription.
    # On disconnect we run the corresponding unsubscribe so the registry
    # does not accumulate dead callbacks. Use ``pop`` to be idempotent.
    sub = _sio_subscriptions.pop(sid, None)
    if sub is not None:
        try:
            sub()
        except Exception as exc:
            logger.warning("subscribe_run cleanup failed for sid {}: {}", sid, exc)


# Socket.io is an invalidation channel, not a data pipe. Each browser tab
# subscribes to exactly one run via ``subscribe_run`` and receives
# ``step_added`` events with the tiny ``{run, step}`` payload. The browser
# then invalidates its TanStack Query keys and refetches via the unified
# ``RunReader`` path.
_sio_subscriptions: dict[str, Any] = {}


@sio.event
async def subscribe_run(sid: str, run: str) -> None:
    """Subscribe a Socket.io client to ``step_added`` notifications for a run.

    Replaces any prior subscription on the same sid (one subscription per
    client tab is the contract). The callback fires from the writer's
    ``push_step`` via ``RunRegistry.notify_subscribers`` and emits a tiny
    ``{run, step}`` payload to the originating client only -- broadcast
    is unnecessary because each tab manages its own subscription.

    For unknown runs ``RunRegistry.subscribe`` returns a no-op unsubscribe
    so the binding still records cleanly. The browser will retry on next
    invalidation if the run becomes known later.
    """
    if not isinstance(run, str) or not run:
        return
    reader = _get_reader()
    if reader is None:
        return

    # Drop any prior subscription on this sid before installing the new one.
    prior = _sio_subscriptions.pop(sid, None)
    if prior is not None:
        try:
            prior()
        except Exception as exc:
            logger.warning("subscribe_run prior cleanup failed for sid {}: {}", sid, exc)

    loop = _sio_loop

    def _on_step(step: int) -> None:
        """Bridge a registry notification onto the Socket.io event loop.

        Notifications are dispatched outside the registry lock and may
        run on the writer's thread. We must hop back to the asyncio
        loop via ``run_coroutine_threadsafe`` to call ``sio.emit``.
        """
        import asyncio

        if loop is None or not loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                sio.emit("step_added", {"run": run, "step": step}, to=sid),
                loop,
            )
        except Exception:
            # Socket errors must not break the writer's push.
            pass

    unsubscribe = reader.subscribe(run, _on_step)
    _sio_subscriptions[sid] = unsubscribe
    logger.debug("Dashboard client {} subscribed to run {}", sid, run)


@sio.event
async def unsubscribe_run(sid: str, run: str | None = None) -> None:
    """Drop the per-sid run subscription, if any.

    Idempotent: missing or already-unsubscribed sids are no-ops. The
    ``run`` argument is ignored because each sid has at most one
    subscription -- it exists only to mirror the client API symmetry.
    """
    sub = _sio_subscriptions.pop(sid, None)
    if sub is None:
        return
    try:
        sub()
    except Exception as exc:
        logger.warning("unsubscribe_run failed for sid {}: {}", sid, exc)
    logger.debug("Dashboard client {} unsubscribed from run {}", sid, run)


_sio_loop: Any = None  # set when uvicorn starts


# ---------------------------------------------------------------------------
# Game lifecycle
# ---------------------------------------------------------------------------


def _start_live_session(run_name: str) -> None:
    """Start a live session for a game run.

    Creates a ``RunWriter`` against the in-process ``DuckLakeStore`` and
    installs it as the active writer. The game thread pushes step data via
    ``push_step_from_game`` which routes through the active writer:

    1. Mirrors the step into ``StepCache`` (write-through cache).
    2. Advances the registry's ``max`` step.
    3. Notifies subscribers (Socket.io ``step_added`` for invalidation).

    Phase 0 invariant: the writer's ``DuckLakeStore`` is the SINGLE store
    for this run -- it doubles as the read store for in-process reads via
    ``RunRegistry.attach_writer_store``. Do not construct a separate
    ``read_only=True`` instance for the same catalog file while a writer
    is open.
    """
    global _active_writer
    from roc.reporting.observability import Observability

    store = Observability.get_ducklake_store()
    if store is None:
        logger.warning("Live session start aborted: no DuckLakeStore available")
        return
    registry = _get_registry()
    if registry is None:
        logger.warning("Live session start aborted: no RunRegistry initialized")
        return
    writer = RunWriter(run_name, registry, _step_cache, None, store)
    _writers[run_name] = writer
    _active_writer = writer
    logger.info("Started live session for run {}", run_name)


def _stop_live_session() -> None:
    """Tear down the live session and detach the active writer.

    The writer's ``close()`` is the single point where
    ``RunRegistry.detach_writer_store`` is called, so
    ``StepRange.tail_growing`` flips back to ``False`` exactly once per
    session. Idempotent: a second call with no active writer is a no-op.
    """
    global _active_writer
    writer = _active_writer
    _active_writer = None
    if writer is not None:
        _writers.pop(writer.run, None)
        try:
            writer.close()
        except Exception as exc:
            logger.warning("writer close failed: {}", exc)
    logger.debug("Stopped live session")


def _emit_game_state_changed(status: dict[str, Any]) -> None:
    """Emit game_state_changed Socket.io event and manage live session."""
    import asyncio

    state = status.get("state", "idle")
    run_name = status.get("run_name")
    exit_code = status.get("exit_code")
    error = status.get("error")

    # Start/stop live session based on game state. The run_name guard
    # ensures we don't start before the game reports its name. The
    # active-writer guard prevents duplicate sessions on repeated state
    # notifications.
    if state == "running" and run_name:
        if _active_writer is None or _active_writer.run != run_name:
            _start_live_session(run_name)
    elif state == "idle":
        _stop_live_session()

    # Emit Socket.io event. The payload mirrors the REST
    # /api/game/status response so a single consumer (useGameState in
    # the browser) can maintain the full shape from either source,
    # eliminating the prior pattern where MenuBar held a parallel copy
    # of the game status just to keep ``error`` visible across Socket
    # updates.
    try:
        if _sio_loop is not None and _sio_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                sio.emit(
                    "game_state_changed",
                    {
                        "state": state,
                        "run_name": run_name,
                        "exit_code": exit_code,
                        "error": error,
                    },
                ),
                _sio_loop,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_started = False


def start_dashboard() -> None:
    """Start the in-process FastAPI dashboard server.

    Used by ``roc.init()`` for standalone ``uv run play``. Initializes
    the registry against the run directory's parent and creates a
    ``RunWriter`` for the active run. The game thread pushes step data
    via ``push_step_from_game`` which routes through the active writer.

    Short-circuits when running under ``server_cli.py``: that path
    already owns the uvicorn server lifecycle and manages live sessions
    via ``_emit_game_state_changed`` -> ``_start_live_session``. The
    ``_started`` flag prevents double-start.
    """
    global _started
    if _started:
        return

    from roc.framework.config import Config
    from roc.reporting.observability import Observability

    cfg = Config.get()
    if not cfg.dashboard_enabled:
        return

    # When running under server_cli.py, the dashboard server is already
    # running (uvicorn was started by the CLI) and the live session is
    # managed by _emit_game_state_changed via _start_live_session. Nothing
    # more for this function to do.
    if get_game_manager() is not None:
        _started = True
        return

    store = Observability.get_ducklake_store()
    if store is None:
        logger.warning("Dashboard enabled but no DuckLakeStore available; skipping.")
        return

    init_data_dir(store.run_dir.parent)
    _start_live_session(store.run_dir.name)

    # Enable gzip compression for API responses (helps with bulk step prefetch)
    if cfg.dashboard_gzip:
        from starlette.middleware.gzip import GZipMiddleware

        app.add_middleware(GZipMiddleware, minimum_size=1000)
        logger.info("Dashboard gzip compression enabled")

    # Mount the ASGI app (FastAPI + Socket.io)
    sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

    # Try to mount React static build if it exists
    dist_dir = Path(__file__).parent.parent.parent / "dashboard-ui" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    # Start uvicorn in a background thread and wait for the event loop
    # to be fully initialized before returning. This prevents the game
    # loop from pushing through the writer (which triggers Socket.io
    # emits via run_coroutine_threadsafe) before the async infrastructure
    # is ready.
    config = uvicorn.Config(
        sio_app,
        host="0.0.0.0",  # nosec B104
        port=cfg.dashboard_port,
        log_level="warning",
        ssl_certfile=cfg.ssl_certfile if cfg.ssl_certfile else None,
        ssl_keyfile=cfg.ssl_keyfile if cfg.ssl_keyfile else None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Block until the FastAPI startup event has captured the event loop.
    # Timeout after 10s to avoid hanging if uvicorn fails to start.
    if not _server_ready.wait(timeout=10):
        logger.warning("Dashboard server did not become ready within 10s")

    _started = True
    proto = "https" if cfg.ssl_certfile else "http"
    logger.info(f"Dashboard API at {proto}://0.0.0.0:{cfg.dashboard_port}")


def stop_dashboard() -> None:
    """Stop the dashboard and clean up.

    The writer's ``close()`` (called via ``_stop_live_session``) is the
    single point where ``RunRegistry.detach_writer_store`` is called,
    so ``StepRange.tail_growing`` flips back to ``False`` exactly once
    per session.
    """
    _stop_live_session()
    logger.debug("Dashboard API stopped.")
