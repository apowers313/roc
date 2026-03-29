"""FastAPI + Socket.io server for the React debug dashboard.

Provides a clean separation of concerns:
- REST endpoints serve step data from DataStore (live buffer or DuckLake)
- Socket.io pushes live step metadata to connected browsers
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

from roc.logger import logger
from roc.reporting.data_store import DataStore, RunSummary
from roc.reporting.run_store import StepData
from roc.reporting.step_buffer import StepBuffer, register_step_buffer

# ---------------------------------------------------------------------------
# Socket.io server
# ---------------------------------------------------------------------------

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ROC Debug Dashboard API")

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
    doesn't push to the StepBuffer before the async infrastructure
    can handle cross-thread coroutine submissions.
    """
    import asyncio

    global _sio_loop, _summary_thread
    _sio_loop = asyncio.get_running_loop()
    _server_ready.set()

    # Populate run summary cache in a background thread so the first
    # /api/runs response is fast (returns placeholders until cached).
    if _data_store is not None and _summary_thread is None:
        _summary_thread = threading.Thread(
            target=_data_store.populate_run_summaries,
            daemon=True,
            name="run-summaries",
        )
        _summary_thread.start()


# Single data store instance -- set by start_dashboard() or CLI entry points
_data_store: DataStore | None = None

_summary_thread: threading.Thread | None = None


@app.exception_handler(FileNotFoundError)
async def _handle_file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Convert FileNotFoundError (from DataStore) to a 404 response."""
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


class Bookmark(BaseModel):
    step: int
    game: int
    annotation: str
    created: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@app.get("/api/runs")
def list_runs(min_steps: int = 10) -> list[RunSummary]:
    """List available runs with metadata.

    ``min_steps`` filters out short-lived / crashed runs (default 10).
    Pass ``min_steps=0`` to include all runs.
    """
    if _data_store is None:
        return []
    return _data_store.list_runs(min_steps)


@app.get("/api/runs/{run_name}/games", responses=_NOT_INITIALIZED_RESPONSE)
def list_games(run_name: str) -> list[GameSummary]:
    """List games in a run."""
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    games = _data_store.list_games(run_name)
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
    """Get all data for a specific step."""
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    t0 = time.monotonic()
    step_data = _data_store.get_step_data(run_name, step)
    t1 = time.monotonic()
    data = _convert_numpy(dataclasses.asdict(step_data))
    t2 = time.monotonic()
    logger.debug(
        "GET step {} fetch={:.1f}ms serialize={:.1f}ms total={:.1f}ms",
        step,
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
        (t2 - t0) * 1000,
    )
    return JSONResponse(content=data)


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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    t0 = time.monotonic()
    step_nums = [int(s.strip()) for s in steps.split(",") if s.strip()]
    try:
        batch = _data_store.get_steps_batch(run_name, step_nums)
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
    """Get min/max step for a run or game."""
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    min_step, max_step = _data_store.get_step_range(run_name, game)
    return StepRange(min=min_step, max=max_step)


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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None
    data = _data_store.get_metrics_history(run_name, game, field_list)
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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = _data_store.get_graph_history(run_name, game)
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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = _data_store.get_event_history(run_name, game)
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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = _data_store.get_intrinsics_history(run_name, game)
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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = _data_store.get_action_history(run_name, game)
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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = _data_store.get_resolution_history(run_name, game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/all-objects",
    responses=_NOT_INITIALIZED_RESPONSE,
)
def get_all_objects(
    run_name: str,
    game: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    """Get all resolved objects with attributes, match counts, and creation step."""
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    data = _data_store.get_all_objects(run_name, game)
    return JSONResponse(content=_convert_numpy(data))


@app.get(
    "/api/runs/{run_name}/schema",
    responses={
        404: {"description": "Resource not found"},
        503: {"description": "Dashboard not initialized"},
    },
)
def get_schema(run_name: str) -> JSONResponse:
    """Get the graph database schema for a run."""
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    schema = _data_store.get_schema(run_name)
    if schema is None:
        raise HTTPException(status_code=404, detail="No schema found for this run")
    return JSONResponse(content=schema)


@app.get("/api/runs/{run_name}/action-map", responses=_NOT_INITIALIZED_RESPONSE)
def get_action_map(run_name: str) -> JSONResponse:
    """Get the full action-id-to-name mapping for a run."""
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    action_map = _data_store.get_action_map(run_name)
    if action_map is None:
        return JSONResponse(content=[])
    return JSONResponse(content=action_map)


def _collect_object_states(obj: Any) -> list[dict[str, Any]]:
    """Collect per-tick observation dicts from ObjectInstance nodes attached to an Object."""
    from roc.object_instance import ObjectInstance, ObservedAs

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
    from roc.object_transform import ObjectHistory, ObjectTransform

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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)

    try:
        from roc.graphdb import NodeId
        from roc.object import Object

        obj = Object.load(NodeId(object_id))
    except Exception:
        raise HTTPException(status_code=404, detail="Object not found")

    states = _collect_object_states(obj)
    transforms = _collect_object_transforms(obj)
    info: dict[str, Any] = {
        "uuid": obj.uuid,
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
    if _data_store is None:
        return []
    bookmarks_file = _data_store.data_dir / run_name / "bookmarks.json"
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
    if _data_store is None:
        raise HTTPException(status_code=503, detail=_NOT_INITIALIZED)
    bookmarks_file = _data_store.data_dir / run_name / "bookmarks.json"
    bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
    bookmarks_file.write_text(json.dumps([b.model_dump() for b in bookmarks], indent=2))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Live mode endpoints
# ---------------------------------------------------------------------------


class LiveStatus(BaseModel):
    active: bool
    run_name: str | None = None
    step: int
    game_number: int
    step_min: int
    step_max: int
    game_numbers: list[int]


@app.get("/api/live/status")
def live_status() -> LiveStatus:
    """Get current live run status."""
    if _data_store is not None:
        status = _data_store.get_live_status()
        if status["active"]:
            return LiveStatus(**status)

    # Game manager is running but no steps received yet
    if _game_manager is not None and _game_manager.state in ("initializing", "running"):
        ds_run_name = _data_store.live_run_name if _data_store else None
        if ds_run_name is not None:
            return LiveStatus(
                active=True,
                run_name=ds_run_name,
                step=0,
                game_number=0,
                step_min=0,
                step_max=0,
                game_numbers=[],
            )

    return LiveStatus(
        active=False,
        run_name=None,
        step=0,
        game_number=0,
        step_min=0,
        step_max=0,
        game_numbers=[],
    )


@app.get("/api/live/step/{step}", responses={404: {"description": "Resource not found"}})
def live_step(step: int) -> JSONResponse:
    """Get step data from the live StepBuffer (in-memory)."""
    if _data_store is None or _data_store.live_buffer is None:
        raise HTTPException(status_code=404, detail="No live session")
    step_data = _data_store.get_live_step(step)
    if step_data is None:
        raise HTTPException(status_code=404, detail=f"Step {step} not in buffer")
    data = _convert_numpy(dataclasses.asdict(step_data))
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# Game lifecycle endpoints
# ---------------------------------------------------------------------------

# Set by server_cli.py when running in unified server mode.
_game_manager: Any = None


class GameStatus(BaseModel):
    state: str
    run_name: str | None = None
    exit_code: int | None = None
    error: str | None = None


@app.get("/api/game/status")
def game_status() -> GameStatus:
    """Get current game state."""
    if _game_manager is None:
        return GameStatus(state="idle")
    status = _game_manager.get_status()
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
    """Start a new game subprocess."""
    if _game_manager is None:
        raise HTTPException(status_code=503, detail="Game manager not initialized")
    try:
        result = _game_manager.start_game(num_games=num_games)
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
    if _game_manager is None:
        raise HTTPException(status_code=503, detail="Game manager not initialized")
    try:
        _game_manager.stop_game()
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
    """Client disconnected."""
    logger.debug("Dashboard client disconnected: {}", sid)


_sio_loop: Any = None  # set when uvicorn starts


def _notify_new_step(step_data: StepData) -> None:
    """Push full step data to all connected clients (called from game thread).

    Sends the complete StepData so the browser can render immediately
    without a REST round-trip. This eliminates flicker in live-following mode.
    """
    import asyncio

    try:
        if _sio_loop is not None and _sio_loop.is_running():
            data = _convert_numpy(dataclasses.asdict(step_data))
            asyncio.run_coroutine_threadsafe(
                sio.emit("new_step", data),
                _sio_loop,
            )
    except Exception:
        pass  # socket errors must not break the game loop


# ---------------------------------------------------------------------------
# Subprocess-based game lifecycle
# ---------------------------------------------------------------------------


def _start_live_session(run_name: str) -> None:
    """Start a live session for a subprocess-based game run."""
    if _data_store is None:
        return
    buf = StepBuffer(capacity=100_000)
    _data_store.set_live_session(run_name=run_name, buffer=buf)
    # Socket.io notifications come from receive_step(), not a listener
    logger.info("Started live session for run {}", run_name)


def _stop_live_session() -> None:
    """Clear the live session state."""
    if _data_store is not None:
        _data_store.clear_live_session()
    logger.debug("Stopped live session")


@app.post("/api/internal/step")
def receive_step(request: dict[str, Any]) -> dict[str, Any]:
    """Receive a step from the game subprocess and broadcast via Socket.io.

    This is an internal endpoint used by the game subprocess to push
    live step data to the dashboard server. The response may include
    ``{"stop": true}`` to request cooperative shutdown.
    """
    step_data = StepData(**{k: v for k, v in request.items() if k in StepData.__dataclass_fields__})
    if _data_store is not None:
        _data_store.push_live_step(step_data)
    _notify_new_step(step_data)
    response: dict[str, Any] = {"status": "ok"}
    if _game_manager is not None and _game_manager.is_stop_requested():
        response["stop"] = True
    return response


@app.post("/api/internal/action-map")
def receive_action_map(request: dict[str, Any]) -> dict[str, str]:
    """Receive the full action map from the game subprocess.

    Called once at game start so the dashboard knows all action names
    without waiting for them to appear in step data.
    """
    run_name = request.get("run_name", "")
    action_map = request.get("action_map", [])
    if _data_store is not None and run_name and isinstance(action_map, list):
        _data_store.set_action_map(run_name, action_map)
    return {"status": "ok"}


def _emit_game_state_changed(status: dict[str, Any]) -> None:
    """Emit game_state_changed Socket.io event and manage polling."""
    import asyncio

    state = status.get("state", "idle")
    run_name = status.get("run_name")

    # Start/stop live session based on game state
    if state == "running" and run_name:
        _start_live_session(run_name)
    elif state == "idle":
        _stop_live_session()

    # Emit Socket.io event
    try:
        if _sio_loop is not None and _sio_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                sio.emit(
                    "game_state_changed",
                    {
                        "state": state,
                        "run_name": run_name,
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
    """Start the FastAPI dashboard server.

    Creates a DataStore and StepBuffer, registers the buffer globally so
    the game loop can push StepData.  Socket.io broadcasts push
    notifications to all connected browser clients.
    """
    global _started, _data_store
    if _started:
        return

    from roc.config import Config
    from roc.reporting.observability import Observability

    cfg = Config.get()
    if not cfg.dashboard_enabled:
        return

    store = Observability.get_ducklake_store()
    if store is None:
        logger.warning("Dashboard enabled but no DuckLakeStore available; skipping.")
        return

    ds = DataStore(data_dir=store.run_dir.parent)
    buf = StepBuffer(capacity=100_000)
    register_step_buffer(buf)
    ds.set_live_session(run_name=store.run_dir.name, buffer=buf, store=store)
    buf.add_listener(lambda: _notify_new_step(buf.get_latest()))  # type: ignore[arg-type]
    _data_store = ds

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
    # loop from pushing to the StepBuffer (which triggers Socket.io emits
    # via run_coroutine_threadsafe) before the async infrastructure is ready.
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
    """Stop the dashboard and clean up."""
    from roc.reporting.step_buffer import clear_step_buffer

    clear_step_buffer()
    if _data_store is not None:
        _data_store.clear_live_session()
    logger.debug("Dashboard API stopped.")
