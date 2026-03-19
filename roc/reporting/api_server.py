"""FastAPI + Socket.io server for the React debug dashboard.

Provides a clean separation of concerns:
- REST endpoints serve step data from StepBuffer (live) or DuckLake (historical)
- Socket.io pushes live step metadata to connected browsers
- In production, serves the React static build from dashboard-ui/dist/
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from roc.logger import logger
from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_store import RunStore, StepData
from roc.reporting.step_buffer import StepBuffer, register_step_buffer

# ---------------------------------------------------------------------------
# Socket.io server
# ---------------------------------------------------------------------------

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ROC Debug Dashboard API")


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
    if _data_dir is not None and _summary_thread is None:
        _summary_thread = threading.Thread(
            target=_populate_run_summaries, daemon=True, name="run-summaries",
        )
        _summary_thread.start()


# Module-level state set by start_dashboard()
_data_dir: Path | None = None
_step_buffer: StepBuffer | None = None
_live_run_name: str | None = None
_live_store: DuckLakeStore | None = None
_run_stores: dict[str, RunStore] = {}
_store_lock = threading.Lock()
_run_summary_cache: dict[str, RunSummary] = {}


def _get_store(run_name: str) -> RunStore:
    """Get or create a RunStore for a run.

    For the live run, shares the game's DuckLakeStore (concurrent
    read+write through the store's lock).  For historical runs,
    opens a read-only DuckLakeStore with a unique alias.
    """
    if _data_dir is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")

    with _store_lock:
        if run_name not in _run_stores:
            # For the live run, reuse the game's DuckLakeStore
            if run_name == _live_run_name and _live_store is not None:
                _run_stores[run_name] = RunStore(_live_store)
            else:
                run_dir = _data_dir / run_name
                if not run_dir.is_dir():
                    raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
                # Unique alias per run avoids DuckDB catalog collisions
                safe_alias = "r_" + run_name.replace("-", "_").replace(".", "_")
                dl_store = DuckLakeStore(run_dir, read_only=True, alias=safe_alias)
                _run_stores[run_name] = RunStore(dl_store)
        return _run_stores[run_name]


def _get_step_data(run_name: str, step: int) -> StepData:
    """Get step data -- from StepBuffer if live, else from parquet/DuckLake.

    For the live run, tries StepBuffer first (in-memory, instant).
    Falls back to a separate DuckDB connection reading parquet files
    directly (bypasses the game's DuckLakeStore to avoid thread-safety issues).

    StepBuffer data lacks ``logs`` (loguru output goes to a separate OTel
    logger, not the game loop).  When the buffer hit has no logs, we
    supplement from DuckLake so the LogMessages panel works during live play.
    """
    # Try StepBuffer first for the live run
    if run_name == _live_run_name and _step_buffer is not None:
        buf_data = _step_buffer.get_step(step)
        if buf_data is not None:
            # Supplement missing logs from DuckLake
            if buf_data.logs is None:
                try:
                    store = _get_store(run_name)
                    logs_df = store.get_step(step, "logs")
                    if len(logs_df) > 0:
                        from typing import cast

                        buf_data.logs = cast(list[dict[str, Any]], logs_df.to_dict("records"))
                except Exception:
                    pass  # logs are best-effort
            return buf_data
        # Step evicted from buffer -- fall through to parquet

    # Read from parquet via RunStore (separate DuckDB connection, thread-safe)
    store = _get_store(run_name)
    return store.get_step_data(step)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    name: str
    games: int
    steps: int


class GameSummary(BaseModel):
    game_number: int
    steps: int
    start_ts: int | None
    end_ts: int | None


class StepRange(BaseModel):
    min: int
    max: int


class Bookmark(BaseModel):
    step: int
    game: int
    annotation: str
    created: str


# ---------------------------------------------------------------------------
# Run summary cache -- populated in a background thread to avoid blocking
# the API while opening hundreds of DuckDB connections.
# ---------------------------------------------------------------------------

_summary_thread: threading.Thread | None = None


def _populate_run_summaries() -> None:
    """Background thread: scan all runs and cache their game/step counts.

    Processes newest runs first (most likely to be requested).  Uses
    temporary DuckDB connections that are closed immediately after each
    query to avoid holding locks that block request-serving threads.
    Sleeps briefly between runs to yield CPU to request handlers.
    """
    import time

    if _data_dir is None:
        return
    names = RunStore.list_runs(_data_dir)
    names.reverse()  # newest first
    for name in names:
        if name in _run_summary_cache:
            continue
        if name == _live_run_name:
            continue
        try:
            run_dir = _data_dir / name
            if not run_dir.is_dir():
                continue
            # Open a temporary connection -- don't use _get_store() which
            # caches connections and holds _store_lock, starving requests.
            safe_alias = "sum_" + name.replace("-", "_").replace(".", "_")
            dl_store = DuckLakeStore(run_dir, read_only=True, alias=safe_alias)
            try:
                store = RunStore(dl_store)
                games_df = store.list_games()
                games = len(games_df)
                steps = int(games_df["steps"].sum()) if games > 0 else store.step_count()
                _run_summary_cache[name] = RunSummary(name=name, games=games, steps=steps)
            finally:
                dl_store.close()
        except Exception:
            _run_summary_cache[name] = RunSummary(name=name, games=0, steps=0)
        # Yield to request-serving threads between runs
        time.sleep(0.05)


def _get_run_summary(name: str) -> "RunSummary":
    """Return a RunSummary, from cache if available.

    For uncached historical runs, returns games=0/steps=0 (the background
    thread will fill in the real values).  The live run is always queried
    fresh since its step count grows.
    """
    is_live = name == _live_run_name
    if not is_live and name in _run_summary_cache:
        return _run_summary_cache[name]
    if is_live:
        try:
            store = _get_store(name)
            games_df = store.list_games()
            games = len(games_df)
            steps = int(games_df["steps"].sum()) if games > 0 else store.step_count()
            return RunSummary(name=name, games=games, steps=steps)
        except Exception:
            return RunSummary(name=name, games=0, steps=0)
    # Not yet cached -- return placeholder; background thread will fill it in
    return RunSummary(name=name, games=0, steps=0)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs(min_steps: int = 10) -> list[RunSummary]:
    """List available runs with metadata.

    ``min_steps`` filters out short-lived / crashed runs (default 10).
    Pass ``min_steps=0`` to include all runs.
    """
    if _data_dir is None:
        return []
    names = RunStore.list_runs(_data_dir)
    names.reverse()  # newest first
    results: list[RunSummary] = []
    for name in names:
        summary = _get_run_summary(name)
        if summary.steps >= min_steps:
            results.append(summary)
    return results


@app.get("/api/runs/{run_name}/games")
def list_games(run_name: str) -> list[GameSummary]:
    """List games in a run."""
    # For live run, use step buffer game data
    if run_name == _live_run_name and _step_buffer is not None:
        counts = _step_buffer.steps_per_game()
        return [
            GameSummary(game_number=g, steps=counts.get(g, 0), start_ts=None, end_ts=None)
            for g in sorted(counts.keys())
        ]
    store = _get_store(run_name)
    df = store.list_games()
    return [
        GameSummary(
            game_number=int(row["game_number"]),
            steps=int(row["steps"]),
            start_ts=int(row["start_ts"]) if row.get("start_ts") is not None else None,
            end_ts=int(row["end_ts"]) if row.get("end_ts") is not None else None,
        )
        for _, row in df.iterrows()
    ]


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


@app.get("/api/runs/{run_name}/step/{step}")
def get_step(
    run_name: str,
    step: int,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get all data for a specific step."""
    t0 = time.monotonic()
    try:
        step_data = _get_step_data(run_name, step)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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


@app.get("/api/runs/{run_name}/steps")
def get_steps_batch(
    run_name: str,
    steps: str = Query(description="Comma-separated step numbers"),
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get data for multiple steps in a single request.

    Returns a dict mapping step number (as string key) to StepData.
    Steps that fail to load are silently skipped.
    """
    t0 = time.monotonic()
    step_nums = [int(s.strip()) for s in steps.split(",") if s.strip()]
    result: dict[str, Any] = {}
    for step in step_nums:
        try:
            step_data = _get_step_data(run_name, step)
            result[str(step)] = _convert_numpy(dataclasses.asdict(step_data))
        except Exception:
            pass  # skip steps that fail
    t1 = time.monotonic()
    logger.debug(
        "GET steps batch ({} steps) total={:.1f}ms",
        len(result),
        (t1 - t0) * 1000,
    )
    return JSONResponse(content=result)


@app.get("/api/runs/{run_name}/step-range")
def get_step_range(
    run_name: str,
    game: int | None = Query(default=None),
) -> StepRange:
    """Get min/max step for a run or game."""
    # For live run, use step buffer range
    if run_name == _live_run_name and _step_buffer is not None and len(_step_buffer) > 0:
        if game is not None:
            smin, smax = _step_buffer.step_range_for_game(game)
        else:
            smin, smax = _step_buffer.min_step, _step_buffer.max_step
        return StepRange(min=smin, max=smax)
    try:
        store = _get_store(run_name)
        min_step, max_step = store.step_range(game)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return StepRange(min=min_step, max=max_step)


@app.get("/api/runs/{run_name}/metrics-history")
def get_metrics_history(
    run_name: str,
    game: int | None = Query(default=None),
    fields: str | None = Query(default=None, description="Comma-separated field names"),
) -> JSONResponse:
    """Get game_metrics for all steps in a game (for charting trends)."""
    store = _get_store(run_name)
    field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None
    data = store.get_metrics_history(game, field_list)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/graph-history")
def get_graph_history(
    run_name: str,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get graph_summary for all steps in a game (for charting cache utilization)."""
    store = _get_store(run_name)
    data = store.get_graph_history(game)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/event-history")
def get_event_history(
    run_name: str,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get event_summary for all steps in a game (for charting event activity)."""
    store = _get_store(run_name)
    data = store.get_event_history(game)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/intrinsics-history")
def get_intrinsics_history(
    run_name: str,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get intrinsics for all steps in a game (for charting trends)."""
    store = _get_store(run_name)
    data = store.get_intrinsics_history(game)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/action-history")
def get_action_history(
    run_name: str,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get action taken for all steps in a game (for charting action distribution)."""
    store = _get_store(run_name)
    data = store.get_action_history(game)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/resolution-history")
def get_resolution_history(
    run_name: str,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get resolution accuracy history for charting correct/incorrect/new counts."""
    store = _get_store(run_name)
    data = store.get_resolution_history(game)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/all-objects")
def get_all_objects(
    run_name: str,
    game: int | None = Query(default=None),
) -> JSONResponse:
    """Get all resolved objects with attributes, match counts, and creation step."""
    store = _get_store(run_name)
    data = store.get_all_objects(game)
    return JSONResponse(content=_convert_numpy(data))


@app.get("/api/runs/{run_name}/bookmarks")
def get_bookmarks(run_name: str) -> list[Bookmark]:
    """Get bookmarks for a run."""
    if _data_dir is None:
        return []
    bookmarks_file = _data_dir / run_name / "bookmarks.json"
    if not bookmarks_file.exists():
        return []
    try:
        data = json.loads(bookmarks_file.read_text())
        return [Bookmark(**b) for b in data]
    except Exception:
        return []


@app.post("/api/runs/{run_name}/bookmarks")
def save_bookmarks(run_name: str, bookmarks: list[Bookmark]) -> dict[str, str]:
    """Save bookmarks for a run."""
    if _data_dir is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    bookmarks_file = _data_dir / run_name / "bookmarks.json"
    bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
    bookmarks_file.write_text(json.dumps([b.model_dump() for b in bookmarks], indent=2))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Live mode endpoints
# ---------------------------------------------------------------------------


class LiveStatus(BaseModel):
    active: bool
    run_name: str | None
    step: int
    game_number: int
    step_min: int
    step_max: int
    game_numbers: list[int]


@app.get("/api/live/status")
def live_status() -> LiveStatus:
    """Get current live run status."""
    if _step_buffer is None or len(_step_buffer) == 0:
        return LiveStatus(
            active=False,
            run_name=None,
            step=0,
            game_number=0,
            step_min=0,
            step_max=0,
            game_numbers=[],
        )
    latest = _step_buffer.get_latest()
    return LiveStatus(
        active=True,
        run_name=_live_run_name,
        step=latest.step if latest else 0,
        game_number=latest.game_number if latest else 0,
        step_min=_step_buffer.min_step,
        step_max=_step_buffer.max_step,
        game_numbers=_step_buffer.game_numbers,
    )


@app.get("/api/live/step/{step}")
def live_step(step: int) -> JSONResponse:
    """Get step data from the live StepBuffer (in-memory)."""
    if _step_buffer is None:
        raise HTTPException(status_code=404, detail="No live session")
    step_data = _step_buffer.get_step(step)
    if step_data is None:
        raise HTTPException(status_code=404, detail=f"Step {step} not in buffer")
    data = _convert_numpy(dataclasses.asdict(step_data))
    return JSONResponse(content=data)


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
# Server lifecycle
# ---------------------------------------------------------------------------

_started = False


def start_dashboard() -> None:
    """Start the FastAPI dashboard server.

    Creates a StepBuffer and registers it globally so the game loop
    can push StepData. Socket.io broadcasts push notifications to
    all connected browser clients.
    """
    global _started, _data_dir, _step_buffer, _live_run_name, _live_store
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

    _data_dir = store.run_dir.parent
    _live_run_name = store.run_dir.name
    _live_store = store

    # Create step buffer for live push.
    # Large capacity so live-following mode can read from memory (instant).
    # Evicted steps fall through to DuckLake catalog reads (~8ms).
    _step_buffer = StepBuffer(capacity=100_000)
    register_step_buffer(_step_buffer)
    _step_buffer.add_listener(lambda: _notify_new_step(_step_buffer.get_latest()))  # type: ignore[arg-type]

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
        host="0.0.0.0",
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
    logger.debug("Dashboard API stopped.")
