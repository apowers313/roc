# Dashboard Server Redesign

## Problem Statement

The current ROC dashboard has two separate server modes that conflict with each
other:

1. **Game mode** (`uv run play`): Runs the game loop on the main thread, starts
   FastAPI + Socket.io in a daemon thread. Serves live data via StepBuffer and
   historical data via DuckLake.
2. **Standalone dashboard** (`uv run dashboard`): Runs FastAPI synchronously on
   the main thread. Historical data only, no game loop.

These modes fight over ports, require separate processes, and have no shared
management. The Vite dev server adds a third process with hardcoded ports and
manual proxy configuration. The result:

- Port conflicts between game, dashboard, and Vite dev server
- Orphan processes that block ports after crashes
- No consistent way to start/stop/switch between modes
- User must remember which URL to open (different ports for dev vs prod)
- No way to start a game from the dashboard UI
- Navigating to a historical run URL gets overridden by live run auto-selection

## Goals

1. **One server, one URL.** A single persistent process serves both the dashboard
   UI and the API. The user always opens the same URL.
2. **Game lifecycle via API.** Start and stop games from the dashboard UI or via
   REST endpoints. No server restarts to switch modes.
3. **Managed by servherd.** One `make` command starts everything. Ports are
   auto-assigned. No hardcoded ports anywhere.
4. **HMR preserved.** The Vite dev server still provides hot module replacement
   during frontend development.
5. **Clean separation.** The server is always available for historical browsing.
   Starting a game adds live data; stopping it returns to browse-only mode.
6. **Respect user intent.** URL parameters for historical runs are never
   overridden by live run detection.

## Architecture

### Current Architecture

```
Mode 1: Game (uv run play)
  main thread: roc.init() -> NethackGym.start() [blocking game loop]
  daemon thread: FastAPI + Socket.io on dashboard_port

Mode 2: Dashboard (uv run dashboard)
  main thread: FastAPI + Socket.io on dashboard_port [blocking uvicorn]
  (no game loop, no StepBuffer, DuckLake only)

Mode 3: Frontend dev
  separate process: Vite dev server on port 9044
  proxies /api and /socket.io to port 9043
```

### Proposed Architecture

```
Persistent server process (managed by servherd as "roc-server"):
  main thread: FastAPI + Socket.io + uvicorn [always running]
  game subprocess: uv run play [started/stopped via API, full process isolation]

Frontend (managed by servherd as "roc-ui"):
  Vite dev server, port auto-assigned
  reads API port from servherd: {{$ "roc-server" "port"}}
```

The server is always available. Starting a game spawns a subprocess. Stopping a
game kills the subprocess. The dashboard UI works at all times -- browsing
historical runs when no game is active, showing live data when a game is running.

## Detailed Design

### 1. Game Lifecycle API

New REST endpoints on the existing FastAPI app:

```
POST /api/game/start    Start a new game (returns immediately)
POST /api/game/stop     Stop the running game (graceful shutdown)
GET  /api/game/status   Current game state (idle/running/stopping)
```

#### POST /api/game/start

Query parameter: `num_games` (optional, default 5).

Response:

```json
{
  "status": "starting"
}
```

Behavior:
- If a game is already running, returns 409 Conflict.
- Spawns `uv run play` as a subprocess with `--no-dashboard-enabled` (the
  game uses the server's existing API, not its own) and
  `--dashboard-callback-url=<server>/api/internal/step` for live data push.
- Returns immediately. The game runs asynchronously.
- Live data flows via HTTP callback: the game subprocess POSTs each StepData
  to the server's `/api/internal/step` endpoint, which stores it in a
  StepBuffer and broadcasts via Socket.io.

#### POST /api/game/stop

Response:

```json
{
  "status": "stopping"
}
```

Behavior:
- Sends SIGTERM to the game subprocess.
- The game loop exits cleanly, flushes OTel/DuckLake state.
- Once the subprocess exits, status transitions to "idle".
- If no game is running, returns 409 Conflict.

#### GET /api/game/status

Response:

```json
{
  "state": "running",
  "run_name": "20260321-120000-adjective-name-name"
}
```

Optional fields (present only when applicable): `exit_code`, `error`.

States: `idle`, `initializing`, `running`, `stopping`.

Note: `game_number` and `step` are not included in the status response.
These are delivered in real-time via the live step data flow (Socket.io
`new_step` events) rather than polled from the status endpoint.

#### POST /api/internal/step (Internal)

Receives StepData from the game subprocess via HTTP callback. The game
subprocess POSTs each step's data as JSON to this endpoint. The server
stores it in its StepBuffer and broadcasts via Socket.io.

This endpoint exists because DuckDB uses file-level locking, which prevents
cross-process concurrent access to the DuckLake catalog. The HTTP callback
approach avoids this limitation while providing lower latency (~10ms) than
polling would.

### 2. Game Process Management

#### Why Subprocess Instead of Threading

The codebase has ~20 module-level singletons (component registry, GraphDB
caches, event buses, OTel providers, NLE state, etc.). Building a reliable
`roc.reset()` to clean up between runs would be fragile and incomplete --
especially for NLE's C extension internals which are opaque to Python.

A subprocess provides full process isolation:
- Every game starts with a clean Python interpreter. No singleton cleanup needed.
- NLE's C extension gets a fresh address space. No thread safety concerns.
- A crashed game cannot corrupt the server's state.
- The existing `uv run play` command works unchanged as the subprocess.

#### GameManager

New module: `roc/game_manager.py`

```python
class GameManager:
    """Manages game lifecycle via subprocess."""

    _process: subprocess.Popen | None
    _state: Literal["idle", "initializing", "running", "stopping"]
    _current_run_name: str | None
    _monitor_thread: Thread | None

    def start_game(self, config_overrides: dict | None = None) -> str:
        """Spawn uv run play as a subprocess."""

    def stop_game(self) -> None:
        """Send SIGTERM to the game subprocess."""

    def get_status(self) -> dict:
        """Return current game state, run name, step number."""
```

#### Subprocess Communication

The game subprocess and the API server need to share live step data.

**Implemented: HTTP callback (POST to /api/internal/step)**

The original design proposed DuckLake polling as the simplest option, with
socket IPC as a lower-latency alternative. During implementation, DuckDB's
file-level locking prevented cross-process access to the DuckLake catalog
(the game subprocess holds a write lock, blocking the server's reads).

The implemented solution uses HTTP callbacks:
- The server passes `--dashboard-callback-url=<server>/api/internal/step`
  to the game subprocess.
- The game subprocess POSTs each StepData as JSON to this endpoint after
  each game step (using `urllib.request` with a 2-second timeout).
- The server receives it, stores in a live StepBuffer, and broadcasts via
  Socket.io.
- Latency: ~10ms. Same real-time feel as the in-process StepBuffer.
- SSL: For HTTPS servers, the subprocess uses `ssl.create_default_context()`
  with certificate verification disabled (localhost self-signed certs).

The `dashboard_callback_url` config field in `roc/config.py` controls this.
The `roc/gymnasium.py` game loop conditionally assembles and POSTs StepData
when this URL is set.

### 3. Unified Server Entry Point

New entry point: `server` in pyproject.toml:

```toml
[project.scripts]
play = "roc.script:cli"           # keep for backward compat
dashboard = "roc.dashboard_cli:main"  # keep for backward compat
server = "roc.server_cli:main"    # new unified entry point
```

The `server` command:
- Starts FastAPI + Socket.io on the configured port (main thread, blocking).
- Mounts static files from `dashboard-ui/dist/` if present.
- Registers the game lifecycle endpoints (`/api/game/*`).
- Registers the GameManager singleton.
- Does NOT start a game automatically.
- Accepts config flags for port, data-dir, SSL, etc.

The existing `play` and `dashboard` commands continue to work unchanged for
backward compatibility, scripting, and direct game runs without the server.

### 4. Vite Configuration Changes

Replace hardcoded ports with environment variables in `vite.config.ts`:

```typescript
const SSL_CERT = process.env.VITE_SSL_CERT || "/home/apowers/ssl/atoms.crt";
const SSL_KEY = process.env.VITE_SSL_KEY || "/home/apowers/ssl/atoms.key";
const HOST = process.env.VITE_HOST || "dev.ato.ms";
const PORT = parseInt(process.env.VITE_DEV_PORT || "9044", 10);
const API_PORT = parseInt(process.env.VITE_API_PORT || "9043", 10);
```

This allows servherd to pass the API port dynamically via its cross-server
reference syntax `{{$ "roc-server" "port"}}`. SSL certificate paths are also
configurable for environments with different cert locations.

### 5. Servherd Integration

Two servherd-managed servers, started by Makefile targets:

**Backend (`roc-server`)**:
```bash
servherd start -n roc-server \
  -e roc_dashboard_port={{port}} \
  -- uv run server
```

**Frontend (`roc-ui`)**:
```bash
servherd start -n roc-ui \
  -e 'VITE_API_PORT={{$ "roc-server" "port"}}' \
  -e VITE_DEV_PORT={{port}} \
  -e VITE_HOST={{hostname}} \
  -- pnpm -C dashboard-ui dev
```

The frontend reads the backend port from servherd's cross-server reference.
Both ports are auto-assigned -- no hardcoded values, no conflicts.

Old servherd entries to clean up: `roc-game`, `roc-dashboard`,
`roc-dashboard-api`, `panel-debug`.

### 6. Makefile Targets

```makefile
# Start the unified server + Vite dev frontend
run:
	@npx servherd start -n roc-server \
	  -e roc_dashboard_port={{port}} \
	  -- uv run server --port {{port}} >/dev/null
	@npx servherd start -n roc-ui \
	  -e 'VITE_API_PORT={{$ "roc-server" "port"}}' \
	  -e VITE_DEV_PORT={{port}} \
	  -e VITE_HOST={{hostname}} \
	  -- pnpm -C dashboard-ui dev >/dev/null
	@echo "Dashboard: $$(npx servherd info roc-ui --json 2>/dev/null | jq -r .data.url)"

# Stop everything
stop:
	@npx servherd stop roc-ui 2>/dev/null || true
	@npx servherd stop roc-server 2>/dev/null || true
```

### 7. Dashboard UI Changes

#### Game Menu

A Mantine `Menu` dropdown in the transport bar, triggered by a gamepad
`ActionIcon`. The icon color reflects state (green=running, yellow=stopping,
gray=idle). The menu provides:

- **Status badge**: Green "Game Running", yellow "Stopping...", or gray
  "No game running" text.
- **Error display**: Red text showing the last game's error message (crash
  exit code, signal name), visible when idle after a failed game.
- **Number of games**: `NumberInput` (1-100, default 5) for configuring
  the next game run. Only shown when idle.
- **Start Game**: Calls `POST /api/game/start?num_games=N`. Disabled while
  loading. Only shown when idle.
- **Stop Game**: Calls `POST /api/game/stop`. Disabled while loading. Only
  shown when running. No confirmation dialog (one-click stop for fast
  iteration during debugging).
- **Auto-refresh**: Status refreshes on menu open via `GET /api/game/status`.

Implemented in `dashboard-ui/src/components/transport/GameMenu.tsx`,
imported by `TransportBar.tsx`.

#### Fix: Live Run URL Override (Bug)

**Current bug**: When a user navigates to a URL like
`/?run=historical-run&game=2&step=100`, the auto-select effect in `App.tsx`
(lines 357-369) overwrites the URL params with the live run:

```typescript
// App.tsx:357-369 -- current buggy behavior
useEffect(() => {
    if (liveStatus?.active && liveStatus.run_name && !liveRunSelected.current) {
        liveRunSelected.current = true;
        setRun(liveStatus.run_name);       // overrides URL param
        setGame(liveStatus.game_number);
        setStep(liveStatus.step);
        dispatchPlayback({ type: "GO_LIVE" });
    }
}, [liveStatus, setRun, setGame, setStep, dispatchPlayback]);
```

The `liveRunSelected` ref only tracks "have we tried yet?" -- it does not check
whether the user explicitly provided URL params.

**Fix**: Check whether the initial URL contained a `run` param. If so, do NOT
auto-select the live run:

```typescript
const initialUrlHadRun = useRef(
    new URLSearchParams(window.location.search).has("run"),
);
const prevLiveRunName = useRef<string | null>(null);

useEffect(() => {
    if (!liveStatus?.active || !liveStatus.run_name) return;

    const isNewRun = liveStatus.run_name !== prevLiveRunName.current;
    prevLiveRunName.current = liveStatus.run_name;

    // Auto-navigate on first load (unless URL had explicit run) or
    // when a new game run appears (user started a game from the menu).
    if (isNewRun && !initialUrlHadRun.current) {
        liveRunSelected.current = true;
        setRun(liveStatus.run_name);
        setGame(liveStatus.game_number);
        setStep(liveStatus.step);
        dispatchPlayback({ type: "GO_LIVE" });
    } else if (!liveRunSelected.current && !initialUrlHadRun.current) {
        // First-time auto-select for existing live session
        liveRunSelected.current = true;
        setRun(liveStatus.run_name);
        setGame(liveStatus.game_number);
        setStep(liveStatus.step);
        dispatchPlayback({ type: "GO_LIVE" });
    }
}, [liveStatus, setRun, setGame, setStep, dispatchPlayback]);
```

The `initialUrlHadRun` ref uses inline `URLSearchParams` rather than
importing `readUrlParams()` -- functionally identical, avoids an import
dependency for a one-line check.

### 8. Socket.io Events

Add new events for game lifecycle:

```
Server -> Client:
  game_state_changed: {
    state: "idle" | "initializing" | "running" | "stopping",
    run_name?: string
  }
```

This allows the UI to react immediately to game state changes without waiting
for the next 3-second poll interval. The `useLiveUpdates` hook listens for
this event and triggers an immediate poll of `/api/live/status`, which
provides the authoritative run/game/step information.

Note: `game_number` and `step` are not included in this event. The event
serves as a lightweight signal to re-poll; the poll response carries the
full state.

## Implementation Record

All four phases have been implemented and verified. 815 unit tests pass,
mypy clean (139 files), ruff clean.

### Phase 1: Server Entry Point + Vite + Servherd (Complete)

1. Created `roc/server_cli.py` -- unified server entry point.
2. Added `server = "roc.server_cli:main"` to `pyproject.toml`.
3. Updated `vite.config.ts` with `VITE_API_PORT`, `VITE_DEV_PORT`,
   `VITE_HOST`, `VITE_SSL_CERT`, `VITE_SSL_KEY` env vars.
4. Added `run`, `stop` targets to `Makefile` using servherd.
5. Cleaned up old servherd entries (roc-game, roc-dashboard, etc.).
6. Fixed live run URL override bug with `initialUrlHadRun` ref in App.tsx.

### Phase 2: Game Manager + API Endpoints (Complete)

1. Created `roc/game_manager.py` with subprocess-based GameManager.
2. Added game lifecycle endpoints to `api_server.py`.
3. Game subprocess launched with `--no-dashboard-enabled` and
   `--dashboard-callback-url` flags.
4. Implemented HTTP callback IPC (not DuckLake polling -- see Technical
   Risks section for why).
5. Added `game_state_changed` Socket.io event.
6. Added `dashboard_callback_url` config field to `roc/config.py`.
7. Added HTTP POST callback logic to `roc/gymnasium.py`.

### Phase 3: Dashboard UI Game Controls (Complete)

1. Created `GameMenu.tsx` -- Mantine Menu with gamepad ActionIcon.
2. Wired Start/Stop to API endpoints with loading states.
3. Status indicator via icon color and badge text.
4. Auto-selects live run on new game start (respects URL override fix).

### Phase 4: Hardening (Complete)

1. Crash detection: monitor thread captures exit code, signal name.
2. Concurrent start rejection: returns 409 if state != idle.
3. Log streaming: subprocess stdout/stderr forwarded to loguru with
   `[game]` prefix.
4. SIGKILL escalation: 10-second watchdog after SIGTERM.
5. Error display: Game Menu shows last crash details in red text.

## Technical Risks (Post-Implementation Notes)

### 1. DuckDB File-Level Locking (Realized -- Mitigated)

**Risk materialized**: DuckDB uses file-level locking. The game subprocess
holds a write lock on the DuckLake catalog.duckdb file, which blocks the
server from reading it. This made DuckLake polling (the original Option A)
impossible.

**Mitigation**: Implemented HTTP callback IPC. The game subprocess POSTs
each StepData to the server via HTTP, bypassing DuckDB entirely for live
data. Historical data still reads from DuckLake after the game exits and
releases the lock. Latency is ~10ms, matching the in-process StepBuffer.

### 2. Subprocess Lifecycle Edge Cases (Low-Medium Risk -- Mitigated)

Game subprocess might:
- Crash without flushing DuckLake (lost steps at the end).
- Hang on NLE C extension (unkillable with SIGTERM).
- Leave orphan child processes.

**Implemented mitigations**:
- Monitor thread detects subprocess exit and captures exit code.
- SIGKILL watchdog escalates after 10 seconds if SIGTERM is ignored.
- Crash details (exit code, signal name) displayed in the Game Menu.
- Subprocess stdout/stderr streamed to server logs with `[game]` prefix.

## Out of Scope

- **Multiple concurrent games**: Only one game at a time.
- **Remote game execution**: Games run on the same machine as the server.
- **Authentication/authorization**: Local development tool only.
- **Production deployment**: No Docker, no reverse proxy, no load balancing.
- **In-process game threading**: Rejected due to singleton cleanup complexity
  and NLE C extension opacity. Subprocess isolation is simpler and more robust.

## Appendix A: Current Port Usage

| Component | Current Port | Source |
|-----------|-------------|--------|
| FastAPI (game mode) | 9042 (default) | `config.py:131` |
| FastAPI (overridden) | 9043 (common) | CLI flag or env var |
| Vite dev server | 9044 (hardcoded) | `vite.config.ts:11` |
| Vite proxy target | 9043 (hardcoded) | `vite.config.ts:12` |
| debugpy | 5678 | `config.py:206` |
| Memgraph | 7687 | `config.py:94` |
| Remote logger | 9080 | `config.py:210` |

After this redesign, only `roc-server` and `roc-ui` will have ports, both
auto-assigned by servherd.

## Appendix B: Key Files

| File | Role | Status |
|------|------|--------|
| `roc/server_cli.py` | Unified server entry point | Created |
| `roc/game_manager.py` | Subprocess game lifecycle | Created |
| `roc/reporting/api_server.py` | FastAPI app, game + internal endpoints | Modified |
| `roc/config.py` | Added `dashboard_callback_url` field | Modified |
| `roc/gymnasium.py` | HTTP callback POST logic | Modified |
| `dashboard-ui/vite.config.ts` | Env-driven ports + SSL | Modified |
| `dashboard-ui/src/App.tsx` | URL override fix, live auto-select | Modified |
| `dashboard-ui/src/hooks/useLiveUpdates.ts` | game_state_changed listener | Modified |
| `dashboard-ui/src/components/transport/GameMenu.tsx` | Game lifecycle menu | Created |
| `dashboard-ui/src/components/transport/TransportBar.tsx` | Added GameMenu | Modified |
| `dashboard-ui/src/components/panels/AuralPerception.tsx` | Legacy format fix | Modified |
| `Makefile` | Added `run`, `stop` targets | Modified |
| `pyproject.toml` | Added `server` entry point | Modified |

## Appendix C: Global State Inventory

These singletons are why subprocess isolation was chosen over in-process
threading for game runs. Each would need a custom reset path if we used
threading:

| File | Singleton(s) | Reset Exists? |
|------|-------------|---------------|
| `component.py` | `loaded_components`, `component_set`, `component_registry` | Partial (`reset()`) |
| `config.py` | `_config_singleton` | Yes (`reset()`) |
| `event.py` | `pool_scheduler`, `eventbus_names`, `Event._step_counts` | Partial |
| `graphdb.py` | `graph_db_singleton`, `node_cache`, `edge_cache`, ID counters | No |
| `object.py` | `_feature_to_objects` | Manual `.clear()` |
| `sequencer.py` | `tick` | No |
| `reporting/state.py` | `states`, `_state_init_done` | No |
| `reporting/step_buffer.py` | `_step_buffer` | Yes (`clear_step_buffer()`) |
| `reporting/api_server.py` | `_step_buffer`, `_live_run_name`, `_live_store` | Partial |
| `reporting/observability.py` | `ObservabilityBase._instances`, `instance_id` | Partial |
| `expmod.py` | `expmod_registry`, `expmod_modtype_current` | No |
| `breakpoint.py` | `_breakpoints_dict` | Manual `.clear()` |
