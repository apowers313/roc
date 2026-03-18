# React Dashboard Design

## Why React Replaces Panel

The Panel/Bokeh dashboard had a fundamental architectural flaw: two control paths for the
same state. Button clicks change widget properties via the Bokeh websocket protocol, while
server push updates change the same properties from Python. These race because Bokeh's
protocol doesn't distinguish "user action" from "server update."

React eliminates this by cleanly separating concerns:
- **Client owns UI state** -- playback controls, current step, play/pause are all React state.
  No server round-trip for button clicks.
- **Server owns data** -- REST API for historical queries, Socket.io for live push. The server
  never touches UI state.
- **No race condition** -- the server pushes *data*, the client decides when/how to render it
  based on its own playback state.

## Requirements

### Navigation and Synchronization
- **R1: Synchronized step navigation** -- a single step control (slider + buttons) updates all
  panels simultaneously. First/Prev/Play/Pause/Next/Last buttons, slider scrubbing, and
  keyboard shortcuts all operate on one shared step cursor.
- **R14: Run selection** -- dropdown to switch between completed runs (newest first). Each run
  is a timestamped directory of parquet files under `data_dir`.
- **R16: Keyboard shortcuts** -- Left/Right for step, Space for play/pause, Home/End for
  first/last, Shift+Left/Right for -10/+10 steps.
- **R17: Bookmarks** -- mark interesting steps with optional annotations. Navigate between
  bookmarks with keyboard. Persisted as JSON alongside run data.
- **R19: Multi-game runs** -- game selector dropdown filters steps to a single game within a
  run. Step range updates to show only that game's steps.
- **R20: Performance** -- step transitions under 200ms. Achieved via TanStack Query caching
  (`staleTime: Infinity` since step data is immutable), prefetch of adjacent steps, and
  debounced slider input.

### Live Mode
- **R18: Live mode** -- when the game loop is running, Socket.io pushes full StepData to the
  browser. The dashboard auto-follows at the live edge unless the user navigates away.
  - **Following**: renders push data directly (no REST round-trip), step counter auto-advances.
  - **Paused**: shows PAUSED badge, REST fetches historical data, live pushes update stepMax
    only.
  - **Catch-up**: auto-plays from paused position toward live edge, transitions to following
    when caught up.
  - **Historical**: no live session active, browse completed runs via REST only.

### Data Panels

Each panel renders a section of the StepData dataclass. All panels update together when the
step changes.

- **R2: Game screen** -- 24x80 colored character grid (NetHack tty output). Rendered as HTML
  spans with per-cell foreground/background colors using `dangerouslySetInnerHTML` for
  performance. Fixed-height container (260px) prevents layout shifts.
- **R3: Saliency map** -- same grid format as game screen but with heatmap colors showing
  attention weights. Rendered by the same CharGrid component.
- **R4: Features** -- feature extraction counts per type (Flood, Line, Single, Distance, Color,
  Shape, Delta, Motion). Always shows all 8 feature types with "--" for missing values to
  prevent layout shifts.
- **R5: Objects** -- object info from the resolution pipeline. Raw string display.
- **R6: Focus points** -- attention focus point coordinates. Raw string display.
- **R7: Game metrics** -- key-value table: score, hp, hp_max, energy, energy_max, depth, gold,
  x, y, hunger, xp_level, experience, ac. Right-aligned tabular numbers.
- **R8: Attenuation** -- saliency attenuation parameters (excluding large grid data).
- **R9: Graph DB summary** -- node/edge cache counts and limits.
- **R10: Event summary** -- per-step event bus activity counts.
- **R11: Resolution metrics** -- object resolution decision details.
- **R12: Log messages** -- OTel log records for the step. Severity-colored rows.

### Infrastructure
- **R13: Parquet + DuckDB storage** -- game data is stored as parquet files via DuckLake
  (OTel LogExporter -> ParquetExporter). RunStore queries via DuckDB. StepBuffer (100K
  capacity) provides in-memory access for the live run.
- **R15: Pipeline-organized layout** -- panels grouped by processing stage in collapsible
  accordion sections: Game State, Perception, Attention, Object Resolution, Log Messages.
- **R21: Server management** -- API server starts as a daemon thread in the game process.
  Synchronized startup via `threading.Event` ensures the asyncio event loop is ready before
  the game loop begins pushing data.

## Design Principles

From the @graphty/compact-mantine theme:
- **Dark background**: `#1a1b1e` base with `#25262b` surface cards
- **Dense layout**: minimal padding, tight line-height, small fonts (xs = 11px)
- **Information-rich**: show data, not chrome. Compact KV tables with titled cards, fixed
  column widths, tabular-nums for stable number alignment
- **Stable rendering**: fixed-height containers, always-present rows (null = "--"), fixed table
  layouts to prevent content shifts between frames
- **No dead space**: 6px card padding, 0px table cell padding, 150px max-width KV tables

## Technology Stack

### Frontend (dashboard-ui/)
| Category | Choice | Why |
|----------|--------|-----|
| Framework | React 18+ / TypeScript | Standard, large ecosystem |
| UI Library | Mantine 8.x | Peer dep of compact-mantine, comprehensive |
| Theme | @graphty/compact-mantine | Dense UI, dark mode, 24px components, 11px fonts |
| Build | Vite | Fast, per project preference |
| Test | Vitest + @testing-library/react | Per project preference |
| Data fetching | @tanstack/react-query | Step caching, prefetch, loading states |
| Real-time | socket.io-client | Auto-reconnect, robust WebSocket |
| Charts | Recharts | React-first, simple API |
| Icons | lucide-react | Consistent with graphty ecosystem |
| Keyboard | react-hotkeys-hook | Clean API, focus-aware |
| Tables | Mantine Table | Built-in, themed, sufficient for our needs |
| State | React useReducer | 4-state playback machine, no external lib needed |
| Layout | Mantine AppShell + Accordion | Header/main areas, collapsible sections |
| CSS | Mantine props + inline styles | No additional CSS-in-JS, no global overrides |

### Backend (roc/reporting/)
| Category | Choice | Why |
|----------|--------|-----|
| API Server | FastAPI + uvicorn | Async, WebSocket support, lightweight |
| Real-time | python-socketio | Server-side Socket.io |
| Data layer | RunStore (DuckDB/Parquet) | Reads parquet files via DuckDB views |
| Live push | StepBuffer (100K ring buffer) | Thread-safe deque, in-memory access for live run |
| Storage | DuckLake + ParquetExporter | OTel logs -> parquet archival files |

## Project Structure

```
roc/
  roc/
    reporting/
      api_server.py               # FastAPI + Socket.io server
      run_store.py                # DuckDB query layer over parquet files
      step_buffer.py              # Thread-safe ring buffer for live data
      ducklake_store.py           # DuckLake catalog + parquet write path
      parquet_exporter.py         # OTel LogExporter -> DuckLake
      screen_renderer.py          # Screen dict -> {chars, fg, bg} conversion
      metrics.py                  # OTel histogram/counter dispatch
      observability.py            # OTel setup (traces, metrics, logs)
      state.py                    # Runtime state tracking + OTel emission

dashboard-ui/
  package.json
  tsconfig.json
  vite.config.ts                  # Dev server with HTTPS + API proxy
  vitest.config.ts
  index.html
  src/
    main.tsx                      # Entry point, providers (Mantine, TanStack, Context)
    App.tsx                       # AppShell layout, data flow orchestration
    api/
      client.ts                   # REST API client (typed fetch wrappers)
      queries.ts                  # TanStack Query hooks (useStepData, useRuns, etc.)
    state/
      playback.ts                 # useReducer playback state machine (4 states)
      playback.test.ts            # State transition tests
      context.tsx                 # DashboardContext (step, game, run, playback, range)
    components/
      transport/
        TransportBar.tsx          # Run/game selectors, step controls, slider, speed
      status/
        StatusBar.tsx             # HP bar, Score, Depth, Gold, Energy, Hunger, LIVE badge
      panels/
        GameScreen.tsx            # Fixed-height wrapper around CharGrid
        SaliencyMap.tsx           # CharGrid with heatmap colors
        FeatureTable.tsx          # Fixed 8-row feature counts (stable layout)
        GameMetrics.tsx           # KVTable card with game stats
        LogMessages.tsx           # Severity-colored log table
        FocusPoints.tsx           # Focus point display
        GraphSummary.tsx          # Node/edge counts
      common/
        KVTable.tsx               # Reusable key-value table with optional titled Card
        CharGrid.tsx              # Colored character grid (dangerouslySetInnerHTML)
    hooks/
      useDebouncedValue.ts        # Debounce hook for slider/step changes
      useLiveUpdates.ts           # Socket.io connection + live status polling
    types/
      step-data.ts                # StepData TypeScript interface (mirrors Python dataclass)
      api.ts                      # API response types (RunSummary, GameSummary, etc.)
```

## Architecture

### Data Flow

```
Live following mode:
  Game loop pushes StepData to StepBuffer
    -> StepBuffer listener calls _notify_new_step()
    -> asyncio.run_coroutine_threadsafe(sio.emit("new_step", data))
    -> Socket.io pushes full StepData to browser
    -> onNewStep callback: setLiveData(pushData), setStep(pushData.step)
    -> data = liveData (instant, no REST round-trip)
    -> All panels re-render

Historical / paused mode:
  User clicks step control
    -> React state update (immediate, no server)
    -> useDebouncedValue (150ms) -> useStepData query fires
    -> TanStack Query: fetch(/api/runs/{run}/step/{n})
    -> Cache hit? Render immediately. Cache miss? Show previous data (placeholderData),
       fetch in background, render when ready.
    -> All panels re-render with new StepData

REST API path (server side):
  GET /api/runs/{run}/step/{n}
    -> _get_step_data(run, n)
    -> Try StepBuffer first (in-memory, instant) for live run
    -> Fall back to RunStore (DuckDB over parquet files) for evicted/historical steps
    -> dataclasses.asdict() + _convert_numpy() -> JSON response
```

### Playback State Machine (Client-Side useReducer)

```
                    GO_LIVE
  historical ─────────────────> live_following
       |                            |
       | (no live session)          | USER_NAVIGATE / PAUSE
       |                            v
       |                       live_paused
       |                            |
       |                            | RESUME
       |                            v
       |                       live_catchup
       |                            |
       |                            | PUSH_ARRIVED(atEdge=true)
       |                            v
       └────────────────────── live_following
```

Transitions:
- `historical` + `GO_LIVE` -> `live_following` (auto-select on first live status)
- `live_following` + `USER_NAVIGATE` -> `live_paused` (user clicked step/slider)
- `live_paused` + `JUMP_TO_END` -> `live_following` (return to live edge)
- `live_paused` + `RESUME` -> `live_catchup` (auto-play toward live)
- `live_catchup` + `PUSH_ARRIVED(atEdge)` -> `live_following` (caught up)

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/runs` | List runs (newest first) |
| GET | `/api/runs/{run}/games` | List games with step counts |
| GET | `/api/runs/{run}/step/{n}` | Full StepData for step n |
| GET | `/api/runs/{run}/step-range` | Min/max step (optional game filter) |
| GET | `/api/runs/{run}/bookmarks` | Bookmarks for a run |
| POST | `/api/runs/{run}/bookmarks` | Save bookmarks |
| GET | `/api/live/status` | Live session info (polled every 3s) |
| GET | `/api/live/step/{n}` | Step from StepBuffer (in-memory only) |
| WS | `/socket.io` | Socket.io: emits `new_step` with full StepData |

### Server Lifecycle

1. `roc.init()` calls `start_dashboard()`
2. Creates StepBuffer (100K capacity), registers globally
3. Adds listener: on push, emit via Socket.io
4. Starts uvicorn in daemon thread
5. **Waits** for `threading.Event` from FastAPI startup handler (prevents pthread crash)
6. Returns to game loop -- StepBuffer pushes are now safe
7. On game end, `stop_dashboard()` clears the buffer

### Key UI Patterns

**KVTable card**: Reusable `<KVTable data={...} title="Metrics" />` component. When `title`
is provided, wraps in a bordered Mantine Card. Fixed 150px max-width, zero cell padding,
tabular-nums for stable number columns. Used for Metrics, Feature Counts, Attenuation, and
Resolution panels.

**CharGrid**: Renders 24x80 colored spans via `dangerouslySetInnerHTML`. No React.memo (parent
controls re-renders via data flow). DejaVu Sans Mono at 9px.

**Stable layout**: GameScreen has 260px min-height. FeatureTable always renders all 8 feature
types (missing = "--"). Tables use `layout="fixed"`. All prevents accordion content from
jumping between frames.

**Flicker-free scrubbing**: TanStack Query `placeholderData: keepPreviousData` shows the old
step's data while the new step loads. Historical mode never falls back to liveData.

## Implementation Status

### Completed
- [x] Phase 1: Skeleton + Screen (MVP) -- FastAPI, REST, TanStack Query, CharGrid
- [x] Phase 2: Data Panels -- StatusBar, FeatureTable, SaliencyMap, GameMetrics, LogMessages,
  KVTable card component, Accordion layout
- [x] Phase 3: Live Mode -- Socket.io push, playback state machine, auto-follow, synchronized
  server startup
- [x] Panel/wandb removal -- all Panel, Bokeh, and wandb code deleted

- [x] Phase 4: Keyboard Shortcuts + Bookmarks
  - react-hotkeys-hook for Left/Right/Space/Home/End/Shift+arrows
  - Keyboard help overlay (Mantine Modal, toggled by ?)
  - Bookmark toggle (B), navigate next/prev (]/[)
  - Visual bookmark markers on slider track
  - Vite resolve alias to deduplicate React (pnpm hoisting fix)

- [x] Phase 5: Polish
  - Recharts LineChart for HP/score/energy trends (MetricsChart + metrics-history API)
  - Recharts BarChart for event activity (EventSummary)
  - Object Resolution inspector with structured table (auto-detected columns)
  - Error boundaries wrapping every panel
  - Production build (`vite build` served from FastAPI static mount)
  - Log message severity filtering (dropdown + color-coded rows)
  - Responsive layout (Grid.Col with `base: 12, md: 8/4` breakpoints)
