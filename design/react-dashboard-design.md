# React Dashboard Design

## Why React Replaces Panel

The Panel/Bokeh dashboard has a fundamental architectural flaw: two control paths for the
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

All requirements from `design/panel-design.md` (R1-R21) carry over. The data layer
(ParquetExporter, RunStore, StepBuffer, StepData) is unchanged. Only the UI layer and
server layer are replaced.

Key requirements summary:
- R1: Synchronized step navigation (single control updates all panels)
- R2-R6: Data panels (screen, saliency, features, objects, focus points)
- R7-R12: Metrics panels (game metrics, attenuation, graph DB, events, resolution)
- R13: Parquet + DuckDB storage (unchanged)
- R14: Run selection
- R15: Pipeline-organized layout
- R16: Keyboard shortcuts
- R17: Bookmarks
- R18: Live mode with push updates
- R19: Multi-game runs
- R20: Performance (<200ms step transitions)
- R21: Server management

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
| CSS | Mantine props + CSS modules | No additional CSS-in-JS |

### Backend (roc/reporting/)
| Category | Choice | Why |
|----------|--------|-----|
| API Server | FastAPI + uvicorn | Async, WebSocket support, lightweight |
| Real-time | python-socketio | Server-side Socket.io |
| Data layer | RunStore (DuckDB/Parquet) | Unchanged from current |
| Live push | StepBuffer | Unchanged from current |

### Dependencies to Add

**Python (pyproject.toml)**:
- `fastapi`
- `python-socketio`
- `uvicorn[standard]`

**JavaScript (dashboard-ui/package.json)**:
```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "@mantine/core": "^8.0.0",
    "@mantine/hooks": "^8.0.0",
    "@graphty/compact-mantine": "^0.5.1",
    "@tanstack/react-query": "^5.0.0",
    "recharts": "^2.15.0",
    "socket.io-client": "^4.8.0",
    "react-hotkeys-hook": "^4.6.0",
    "lucide-react": "^0.500.0"
  },
  "devDependencies": {
    "vite": "^6.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.7.0",
    "vitest": "^3.0.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "jsdom": "^26.0.0"
  }
}
```

### Dependencies to Remove (after migration complete)

**Python**:
- `panel`
- `bokeh`
- `python-statemachine` (playback machine moves to React useReducer)

## Project Structure

```
roc/                              # Python project root
  roc/
    reporting/
      api_server.py               # NEW: FastAPI + Socket.io server
      run_store.py                # Unchanged
      step_buffer.py              # Unchanged
      parquet_exporter.py         # Unchanged
      screen_renderer.py          # Unchanged (used by API to pre-render)
      observability.py            # Unchanged
      panel_debug.py              # DELETE (replaced by React)
      playback_machine.py         # DELETE (replaced by React useReducer)
      dashboard_server.py         # DELETE (replaced by api_server.py)
      components/                 # DELETE (Panel components)

dashboard-ui/                     # NEW: React project root
  package.json
  tsconfig.json
  vite.config.ts
  vitest.config.ts
  index.html
  src/
    main.tsx                      # Entry point, providers
    App.tsx                       # AppShell layout, routing
    api/
      client.ts                   # REST API client (fetch wrappers)
      socket.ts                   # Socket.io client setup
      queries.ts                  # TanStack Query hooks (useStepData, useRuns, etc.)
    state/
      playback.ts                 # useReducer playback state machine
      context.ts                  # DashboardContext provider (step, game, run, playback)
    components/
      transport/
        TransportBar.tsx          # Step controls (play/pause/speed/slider)
        StepSlider.tsx            # Range slider with current position
        SpeedSelector.tsx         # Playback speed dropdown
      status/
        StatusBar.tsx             # HP, Score, Depth, etc. with Mantine Progress
        LiveBadge.tsx             # LIVE indicator + new data badge
      panels/
        GameScreen.tsx            # 21x79 colored character grid
        SaliencyMap.tsx           # Heatmap character grid
        FeatureTable.tsx          # Feature extraction KV table
        FocusPoints.tsx           # Focus point list
        AttenuationDetails.tsx    # Saliency attenuation KV table
        ObjectResolution.tsx      # Resolution decision + candidates
        GameMetrics.tsx           # Vitals KV table
        GraphSummary.tsx          # Node/edge counts
        EventActivity.tsx         # Event bus bar chart (Recharts)
        LogMessages.tsx           # Filterable log table
      layout/
        SectionAccordion.tsx      # Accordion wrapper with summary headers
      common/
        KVTable.tsx               # Reusable key-value Mantine Table
        CharGrid.tsx              # Colored character grid renderer
      bookmarks/
        BookmarkList.tsx          # Bookmark panel
        BookmarkButton.tsx        # Toggle bookmark on current step
      help/
        KeyboardHelp.tsx          # Shortcut overlay (Mantine Modal)
    hooks/
      useKeyboardShortcuts.ts     # react-hotkeys-hook wrappers
      useLiveUpdates.ts           # Socket.io subscription hook
      useBookmarks.ts             # Bookmark CRUD + navigation
    types/
      step-data.ts                # StepData TypeScript type (mirrors Python dataclass)
      api.ts                      # API response types
    utils/
      colors.ts                   # HP color thresholds, severity colors
      grid-renderer.ts            # Character grid HTML generation
```

## Architecture

### Data Flow

```
Historical mode:
  User clicks step control
    -> React state update (immediate, no server)
    -> TanStack Query: fetch(/api/runs/{run}/step/{n})
    -> Cache hit? Render immediately. Cache miss? Show loading, fetch, render.
    -> All panels re-render with new StepData

Live mode (following):
  Socket.io receives "new_step" event
    -> Update step range in React state
    -> If following: auto-advance current step, fetch new data
    -> If paused: show "new data available" badge, don't advance

Live mode (user navigates away):
  User clicks prev/slider/game change
    -> React state: transition to "live_paused" (instant, client-side)
    -> Fetch historical step data via REST
    -> Socket.io still receives pushes, updates badge count
```

### Playback State Machine (Client-Side)

```typescript
type PlaybackState = "historical" | "live_following" | "live_paused" | "live_catchup";

type PlaybackAction =
    | { type: "GO_LIVE" }
    | { type: "PAUSE" }
    | { type: "RESUME" }
    | { type: "JUMP_TO_END" }
    | { type: "USER_NAVIGATE" }
    | { type: "PUSH_ARRIVED"; atEdge: boolean }
    | { type: "TOGGLE_PLAY" };

function playbackReducer(state: PlaybackState, action: PlaybackAction): PlaybackState {
    switch (state) {
        case "historical":
            switch (action.type) {
                case "GO_LIVE": return "live_following";
                case "TOGGLE_PLAY": return "historical"; // no-op, handled by timer
                case "USER_NAVIGATE": return "historical"; // no-op
                case "JUMP_TO_END": return "historical"; // no-op
                default: return state;
            }
        case "live_following":
            switch (action.type) {
                case "PAUSE": return "live_paused";
                case "USER_NAVIGATE": return "live_paused";
                case "PUSH_ARRIVED": return "live_following"; // stay
                case "JUMP_TO_END": return "live_following"; // no-op
                default: return state;
            }
        case "live_paused":
            switch (action.type) {
                case "RESUME": return "live_catchup";
                case "JUMP_TO_END": return "live_following";
                case "PUSH_ARRIVED": return "live_paused"; // stay, update badge
                case "USER_NAVIGATE": return "live_paused"; // no-op
                default: return state;
            }
        case "live_catchup":
            switch (action.type) {
                case "PAUSE": return "live_paused";
                case "USER_NAVIGATE": return "live_paused";
                case "JUMP_TO_END": return "live_following";
                case "PUSH_ARRIVED":
                    return action.atEdge ? "live_following" : "live_catchup";
                default: return state;
            }
    }
}
```

### API Server (FastAPI)

```python
# roc/reporting/api_server.py (~200 lines)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import socketio

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
sio_app = socketio.ASGIApp(sio, other_app=app)

@app.get("/api/runs")
async def list_runs() -> list[RunSummary]:
    """List available runs with metadata."""

@app.get("/api/runs/{run_name}/games")
async def list_games(run_name: str) -> list[GameSummary]:
    """List games in a run with step counts."""

@app.get("/api/runs/{run_name}/step/{step}")
async def get_step(run_name: str, step: int, game: int | None = None) -> StepDataResponse:
    """Get all data for a specific step."""

@app.get("/api/runs/{run_name}/step-range")
async def get_step_range(run_name: str, game: int | None = None) -> StepRange:
    """Get min/max step for a run or game."""

@app.get("/api/runs/{run_name}/bookmarks")
async def get_bookmarks(run_name: str) -> list[Bookmark]:
    """Get bookmarks for a run."""

@app.post("/api/runs/{run_name}/bookmarks")
async def save_bookmarks(run_name: str, bookmarks: list[Bookmark]) -> None:
    """Save bookmarks for a run."""

# Socket.io: push new step data to connected clients
@sio.event
async def connect(sid, environ):
    """Client connected."""

def notify_new_step(step_data: StepData) -> None:
    """Called by StepBuffer listener when new data arrives."""
    sio.start_background_task(sio.emit, "new_step", step_data.to_dict())

# Serve React static build in production
app.mount("/", StaticFiles(directory="dashboard-ui/dist", html=True))
```

### TanStack Query Integration

```typescript
// api/queries.ts

export function useStepData(run: string, step: number, game?: number) {
    return useQuery({
        queryKey: ["step", run, step, game],
        queryFn: () => fetchStep(run, step, game),
        staleTime: Infinity, // Step data never changes
        // Prefetch adjacent steps for smooth scrubbing
        placeholderData: keepPreviousData,
    });
}

export function useRuns() {
    return useQuery({
        queryKey: ["runs"],
        queryFn: fetchRuns,
        refetchInterval: 5000, // Poll for new runs
    });
}

export function useGames(run: string) {
    return useQuery({
        queryKey: ["games", run],
        queryFn: () => fetchGames(run),
    });
}

// Prefetch adjacent steps when user is scrubbing
export function usePrefetchAdjacentSteps(run: string, step: number, game?: number) {
    const queryClient = useQueryClient();
    useEffect(() => {
        for (const offset of [-2, -1, 1, 2]) {
            queryClient.prefetchQuery({
                queryKey: ["step", run, step + offset, game],
                queryFn: () => fetchStep(run, step + offset, game),
            });
        }
    }, [run, step, game]);
}
```

### Layout (Mantine AppShell + Accordion)

```tsx
// App.tsx

<MantineProvider theme={compactTheme} defaultColorScheme="dark">
    <QueryClientProvider client={queryClient}>
        <DashboardProvider>
            <AppShell header={{ height: 120 }}>
                <AppShell.Header>
                    <TransportBar />
                </AppShell.Header>
                <AppShell.Main>
                    <StatusBar />
                    <Accordion multiple defaultValue={["game-state", "perception"]}>
                        <Accordion.Item value="game-state">
                            <Accordion.Control>Game State</Accordion.Control>
                            <Accordion.Panel>
                                <GameScreen /> <GameMetrics /> <GraphSummary />
                            </Accordion.Panel>
                        </Accordion.Item>
                        <Accordion.Item value="perception">
                            <Accordion.Control>Perception</Accordion.Control>
                            <Accordion.Panel>
                                <FeatureTable />
                            </Accordion.Panel>
                        </Accordion.Item>
                        {/* ... more sections */}
                    </Accordion>
                </AppShell.Main>
            </AppShell>
        </DashboardProvider>
    </QueryClientProvider>
</MantineProvider>
```

### Character Grid Component

```tsx
// components/common/CharGrid.tsx

interface CharGridProps {
    chars: number[][];   // 21x79 character codes
    fg: string[][];      // 21x79 hex foreground colors
    bg: string[][];      // 21x79 hex background colors
}

export const CharGrid = React.memo(function CharGrid({ chars, fg, bg }: CharGridProps) {
    const html = useMemo(() => {
        const rows: string[] = [];
        for (let r = 0; r < chars.length; r++) {
            const spans: string[] = [];
            for (let c = 0; c < chars[r].length; c++) {
                const ch = String.fromCharCode(chars[r][c]);
                const fgColor = fg[r][c];
                const bgColor = bg[r][c];
                const escaped = ch === "<" ? "&lt;" : ch === ">" ? "&gt;" : ch === "&" ? "&amp;" : ch;
                spans.push(`<span style="color:${fgColor};background:${bgColor}">${escaped}</span>`);
            }
            rows.push(spans.join(""));
        }
        return rows.join("<br/>");
    }, [chars, fg, bg]);

    return (
        <div
            style={{
                fontFamily: "'DejaVu Sans Mono', monospace",
                fontSize: "9px",
                lineHeight: 1.15,
                background: "#000",
                padding: "4px",
                whiteSpace: "pre",
            }}
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
});
```

## Implementation Phases

### Phase 1: Skeleton + Screen (MVP)

**Goal**: React app that displays the game screen synced to a step slider. Proves the
full pipeline: FastAPI -> REST -> TanStack Query -> React render.

**Backend**:
- `api_server.py`: FastAPI with GET /api/runs, /api/runs/{run}/games,
  /api/runs/{run}/step/{n}, /api/runs/{run}/step-range
- Serve React dev build via Vite proxy

**Frontend**:
- Vite + React + TypeScript project scaffold
- MantineProvider with compactTheme
- AppShell with header (run selector, game selector, step slider)
- CharGrid component rendering game screen
- TanStack Query for step data fetching
- useReducer playback state (historical mode only)

**Tests**:
- API: test endpoints return correct data shapes
- CharGrid: renders correct number of spans
- Playback reducer: state transitions
- Step data query: cache behavior

**Deliverable**: Browse to dashboard, select run/game, scrub through steps, see game screen
update.

### Phase 2: All Data Panels

**Goal**: Add all remaining data panels to match the Panel dashboard feature set.

- StatusBar (HP, Score, Depth with Mantine Progress for HP/Energy)
- FeatureTable (KV table)
- SaliencyMap (CharGrid with heatmap colors + legend)
- FocusPoints (KV table)
- AttenuationDetails (KV table)
- ObjectResolution (decision + candidates table)
- GameMetrics (KV table)
- GraphSummary (KV table)
- EventActivity (Recharts horizontal bar chart)
- LogMessages (filterable Mantine Table with severity coloring)
- Accordion sections with summary headers when collapsed

**Tests**:
- Each panel component renders with mock data
- Each panel handles null/missing data gracefully
- Log level filter works

### Phase 3: Live Mode

**Goal**: Socket.io push updates, LIVE badge, follow/pause/catchup behavior.

**Backend**:
- Add python-socketio to api_server.py
- Wire StepBuffer listener to emit Socket.io events
- Emit "new_step" with step number and metadata (full data fetched on demand)

**Frontend**:
- Socket.io client connection in useLiveUpdates hook
- Full playback state machine (all 4 states)
- LIVE badge component
- "New data available" badge when paused
- Auto-advance when following
- Step range expansion on push

**Tests**:
- Playback reducer: all state transitions including PUSH_ARRIVED
- Live badge visibility per state
- Socket reconnection behavior

### Phase 4: Keyboard Shortcuts + Bookmarks

**Goal**: Keyboard-driven navigation and persistent bookmarks.

- react-hotkeys-hook for all shortcuts (R16)
- Keyboard help overlay (Mantine Modal, toggled by ? or h)
- Bookmark CRUD (toggle, navigate next/prev, annotate)
- Bookmark persistence via API (GET/POST /api/runs/{run}/bookmarks)
- Bookmark list panel
- Visual bookmark indicators on slider

**Tests**:
- Shortcut key -> correct action dispatched
- Bookmark toggle adds/removes
- Bookmark navigation (next/prev wrapping)
- Bookmark persistence round-trip

### Phase 5: Polish + Panel Removal

**Goal**: Production-ready, remove all Panel code.

- Error boundaries and loading states
- Responsive layout (test on iPad width)
- Performance optimization (prefetch adjacent steps, React.memo)
- ServHerd integration for api_server.py
- Entry point: `uv run dashboard` starts FastAPI serving React build
- Delete: panel_debug.py, playback_machine.py, dashboard_server.py, components/
- Delete: tests/unit/test_panel_debug.py, tests/unit/test_playback_machine.py
- Remove Python deps: panel, bokeh, python-statemachine
- Update CLAUDE.md panel dashboard section

**Tests**:
- Integration test: start server, fetch step, verify response
- No remaining Panel imports in codebase
