# Dashboard UI

## Why This Design

The dashboard is a debugging and analysis tool for understanding ROC runs, especially graph data construction. It replaced a Panel/Bokeh predecessor that had a fundamental flaw: two control paths (button clicks via Bokeh websocket and server push updates from Python) raced because Bokeh's protocol could not distinguish user actions from server updates.

The fix is a strict separation: client owns all UI state (playback, current step, navigation), server owns all data (REST + Socket.io push). The server never touches UI state. This eliminates the race condition class entirely.

The design is information-dense and compact -- a debugging tool, not a consumer product. Every panel is custom-tailored for human understanding of its specific pipeline stage. NetHack symbols, words, and actions must appear in their human-facing form (glyphs, not numeric IDs) to enable rapid comprehension.

There is no pre-planned final design. The dashboard expands indefinitely as new processing stages and data types are added. This makes pattern reuse and component extraction critical -- without standardized patterns, each new panel reinvents layout and styling, creating recurring manual fix-up work.

## Key Decisions

- **TanStack Query with staleTime: Infinity for step data** -- step data is immutable for a given run/step/game. Infinite stale time means cached steps never refetch, giving <15ms navigation for visited steps.
- **Prefetch window: radius=100, batch=50, debounce=50ms** -- empirically tuned. Fills 200-step cache in ~2 seconds (4 HTTP requests). AbortController cancels in-flight batches on navigation to prevent stale request buildup.
- **StepBuffer for live data (bypasses DuckDB)** -- DuckDB uses file-level locking. The game subprocess holds a write lock on the DuckLake catalog, adding 300-600ms to API reads. StepBuffer (100K ring buffer) serves live steps directly, keeping latency at ~10ms.
- **Mantine Drawer for popout panels (not Modal or draggable)** -- zero new dependencies, non-blocking (main accordion stays interactive), shares DashboardContext so click-to-navigate works unchanged. If drag-and-drop is needed later, only the container wrapper changes.
- **HTTP callback IPC (not DuckLake polling)** -- DuckDB file-level locking prevents cross-process reads while the game writes. Game subprocess POSTs StepData to the server's /api/internal/step endpoint instead.

## Invariants

- **Live/historical feature parity.** Every new panel or feature must work in both live-following mode (Socket.io push) and historical mode (REST fetch from DuckLake). Validate both with Playwright MCP after any UI change. Write tests covering both modes. Failure to test both has repeatedly caused multi-iteration debugging sessions to achieve feature parity after the fact.

- **No inline styles.** Never use `style={{}}` for sizing, spacing, or colors. Use Mantine component props or app-level theme overrides. When the compact-mantine theme does not provide the right default, fix it at the theme level or create a reusable wrapper component. Inline styles bypass the theme system, creating per-component inconsistencies that compound across 40+ panels.

- **Stable rendering.** Panels must not cause layout shifts during step navigation. Use fixed-height containers (e.g., GameScreen minHeight 260px). Show "--" for null/missing values -- never omit rows. Use `layout="fixed"` on tables. Use `fontVariantNumeric: "tabular-nums"` for numeric columns. Use `useRatchetHeight()` for accordion content that grows. Violating these causes the entire page to reflow on every step change.

- **URL parameter sovereignty.** Never override the user's explicit URL navigation (`?run=X&game=Y&step=Z`) with auto-detected live run state. Auto-select only when no explicit `run` param is present. Violating this silently teleports the user away from what they are examining.

- **One UI server, one API server.** Never start additional servers. `roc-ui` (Vite) and `roc-server` (FastAPI) are managed by servherd. Use `make run` / `make stop`. Adding servers creates port conflicts and URL confusion. See design/dashboard-server-redesign.md.

## Playback State Machine

Four states govern live/historical mode transitions. Getting transitions wrong causes the dashboard to stop updating or to fight user navigation.

```
historical --GO_LIVE--> live_following
live_following --USER_NAVIGATE/PAUSE--> live_paused
live_paused --RESUME--> live_catchup
live_catchup --PUSH_ARRIVED(atEdge)--> live_following
```

- `live_following`: renders Socket.io push data directly (no REST round-trip)
- `live_paused`: user navigated away from live edge; REST fetches like historical
- `live_catchup`: playing forward toward live edge; transitions to following only when caught up
- `historical`: no live game; all data from REST

The `PUSH_ARRIVED` transition checks `atEdge` -- whether the displayed step equals the live edge. Only transition to `live_following` when truly caught up, otherwise the UI skips steps.

## Component Extraction Rules

Extract a reusable component when a pattern appears in 2 or more panels. Even at 2 uses, extraction establishes a discoverable design pattern for future panels. The existing primitives are the model:

- **Section**: accordion item with icon, color, toolbar, ErrorBoundary -- every panel uses it
- **KVTable**: compact key-value table (150px max-width, zero cell padding, tabular-nums)
- **ClickableChart**: wrapper adding click-to-navigate-step on Recharts charts
- **PopoutPanel**: right-side Drawer wrapper for aggregate/detail views
- **ErrorBoundary**: panel-level crash isolation (every Section includes one)

When adding a new panel, check existing primitives first. If your layout resembles KVTable, use KVTable. If you are building a chart, wrap it in ClickableChart. If no primitive fits and you see the same pattern elsewhere, extract a new one.

## Layout Rules

The dashboard optimizes for information density. Follow these to avoid recurring layout pain:

- **Tables**: narrow enough that data is near its label, no horizontal scrolling. Use `layout="fixed"` and constrained column widths.
- **Spacing**: tight between components. Use Mantine's `xs` (4px) or `sm` (8px) spacing between dashboard elements. Do not use `md` or larger.
- **Grid breakpoints**: `base: 12` (mobile/narrow), `md: 8/4` or `md: 6/6` (desktop splits). All panels must be responsive.
- **Cards**: minimal padding (6px). Do not use Mantine's default card padding.
- **Alignment**: components within a section must be vertically and horizontally aligned. No ragged layouts.
- **Overflow**: content must not overflow containers. Use text truncation, never horizontal scroll.

## Validation Process

After any UI change -- new panel, modified layout, data flow change:

1. Capture screenshots in both historical and live modes using Playwright MCP
2. Use Nanobanana MCP to verify rendering -- ask objective yes/no questions, not leading ones
3. Verify that data flows correctly into components (not just that they render)
4. Check compact layout rules: no overflow, proper alignment, no excessive whitespace

## Stale Closure Pattern

Socket.io callbacks capture state at registration time. All mutable state accessed in Socket.io handlers (run, game, step, callbacks) must go through refs (`useRef`) updated in effects, not direct state variables. Accessing stale state causes the dashboard to silently process events with outdated context -- e.g., pushing live data to the wrong run.

## Performance Targets

| Scenario | Target |
|----------|--------|
| Step from cache | <15ms |
| Cold fetch (no game) | 430-500ms |
| Cold fetch (game running) | 800-1100ms |
| Full prefetch (200 steps) | ~2 seconds |
| Perceived step transition | <200ms |

History queries (metrics, events, intrinsics) use staleTime of 5 minutes -- they grow during live play but do not need real-time freshness.
