# Dashboard UI

## Why This Design

The dashboard is a debugging and analysis tool for understanding ROC runs, especially graph data construction. It replaced a Panel/Bokeh predecessor that had a fundamental flaw: two control paths (button clicks via Bokeh websocket and server push updates from Python) raced because Bokeh's protocol could not distinguish user actions from server updates.

The fix is a strict separation: client owns all UI state (playback, current step, navigation), server owns all data (REST + Socket.io push). The server never touches UI state. This eliminates the race condition class entirely.

The design is information-dense and compact -- a debugging tool, not a consumer product. Every panel is custom-tailored for human understanding of its specific pipeline stage. NetHack symbols, words, and actions must appear in their human-facing form (glyphs, not numeric IDs) to enable rapid comprehension.

There is no pre-planned final design. The dashboard expands indefinitely as new processing stages and data types are added. This makes pattern reuse and component extraction critical -- without standardized patterns, each new panel reinvents layout and styling, creating recurring manual fix-up work.

## Key Decisions

- **TanStack Query with staleTime: Infinity for step data** -- step data is immutable for a given run/step/game. Infinite stale time means cached steps never refetch, giving <15ms navigation for visited steps.
- **Prefetch window: radius=100, batch=50, debounce=50ms** -- empirically tuned. Fills 200-step cache in ~2 seconds (4 HTTP requests). AbortController cancels in-flight batches on navigation to prevent stale request buildup.
- **TanStack Query is the only client store for server data** -- Components never hold parallel state for step data, step ranges, run lists, or live status. Socket.io is invalidation-only -- it tells the query cache when to refetch but never writes data directly into React state.
- **Mantine Drawer for popout panels (not Modal or draggable)** -- zero new dependencies, non-blocking (main accordion stays interactive), shares DashboardContext so click-to-navigate works unchanged. If drag-and-drop is needed later, only the container wrapper changes.
- **In-process game thread (not subprocess)** -- the game runs as a daemon thread inside the server process. StepData flows through `push_step_from_game()` -> `RunWriter` with no serialization or IPC. All caches (StepCache, GraphCache) are shared via the process heap.
- **Playing state is NOT persisted to sessionStorage** -- speed, autoFollow, and scroll position are persisted for iOS page-discard recovery, but `playing` is always `false` on cold load. Auto-play on a cold page causes blank-screen flickering because the TanStack Query cache is empty and each step advance shows a blank screen while data fetches (1-3s per step). The user presses play when ready.

## Invariants

- **No inline styles.** Never use `style={{}}` for sizing, spacing, or colors. Use Mantine component props or app-level theme overrides. When the compact-mantine theme does not provide the right default, fix it at the theme level or create a reusable wrapper component. Inline styles bypass the theme system, creating per-component inconsistencies that compound across 40+ panels.

- **Stable rendering.** Panels must not cause layout shifts during step navigation. Use fixed-height containers (e.g., GameScreen minHeight 260px). Show "--" for null/missing values -- never omit rows. Use `layout="fixed"` on tables. Use `fontVariantNumeric: "tabular-nums"` for numeric columns. Use `useRatchetHeight()` for accordion content that grows. Violating these causes the entire page to reflow on every step change.

- **URL parameter sovereignty.** Never override the user's explicit URL navigation (`?run=X&game=Y&step=Z`) with auto-detected live run state. Auto-select only when no explicit `run` param is present. Violating this silently teleports the user away from what they are examining.

- **One UI server, one API server.** Never start additional servers. `roc-ui` (Vite) and `roc-server` (FastAPI) are managed by servherd. Use `make run` / `make stop`. Adding servers creates port conflicts and URL confusion. See design/dashboard-server-redesign.md.

- **Single source of truth for each server concept.** Every category of server-owned data has exactly one hook or query that owns it, and every consumer reads through that one door. This is mechanically enforced by `src/architecture.test.ts`. The current single sources are:
  - **Game state** (state/run_name/exit_code/error): `useGameState()` in `hooks/useRunSubscription.ts`. REST via one-shot `/api/game/status` fetch on mount, Socket.io `game_state_changed` for updates. No other component may fetch `/api/game/status` or hold a parallel copy.
  - **Step range** (min/max/tail_growing): `useStepRange(run, game)` in `api/queries.ts`. TransportBar is the single owner that syncs the query to context. App.tsx's auto-follow effect only advances `step`; it never writes `stepRange`.
  - **Step data** (one game step): `useStepData(run, step, game)` in `api/queries.ts`. Immutable, `staleTime: Infinity`.
  - **Run list**: `useRuns(includeAll)` in `api/queries.ts`. Polled every 10s.
  - **Games per run**: `useGames(run)` in `api/queries.ts`. Polled every 10s.
  - **Socket.io client**: `getSocket()` singleton inside `useRunSubscription.ts`. No other file calls `io({...})`.
  Adding a new piece of server data? Add one hook, one queryKey, one owner. Two components duplicating the fetch is how TC-GAME-004 happened: the step range was mirrored in context and a TransportBar effect, guarded differently in each place, and clicking GO LIVE read a stale closure. The Phase-3 live-vs-historical split had the same flavor and was fixed by the unified-run architecture (see `design/unified-run-architecture.md`).

- **Socket.io is invalidation-only.** Server never pushes data through it; it only says "the N-th step exists now" so TanStack Query knows to refetch. A second `io({...})` call in any file is a smell -- almost always a component trying to receive data directly, the Phase-3 anti-pattern. The architecture test fails on this.

## Playback Model

Two booleans drive playback: `playing` (timer is ticking) and `autoFollow` (snap to head as
it grows). The four old states map to combinations:

| Old | `playing` | `autoFollow` |
|---|---|---|
| historical | t/f | false |
| live_following | n/a | true |
| live_paused | false | false |
| live_catchup | true | true |

Transitions are direct mutations of the two booleans -- no reducer, no state machine. Any
explicit user navigation sets `autoFollow=false`. "GO LIVE" sets `autoFollow=true` and snaps
to `range.max`. The auto-follow effect watches `range.max` and pulls the user along while
`autoFollow && range.tail_growing`.

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

## Performance Targets

| Scenario | Target |
|----------|--------|
| Step from cache | <15ms |
| Cold fetch (no game) | 430-500ms |
| Cold fetch (game running) | 800-1100ms |
| Full prefetch (200 steps) | ~2 seconds |
| Perceived step transition | <200ms |

History queries (metrics, events, intrinsics) use staleTime of 5 minutes -- they grow during live play but do not need real-time freshness.

## Error Handling

Errors must be surfaced, not swallowed. Every `.catch()` block must either:
1. Show the error to the user (inline text, menu label, or console.error for remote logger visibility)
2. Log to `console.error`/`console.warn` so the remote logger captures it

Never write `.catch(() => {})` -- this is the anti-pattern that made the dashboard "unreliable" by hiding every failure as "no data". Specific rules:

- **fetchJson** reads the response body on error and includes server detail in the thrown Error message.
- **Game start/stop** (MenuBar) shows the error in the Game dropdown menu.
- **Socket.io** has `connect`/`disconnect`/`connect_error` listeners. The connection dot in the TransportBar reflects actual socket state, not hardcoded `true`.
- **Bookmark save** awaits the POST and logs on failure.
- **Step-range fetches** log to console.error with context about which operation failed.

## Connectivity Validation

External dependencies must be health-checked on init:

- **Remote logger**: `useRemoteLogger` fetches `/status` on client init. If unreachable, logs `console.warn` so the failure is visible even without remote logging.
- **Socket.io**: `useSocketConnected()` exposes reactive connection state via `useSyncExternalStore`. On reconnect, all step-range and run-list queries are invalidated.
