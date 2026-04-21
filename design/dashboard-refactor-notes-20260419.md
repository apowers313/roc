# Dashboard Refactor Research Notes -- 2026-04-19

## Current Dashboard Stack
React + Mantine UI + TanStack Query + Socket.io + Recharts + Vite

## Recurring Bug Categories

### 1. Race Conditions
- Socket.io connection state not tracked (hardcoded `connected = true`)
- Stale closures in timer callbacks (playback speed changes)
- Concurrent access to shared state (GameManager without locking)
- TOCTOU bugs in connection lifecycle

### 2. Rendering Errors
- Stale data displayed after reconnection (no cache invalidation on reconnect)
- Auto-play on cold load causing blank screens (playing state persisted to sessionStorage)
- Connection indicator never updating (no real connection tracking)

### 3. Slow Data Loading
- Prefetch window centered on range midpoint instead of current step
- No request cancellation (AbortSignal not threaded through)
- Error details swallowed (`.catch(() => {})` everywhere)
- No retry logic on transient failures

## Root Cause Analysis

### Boolean Soup
~8 independent `useState` hooks + module-level singletons create ~256 possible
state combinations, most of which are impossible states that code must guard
against at every use site. The guards are incomplete, leading to race conditions.

Affected files: `playback.ts`, `context.tsx`, `useRunSubscription.ts`, `TransportBar.tsx`

### Fire-and-Forget Async
Multiple `.then()` chains without AbortController or cancellation tokens.
Rapid navigation causes out-of-order resolution.

### Timer-Based Polling
`setTimeout` loop in TransportBar polls `stepDataReadyRef` every 5ms.
Creates stale closure risks and timer races.

## Recommended Fixes (Component-Level)

### 1. XState for Connection + Playback State
Replace boolean soup with a formal state machine:
```
connection:
  disconnected -> connecting -> connected -> disconnected
playback (nested under connected):
  idle -> playing -> paused
liveness (parallel region):
  historical | live_following | live_not_following
```

Benefits:
- Impossible states become unrepresentable
- Socket.io connection becomes an invoked actor (created/disposed with state)
- Reconnection + cache invalidation = single transition
- Visual debugging via stately.ai/viz

Similar projects: Kaltura media player, Foxglove Studio

Files affected: `playback.ts` (replaced by machine), `context.tsx` (wraps useMachine),
`useRunSubscription.ts` (Socket.io singleton becomes invoked actor)

### 2. useReducer for Coordinated State Transitions
Replace 4-5 setter calls on run selection with atomic dispatch:
```typescript
case 'SELECT_RUN': return { ...state, run, step: 1, game: 1, stepRangeReady: false, autoFollow: false };
case 'STEP_RANGE_UPDATED': return { ...state, stepMin, stepMax, stepRangeReady: true };
case 'GO_LIVE': return { ...state, step: state.stepMax, autoFollow: true };
```

### 3. Data-Driven Step Advancement
Replace setTimeout polling with useEffect on query settlement:
```typescript
useEffect(() => {
  if (playing && nextStepData && !nextStepIsLoading) {
    setStep(step + 1);
  }
}, [playing, nextStepData, nextStepIsLoading]);
```

### 4. Asymmetric Prefetch During Playback
Bias forward (+150/-50 instead of +/-100) during playback.
Make step N+1 fetch non-cancellable.

## Comparable Open-Source Projects

### Foxglove Studio (closest match)
- Robotics data viewer with timeline scrubbing, live streaming, panel-based layout
- React + TypeScript
- IterablePlayer handles live and recorded data through same code path
- MessagePipeline fans out data to panels
- Player state machine for playback
- github.com/foxglove/studio (MPL-2.0)

### Rerun
- Multimodal data viewer
- ECS architecture with time-aware components
- Apache Arrow for data handling
- Multiple time domains with synchronized scrubbing
- Rust backend + WASM viewer
- github.com/rerun-io/rerun

### Replay.io
- Time-travel debugging with timeline
- Step-through capability
- React DevTools integration
- github.com/replayio/devtools

### OpenReplay
- Session replay with state mutation tracking
- React frontend + Go/Python backend
- github.com/openreplay/openreplay

### Perfetto UI
- Chrome's trace viewer
- WASM + WebWorkers for heavy queries
- Columnar in-memory database
- On-demand SQL queries for visible window
- github.com/google/perfetto

### Speedscope
- Flamegraph viewer
- GPU-accelerated rendering + LRU cache
- Custom pan/zoom
- github.com/jlfwong/speedscope

## Key Insight from Research

The Socket.io + TanStack Query invalidation pattern is correct (confirmed by TkDodo).
The transport layer doesn't need changing. The bugs live in state management and
lifecycle coordination, not in data fetching or real-time transport.

## What Was NOT Recommended

- Replacing Socket.io with SSE, Ably, Liveblocks, or PartyKit (overkill, wrong problem)
- Timeline scrubbing libraries (react-timeline-editor, etc.) -- the scrubber itself works fine
- Grafana or Jupyter as replacements (90% of usage is scrubbing, which neither handles well)

---

## Framework Landscape (Comprehensive Research -- 2026-04-19)

### A. Full Application Frameworks (could replace the whole dashboard)

| Framework | Domain | Key Features | License | Fit |
|-----------|--------|-------------|---------|-----|
| **Flora** (Foxglove fork) | Robotics | react-mosaic panels + transport + extensions + data playback | MPL-2.0 | HIGH but ROS-coupled |
| **Foxglove Studio** v1.87 | Robotics | Same as Flora (frozen open-source) | MPL-2.0 | HIGH but frozen |
| **OHIF Viewer** | Medical imaging | Extension/mode system + cine player + Cornerstone prefetch/cache | MIT | HIGH but DICOM-coupled |
| **Rerun** | Multi-modal | Timeline + multi-panel + ECS data model | Apache/MIT | MEDIUM - monolithic WASM |
| **Remotion** | Video creation | Frame-accurate React rendering + Player + transport | Custom | MEDIUM - different purpose |
| **Theatre.js** | Animation | Timeline + scrubbing + inspector + studio UI | Apache/AGPL | CREATIVE - repurposable? |
| **Playwright Trace Viewer** | Testing | Step-through + film strip + inspector panels | Apache-2.0 | HIGH conceptually |
| **Grafana Scenes** | Observability | Data mgmt + panels + variables + time ranges | Apache-2.0 | LOW - needs Grafana runtime |
| **Refine** | CRUD/admin | Data fetching + caching + real-time + Mantine support | MIT | MEDIUM - CRUD-focused |

### B. Panel Layout Frameworks

| Framework | Stars | Features | License |
|-----------|-------|----------|---------|
| **Dockview** | 3.1k | Tabs, docking, floating, popout, serialize, zero deps | MIT |
| **react-mosaic** | 4.7k | N-ary tree layout, used by Foxglove | Apache-2.0 |
| **FlexLayout** | 1.3k | Tabs + splitters, recommended over Golden Layout | MIT |
| **react-grid-layout** | 22.2k | Drag/drop grid, responsive breakpoints | MIT |
| **Golden Layout** | 6.6k | Multi-window IDE-style (declining momentum) | MIT |
| **rc-dock** | - | Dock layout with tab API | Apache-2.0 |
| **Allotment** | - | VS Code-style split views | MIT |
| **react-resizable-panels** | - | Split views (by React core team member) | MIT |
| **Gridstack.js** | - | Mobile-friendly widget grid | MIT |

### C. Transport / Timeline / Scrubbing

| Framework | Domain | Key Feature | License |
|-----------|--------|------------|---------|
| **react-player-controls** | Media | Pure UI play/pause/slider, media-agnostic | ISC |
| **react-timeline-editor** | Video/NLE | Multi-track timeline + cursor + effects | MIT |
| **animation-timeline-control** | Animation | Canvas timeline, zero deps | MIT |
| **Tone.js Transport** | DAW | Master transport with schedule/seek/events | MIT |
| **Waveform Playlist** | DAW | React hooks: usePlaylistControls, usePlaybackAnimation | MIT |
| **Cornerstone.js Cine** | Medical | Frame-rate play, direction, loop, prefetch-100-ahead | MIT |
| **GSAP GSDevTools** | Animation | Visual timeline debugger with scrubber | Commercial |
| **react-scrubber** | Media | Touch-friendly scrubber slider | MIT |

### D. State Management / Time-Travel

| Framework | Key Feature | License |
|-----------|------------|---------|
| **XState** | Formal state machines + actors + visualizer (29k stars) | MIT |
| **use-travel / Travels** | React hook with back/forward/go(n), JSON patches | MIT |
| **zundo** | Zustand undo/redo middleware (<700B) | MIT |
| **zustand-travel** | Zustand time-travel with JSON patches (Mutative-based) | MIT |
| **MobX-State-Tree** | Snapshots + time-travel + middleware | MIT |
| **Reactime** | Chrome ext: playback + speed + jump + diff | MIT |

### E. Data / Streaming

| Framework | Key Feature | License |
|-----------|------------|---------|
| **TanStack Query** (current) | REST caching + stale-while-revalidate | MIT |
| **FINOS Perspective** | WASM streaming analytics + pivot + charts | Apache-2.0 |
| **Cube.js** | Semantic API + pre-aggregation + WebSocket transport | MIT |
| **AG Grid** | 150k+ updates/sec streaming | MIT/Commercial |
| **Glide Data Grid** | 100M rows at 60fps, canvas-based | MIT |
| **uPlot** | 50KB, 166k pts in 25ms, 10% CPU at 60fps | MIT |

### F. Game Replay / Move Viewers

| Framework | Domain | License |
|-----------|--------|---------|
| **OpenDota/Rapier** | Dota 2 React/Redux replay dashboard | MIT |
| **Chessground** (Lichess) | Chess board + move stepping (10KB) | GPL-3.0 |
| **pgn-viewer** | Chess step forward/back/autoplay | GPL-3.0 |
| **WGo.js** | Go/Baduk SGF step-through | Unspecified |

### G. Emulator Debuggers

| Framework | Key Feature | License |
|-----------|------------|---------|
| **8bitworkshop** | Step/continue/breakpoint + multi-panel | GPL-3.0 |
| **WasmBoy** | Preact debugger shell wrapping WASM core | Apache-2.0 |
| **NesJs** | Single-step + multi-panel inspection | MIT |

### H. ML/AI Pipeline Trace Viewers

| Framework | Key Feature | License |
|-----------|------------|---------|
| **Langfuse** | Step-by-step pipeline trace, OTel-based | MIT |
| **Phoenix (Arize)** | AI pipeline tracing, OTel/OpenInference | Apache/Elastic |
| **Aim** | React experiment tracker, 100k+ sequences | Apache-2.0 |
| **MLflow** | React frontend, run comparison, metrics | Apache-2.0 |

---

## Evaluation of Top Candidates

### Tier 1: Adopt an existing application framework

**Flora (Foxglove fork)** -- closest architectural match
- Has: react-mosaic panels, transport controls, chunked caching with message lookback
  on seek, extension SDK for custom React panels, live + historical through same path
- Catch: deeply coupled to ROS bags/MCAP/Protobuf message schemas. Would need to
  write a custom data source adapter or fork and replace the data layer.
- Risk: MPL-2.0 license, unmaintained community fork of frozen codebase

**OHIF Viewer** -- similar architecture from medical imaging
- Has: extension/mode system (plugins define panels), Cornerstone cine player with
  prefetch-100-ahead and LRU cache, React + MIT license
- Catch: deeply coupled to DICOM metadata and medical imaging workflows

**Remotion Player** -- unexpected option
- Has: frame-accurate React rendering, Player with seek/scrub/play/pause, React-native
- Each ROC step = a React "frame" that renders all dashboard panels
- Catch: designed for video creation/export. Paid Timeline component.

### Tier 2: Compose from best-in-class pieces

Replace ad-hoc layout + state management with purpose-built libraries:
- **Dockview** (layout) -- real docking panels with tabs, floating, serializable. MIT.
- **XState** (state) -- formal state machine for connection/playback/liveness. MIT, 29k stars.
- **TanStack Query + Socket.io** (data) -- keep current stack. Pattern confirmed correct.
- **uPlot** (charts, optional) -- if Recharts perf becomes issue. 50KB, 10x less CPU.

Addresses actual failure modes. Incremental migration possible (panel by panel).

### Tier 3: Creative domain-crossing

- **Theatre.js Studio** -- repurpose animation timeline for pipeline data stepping
- **react-player-controls + use-travel** -- media-agnostic transport + state history
- **Cornerstone cine pattern** -- adopt the prefetch-100-ahead + LRU cache architecture
  without the DICOM viewer (just the caching/transport logic)

---

## Key Finding

No single framework bundles panel layout + transport + data fetching + caching + real-time.
The closest are Flora/Foxglove and OHIF, but both are deeply domain-coupled.

The most practical path is composing purpose-built pieces (Tier 2), borrowing specific
architectural patterns from Foxglove (chunked caching, message lookback on seek) and
Cornerstone (asymmetric directional prefetch, LRU eviction).
