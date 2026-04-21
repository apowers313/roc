# Code Review Report - 2026-04-18

## Executive Summary

- Files reviewed: ~90 (19 Python backend, ~70 TypeScript frontend, configs)
- Critical issues: 3
- High priority issues: 4
- Medium priority issues: 5
- Low priority issues: 3

**Root cause assessment**: The recurring dashboard instability (hangs, freezes,
glitches) traces primarily to three interacting problems:

1. Socket.io subscriptions are silently lost on reconnect, causing the
   dashboard to stop receiving updates with no visual feedback.
2. Production console logging with remote forwarding creates 120+ HTTP
   requests/sec at high playback speeds, blocking the UI thread.
3. Unbounded memory growth in both frontend (Cytoscape accumulation) and
   backend (GraphService cache) degrades performance over long sessions.

These are not independent: the logging overhead makes reconnection stalls
worse, and memory pressure makes the logging overhead worse. Fixing all three
is needed for stable behavior.

---

## File Inventory

### Production Code -- Frontend (dashboard-ui/src/)

| Category | Files | Key files |
|----------|-------|-----------|
| Entry/Layout | 2 | App.tsx, main.tsx |
| State | 2 | state/context.tsx, state/highlight.tsx |
| API layer | 2 | api/client.ts, api/queries.ts |
| Hooks | 8 | useRunSubscription, usePrefetchWindow, useCacheInvalidation, useBookmarks, useDebouncedValue, useKeyboardShortcuts, useRemoteLogger, useRatchetHeight |
| Panels | ~40 | GraphVisualization, GameScreen, SaliencyMap, ResolutionInspector, etc. |
| Common | 5 | Section, KVTable, ClickableChart, PopoutPanel, ErrorBoundary |
| Transport | 4 | TransportBar, StatusBar, MenuBar, BookmarkBar |
| Types | 3 | types/api.ts, types/step-data.ts, types/cytoscape.d.ts |

### Production Code -- Backend (roc/reporting/)

| Category | Files | Key files |
|----------|-------|-----------|
| Data storage | 3 | ducklake_store.py, parquet_exporter.py, step_cache.py |
| Run lifecycle | 3 | run_registry.py, run_writer.py, run_reader.py |
| Query layer | 1 | run_store.py (1015 lines) |
| API server | 2 | api_server.py (1084 lines), graph_api.py |
| Graph | 1 | graph_service.py |
| Observability | 3 | observability.py, metrics.py, resource_metrics.py |
| Other | 4 | state.py, types.py, screen_renderer.py, remote_logger_exporter.py |

### Test Code

| Category | Files | Lines |
|----------|-------|-------|
| Backend unit tests | 18 | ~9,000 |
| Backend integration | 5 | ~1,000 |
| Frontend unit tests | 55+ | ~11,000 |
| Frontend E2E (Playwright) | 5 | ~2,150 |
| UAT manual plan | 1 | 824 |

### Configuration

- `dashboard-ui/vite.config.ts`, `vitest.config.ts`, `tsconfig.json`
- `pyproject.toml` (pytest config, ruff, mypy)
- `servherd.config.json`

---

## Critical Issues (Fix Immediately)

### C1. Socket.io subscription lost on reconnect -- dashboard freezes

- **Files**: `dashboard-ui/src/hooks/useRunSubscription.ts:65-86`, `dashboard-ui/src/App.tsx:329`
- **Description**: When the Socket.io transport drops and reconnects (which
  gives the client a new SID), the `subscribe_run` event is never re-emitted.
  The server-side subscription is keyed by SID -- the old SID's subscription
  is cleaned up on disconnect, and the new SID has no subscription.
  `step_added` events stop flowing. The dashboard appears completely frozen
  with no error indicator. This is compounded by `App.tsx:329` where
  `connected` is hardcoded to `true`, so the green "Connected" dot in the
  transport bar never changes.

- **Why it's critical**: This is the most likely root cause of the "dashboard
  hangs" reports. Any transient network hiccup (WiFi switch, server restart,
  sleep/wake on iPad) triggers this. Once triggered, the only recovery is a
  full page reload.

- **Example**: `useRunSubscription.ts:68-86`
```typescript
useEffect(() => {
    if (!run) return;
    const socket = getSocket();
    socket.emit("subscribe_run", run);
    // ...
    socket.on("step_added", onStepAdded);
    return () => {
        socket.off("step_added", onStepAdded);
        socket.emit("unsubscribe_run", run);
    };
}, [run, cache]);
// Effect only re-runs when `run` or `cache` changes.
// Reconnection (new SID) does not trigger re-run.
```

- **Fix**:
```typescript
useEffect(() => {
    if (!run) return;
    const socket = getSocket();

    const subscribe = () => {
        socket.emit("subscribe_run", run);
    };

    subscribe();
    // Re-subscribe after reconnection (server assigns new SID)
    socket.on("connect", subscribe);

    const onStepAdded = (payload: StepAddedPayload) => {
        if (!payload || payload.run !== run) return;
        cache.invalidateStepRange(run);
    };
    socket.on("step_added", onStepAdded);

    return () => {
        socket.off("connect", subscribe);
        socket.off("step_added", onStepAdded);
        socket.emit("unsubscribe_run", run);
    };
}, [run, cache]);
```

Also fix `App.tsx:329` -- expose actual socket connection state:
```typescript
// Replace: const connected = true;
// With a hook that tracks socket.connected and connect/disconnect events
const [connected, setConnected] = useState(false);
useEffect(() => {
    const socket = getSocket();
    setConnected(socket.connected);
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);
    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    return () => {
        socket.off("connect", onConnect);
        socket.off("disconnect", onDisconnect);
    };
}, []);
```

---

### C2. Console logging in production creates 120+ HTTP requests/sec

- **Files**: `dashboard-ui/src/App.tsx:525-531, 591-594`, `dashboard-ui/src/hooks/useRemoteLogger.ts:69-75`
- **Description**: Two `console.log` calls fire on every render cycle --
  one in the auto-nav effect (line 526) and one after `useStepData` (line
  591). Each call triggers `JSON.stringify` on a data object, then
  `useRemoteLogger` intercepts it and POSTs the serialized message to the
  remote log server. At 10x playback speed (~60 steps/sec), this produces
  120 `JSON.stringify` calls + 120 HTTP POSTs per second on the UI thread.

- **Why it's critical**: `JSON.stringify` is synchronous and blocks the main
  thread. Combined with the HTTP POST (which queues microtasks), this creates
  frame drops, stutter, and the appearance of the dashboard "hanging" during
  fast playback. The remote logger is valuable for iPad debugging but
  should not process high-frequency per-step telemetry.

- **Example**: `App.tsx:591-594`
```typescript
// This runs on EVERY render where step/game changes:
console.log("[App] useStepData", JSON.stringify({
    step, game, stepIsLoading,
    hasData: restData != null, hasScreen: restData?.screen != null,
}));
```

- **Fix**: Gate debug logging behind dev mode, or use a rate limiter:
```typescript
if (import.meta.env.DEV) {
    console.log("[App] useStepData", JSON.stringify({
        step, game, stepIsLoading,
        hasData: restData != null, hasScreen: restData?.screen != null,
    }));
}
```
Alternatively, modify `useRemoteLogger` to sample high-frequency messages:
```typescript
// In useRemoteLogger.ts, add rate limiting:
let lastSend = 0;
const MIN_INTERVAL_MS = 200;
console[method] = (...args: unknown[]) => {
    orig(...args);
    const now = Date.now();
    if (now - lastSend < MIN_INTERVAL_MS) return;
    lastSend = now;
    // ... send to remote logger
};
```

---

### C3. Error objects serialize to `{}` in remote logger

- **Files**: `dashboard-ui/src/hooks/useRemoteLogger.ts:71-73`
- **Description**: When `console.error("Panel render error:", error)` fires
  (e.g., from ErrorBoundary), `useRemoteLogger` serializes each argument via
  `JSON.stringify(a)`. Error objects have non-enumerable properties (`message`,
  `stack`), so `JSON.stringify(new Error("boom"))` produces `"{}"`. The remote
  log server receives `"Panel render error: {}"` -- losing the actual error
  message and stack trace. This makes iPad crash debugging nearly impossible.

- **Why it's critical**: The remote logger is the only debugging tool for
  iPad/mobile sessions. If it can't capture error details, crashes remain
  invisible.

- **Example**: `useRemoteLogger.ts:71-73`
```typescript
const message = args
    .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
    .join(" ");
```

- **Fix**:
```typescript
function serialize(a: unknown): string {
    if (typeof a === "string") return a;
    if (a instanceof Error) {
        return `${a.name}: ${a.message}${a.stack ? "\n" + a.stack : ""}`;
    }
    try {
        return JSON.stringify(a);
    } catch {
        return String(a);
    }
}
const message = args.map(serialize).join(" ");
```

---

## High Priority Issues (Fix Soon)

### H1. GraphService cache grows without bound

- **File**: `roc/reporting/graph_service.py:60`
- **Description**: `GraphService._cache` stores the parsed NetworkX DiGraph
  for every run ever viewed. Each graph can be several MB. Over a long
  dashboard session exploring many historical runs, this accumulates without
  any eviction policy. The description in `resource_metrics.py:127` even
  acknowledges this: `"grows without bound"`.

- **Impact**: Memory exhaustion on long-running servers. DuckLake stores are
  swept after 5 minutes idle, but graph caches are not.

- **Example**: `graph_service.py:60-75`
```python
def __init__(self, data_dir: Path) -> None:
    self._data_dir = data_dir
    self._cache: dict[str, nx.DiGraph] = {}  # Never evicted

def get_graph(self, run_name: str) -> nx.DiGraph:
    if run_name in self._cache:
        return self._cache[run_name]
    # ... load and cache forever
```

- **Fix**: Use an LRU eviction strategy (same pattern as `StepCache`):
```python
from collections import OrderedDict

class GraphService:
    MAX_CACHED_GRAPHS = 10

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: OrderedDict[str, nx.DiGraph] = OrderedDict()

    def get_graph(self, run_name: str) -> nx.DiGraph:
        if run_name in self._cache:
            self._cache.move_to_end(run_name)
            return self._cache[run_name]
        graph = self._load_graph(run_name)
        self._cache[run_name] = graph
        while len(self._cache) > self.MAX_CACHED_GRAPHS:
            self._cache.popitem(last=False)
        return graph
```

---

### H2. GraphService cache has no thread safety

- **File**: `roc/reporting/graph_service.py:60, 74-75`
- **Description**: The `_cache` dict is accessed from FastAPI's async
  request handlers without any lock. Two concurrent requests for the
  same run can race on cache population -- both loading and parsing the
  same `graph.json` simultaneously. While CPython's GIL prevents dict
  corruption, the duplicate parsing wastes time and memory.

- **Impact**: Under concurrent load (multiple browser tabs, rapid
  navigation), duplicate graph parsing causes latency spikes and
  transient memory spikes.

- **Fix**: Add a `threading.Lock` (consistent with the pattern used by
  `StepCache`, `DuckLakeStore`, and `RunRegistry`):
```python
import threading

class GraphService:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: OrderedDict[str, nx.DiGraph] = OrderedDict()
        self._lock = threading.Lock()

    def get_graph(self, run_name: str) -> nx.DiGraph:
        with self._lock:
            if run_name in self._cache:
                self._cache.move_to_end(run_name)
                return self._cache[run_name]
        graph = self._load_graph(run_name)  # Expensive, outside lock
        with self._lock:
            self._cache[run_name] = graph
            # ... eviction
        return graph
```

---

### H3. DuckLakeStore.execute() returns cursor outside lock

- **File**: `roc/reporting/ducklake_store.py:133-143`
- **Description**: The `execute()` method acquires `_lock`, calls
  `_conn.execute()`, then returns the cursor to the caller -- but the
  lock is released when `execute()` returns. The caller can then iterate
  the cursor while another thread acquires the lock and runs a different
  query on the same connection, corrupting the cursor's internal state.
  The docstring warns about this, but the API design is a footgun.

- **Impact**: Any code path that calls `execute()` and iterates the result
  outside the `with` block risks corrupted reads or segfaults. Currently
  most callers use `query_df()` or `query_one()`, which are safe, but the
  unsafe method exists and could be called.

- **Example**: `ducklake_store.py:133-143`
```python
def execute(self, sql: str, params=None) -> duckdb.DuckDBPyConnection:
    """.. warning:: The returned cursor is only valid while no other
       thread uses this connection."""
    with self._lock:
        if params:
            return self._conn.execute(sql, params)
        return self._conn.execute(sql)
    # Lock released -- cursor is now unsafe!
```

- **Fix**: Audit all callers. If none need the cursor (they all just want
  DataFrames or single rows), deprecate `execute()` and make it private.
  If a caller does need raw cursor access, provide a context manager:
```python
@contextmanager
def execute_locked(self, sql, params=None):
    """Execute a query and yield the cursor with the lock held."""
    with self._lock:
        if params:
            yield self._conn.execute(sql, params)
        else:
            yield self._conn.execute(sql)
```

---

### H4. Cytoscape accumulated data grows without bound

- **Files**: `dashboard-ui/src/components/panels/GraphVisualization.tsx:2258-2267`
- **Description**: Every node expansion fetches neighbor data and merges it
  into `accumulatedData` via `mergeGraphData()`. Data is never pruned.
  Over a session with many expand/collapse cycles on large graphs, the
  accumulated state can grow to 10+ MB. Cytoscape.js also maintains its
  own internal copy of every element.

- **Impact**: Memory pressure on iPad/mobile causes browser tab kills.
  On desktop, GC pauses cause visible stutter during graph interaction.

- **Example**: `GraphVisualization.tsx:2258-2267`
```typescript
setAccumulatedData((prev) => {
    const merged = prev ? mergeGraphData(prev, neighbors) : neighbors;
    // merged grows monotonically -- collapsed nodes are hidden
    // but their data is never removed from the accumulated set
    return merged;
});
```

- **Fix**: Reset accumulated data on run/game/root change (partially done
  at line 2026 for root changes), and add a max-elements cap:
```typescript
const MAX_ACCUMULATED_NODES = 500;
setAccumulatedData((prev) => {
    const merged = prev ? mergeGraphData(prev, neighbors) : neighbors;
    if (merged.elements.nodes.length > MAX_ACCUMULATED_NODES) {
        // Keep only nodes reachable from currently expanded set
        return pruneToReachable(merged, expandedNodesRef.current);
    }
    return merged;
});
```

---

## Medium Priority Issues (Technical Debt)

### M1. run_coroutine_threadsafe silent failure

- **File**: `roc/reporting/api_server.py:850-856`
- **Description**: The `_on_step` callback bridges game-thread notifications
  to the asyncio event loop via `run_coroutine_threadsafe`. If the loop
  is stopped (server shutting down, uncaught exception in the async loop),
  the `except Exception: pass` silently swallows the failure. Socket.io
  notifications stop flowing with no log message.

- **Impact**: During server restarts or edge-case async loop failures,
  the dashboard silently stops updating. Combined with C1 (no visual
  disconnect indicator), users see a frozen dashboard with no clue why.

- **Example**: `api_server.py:850-856`
```python
try:
    asyncio.run_coroutine_threadsafe(
        sio.emit("step_added", {"run": run, "step": step}, to=sid),
        loop,
    )
except Exception:
    pass  # Silent -- no logging
```

- **Fix**: Log the failure so it appears in the remote logger:
```python
except Exception:
    logger.debug("Socket.io emit failed for sid {} step {}", sid, step)
```

---

### M2. Subscriber callbacks block the game thread

- **File**: `roc/reporting/run_registry.py:360-369`
- **Description**: `notify_subscribers` dispatches outside the registry
  lock (correct for deadlock prevention), but the callbacks run
  synchronously on the game thread. If a callback (e.g., the Socket.io
  bridge in `_on_step`) is slow, it blocks `push_step()` and stalls the
  game loop.

- **Impact**: Under network congestion, `run_coroutine_threadsafe` may
  block briefly. Multiple subscribers compound the delay. This manifests
  as the game loop stuttering -- steps arrive in bursts instead of
  smoothly.

- **Example**: `run_registry.py:365-369`
```python
for cb in subs:
    try:
        cb(step)  # Runs on game thread
    except Exception as exc:
        logger.warning("subscriber error for run {}: {}", name, exc)
```

- **Fix**: Add a timeout guard, or dispatch callbacks to a thread pool:
```python
import concurrent.futures
_notify_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                      thread_name_prefix="notify")

def notify_subscribers(self, name: str, step: int) -> None:
    with self._lock:
        entry = self._entries.get(name)
        subs = list(entry.subscribers) if entry else []
    for cb in subs:
        _notify_pool.submit(cb, step)
```

---

### M3. useBookmarks fetch not cancellable

- **File**: `dashboard-ui/src/hooks/useBookmarks.ts:23-34`
- **Description**: The bookmark load effect uses a `cancelled` flag to
  prevent stale state updates, but the underlying fetch request continues
  running. On rapid run switches, orphaned requests accumulate.

- **Impact**: Minor bandwidth waste and closure retention over long
  sessions. Not a crash risk, but adds to the general "sluggishness"
  complaint.

- **Fix**: Use AbortController:
```typescript
useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    fetchBookmarks(run, controller.signal)
        .then((data) => setBookmarks(data))
        .catch(() => {
            if (!controller.signal.aborted) setBookmarks([]);
        });
    return () => controller.abort();
}, [run]);
```

---

### M4. ParquetExporter 5s shutdown timeout

- **File**: `roc/reporting/parquet_exporter.py:146-149`
- **Description**: `shutdown()` joins the background writer thread with a
  5-second timeout. If `_drain()` or `checkpoint()` takes longer (e.g.,
  large accumulated queue, slow disk), the thread is orphaned and queued
  records are lost.

- **Impact**: Data loss at the tail of a run. The last few steps may not
  be written to the archive. This is particularly bad for runs that crash
  -- the crash-adjacent data is the most valuable and the most likely to
  be lost.

- **Fix**: Increase timeout to 30s (this is a clean shutdown path, not a
  latency-sensitive operation), and log when the timeout is exceeded:
```python
def shutdown(self) -> None:
    self._shutdown_event.set()
    self._data_ready.set()
    if self._thread is not None:
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            logger.warning("ParquetExporter thread did not exit within 30s")
    self._drain()
    if self._store:
        self._store.checkpoint()
```

---

### M5. ParquetExporter silently swallows all export errors

- **File**: `roc/reporting/parquet_exporter.py:106-120`
- **Description**: The `export()` method catches all exceptions and returns
  `FAILURE` without logging. This is intentional (the game loop must never
  block), but it makes diagnosing data loss impossible. Missing data in
  the dashboard shows up as "no data for step N" with no trail to follow.

- **Impact**: Silent data corruption or loss that manifests as dashboard
  glitches ("data represented incorrectly") with no root-cause trail.

- **Fix**: Log at debug level so the remote logger captures it:
```python
except Exception:
    logger.opt(exception=True).debug("ParquetExporter.export() failed")
    return LogRecordExportResult.FAILURE
```

---

## Low Priority Issues (Nice to Have)

### L1. Step not clamped when max <= 0

- **File**: `dashboard-ui/src/state/context.tsx:198-205`
- **Description**: `setStepRange` skips clamping when `max <= 0`. If a
  range transitions from valid to empty (e.g., corrupt catalog), the step
  can remain pointing at a non-existent step.
- **Fix**: Always clamp, defaulting to `min` when `max <= 0`.

### L2. saveBookmarks errors silently swallowed

- **File**: `dashboard-ui/src/hooks/useBookmarks.ts:39`
- **Description**: `void saveBookmarks(run, updated)` discards the
  Promise. If the save fails, bookmarks appear saved in the UI but are
  lost on reload.
- **Fix**: Add `.catch()` with a user-visible notification.

### L3. Remote logger exporter silently drops on network failure

- **File**: `roc/reporting/remote_logger_exporter.py:110-111`
- **Description**: HTTP POST failures are caught and discarded. During
  network outages, log records are silently lost with no retry.
- **Impact**: Acceptable tradeoff (game loop must not block), but means
  the remote logger is unreliable for crash investigation during network
  instability.

---

## Positive Findings

These patterns are well-designed and should be replicated:

- **StepCache as mandatory hot cache**: The LRU cache absorbs ~95% of reads,
  preventing DuckDB lock contention. The "transparent optimization, not a
  separate API" principle is clean.

- **RunRegistry single-owner pattern**: One entry per run, one store per
  entry, `tail_growing` as the only live signal. This is a textbook example
  of eliminating a category of bugs (live-vs-historical branching) by
  architectural constraint.

- **Socket.io invalidation-only pattern**: Server pushes `{run, step}`,
  client invalidates TanStack Query. Clean separation of concerns. The
  `architecture.test.ts` that enforces "one Socket.io client, one source
  per data type" is excellent.

- **Write-through cache + background writer**: Game thread never blocks on
  DB writes. The ParquetExporter queue + background thread + silent error
  handling keeps the experiment running regardless of storage failures.

- **`useCacheInvalidation` centralized invalidation**: All query key
  management in one file prevents the "stale cache" class of bugs.

- **Two-boolean playback model**: `playing` x `autoFollow` replaces a
  4-state machine. Simpler to reason about and fewer edge cases.

---

## Recommendations

### Immediate (this week)

1. **Fix C1 (Socket.io reconnection)** -- This is the most impactful single
   fix. Add `connect` event listener to re-subscribe, and expose real
   connection state to the UI. Estimated effort: 1-2 hours.

2. **Fix C2 (console logging)** -- Gate `App.tsx` debug logs behind
   `import.meta.env.DEV`. This immediately eliminates the high-playback
   stutter. Estimated effort: 15 minutes.

3. **Fix C3 (Error serialization)** -- Add Error-aware serialization to
   `useRemoteLogger`. This enables diagnosis of all other bugs via iPad.
   Estimated effort: 30 minutes.

### Short-term (next 1-2 weeks)

4. **Fix H1+H2 (GraphService cache)** -- Add LRU eviction and lock. Follow
   the `StepCache` pattern exactly. Estimated effort: 1 hour.

5. **Fix H3 (DuckLakeStore.execute)** -- Audit callers, deprecate or make
   private. Estimated effort: 1 hour.

6. **Fix H4 (Cytoscape memory)** -- Add element cap and prune logic.
   Estimated effort: 2-3 hours.

### Medium-term (next month)

7. **Add Socket.io reconnection E2E test** -- Simulate disconnect/reconnect
   in Playwright and verify step updates resume. This is the biggest gap
   in the current test suite.

8. **Add concurrent access tests** for backend data stores -- The current
   tests are all single-threaded. Thread contention is a documented risk
   that has no test coverage.

9. **Instrument the remote logger with sampling** -- Instead of removing
   logging, add intelligent sampling so high-frequency messages (step
   changes) are rate-limited while low-frequency messages (errors, state
   transitions) always flow through.
