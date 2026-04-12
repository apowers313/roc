# Unified Run Architecture

- **Status:** Proposed
- **Date:** 2026-04-08
- **Supersedes (in part):** `dashboard-server-redesign.md`, `live-step-performance-fix.md`, `playback-state-machine.md`
- **Related:** `dashboard-step-performance.md`

## Problem Statement

The dashboard has two parallel data paths -- "live" (StepBuffer + Socket.io push +
in-memory indices + frontend `liveData` state) and "historical" (DuckLake reads +
TanStack Query) -- with merge logic at every layer that tries to reconcile them.
This duality is the source of a recurring class of UI bugs:

- Steps silently disappear from the dropdown (filtered out without explanation)
- Step data fetches return 500 but the UI shows "no data" indistinguishably from
  a real missing step (`pd.NA` serialization, encountered 2026-04-08)
- "Corrupt run" markers appear and disappear (DuckLake alias collision under
  populate-thread race, also 2026-04-08)
- The four-state playback machine
  (`historical`/`live_following`/`live_paused`/`live_catchup`) gets stuck or
  transitions incorrectly under push pressure
- Stale-closure bugs in `App.tsx` Socket.io callbacks require a pile of refs
  (`runRef`, `gameRef`, `stepRef`, `stepMinRef`, `stepMaxRef`, `isFollowingRef`,
  `liveRunNameRef`) to paper over

A 24-finding architectural review on 2026-04-08 traced all of these to the same
root: **the system has multiple writers into the same logical state, with errors
uniformly treated as "missing data" instead of "backend failure"**. Most
findings (silent error swallowing, stale closures, race conditions, mismatched
contracts) reduce to "live and historical are kept as two parallel
representations that need to be merged."

## How We Got Here

The current architecture grew incrementally. Each step was reasonable in
isolation; together they multiply complexity:

1. **DuckLake-only** -- Originally everything went through DuckLake. Each
   `INSERT` produced a Parquet file, leading to ~7000 tiny files per run and
   5-11s queries.
2. **Data inlining** added (`data_inlining_row_limit=500`) -- query times drop
   to ~8ms per step. The original perf problem is solved.
3. **StepBuffer** added -- DuckLake's file-level lock blocks readers for
   300-600ms while a writer is active. The dashboard wanted sub-100ms step
   navigation during live play, so an in-memory ring buffer was added to bypass
   DuckLake entirely for live data.
4. **`DataStore._indices`** added -- to avoid re-querying DuckLake for chart
   history (graph_history, metrics_history, etc.) during live play, indices are
   accumulated in memory from each Socket.io push.
5. **`_live_run_name` / `_live_buffer` / `_live_store`** added -- DataStore now
   needs to know which run is "live" to route reads to the right tier.
6. **`is_live()` branches** added at every API endpoint that needs to choose
   between cache and DuckLake.
7. **`_run_summary_cache` / `_run_stores`** added as separate caches inside
   DataStore, with their own race conditions.
8. **Frontend `liveData` state** added -- the dashboard accepts Socket.io
   pushes containing full StepData. App.tsx merges this with REST data via
   `data = isFollowing ? liveData ?? restData : restData`.
9. **Ref pile** added (`runRef`, `gameRef`, etc.) -- the Socket.io callback
   needs to read mutable state without triggering re-renders, so refs are
   manually synced on every render.
10. **Four-state playback machine** added
    (`historical`/`live_following`/`live_paused`/`live_catchup`) -- to manage
    the different ways the user can move between live and historical viewing
    modes.
11. **`liveRunName`/`liveGameNumber`/`liveGameActive`** added to
    `DashboardContext` -- shared state for the playback machine.
12. **"URL parameter sovereignty" rule** added to `dashboard-ui/CLAUDE.md` --
    because auto-detected live state was overriding explicit user navigation.
13. **"Stale Closure Pattern" rule** added to `dashboard-ui/CLAUDE.md` --
    because Socket.io callbacks were reading stale state without refs.
14. **"Live/historical feature parity" rule** added to
    `dashboard-ui/CLAUDE.md` -- because every new panel had to be tested in
    both modes, since they have different code paths.

The accumulated cost: ~1000 lines of dual-path code, three new "rules" in
`CLAUDE.md` to prevent its known failure modes, and a recurring bug class that
the architectural review boxed in.

## Key Insight

**Live and historical are not different domains. They are the same domain with
one binary distinction: is the tail still growing?**

A run is a sequence of steps. Whether more steps may arrive in the next second
is a property of the run, not a different kind of run. The current dual code
path exists because of one historical implementation constraint -- DuckLake
file locking under contention -- which is no longer the same problem it was
when StepBuffer was introduced. The original problem (5-11s queries) was fixed
by data inlining. The remaining contention (300-600ms reads while writing) is
addressed by making the cache transparent: every write goes through `StepCache`
write-through, and reads of active runs are served from the cache or, on miss,
through the writer's existing `DuckLakeStore` instance under its existing
`_lock`. Phase 0 confirmed empirically that opening a *separate* read-only
`DuckLakeStore` against the same catalog file in the same process is not
possible (DuckDB rejects it as a "Unique file handle conflict"); see
`tmp/ducklake_concurrency_spike.py` for the measurement.

The unified mental model:

> A run is a tape. Some tapes are still being written to (a game subprocess is
> recording). Most tapes are sitting in a drawer. Reading from a tape is the
> same operation regardless of whether it's still being recorded.

Two states for any run:

```
tail_growing = True   # game subprocess is currently writing into this run
tail_growing = False  # tape is in the drawer, no writer attached
```

That's the entire live/historical distinction. It is one boolean, not a
different code path.

## Goals

1. **One read path.** A single `RunReader` facade serves all reads. No
   `is_live()` branching anywhere above the cache layer.
2. **One write path.** A single `RunWriter` is the only thing that mutates
   server-side state for a run.
3. **One client store.** TanStack Query is authoritative on the frontend.
   Socket.io is invalidation-only -- it never delivers data into a parallel
   state container.
4. **`tail_growing` is the only signal at the API boundary.** No
   `liveRunName`, no `is_live`, no separate live status endpoint that the
   frontend has to reconcile with step-range queries.
5. **Collapse the four-state playback machine to a single boolean.**
   `autoFollow: bool` replaces the entire state enum.
6. **Delete ~1000 lines of dual-path code** along with three architectural
   rules whose only purpose is to police that code.

## Non-Goals

- **Replacing DuckLake.** It is still the persistence layer. The catalog and
  Parquet files stay exactly as they are.
- **Replacing TanStack Query.** It is still the client cache. We are removing
  the *parallel* client state (`liveData`, refs, context fields).
- **Cross-run merging.** Each run is independent. Joining data across runs is
  out of scope.
- **Backward compatibility with old API contracts.** This is a solo
  experimental project with no external clients. Breaking changes are fine.
- **Multi-process write coordination.** Only one writer per run at a time, as
  today (the game subprocess).

## Architecture

### Current

```
                      ┌─────────────┐
   Game subprocess ───┤StepBuffer   │──Socket.io──> liveData state ──┐
                      │(in-memory)  │                                ├─> isFollowing? merge
                      └─────────────┘                                │
                            │                                        │
                            └──flush──> DuckLake catalog ──REST──> restData ─────┘
                                              │                          │
                                              │                          ▼
                                              │                  React component
                                              │
                                              └──> _indices (per-game accumulator)
                                                     │
                                                     └──REST──> useGraphHistory etc.

   _is_live() branches in every endpoint, every DataStore method
   _live_run_name, _live_buffer, _live_store, _run_stores, _run_summary_cache
   liveRunName, liveGameNumber, liveGameActive in DashboardContext
   runRef, gameRef, stepRef, stepMinRef, stepMaxRef, isFollowingRef, liveRunNameRef
   playbackReducer with 4 states
```

### Proposed

```
                          ┌────────────────────────┐
   Game subprocess ─POST─>│  StepCache (private)   │──flush──> DuckLake
   (only if recording)    │  LRU, all runs         │           catalog.duckdb
                          └────────────────────────┘                  │
                                     ▲                                ▼
                                     │ get(run, step)                 │
                                     │ ┌──────────────┐               │
                                     └─┤  RunReader   │◄──────────────┘
                                       │  (unified)   │  (read-only)
                                       └──────────────┘
                                              ▲
                                              │
                                       ┌──────────────┐
                                       │  REST /api   │
                                       └──────────────┘
                                              ▲
                                              │      ┌──────────────────┐
                                              │      │ Socket.io        │
                                              │      │ {run, max_step}  │
                                              │      │ (invalidation)   │
                                              │      └────────┬─────────┘
                                              │               │
                                              │               ▼
                                              │      queryClient.invalidate
                                              │               │
                                              └───TanStack Query cache (frontend)
                                                              │
                                                              ▼
                                                      React component
                                                  (single useStepData hook)
```

One read path. One cache. One write path (active only when a game is
running). Notifications are an optional signal layered on top.

## Detailed Design

### Backend

#### `RunReader`

The single public read facade. Lives in `roc/reporting/run_reader.py`. Replaces
`DataStore`'s read methods. Internally chooses between `StepCache` (hot) and
`DuckLakeStore` (cold).

```python
class RunReader:
    """Single read facade for any run.

    Callers do not see the live/historical distinction. The reader
    routes internally to a hot in-memory cache (recently-written
    steps) or to DuckLake (everything else). Both paths return the
    same shape of data.

    StepCache is MANDATORY, not an optimization. Phase 0 proved that a
    separate read_only=True DuckLakeStore cannot be opened in the same
    process while the writer is active (BinderException: Unique file
    handle conflict). For active runs, the only DuckLake-side fallback
    is the writer's own DuckLakeStore instance, accessed via
    _RunEntry.store under its existing _lock. The cache absorbs the
    common case so reads almost never need to take that lock.
    """

    def __init__(self, registry: RunRegistry, cache: StepCache) -> None:
        self._registry = registry
        self._cache = cache

    def get_step(self, run: str, step: int) -> StepResponse:
        """Single step lookup. Returns a typed envelope -- never a fake placeholder."""
        entry = self._registry.get(run)
        if entry is None:
            return StepResponse(status="run_not_found", data=None)
        if step < 1 or step > entry.range.max:
            return StepResponse(
                status="out_of_range",
                data=None,
                range=entry.range,
            )
        cached = self._cache.get(run, step)
        if cached is not None:
            return StepResponse(status="ok", data=cached, range=entry.range)
        # entry.store is the writer's DuckLakeStore for active runs and a
        # lazily-opened read_only=True DuckLakeStore for closed runs;
        # RunRegistry.get() guarantees it is non-None on return. The store's
        # internal _lock serializes this read against any concurrent writes.
        try:
            data = entry.store.get_step_data(step)
        except Exception as exc:
            logger.warning("step read failed run={} step={}: {}", run, step, exc)
            return StepResponse(
                status="error",
                data=None,
                error=f"{type(exc).__name__}: {exc}",
                range=entry.range,
            )
        if data is None:
            return StepResponse(status="not_emitted", data=None, range=entry.range)
        self._cache.put(run, step, data)
        return StepResponse(status="ok", data=data, range=entry.range)

    def get_step_range(self, run: str) -> StepRange:
        """Returns {min, max, tail_growing}. The single source of truth for navigation."""
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        return entry.range  # already includes tail_growing

    def get_history(
        self, run: str, kind: HistoryKind, game: int | None = None
    ) -> list[dict]:
        """All history queries (graph_history, event_history, etc.) hit DuckLake.

        The data inlining row limit means typical queries complete in <100ms.
        We rely on the TanStack Query cache (5-minute staleTime) on the client
        to avoid re-fetching the same series during normal interaction.
        """
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        return entry.store.get_history(kind, game)

    def list_runs(self, *, include_all: bool = False) -> list[RunSummary]:
        """Run list with status field. Already implemented in this conversation."""
        return self._registry.list(include_all=include_all)

    def subscribe(self, run: str, callback: Callable[[int], None]) -> Unsubscribe:
        """Notification stream for new steps.

        For runs with tail_growing=True, fires whenever a new step is written.
        For runs with tail_growing=False, returns a no-op unsubscribe immediately.
        Callers (Socket.io handler) bind a callback and forget about it.
        """
        return self._registry.subscribe(run, callback)
```

#### `RunWriter`

The single write entry point. Lives in `roc/reporting/run_writer.py`. Used by
the HTTP callback handler (`/api/internal/step`) and the in-process game loop.

```python
class RunWriter:
    """Active for the lifetime of one game run.

    Created when a game starts; closed when it stops. Writes are
    append-only and go to both the hot cache (instant) and the
    background ParquetExporter (eventual persistence to DuckLake).

    The writer owns the SINGLE DuckLakeStore for its run. Phase 0 proved
    that DuckDB will not let RunReader open a separate read_only=True
    instance against the same catalog file in the same process, so the
    writer's store is the read store as well. RunRegistry.attach_writer_store
    installs it as _RunEntry.store; RunRegistry.detach_writer_store drops
    it on close so the next read can lazily reopen read_only=True.
    """

    def __init__(
        self,
        run_name: str,
        registry: RunRegistry,
        cache: StepCache,
        exporter: ParquetExporter,
        store: DuckLakeStore,  # the single read+write store for this run
    ) -> None:
        self._run = run_name
        self._registry = registry
        self._cache = cache
        self._exporter = exporter
        self._store = store
        # Install our store as the read store and flip tail_growing on.
        self._registry.attach_writer_store(run_name, store)

    def push_step(self, data: StepData) -> None:
        """Single write entry point. Synchronous, microsecond-fast."""
        self._cache.put(self._run, data.step, data)             # hot cache
        self._exporter.queue(data)                              # async DuckLake
        self._registry.update_max_step(self._run, data.step)    # range update
        self._registry.notify_subscribers(self._run, data.step) # frontend invalidation

    def close(self) -> None:
        """End of game. Tape goes back in the drawer."""
        self._exporter.flush()
        # Drop our store from the registry so the next read reopens
        # read_only=True against the now-quiescent catalog file.
        self._registry.detach_writer_store(self._run)
```

#### `RunRegistry`

Replaces the three caches in `DataStore` (`_run_stores`, `_run_summary_cache`,
`_indices`) with a single store-of-runs. Lives in `roc/reporting/run_registry.py`.

```python
@dataclass
class _RunEntry:
    name: str
    summary: RunSummary       # with the status field added in this conversation
    store: RunStore           # the ONE DuckLakeStore for this run
    range: StepRange          # {min, max, tail_growing}
    mtime: float              # filesystem mtime; invalidates the cache
    subscribers: list[Callable[[int], None]] = field(default_factory=list)
    # Provenance of `store` -- enforced by Phase 0's measurement
    # (see Risk #2 / tmp/ducklake_concurrency_spike.py):
    #   tail_growing == True  -> store is the SAME DuckLakeStore instance
    #                            held by the active RunWriter. Do NOT
    #                            construct a separate read_only=True
    #                            instance for the same catalog file --
    #                            DuckDB will reject the second attach
    #                            with "Unique file handle conflict".
    #   tail_growing == False -> store is a fresh read_only=True
    #                            DuckLakeStore opened lazily by
    #                            RunRegistry._load(). On the next
    #                            mark_growing(name, growing=True),
    #                            this read-only store is replaced with
    #                            the writer's instance.


class RunRegistry:
    """Single owner of all per-run state. One lock for everything.

    Owns the single DuckLakeStore per run. The active-run case shares
    the writer's instance; the closed-run case opens a fresh
    read_only=True instance. See _RunEntry.store comment for why.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._entries: dict[str, _RunEntry] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> _RunEntry | None:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                entry = self._load(name)
                if entry is not None:
                    self._entries[name] = entry
            elif self._stale(entry):
                entry = self._reload(entry)
            return entry

    def attach_writer_store(self, name: str, store: DuckLakeStore) -> None:
        """Called by RunWriter.__init__ to install the writer's store
        as the entry's read store. Replaces any existing read_only
        store for the run. The writer's _lock serializes all access
        for the lifetime of the writer.
        """
        with self._lock:
            entry = self._entries.get(name) or self._load(name)
            if entry is None:
                entry = self._build_entry(name, store)
            else:
                entry.store = store
            entry.range = entry.range.with_tail_growing(True)
            self._entries[name] = entry

    def detach_writer_store(self, name: str) -> None:
        """Called by RunWriter.close(). Drops the writer-owned entry
        and replaces it with a fresh entry whose store is a brand-new
        read_only=True DuckLakeStore. Phase 0 confirmed this open
        succeeds against a quiescent catalog. The replacement happens
        atomically under the lock so any get() in flight either sees
        the writer's store or the new read-only store -- never a
        half-built entry.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            # Build a fresh closed-run entry. _load() opens
            # read_only=True; this is the validated closed-run path.
            self._entries[name] = self._load(name)

    def list(self, *, include_all: bool = False) -> list[RunSummary]:
        # Same semantics as DataStore.list_runs() today, with status field.
        ...

    def mark_growing(self, name: str, *, growing: bool) -> None:
        with self._lock:
            entry = self._entries.get(name) or self._load(name)
            if entry:
                entry.range = entry.range.with_tail_growing(growing)
                self._entries[name] = entry

    def update_max_step(self, name: str, step: int) -> None:
        with self._lock:
            entry = self._entries.get(name)
            if entry:
                entry.range = entry.range.with_max(step)

    def notify_subscribers(self, name: str, step: int) -> None:
        # Snapshot under lock, dispatch outside lock to avoid deadlock
        with self._lock:
            entry = self._entries.get(name)
            subs = list(entry.subscribers) if entry else []
        for cb in subs:
            try:
                cb(step)
            except Exception as exc:
                logger.warning("subscriber error for run {}: {}", name, exc)

    def subscribe(self, name: str, cb: Callable[[int], None]) -> Unsubscribe:
        with self._lock:
            entry = self._entries.get(name) or self._load(name)
            if entry is None:
                return lambda: None
            entry.subscribers.append(cb)
        return lambda: self._unsubscribe(name, cb)
```

#### `StepCache`

Generalizes `StepBuffer` to a process-wide LRU keyed by `(run, step)`. Lives
in `roc/reporting/step_cache.py`. Not exposed in any public API -- private to
`RunReader`/`RunWriter`.

```python
class StepCache:
    """Process-wide LRU of recently-touched steps. Not 'live' -- just hot."""

    def __init__(self, capacity: int = 5000):
        self._cache: OrderedDict[tuple[str, int], StepData] = OrderedDict()
        self._lock = threading.Lock()
        self._capacity = capacity

    def get(self, run: str, step: int) -> StepData | None:
        with self._lock:
            key = (run, step)
            data = self._cache.get(key)
            if data is not None:
                self._cache.move_to_end(key)
            return data

    def put(self, run: str, step: int, data: StepData) -> None:
        with self._lock:
            self._cache[(run, step)] = data
            self._cache.move_to_end((run, step))
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def invalidate_run(self, run: str) -> None:
        with self._lock:
            for key in [k for k in self._cache if k[0] == run]:
                del self._cache[key]
```

The current `StepBuffer` (`roc/reporting/step_buffer.py`) becomes the
implementation of `StepCache`. The 100K capacity becomes a configurable LRU
size; 5000 is plenty because the cache only needs to absorb the gap between
write time and DuckLake flush.

#### API Endpoints

Every endpoint becomes a 1-2 line wrapper around `RunReader`. No
`is_live()`, no merge logic, no special cases.

```python
@app.get("/api/runs/{run}/step/{n}")
def get_step(run: str, n: int) -> StepResponse:
    return reader.get_step(run, n)

@app.get("/api/runs/{run}/step-range")
def get_step_range(run: str) -> StepRange:
    return reader.get_step_range(run)  # includes tail_growing

@app.get("/api/runs/{run}/graph-history")
def get_graph_history(run: str, game: int | None = None) -> list[dict]:
    return reader.get_history(run, "graph", game)

# ... other history endpoints follow the same pattern

@app.get("/api/runs")
def list_runs(include_all: bool = False) -> list[RunSummary]:
    return reader.list_runs(include_all=include_all)

@sio.event
async def subscribe_run(sid: str, run: str) -> None:
    """Frontend asks for notifications about a specific run."""
    def on_step(step: int) -> None:
        sio.emit("step_added", {"run": run, "step": step}, to=sid)
    unsub = reader.subscribe(run, on_step)
    _subscriptions[sid] = unsub  # cleanup on disconnect
```

The handler code does not contain the words "live" or "historical" anywhere.
Compare to today, where `data_store.py:list_runs`, `data_store.py:list_games`,
`data_store.py:get_step_range`, `data_store.py:get_step_data`, and every
history method has a live branch.

#### Game Subprocess Side

The HTTP callback (`/api/internal/step`) becomes one call:

```python
@app.post("/api/internal/step")
async def receive_step(step: StepData):
    writer = _writers.get(step.run_name)
    if writer is None:
        # First step of a new run -- create the writer.
        # The writer owns a single DuckLakeStore (read+write) that
        # the registry shares with RunReader. Phase 0 proved we
        # cannot open a second read_only=True instance against the
        # same catalog file in the same process; see Risk #2.
        run_dir = data_dir / step.run_name
        store = DuckLakeStore(run_dir, read_only=False)
        writer = RunWriter(step.run_name, registry, cache, exporter, store)
        _writers[step.run_name] = writer
    writer.push_step(step)
```

`GameManager` calls `writer.close()` when the subprocess exits. The
`tail_growing` flag flips off; subscribers stop receiving notifications;
the next read for that run still works (it goes to DuckLake or the cache).

### Frontend

#### Single client store: TanStack Query

Components only ever call query hooks. No `liveData` state, no parallel
caches, no merge ternaries.

```ts
// All step access goes through this. No alternatives.
function useStepData(run: string, step: number, game?: number) {
    return useQuery({
        queryKey: ["step", run, step, game],
        queryFn: () => fetchStep(run, step, game),
        enabled: run !== "" && step > 0,
        staleTime: Infinity,           // step data is immutable
        placeholderData: keepPreviousData,
    });
}

// Step range now includes tail_growing
function useStepRange(run: string, game?: number) {
    return useQuery({
        queryKey: ["step-range", run, game],
        queryFn: () => fetchStepRange(run, game),
        enabled: run !== "",
    });
}
```

#### Socket.io as invalidation only

The Socket.io listener does *not* update React state. It only invalidates
queries.

```ts
// hooks/useRunSubscription.ts
export function useRunSubscription(run: string) {
    const queryClient = useQueryClient();
    useEffect(() => {
        if (!run) return;
        socket.emit("subscribe_run", run);
        const onStepAdded = ({run: r, step}: {run: string, step: number}) => {
            if (r !== run) return;
            // Tell TanStack Query that the step range may have changed.
            // The query will refetch on next access (or immediately if active).
            queryClient.invalidateQueries({queryKey: ["step-range", run]});
            // Optionally, also seed the new step into the cache via setQueryData
            // if the server includes the data in the notification.
        };
        socket.on("step_added", onStepAdded);
        return () => {
            socket.off("step_added", onStepAdded);
            socket.emit("unsubscribe_run", run);
        };
    }, [run, queryClient]);
}
```

This eliminates the entire `onNewStep` callback in `App.tsx:202-236`, the ref
pile in `App.tsx:187-200`, the `liveData` state, and the
`isFollowing ? liveData ?? restData : restData` ternary.

#### Single autoFollow boolean

The four-state playback machine collapses to one flag:

```ts
// state/playback.ts (replacement)
export interface PlaybackState {
    playing: boolean;       // is the playback timer ticking
    autoFollow: boolean;    // when true, snap to range.max as it grows
}

// In App.tsx:
const range = useStepRange(run, game);
const prevMaxRef = useRef(range.data?.max ?? 0);

useEffect(() => {
    if (!autoFollow) return;
    if (!range.data?.tail_growing) return;
    if (range.data.max <= prevMaxRef.current) return;
    // The user was at the head; pull them along.
    if (step === prevMaxRef.current) {
        navigate({step: range.data.max});
    }
    prevMaxRef.current = range.data.max;
}, [range.data?.max, range.data?.tail_growing, autoFollow, step]);
```

The "GO LIVE" button becomes:

```ts
const goLive = () => {
    if (!range.data?.tail_growing) return;
    setAutoFollow(true);
    navigate({step: range.data.max});
};
```

The "user navigated, stop following" behavior is one line in the navigate
handler:

```ts
const navigate = (to: NavTarget) => {
    setAutoFollow(false);          // any explicit navigation drops follow
    setSearchParams(to);
};
```

The four states (`historical`, `live_following`, `live_paused`,
`live_catchup`) collapse to combinations of `(playing, autoFollow)`:

| Old state | `playing` | `autoFollow` |
|---|---|---|
| `historical` | true/false | false |
| `live_following` | n/a | true |
| `live_paused` | false | false |
| `live_catchup` | true | true (auto-snaps when reaching head) |

This is the entire playback model. There is no state machine because the
state space is 2 booleans = 4 combinations, and each combination has a
straightforward interpretation.

#### Navigation: URL is the input, queries are the output

The frontend already has URL-driven state, but it's mediated through
`DashboardContext`. With `DashboardContext` shrunk to UI-only state, the URL
becomes the direct input:

```ts
function useDashboardLocation() {
    const [params, setParams] = useSearchParams();
    return {
        run: params.get("run") ?? "",
        game: Number(params.get("game") ?? 1),
        step: Number(params.get("step") ?? 1),
        navigate: (to: Partial<{run: string; game: number; step: number}>) => {
            const next = new URLSearchParams(params);
            for (const [k, v] of Object.entries(to)) next.set(k, String(v));
            setParams(next, {replace: true});
        },
    };
}
```

`DashboardContext` retains only true UI state: playback flags, speed,
modals, accordion expansion. Everything that comes from the server lives in
TanStack Query.

## What Gets Deleted

### Backend

- `roc/reporting/data_store.py`:
  - `_live_run_name`, `_live_buffer`, `_live_store` (lines 137-139)
  - `_indices`, `_GameIndex` class (lines 97-110, 140)
  - `_run_summary_cache`, `_run_stores` (lines 142-144) -- replaced by `RunRegistry`
  - `set_live_session()`, `clear_live_session()` (lines 166-214)
  - `_is_live()`, `is_live()` (lines 274-279)
  - `_get_live_history()` (lines 281-290)
  - `_supplement_logs()` (lines 439-451)
  - `_on_step_pushed()`, `_index_step()`, `_append_if_present()`,
    `_index_event_summary()` (lines 225-269)
  - All `if self._is_live()` branches in `list_games`, `get_step_range`,
    `get_step_data`, `get_steps_batch`, etc.

- `roc/reporting/api_server.py`:
  - Live-vs-historical routing in every step/range/history handler
  - Branch in `/api/runs/{run}/games` for live runs

- `roc/reporting/step_buffer.py`:
  - File replaced by `roc/reporting/step_cache.py` (LRU, multi-run, shared)
  - `StepBuffer.steps_per_game()`, `step_range_for_game()` -- not needed,
    range tracking moves to `RunRegistry`

- `roc/reporting/graph_api.py`:
  - Live cache vs historical archive routing in graph endpoints

Estimated deletion: **400-500 lines of backend code**, plus the rules in
`reporting/CLAUDE.md` that police it.

### Frontend

- `dashboard-ui/src/App.tsx`:
  - `liveData` state (line 175)
  - `liveRunSelected` ref (line 178)
  - `initialUrlRun` ref (line 182)
  - `runRef`, `gameRef`, `stepRef`, `stepMinRef`, `stepMaxRef`,
    `isFollowingRef`, `liveRunNameRef` (lines 187-200)
  - `onNewStep` callback (lines 202-236)
  - `data = isFollowing ? liveData ?? restData : restData` (line 457)
  - `goLive()` and the GO LIVE badge wiring (lines 322-328) -- replaced by
    a one-line `setAutoFollow(true)` handler
  - The `liveStatus` plumbing through `useLiveUpdates`

- `dashboard-ui/src/state/context.tsx`:
  - `liveRunName`, `liveGameNumber`, `liveGameActive` and their setters
    (lines 92-100, 116-118, 184-189)
  - `stepMin`, `stepMax`, `setStepRange` (lines 83-85, 112-113, 131-148) --
    becomes a query result, not state
  - `setStepRange` and the clamp logic (lines 131-148)
  - URL sync effect (lines 153-159) -- replaced by direct `useSearchParams`

- `dashboard-ui/src/state/playback.ts`:
  - The four-state machine (`historical`, `live_following`, `live_paused`,
    `live_catchup`) and its reducer
  - All transition tests for these states

- `dashboard-ui/src/components/transport/TransportBar.tsx`:
  - `isViewingLiveGame`, `effectiveMin`/`effectiveMax` ternary (lines 67-78)
  - The `setStepRange` sync effect (lines 74-78)

- `dashboard-ui/src/hooks/useLiveUpdates.ts`:
  - `pollLiveStatusRef` initialization race
  - `liveStatus` state -- replaced by `useStepRange().data.tail_growing`

Estimated deletion: **400-500 lines of frontend code**, plus the
"Stale Closure Pattern" and "Live/historical feature parity" rules in
`dashboard-ui/CLAUDE.md`.

## CLAUDE.md Updates

### `/CLAUDE.md` (root)

**Architectural Invariants section** -- Add a new invariant:

> 11. **Run data has one read path through `RunReader`.** The dashboard
>     server's read path goes `RunReader` → `RunRegistry` → `StepCache` ->
>     `DuckLakeStore`. There is no separate "live" read path. The
>     distinction between a run that is currently being recorded and one
>     that is not is encoded as `tail_growing: bool` on the `StepRange`
>     response, not as a separate API or code path. Do not add a parallel
>     read path for live data.

Update the **"React Dashboard"** section's data flow description:

> - **Data flow**: Game subprocess POSTs StepData to server via HTTP
>   callback. Server writes through `RunWriter` to a hot cache (`StepCache`)
>   and asynchronously to DuckLake. Frontend reads via REST through
>   `RunReader`, which checks the cache first then falls back to DuckLake.
>   Socket.io broadcasts step-added notifications that the frontend uses to
>   invalidate query caches; it never delivers data directly into React
>   state.

Remove the "Server Architecture (One Server, One URL)" subsection's mention
of "live data" as a distinct concept; phrase as "the same server reads from
runs whether or not they are currently being written".

### `roc/reporting/CLAUDE.md`

This file needs the most rewriting. Specific changes:

**"Key Decisions" section:**
- **DELETE** the bullet "StepBuffer for live data (bypasses DuckDB)" -- the
  premise is no longer true after this refactor.
- **REPLACE** with: "**StepCache for hot data, regardless of liveness** --
  A process-wide LRU absorbs recently-written steps so reads avoid the
  DuckLake round trip. The cache is a transparent optimization, not a
  separate API. Live and historical reads use the identical code path."
- **ADD** a bullet: "**`tail_growing` is the only live signal at the API
  boundary** -- the dashboard never sees an `is_live` predicate. A run with
  `tail_growing=True` may receive new steps; a run with `tail_growing=False`
  is closed. That's the only distinction."

**"Invariants" section:**
- **REPLACE** "Live data two-tier lookup" with "**Cache-first reads.**
  `RunReader.get_step()` always checks `StepCache` before `DuckLakeStore`.
  This is an internal optimization; callers do not see the difference."
- **DELETE** the paragraph about "live and historical queries hit different
  code paths with different performance characteristics" -- this is the
  exact thing being eliminated.

**"Non-Obvious Behavior" section:**
- **DELETE** the "Live data two-tier lookup" entry.
- **DELETE** the paragraph about `_supplement_logs` (function is removed).

**"Anti-Patterns" section:**
- **REPLACE** "Do not create additional DuckLakeStore instances for the same
  run directory" with: "**Do not bypass `RunReader`/`RunWriter`.** All run
  data access flows through these two facades. Direct `DuckLakeStore`
  construction is allowed only inside `RunRegistry`."
- **ADD**: "**Do not reintroduce `is_live` branches.** If you find yourself
  about to write `if run_name == self._live_run_name`, stop. The unified
  architecture means there is no separate live read path. Use
  `RunReader.get_*()` and let the cache layer handle hot vs cold."

**"Interfaces" section:**
- **REPLACE** "`Game thread -> StepBuffer`" with "`Game subprocess -> POST
  /api/internal/step -> RunWriter -> StepCache + ParquetExporter`".
- **REPLACE** "`API server -> browser`" with: "**`RunReader -> REST -> browser`**
  for data; **`RunRegistry subscribers -> Socket.io -> browser`** for
  invalidation notifications. Socket.io payloads contain only `{run, step}`,
  never full `StepData`."

### `dashboard-ui/CLAUDE.md`

This file also needs major rewriting. Specific changes:

**"Why This Design" section:**
- The first paragraph about race conditions between Bokeh control paths is
  still accurate context. Keep.
- The second paragraph ("strict separation: client owns all UI state...")
  remains correct but should be expanded: client also owns no server data
  -- TanStack Query is the cache, not React state.

**"Key Decisions" section:**
- **DELETE** the bullet "StepBuffer for live data (bypasses DuckDB)".
- **REPLACE** with: "**TanStack Query is the only client store for server
  data.** Components never hold parallel state for step data, step ranges,
  run lists, or live status. Socket.io is invalidation-only -- it tells the
  query cache when to refetch but never writes data directly into React
  state."

**"Invariants" section:**
- **DELETE** the "Live/historical feature parity" invariant. The unified
  architecture removes the distinction at the data layer; panels can no
  longer render differently in live vs historical mode because there is no
  separate live mode.
- The "URL parameter sovereignty" invariant remains; if anything it is now
  enforceable mechanically because the URL is the direct input to
  `useDashboardLocation()`.

**"Playback State Machine" section:**
- **REPLACE ENTIRELY**. The four-state machine and its transition diagram
  are obsolete. Replace with:

  > ## Playback Model
  >
  > Two booleans drive playback: `playing` (timer is ticking) and
  > `autoFollow` (snap to head as it grows). The four old states map to
  > combinations:
  >
  > | Old | `playing` | `autoFollow` |
  > |---|---|---|
  > | historical | t/f | false |
  > | live_following | n/a | true |
  > | live_paused | false | false |
  > | live_catchup | true | true |
  >
  > Transitions are direct mutations of the two booleans -- no reducer, no
  > state machine. Any explicit user navigation sets `autoFollow=false`.
  > "GO LIVE" sets `autoFollow=true` and snaps to `range.max`. The
  > auto-follow effect watches `range.max` and pulls the user along while
  > `autoFollow && range.tail_growing`.

**"Stale Closure Pattern" section:**
- **DELETE ENTIRELY**. With Socket.io as invalidation-only, the
  `onNewStep` callback no longer reads React state through refs. The ref
  pile (`runRef`, `gameRef`, etc.) is removed entirely. The pattern is
  unenforceable because there is no longer a Socket.io callback that needs
  to read state.

**"Performance Targets" section:**
- The targets remain. Cold/warm fetch latencies are unchanged because the
  cache absorbs hot reads. Document the new path: cache hit ~1ms, DuckLake
  read ~8ms, no Socket.io round-trip needed for already-cached data.

### `roc/game/CLAUDE.md`

Minimal change. Game-side architecture is unchanged. Add a note:

> The game subprocess writes step data via HTTP callback to
> `/api/internal/step`. The receiving handler invokes `RunWriter.push_step`
> which fans out to the in-process cache, the async DuckLake exporter, and
> any subscribers. The game does not need to know whether anyone is
> watching.

## Risks and Unknowns

The following risks are ordered by severity and by how blocking they are.
**Phase ordering is risk-first**: phases that expose blocking risks come
first so we can pivot or abandon the design before sinking work into
dependent phases.

### Risk #1: DuckLake read latency under concurrent writes (RESOLVED -- cache is mandatory)

**The premise.** Today, `StepBuffer` exists *because* DuckLake reads are
slow when contended with writes (300-600ms historically). The unified
architecture originally assumed that a separate read-only secondary
connection plus data inlining would let reads stay fast enough that the
cache could be a pure optimization.

**Phase 0 outcome (2026-04-08).** Tested. The premise is wrong in a
stronger way than the original risk anticipated: in the same process, a
second `DuckLakeStore(read_only=True)` against the catalog of an active
writer cannot be opened *at all*. DuckDB rejects the duplicate attach
with `BinderException: Unique file handle conflict`, regardless of the
DuckLake alias chosen. The measurement (`tmp/ducklake_concurrency_spike.py`)
also confirmed that the closed-run path -- open `read_only=True` *after*
`writer.close()` -- works cleanly. So the risk did not "leave reads slow";
it removed the secondary-connection option entirely for active runs.

**Resulting shape.** The cache is **mandatory**, not optional. For an
active run, `RunReader.get_step()` either hits `StepCache` or falls
through to the *same* `DuckLakeStore` instance held by the active
`RunWriter` (under its existing `_lock`). For a closed run, `RunReader`
opens a separate `read_only=True` `DuckLakeStore`. `RunRegistry` is the
sole owner of this active-vs-closed distinction; nothing above it sees
the difference.

**Mitigation.** Done. Phase 0 measured this; the cache becomes
load-bearing in Phase 1 and beyond. The "what we delete" analysis is
unaffected -- nothing else depends on the secondary-connection idea.

### Risk #2: DuckLake read-only secondary connection support (RESOLVED -- not supported in-process)

**The premise.** A read-only `DuckLakeStore` opened on a catalog file
could coexist with a writer connection on the same file in the same
process.

**Phase 0 outcome (2026-04-08).** Tested. **Not supported.** A second
`DuckLakeStore(run_dir, read_only=True)` constructed in the same Python
process while another `DuckLakeStore(run_dir, read_only=False)` is open
on the same `run_dir` raises:

    BinderException: Failed to attach DuckLake MetaData
    "__ducklake_metadata_<alias>" -- Unique file handle conflict:
    Cannot attach "..._metadata_..." - the database file
    "<run>/catalog.duckdb" is already attached by database
    "__ducklake_metadata_lake"

The error is at the catalog-file level, not the DuckLake alias level
(reproduced with both `alias="lake_reader"` and `alias="lake"`). DuckDB
1.5.0 does not allow the same database file to be attached twice in one
process. This matches the existing `roc/reporting/CLAUDE.md` anti-pattern
("Do not create additional `DuckLakeStore` instances for the same run
directory"); the old code base already worked around this by having the
live writer and the dashboard reader share one store via
`set_live_session`.

**Resulting shape.** `RunRegistry` enforces the rule: for an active run
(`tail_growing=True`), `_RunEntry.store` is the same `DuckLakeStore`
instance held by the `RunWriter`. For a closed run, `_RunEntry.store` is
a fresh `read_only=True` instance opened lazily. `StepCache`
write-through from `RunWriter.push_step` ensures most reads of hot data
never need to acquire the writer's lock at all.

**Closed-run path.** Phase 0 also measured this directly: opening
`DuckLakeStore(run_dir, read_only=True)` *after* `writer.close()`
succeeds and reads back data cleanly. The closed-run code path the
design assumes is therefore validated.

**Mitigation.** Done. See `tmp/ducklake_concurrency_spike.py` for the
measurement. Any future implementer who tries to "simplify" the design
by giving `RunReader` its own connection will reproduce the spike's
failure -- do not do this.

### Risk #3: History query performance via DuckLake (MEDIUM)

**The premise.** Removing `_indices` means `graph_history`, `event_history`,
`metrics_history`, etc. all hit DuckLake directly via `RunStore`. With data
inlining, a 10K-step run should query in <100ms.

**Failure mode if wrong.** History endpoints get slow. Charts take seconds
to load. The mitigation is to add a cache layer in `RunReader.get_history`
similar to `StepCache` -- per-run, evicted on `mtime` change. The design
shape is unchanged.

**Mitigation.** Phase 2 measures this directly. If it fails, we add a
read-through history cache; the public API stays the same.

### Risk #4: Frontend playback regression (LOW-MEDIUM)

**The premise.** The four-state playback machine is well-tested
(see `state/playback.test.ts`, ~100 lines covering all transitions).
Collapsing to two booleans risks missing subtle behaviors that the state
machine encoded.

**Failure mode if wrong.** A specific user interaction (e.g., navigating
during catchup, pausing exactly at the live edge, switching games while
following) breaks in a way the unit tests don't catch.

**Mitigation.** Phase 5 rewrites the playback tests as user-perspective
scenarios first ("GO LIVE → new step arrives → step advances", "user
clicks at step 50 while following → autoFollow drops to false → next
push does NOT advance step") and then performs the refactor against
those tests. End-to-end Playwright tests cover the visible behaviors.

### Risk #5: Test rewrite cost (LOW)

**The premise.** Many existing tests (`test_data_store.py`,
`test_api_server.py`) are coupled to the internal `_is_live` /
`_live_buffer` / `_indices` API. They will break and need rewriting.

**Failure mode if wrong.** Refactor takes longer than expected, with most
of the time spent in test churn.

**Mitigation.** Most coupled tests are simply deleted along with the code
they test. The behavioral tests (e.g., "the API returns the expected step
data") translate directly. Estimate: ~30 tests need rewrites, mostly
mechanical.

### Risk #6: Game subprocess HTTP callback contract (LOW)

**The premise.** The game subprocess uses `POST /api/internal/step` with a
specific payload shape. Changing the receiving handler to invoke
`RunWriter` is internal to the dashboard server and does not change the
contract.

**Failure mode if wrong.** Game subprocess can't talk to the dashboard
server. Live data stops flowing.

**Mitigation.** The HTTP contract is unchanged -- only the internal
handler changes. Phase 1's wedge keeps the existing signature.

## Phased Implementation

Each phase is a vertical slice. After completing a phase, the dashboard
must work end-to-end with manual validation. Phases are ordered so that
the highest-risk assumptions are tested first; if a phase fails, the
rest of the design adapts but is not invalidated.

### Phase 0: DuckLake concurrency spike (de-risks Risks #1, #2, #3)

**Goal.** Verify the central assumption that DuckLake reads stay fast
under concurrent writes from a separate connection. This is a pure
measurement -- no production code changes.

**Implementation.**

A standalone Python script in `tmp/ducklake_concurrency_spike.py` that:

1. Creates a fresh DuckLake catalog in a temporary directory.
2. Spawns a writer thread that inserts step records at 60 Hz (matching
   the game loop's max rate). Each record has the typical schema
   (screen, saliency, metrics, events, logs).
3. Spawns a reader thread (using a separate `DuckLakeStore` instance
   opened with `read_only=True`) that continuously queries random recent
   steps via `query_df`.
4. Runs for 30 seconds.
5. Reports p50, p95, p99 read latencies. Also reports any errors or
   stalls from the reader.
6. Repeats with `data_inlining_row_limit=500` (current default) and
   `data_inlining_row_limit=0` (no inlining) to confirm the inlining is
   what helps.

**Validation criteria.**

- [ ] Reader can open a read-only connection to a catalog being written
      by another connection without raising "unique file handle conflict"
      or similar.
- [ ] p99 read latency under load < 50ms.
- [ ] No torn reads (reader sees consistent step records, never
      half-written rows).
- [ ] No reader-side exceptions over the 30-second run.

**Phase 0 outcome (2026-04-08).**

Criterion #1 failed. The pivot listed below as "if criterion #1 fails"
is now the actual design. Specifically:

- Reader cannot open a `read_only=True` `DuckLakeStore` against the
  catalog file of an active writer in the same Python process. DuckDB
  raises `BinderException: Unique file handle conflict` regardless of
  the DuckLake alias name. Reproduced twice in the spike with both
  `inlining=500` and `inlining=0`.
- Closed-run control: opening `read_only=True` *after* `writer.close()`
  succeeds and reads back data cleanly. The closed-run path is valid.
- Criteria #2, #3, #4 are not measurable while criterion #1 fails (the
  reader thread never starts). Torn reads remain a theoretical concern
  for the shared-store path; this is mitigated by `DuckLakeStore._lock`
  serializing all access on the writer's instance.

Pivot taken (matches the original "if criterion #1 fails" branch but
made stronger by the fact that the reader cannot open *at all*):

1. `RunRegistry._RunEntry.store` is the **single source of truth** for
   "the `DuckLakeStore` for this run." For active runs, that store IS
   the writer's instance (passed in by `RunWriter.__init__`). For
   closed runs, it is a fresh `read_only=True` instance opened lazily
   by `RunRegistry._load`.
2. `RunWriter.push_step` writes through to `StepCache` synchronously,
   so hot reads of an active run hit the cache and never need the
   writer's `_lock` at all.
3. On `RunWriter.close()`, the writer's store is detached from
   `_RunEntry`; the next read of that run lazily opens a fresh
   `read_only=True` instance.

**Original decision branches** (kept for historical reference; criterion
#1 fired and superseded the others):

- *All four pass* -> proceed unchanged; `StepCache` would have been a
  pure optimization. **Did not happen.**
- *Criterion #1 fails* -> the pivot above. **This is what happened.**
- *Criteria #2 or #4 fail* -> same pivot. **Subsumed by #1 failing.**
- *Criterion #3 fails* -> stop and revisit storage layer. **Did not
  happen.**

**Reversible.** Yes -- the spike script (`tmp/ducklake_concurrency_spike.py`)
is the only artifact; it will be deleted at the end of Phase 0 once
the new constraint is documented in `roc/reporting/CLAUDE.md`.

**Effort actually spent.** ~1 hour to write, run, and document the
spike + the design-doc updates that captured the pivot.

---

### Phase 1: Introduce `RunReader` / `RunRegistry` facade (no behavior change)

**Goal.** Add the unified facade as a wrapper over the existing
`DataStore` internals. All API endpoints route through it. The internal
cache/index code stays in place initially.

**Implementation.**

1. Create `roc/reporting/run_registry.py` with `RunRegistry` and
   `_RunEntry`. Initially, `_RunEntry.store` wraps the existing
   `_get_run_store` logic from `DataStore`. The summary cache and the
   live index data move into `_RunEntry` fields.
2. Create `roc/reporting/run_reader.py` with `RunReader`. Implement
   `get_step`, `get_step_range`, `get_history`, `list_runs`, and
   `subscribe`. Initially, `subscribe()` returns a no-op callback for
   all runs (notifications are wired in Phase 4).
3. Update `api_server.py` handlers to call `RunReader` methods instead
   of `DataStore` methods directly. `DataStore` itself becomes a thin
   shim that holds the `RunRegistry` and exposes legacy methods for the
   short list of callers that haven't been migrated yet.
4. Update tests: rename a few `DataStore.list_runs()` call sites to
   `RunReader.list_runs()`. Most existing tests still pass without
   modification because the facade returns the same data shape.

**Validation criteria.**

- [ ] `make test` passes.
- [ ] `pnpm -C dashboard-ui exec vitest run` passes.
- [ ] Manual: open dashboard, browse historical run, step through
      frames, verify data displays correctly.
- [ ] Manual: start a game via "Game" button, verify live data appears
      and updates.
- [ ] Manual: navigate to a historical run while a game is running,
      verify no impact on either run.
- [ ] No `is_live`/`_live_*` references remain in `api_server.py`
      (they may still exist inside `DataStore` -- we delete those in
      Phase 6).

**Why first (after Phase 0).** This is the wedge. Once everything reads
through the facade, removing the alternatives is mechanical. No
behavior change means low risk; if something breaks, the cause is local
to the facade.

**Reversible.** Yes -- delete the new files, revert the `api_server.py`
changes.

**Estimated effort.** One to two days.

---

### Phase 2: Eliminate `_indices`, history reads from DuckLake

**Goal.** Delete `DataStore._indices` and `_GameIndex`. History queries
(`get_graph_history`, `get_event_history`, `get_metrics_history`,
`get_intrinsics_history`, `get_action_history`, `get_resolution_history`)
hit DuckLake directly via `RunStore`.

**Implementation.**

1. In `RunStore`, ensure each history query method exists and is
   correct. Most already exist. Add the few that only existed in
   `DataStore._get_live_history`.
2. In `RunReader.get_history`, dispatch by `kind` to the appropriate
   `RunStore` method.
3. Delete `DataStore._indices`, `_GameIndex`, `_index_step`,
   `_index_event_summary`, `_get_live_history`.
4. Delete the listener that calls `_on_step_pushed` -- it was the only
   thing populating `_indices`.
5. Update tests in `test_data_store.py` that asserted on
   `_indices` directly. Most can be deleted; a few should be rewritten
   as integration tests against the DuckLake-backed query path.

**Validation criteria.**

- [ ] All backend tests pass.
- [ ] Manual: open the Graph & Events panel for a historical run,
      verify the graph history chart renders. Same for the live run.
- [ ] Performance: `get_graph_history` for a 10K-step run completes in
      <500ms. Measure via the existing timing logger or a benchmark.
- [ ] Performance: `get_metrics_history` for the same run completes
      similarly fast.

**Why second.** This validates Risk #3 (history query perf via
DuckLake). If DuckLake history queries are too slow, we add a
read-through cache in `RunReader.get_history` (per-run, mtime-evicted)
and the rest of the design proceeds unchanged. We want to find this out
before doing surgery on the frontend.

**Reversible.** Yes -- restore `_indices` from git history, re-attach
the listener. The change is contained to backend.

**Estimated effort.** One day.

---

### Phase 3: `tail_growing` in API + delete frontend live status

**Goal.** Add `tail_growing` to the step-range response. Delete
`liveRunName`, `liveGameNumber`, `liveGameActive` from `DashboardContext`.
Frontend reads live status from `useStepRange().data.tail_growing`. The
"GO LIVE" badge appears when this is true.

**Implementation.**

1. Backend: `StepRange` model gets a `tail_growing: bool` field, set by
   `RunRegistry` based on whether a `RunWriter` is currently active for
   the run.
2. Backend: Wire `RunWriter.__init__` to call `registry.mark_growing(name,
   growing=True)` and `RunWriter.close()` to call
   `registry.mark_growing(name, growing=False)`.
3. Frontend: Update `useStepRange` return type to include
   `tail_growing`.
4. Frontend: Delete `liveRunName`, `liveGameNumber`, `liveGameActive`
   and their setters from `DashboardContext`.
5. Frontend: Replace `liveGameActive` reads in `StatusBar` (the GO LIVE
   badge condition) with a check on the current run's
   `useStepRange().data.tail_growing`.
6. Frontend: Delete `useLiveUpdates`'s `liveStatus` polling. Game start
   notifications come via Socket.io (wired in Phase 4) or via natural
   refetch of `/api/runs` (every 10s).

**Validation criteria.**

- [ ] Manual: open dashboard with no game running. No GO LIVE badge.
      `useStepRange.data.tail_growing` is false for all runs.
- [ ] Manual: start a game. Within ~1s, `tail_growing` flips true for
      the new run. GO LIVE badge appears when viewing that run.
- [ ] Manual: stop the game. `tail_growing` flips back to false. GO
      LIVE badge disappears.
- [ ] Existing live status tests are rewritten to assert on
      `tail_growing`.

**Why third.** This is the API contract change. If `tail_growing`
semantics are wrong (e.g., race conditions in marking the writer as
active), we want to find out before removing the rest of the
live/historical branching. This phase changes a small surface; the next
phase changes a much larger one.

**Reversible.** Mostly. Context fields can be added back. The harder
revert is the deletion of `useLiveUpdates`, but it's a contained file.

**Estimated effort.** One to two days.

---

### Phase 4: Socket.io as invalidation, delete `liveData` and refs

**Goal.** Change the Socket.io payload from full `StepData` to
`{run, step}` notifications. Frontend handler calls
`queryClient.invalidateQueries`. Delete `liveData` state, the
`isFollowing ?` ternary, and the entire ref pile in `App.tsx`.

**Implementation.**

1. Backend: Add `subscribe_run` / `unsubscribe_run` Socket.io events.
   The handler calls `RunReader.subscribe(run, on_step)`. The `on_step`
   callback emits `step_added` to the requesting socket.
2. Backend: Change the existing `new_step` Socket.io broadcast to emit
   only `{run, step}` (not full `StepData`). Eventually delete the old
   broadcast entirely; transitionally, both can coexist.
3. Frontend: Add `useRunSubscription(run)` hook (sketched above).
   Subscribe on mount, unsubscribe on cleanup.
4. Frontend: In the subscription handler, call
   `queryClient.invalidateQueries({queryKey: ["step-range", run]})` to
   pick up the new max. Optionally also `setQueryData` for the new
   step if the payload includes it (it doesn't, by design, in this
   phase).
5. Frontend: Delete `liveData` state in `App.tsx`. Delete the
   `data = isFollowing ? liveData ?? restData : restData` ternary.
6. Frontend: Delete the entire ref pile (`runRef`, `gameRef`, `stepRef`,
   `stepMinRef`, `stepMaxRef`, `isFollowingRef`, `liveRunNameRef`).
7. Frontend: Delete the `onNewStep` callback. The auto-follow effect
   from Phase 5 (or a temporary inline version) handles the "advance
   step when new data arrives" behavior.

**Validation criteria.**

- [ ] Manual: start a game. Dashboard shows live data updating in
      real time as new steps arrive.
- [ ] Manual: pause playback during live game. Scrub backward.
      Historical steps load correctly.
- [ ] Manual: click GO LIVE. Dashboard catches up to the latest step.
- [ ] Manual: at high speed (10x), the dashboard does not get stuck
      in `live_paused` or `live_catchup` states.
- [ ] E2E (Playwright): simulate rapid step pushes via the existing
      test harness. Verify the dashboard advances correctly.
- [ ] No refs related to playback exist in `App.tsx`.

**Why fourth.** Highest UI risk. The Socket.io callback has been the
source of most stale-closure bugs. Doing this *after* `tail_growing`
means the frontend already has a stable signal for "is there new data
coming"; we're just changing how the data delivery happens.

**Reversible.** Difficult. Touches a lot of frontend state. Would need
to revert Phases 3 and 4 together. Mitigated by aggressive E2E testing.

**Estimated effort.** Two to three days.

---

### Phase 5: Collapse playback state machine to `autoFollow`

**Goal.** Replace the four-state `playbackReducer` with two booleans
(`playing`, `autoFollow`). Update all consumers.

**Implementation.**

1. Frontend: Replace the four-state enum in
   `dashboard-ui/src/state/playback.ts` with the two-boolean shape.
   Delete the reducer; transitions become direct setters.
2. Frontend: Update `DashboardContext` to expose `playing`,
   `setPlaying`, `autoFollow`, `setAutoFollow`. Delete the `playback`
   field.
3. Frontend: Add the auto-follow effect to `App.tsx` (sketched above).
4. Frontend: Replace all `playback === "live_following"` checks with
   `autoFollow`. Replace `dispatchPlayback({type: "USER_NAVIGATE"})`
   with `setAutoFollow(false)`. Replace
   `dispatchPlayback({type: "GO_LIVE"})` with `setAutoFollow(true)`.
5. Frontend: Delete `state/playback.ts` and its tests.
6. Frontend: Add new tests covering all four `(playing, autoFollow)`
   combinations and the auto-follow advance behavior.

**Validation criteria.**

- [ ] Manual: GO LIVE works (sets `autoFollow=true`, snaps to head).
- [ ] Manual: navigation while following drops `autoFollow` to false.
- [ ] Manual: pause/play during live game works.
- [ ] Manual: switching games during live following resets the follow
      target to the new game's head.
- [ ] All replaced playback tests pass under the new shape.

**Why fifth.** Cleanup. By Phase 4, the four-state machine is dead
weight -- Socket.io invalidation plus an `autoFollow` flag does
everything the state machine did. This phase removes the dead code.

**Reversible.** Yes -- small frontend change.

**Estimated effort.** One day.

---

### Phase 6: Final cleanup, delete dead code

**Goal.** Final pass. Delete `_live_run_name`, `_live_buffer`,
`_live_store`, `set_live_session`, `clear_live_session`, all `is_live()`
methods, the legacy `StepBuffer` class (replaced by `StepCache`),
`useLiveUpdates`, and any other now-unused branches.

**Implementation.**

1. Backend: Walk `data_store.py`, delete every method and field whose
   only purpose was the live/historical distinction. The remaining
   `DataStore` is a thin wrapper around `RunRegistry` (or is itself
   deleted in favor of direct `RunRegistry`/`RunReader` usage).
2. Backend: Replace `step_buffer.py` with `step_cache.py`. Update any
   imports.
3. Backend: Delete `set_live_session`, `clear_live_session`,
   `_supplement_logs`, `_is_live`, `is_live`, `_get_live_history`.
4. Backend: Run `make lint` and address any unused-import warnings.
5. Frontend: Delete `useLiveUpdates.ts` if not done in Phase 3.
6. Frontend: Delete the now-orphaned context fields and effects.
7. Update `roc/reporting/CLAUDE.md` and `dashboard-ui/CLAUDE.md` per the
   "CLAUDE.md Updates" section above.
8. Update `/CLAUDE.md` with the new Architectural Invariant #11.

**Validation criteria.**

- [ ] All backend tests pass.
- [ ] All frontend tests pass.
- [ ] `make lint` clean.
- [ ] `pnpm -C dashboard-ui exec tsc --noEmit` clean.
- [ ] Manual smoke test: full game lifecycle (start → play → pause →
      scrub → GO LIVE → stop → browse historical).
- [ ] `grep -r "is_live\|_live_buffer\|_indices" roc/reporting/` returns
      zero matches.
- [ ] `grep -r "liveData\|runRef\|gameRef\|liveRunName" dashboard-ui/src/`
      returns zero matches.
- [ ] CLAUDE.md files reflect the new architecture.

**Reversible.** Doesn't need to be -- by this point everything is
working with the new model. This is a cleanup phase, not a feature
phase.

**Estimated effort.** One day.

---

## Validation Strategy

### Per-phase gates

- All existing tests pass (except those explicitly rewritten or deleted
  as part of the phase).
- Manual smoke test of the affected user flows.
- Performance: dashboard navigation stays under 50ms per cached step,
  under 500ms per cold step (matching current targets in
  `dashboard-ui/CLAUDE.md`).

### End-to-end gates

After every phase (and especially after Phase 6), the following must work:

1. **Cold start.** Open the dashboard with no game running. The run
   dropdown lists historical runs. Click one. The dashboard loads its
   step data within 500ms. Step through frames; navigation is instant
   for cached steps.
2. **Live start.** Click "Game" → "Start". A game subprocess spawns.
   Within ~5 seconds, the new run appears in the dropdown and is
   auto-selected. Step data streams in. The GO LIVE badge appears (or
   the dashboard is already in follow mode).
3. **Live navigation.** During a live game, click the slider to navigate
   backward. Auto-follow drops to false. Historical steps load. Click
   GO LIVE. Dashboard catches up to the latest step.
4. **Live → historical transition.** Stop the game. The run remains in
   the dropdown but `tail_growing` flips false. The GO LIVE badge
   disappears. All previously-loaded data remains accessible. New
   navigation hits DuckLake.
5. **Concurrent runs.** Start a game. While it runs, navigate to a
   different historical run in the dropdown. Both should work; the live
   game continues writing in the background while you browse the other.
6. **Errors are visible.** If an API call fails (simulate via stopping
   the server briefly), the StatusBar shows an ERROR badge. No step
   shows blank without explanation.

### Test strategy

- **Unit tests** for `RunReader`, `RunWriter`, `RunRegistry`, `StepCache`
  with mocked `DuckLakeStore`. These are easier to write than the
  current `DataStore` tests because the surface is smaller.
- **Integration tests** for the cache → DuckLake fall-through path.
  These replace the current `_supplement_logs` tests.
- **Frontend tests** for `useStepData`, `useStepRange`, the auto-follow
  effect, and the `StatusBar` ERROR badge.
- **E2E tests (Playwright)** for the live → historical transition,
  cross-game navigation, and rapid step pushes. Use the existing
  test harness in `tests/e2e/`.

## Open Questions

1. **Should `StepCache` be process-wide or per-run?** Lean: process-wide
   LRU. A 5000-entry shared cache gives ~50 MB of memory (assuming
   ~10 KB per StepData) and serves all runs. Per-run caches add
   complexity (eviction, lifecycle) without obvious benefit since the
   working set is dominated by whatever the user is currently viewing.

2. **Should the Socket.io payload include `StepData` (push) or just the
   notification (pull)?** Lean: pull. Sending only `{run, step}` keeps
   the protocol simple and forces the test harness to use the same
   data path as the production REST endpoint. The cost is one extra
   round trip per step (~10ms), which is invisible to the user. If
   profiling later shows this is a bottleneck, we can add `StepData`
   to the payload as an optimization (frontend uses `setQueryData` to
   seed the cache).

3. **What happens when a game starts on a run name that already exists
   historically?** Today this doesn't happen because run names include
   timestamps (`20260408120000-rancorous-fey-devy`). The unified
   architecture preserves this assumption -- if a name collision did
   occur, the existing entry in `RunRegistry` would be reused, and
   `RunWriter` would append to the existing catalog. This is wrong
   behavior, but the surface area for the bug is limited to "do not
   reuse run names." Document the assumption.

4. **Should `RunReader.subscribe` notifications be coalesced?** If a
   game is writing at 60 Hz, the frontend receives 60 invalidations per
   second per subscribed query. TanStack Query handles this gracefully
   (debounces via `staleTime`), but we could coalesce on the server
   side to reduce Socket.io traffic. Lean: no coalescing initially;
   measure first.

5. **Should `StepCache` support eviction by run lifecycle?** When a
   game ends and `RunWriter.close()` is called, should we proactively
   evict the cached entries for that run? Lean: no -- the LRU naturally
   evicts them as new data comes in, and keeping them around briefly
   makes "look at the run that just ended" instant.

6. **Does this design cleanly support the in-process dashboard mode
   (`uv run play` with `start_dashboard()`)?** Phase 0 inverted this
   question: in-process is no longer a "structurally identical"
   special case but the **only** shape that works for active reads.
   Because DuckDB will not let two `DuckLakeStore` instances attach
   the same catalog file in the same process, both subprocess mode
   and in-process mode share *exactly one* `DuckLakeStore` per
   active run. The single-process case (game thread + dashboard
   server in one process) and the dual-process case (dashboard
   server hosts the writer, game subprocess POSTs steps over HTTP)
   converge on the same internal shape: one `DuckLakeStore` instance
   owned by the active `RunWriter`, shared with `RunReader` via
   `_RunEntry.store`. Phase 1 must verify that `start_dashboard()`'s
   short-circuit (when `_game_manager is not None`) wires the writer
   into the same `RunRegistry` the API server reads from.

## References

- `design/dashboard-server-redesign.md` -- The previous server architecture
  redesign that established the current `roc-server` + `roc-ui` setup.
- `design/live-step-performance-fix.md` -- Previous fix for the
  log-supplementation locking issue. The unified architecture makes the
  underlying problem go away.
- `design/playback-state-machine.md` -- The current four-state playback
  machine design. Will be obsoleted by Phase 5.
- `design/dashboard-step-performance.md` -- Performance baselines for
  step navigation. Phase validation should match or beat these numbers.
- `roc/reporting/CLAUDE.md` -- The reporting subsystem guide. Needs the
  updates listed in the "CLAUDE.md Updates" section.
- `dashboard-ui/CLAUDE.md` -- The dashboard UI guide. Needs the updates
  listed in the "CLAUDE.md Updates" section.
- The architectural review from 2026-04-08 (24 findings, recorded in
  conversation context) -- the analysis that motivated this design.

## Appendix A: Current vs Proposed -- Concrete Diffs

### `data_store.py` shrinks dramatically

Current size: ~750 lines. Estimated after Phase 6: ~80 lines (or
deleted entirely, with `RunRegistry` standing alone).

The deleted parts:

```python
# DELETE - lines ~97-110
@dataclass
class _GameIndex:
    graph_history: list[dict[str, Any]] = field(default_factory=list)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    intrinsics_history: list[dict[str, Any]] = field(default_factory=list)
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    resolution_events: list[dict[str, Any]] = field(default_factory=list)

# DELETE - lines ~137-145
self._live_run_name: str | None = None
self._live_buffer: StepBuffer | None = None
self._live_store: DuckLakeStore | None = None
self._indices: dict[int, _GameIndex] = {}
self._run_stores: dict[str, RunStore] = {}
self._run_summary_cache: dict[str, RunSummary] = {}

# DELETE - lines ~166-214
def set_live_session(self, run_name, buffer, store=None): ...
def clear_live_session(self): ...

# DELETE - lines ~225-269
def _on_step_pushed(self): ...
def _index_step(self, step_data): ...
def _append_if_present(history, step, data): ...
def _index_event_summary(idx, step, event_summary): ...

# DELETE - lines ~274-290
def is_live(self, run_name): ...
def _is_live(self, run_name): ...
def _get_live_history(self, game, field_name): ...

# DELETE - lines ~439-451
def _supplement_logs(self, buf_data, run_name, step): ...

# DELETE the live branches in:
#   list_runs (already mostly gone after the dropdown fix)
#   list_games
#   get_step_range
#   get_step_data
#   get_steps_batch
```

### `App.tsx` shrinks

Current size: ~600 lines. Estimated after Phase 6: ~400 lines.

The deleted parts:

```typescript
// DELETE - line 175
const [liveData, setLiveData] = useState<StepData | null>(null);

// DELETE - lines 178-184
const liveRunSelected = useRef(false);
const initialUrlRun = useRef<string | null>(...);

// DELETE - lines 187-200 (the entire ref pile)
const runRef = useRef(run);
runRef.current = run;
const gameRef = useRef(game);
gameRef.current = game;
// ... (all of them)

// DELETE - lines 202-236 (the onNewStep callback)
const onNewStep = useCallback(...);

// DELETE - line 457
const data = isFollowing ? liveData ?? restData : restData;

// REPLACE with:
const data = restData;  // restData IS the data; cache is invalidated by Socket.io
```

### `context.tsx` shrinks

Current size: ~215 lines. Estimated after Phase 6: ~80 lines.

```typescript
// DELETE - lines 92-100, 116-118
liveRunName, setLiveRunName,
liveGameNumber, setLiveGameNumber,
liveGameActive, setLiveGameActive,

// DELETE - lines 83-85, 112-113, 131-148
stepMin, stepMax,
setStepRange,
// (the clamp logic)

// DELETE - lines 153-159
// URL sync effect (replaced by direct useSearchParams)

// REPLACE with: a much smaller context that only holds UI state.
```

### `playback.ts` is replaced

Current: ~100 lines, four-state reducer.
After Phase 5: deleted entirely. Two booleans (`playing`, `autoFollow`) live
directly in `DashboardContext`.

### `useLiveUpdates.ts` is deleted

Current: ~180 lines. After Phase 6: deleted entirely. Live status comes
from `useStepRange().data.tail_growing`. Subscriptions come from
`useRunSubscription`.

## Appendix B: Mapping of Findings to Phases

The architectural review from 2026-04-08 identified 24 findings. Mapping
each to the phase that addresses it:

| Finding | Description | Addressed by |
|---|---|---|
| 1.1 | Corrupted runs filtered via `min_steps` | **DONE** (this conversation) |
| 1.2 | `get_steps_batch` silently skips failed steps | Phase 1 (replaced by `RunReader`) |
| 1.3 | `_supplement_logs` swallows DuckLake errors | Phase 6 (function deleted) |
| 1.4 | StepBuffer listener exceptions suppressed | Phase 6 (listener removed) |
| 1.5 | `_query_table_for_steps` returns empty on error | Phase 1 (`RunReader` returns typed status) |
| 2.1 | `onNewStep` reads stale `stepMin/stepMax` refs | Phase 4 (`onNewStep` deleted) |
| 2.2 | `pollLiveStatusRef` initialized after listener setup | Phase 3 (`useLiveUpdates` deleted) |
| 2.3 | Multiple refs read sequentially out of sync | Phase 4 (refs deleted) |
| 3.1 | StatusBar shows "Game 0" for out-of-range | Phase 1 (typed `StepResponse`) |
| 3.2 | DuckLake errors return empty df not error | Phase 1 (`RunReader` returns typed status) |
| 4.1 | `_run_summary_cache` accessed unsynchronized | Phase 1 (`RunRegistry` has one lock) |
| 4.2 | DuckLake write lock cascades to API reads | Phase 0 (de-risked) + Phase 6 (cache absorbs) |
| 4.3 | `usePrefetchWindow` AbortController doesn't stop in-flight | Out of scope (separate fix) |
| 5.1 | `metrics ?` truthy for `{}`, NaN math in HP bar | Out of scope (separate fix) |
| 5.2 | Two sources of truth for step range | Phase 3 (`tail_growing`) |
| 5.3 | History indices vs StepBuffer eviction gap | Phase 2 (`_indices` deleted) |
| 6.1 | All `StepData` fields optional → "no data" ambiguous | Phase 1 (typed `StepResponse`) |
| 6.2 | `game_number` int coercion silently truncates | Out of scope (Pydantic) |
| 6.3 | `useActionMap` 404 vs network error indistinguishable | Out of scope (separate fix) |
| 7.1-7.5 | Test gaps | Addressed phase-by-phase as tests are added |

The unified architecture addresses 14 of 19 high/medium-severity findings
directly. The remaining 5 are smaller, independent fixes that can be done
before, during, or after this refactor without conflict.
