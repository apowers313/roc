# Design: Thread-Based Shared Memory Between Server and Game

## Problem Statement

The current architecture runs each game as a **subprocess** (`subprocess.Popen`),
communicating with the dashboard server via HTTP POST callbacks. This creates three
problems:

1. **No live graph access.** The graph (GraphCache with Node/Edge objects) lives in
   the game subprocess's memory space. The server cannot query it. Graph visualization
   currently requires a `graph.json` export written at game end -- no live exploration.

2. **Serialization overhead.** Every step, the game subprocess serializes the entire
   StepData to JSON (~10-500KB), POSTs it over HTTP, and the server deserializes it.
   This is wasted work when the data could be shared directly.

3. **Architectural complexity.** The HTTP callback requires: URL wiring, SSL context
   management, cooperative shutdown via HTTP response body, best-effort error handling,
   numpy-to-JSON conversion, and a separate `/api/internal/step` endpoint. Removing
   this simplifies both codebases.

### What the Prototypes Proved

**Prototype 1: Generic shared memory** (`tmp/thread_shared_memory_experiment.py`,
39/39 tests) demonstrated that a FastAPI server and worker threads can share Python
objects with zero serialization:

- Server reads/writes the same Python objects created by worker threads
- BFS graph traversal works on shared data structures
- Data persists after the worker thread stops
- Multiple workers write to the same shared graph without conflicts
- 20 concurrent reads during active writes produced zero errors

**Prototype 2: NLE in threads** (`tmp/nle_thread_experiment.py`, 63/63 tests)
proved that the real NLE/Gymnasium environment works correctly in threads:

- NLE environment creation and stepping works in daemon threads
- Server reads live game state (HP, score, screen, position) directly from shared
  Python objects while the game runs -- zero serialization
- Full 21x80 TTY screens readable via HTTP while NLE runs
- Frame history (FrameSnapshot objects) persists after game thread exits
- Server can annotate frame objects created by the game thread (bidirectional access)
- **3 sequential game runs** in the same process: all completed, no NLE C state leaks
- **Multi-game run** (2 games via env.reset() in one thread): worked correctly
- **2 concurrent NLE instances** in parallel threads: no conflicts, both clean
- **Stop signal** via threading.Event: cleanly stops game loop mid-play
- **40 concurrent reads** during active NLE gameplay: zero errors
- **Final game after 9 total runs**: NLE still works -- no accumulated C state issues
- `env.close()` + `del env` fully cleans up NLE's C extensions between runs

## Current Architecture

```
+---------------------------+         HTTP POST          +---------------------------+
|    Game Subprocess        |  ----------------------->  |    Server Process          |
|    (uv run play)          |  StepData JSON (~500KB)    |    (uv run server)         |
|                           |  <-----------------------  |                            |
|  NLE Environment          |  {"stop": true/false}      |  FastAPI + Socket.io       |
|  Pipeline (Components)    |                            |  DataStore + StepBuffer    |
|  GraphCache (Node/Edge)   |                            |  DuckLakeStore             |
|  State + OTel emission    |                            |  GraphService (graph.json) |
|  Config singleton         |                            |  GameManager               |
+---------------------------+                            +---------------------------+
     Isolated process                                         Isolated process
     Fresh Python interpreter                                 Long-running
```

### Data Flow (Current)

1. Game subprocess calls `roc.init()` -- creates NLE env, loads Components, starts
   pipeline, starts its own dashboard server in background thread
2. Each step: pipeline runs, `State.emit_state_logs()` emits OTel records, then
   `_push_dashboard_data()` assembles StepData
3. StepData is serialized to JSON and POSTed to `{callback_url}/api/internal/step`
4. Server deserializes, pushes to StepBuffer, indexes in DataStore, emits Socket.io
5. Server response may include `{"stop": true}` for cooperative shutdown
6. At game end: `graph.json` exported (optional), process exits

### What Cannot Be Accessed Live

- **GraphCache contents** -- Node/Edge objects live in subprocess memory
- **Individual node/edge queries** -- no API for querying specific nodes during play
- **BFS subgraph extraction** -- requires graph.json (only exists after game end)
- **Object history traversal** -- same limitation

## Proposed Architecture

```
+-------------------------------------------------------------------+
|                     Server Process (uv run server)                  |
|                                                                     |
|  +--------------------+     shared heap     +--------------------+  |
|  |  Game Thread        | <================> |  FastAPI + Socket.io|  |
|  |                     |                    |                     |  |
|  |  NLE Environment    |    GraphCache      |  REST API handlers  |  |
|  |  Pipeline           |    Node/Edge objs  |  DataStore          |  |
|  |  State + OTel       |    StepBuffer      |  DuckLakeStore      |  |
|  |  Config (shared)    |    DataStore index  |  GraphService(live) |  |
|  +--------------------+                    +--------------------+  |
|       daemon thread                              uvicorn thread     |
+-------------------------------------------------------------------+
```

### Key Change

The game runs as a **daemon thread** within the server process instead of a subprocess.
All Python objects are shared via the process heap. No serialization, no HTTP callbacks,
no IPC.

### Data Flow (Proposed)

1. Server starts, creates shared infrastructure (DataStore, StepBuffer, Config)
2. User triggers game start via `POST /api/game/start`
3. Server spawns game thread: initializes NLE, loads Components, starts pipeline
4. Each step: pipeline runs, StepData assembled in-process, pushed directly to
   StepBuffer (function call, not HTTP)
5. StepBuffer listener fires DataStore indexing (same as today, minus HTTP layer)
6. Socket.io notification via `run_coroutine_threadsafe` (same pattern as today)
7. **NEW**: API endpoints can query GraphCache directly for live graph data
8. Shutdown via `threading.Event` (replaces HTTP response-based signaling)
9. At game end: thread exits, all data remains in server process memory

## Detailed Component Changes

### 1. GameManager -- Thread Instead of Subprocess

**Current**: Spawns `subprocess.Popen(["uv", "run", "play", ...])`, monitors via
filesystem polling and stdout pipe, cooperative shutdown via HTTP response.

**Proposed**: Spawns `threading.Thread(target=game_entry, daemon=True)`, direct
access to thread state, shutdown via `threading.Event`.

```python
class GameManager:
    def __init__(self, data_dir, on_state_change, server_url):
        self._stop_event = threading.Event()
        self._game_thread: threading.Thread | None = None
        self._data_store: DataStore  # shared reference

    def start_game(self, num_games: int = 5) -> None:
        self._stop_event.clear()
        self._game_thread = threading.Thread(
            target=self._run_game,
            args=(num_games,),
            name="game-worker",
            daemon=True,
        )
        self._game_thread.start()

    def _run_game(self, num_games: int) -> None:
        """Game entry point -- runs on the game thread."""
        try:
            self._state = "running"
            self._on_state_change(self.get_status())
            _game_main(
                num_games=num_games,
                stop_event=self._stop_event,
                data_store=self._data_store,
                step_buffer=self._step_buffer,
            )
        except Exception as e:
            self._error_message = str(e)
        finally:
            self._state = "idle"
            self._on_state_change(self.get_status())

    def stop_game(self) -> None:
        self._stop_event.set()
        if self._game_thread:
            self._game_thread.join(timeout=10)
```

**Removed**: subprocess.Popen, monitor thread, log thread, filesystem polling for
run directory, SIGTERM/SIGKILL escalation, callback URL wiring.

**Added**: Direct thread lifecycle, shared `threading.Event` for stop signaling.

### 2. Game Entry Point -- New `_game_main()` Function

The current `roc.init()` + `roc.start()` sequence does too much for a thread context.
A new entry point isolates game-specific initialization from server infrastructure:

```python
def _game_main(
    num_games: int,
    stop_event: threading.Event,
    data_store: DataStore,
    step_buffer: StepBuffer,
) -> None:
    """Game loop entry point for thread-based execution."""
    # 1. Initialize game-specific components
    Config.init(force=False)  # reuse existing config singleton
    gym = NethackGym()
    Component.init()
    ExpMod.init()
    State.init()

    # 2. Run game loop
    try:
        gym.start(stop_event=stop_event, step_buffer=step_buffer)
    finally:
        # 3. Cleanup game-specific state
        Component.reset()
        # GraphCache nodes/edges persist -- intentional!
        # Only pipeline component state is cleaned up
```

**Critical**: `Component.reset()` shuts down pipeline components but does NOT
clear GraphCache. The Node/Edge objects remain accessible to the server after the
game thread exits. This is the key benefit of shared memory.

### 3. gymnasium.py -- Remove HTTP Callback, Push Directly

**Remove**:
- `_push_step_to_server()` -- HTTP POST logic
- `callback_url` parameter and SSL context setup
- `_server_stop` flag (replaced by `stop_event.is_set()`)
- All `urllib.request` imports

**Modify `_push_dashboard_data()`**:

```python
def _push_dashboard_data(
    obs: Any,
    loop_num: int,
    game_num: int,
    step_buffer: StepBuffer,  # direct reference, not HTTP
) -> None:
    """Collect step data and push directly to shared StepBuffer."""
    # ... existing StepData assembly (unchanged) ...
    step_data = StepData(...)
    step_buffer.push(step_data)  # direct function call, zero serialization
```

**Modify `Gym.start()` to accept stop_event**:

```python
def start(self, stop_event: threading.Event, step_buffer: StepBuffer) -> None:
    while game_num <= settings.num_games and not stop_event.is_set():
        # ... existing game loop ...
        _push_dashboard_data(obs, loop_num, game_num, step_buffer)
        # Check stop signal (replaces HTTP response check)
        if stop_event.is_set():
            break
```

### 4. api_server.py -- Remove Internal Endpoint, Add Live Graph

**Remove**:
- `POST /api/internal/step` endpoint
- `_notify_new_step()` can be simplified (StepBuffer listener handles Socket.io)

**Modify live session management**:

```python
def _start_live_session(run_name: str) -> None:
    """Called when game thread starts."""
    buf = StepBuffer(capacity=100_000)
    register_step_buffer(buf)
    _data_store.set_live_session(run_name, buf)
    # Game thread will push directly to this buffer
```

**Add live graph query support to GraphService**:

```python
class GraphService:
    def get_graph(self, run_name: str) -> nx.DiGraph:
        if self._is_live(run_name):
            return self._get_live_graph()  # query GraphCache directly
        # ... existing graph.json loading ...

    def _get_live_graph(self) -> nx.DiGraph:
        """Build a NetworkX view from the live GraphCache."""
        from roc.db.graphdb import Node, Edge
        # Iterate cache, build graph on the fly
```

### 5. GraphService -- Live Graph Access

This is the primary motivation for the architecture change. Currently, GraphService
only reads from `graph.json` files (post-game archives). With shared memory, it can
query the live GraphCache directly.

**Two modes**:

| Mode | Data Source | When |
|------|------------|------|
| Live | GraphCache (Node/Edge LRU cache) | Game thread running or recently stopped |
| Historical | graph.json archive file | Past runs loaded from disk |

**Live graph query implementation**:

```python
def subgraph_from_frame_live(self, tick: int, depth: int = 2) -> nx.DiGraph:
    """BFS subgraph from live GraphCache."""
    from roc.db.graphdb import Node, Edge
    from roc.pipeline.temporal.sequencer import Frame

    # Find the Frame node with matching tick
    frame = Frame.find_one(tick=tick)
    if frame is None:
        raise ValueError(f"No Frame with tick={tick}")

    # BFS through Node/Edge objects directly
    visited_nodes = {}
    visited_edges = {}
    frontier = {frame.id}

    for _ in range(depth):
        next_frontier = set()
        for nid in frontier:
            node = Node.load(nid)
            for edge in node.src_edges:
                visited_edges[edge.id] = edge
                if edge.dst.id not in visited_nodes:
                    visited_nodes[edge.dst.id] = edge.dst
                    next_frontier.add(edge.dst.id)
            for edge in node.dst_edges:
                visited_edges[edge.id] = edge
                if edge.src.id not in visited_nodes:
                    visited_nodes[edge.src.id] = edge.src
                    next_frontier.add(edge.src.id)
        frontier = next_frontier

    # Convert to NetworkX for format compatibility
    G = nx.DiGraph()
    for nid, node in visited_nodes.items():
        G.add_node(nid, **node.to_dict())
    for eid, edge in visited_edges.items():
        G.add_edge(edge.src.id, edge.dst.id, **edge.to_dict())
    return G
```

This gives the dashboard real-time graph exploration -- click a node, fetch its
neighbors, expand on demand -- all from live in-memory data.

### 6. Action Map and Schema -- Direct Access

**Current**: Game subprocess POSTs action map to server via HTTP, saves schema to file.

**Proposed**: Game thread writes directly to DataStore:

```python
# In game initialization (within game thread):
action_map = _build_action_map(gym.actions)
data_store.set_action_map(run_name, action_map)  # direct call

schema = Schema()
data_store.set_schema(run_name, schema.to_dict())  # direct call
```

No HTTP, no file I/O for live runs. Files still written for archival.

### 7. Socket.io Notification

The existing pattern works unchanged. StepBuffer already notifies listeners after
each push. The listener in DataStore already calls `_notify_new_step()` which uses
`asyncio.run_coroutine_threadsafe()` to emit to the Socket.io event loop.

The only change: the push comes from the game thread (same process) instead of
from an HTTP handler thread. The `run_coroutine_threadsafe` pattern is thread-safe
regardless of which thread calls it.

### 8. Cooperative Shutdown

**Current**: Server sets `stop_requested` flag, HTTP response includes
`{"stop": true}`, game subprocess reads response and exits.

**Proposed**: Server sets `threading.Event`, game thread checks
`stop_event.is_set()` each step.

```python
# Server side (GameManager):
def stop_game(self) -> None:
    self._stop_event.set()
    if self._game_thread:
        self._game_thread.join(timeout=10)
        if self._game_thread.is_alive():
            logger.warning("Game thread did not exit within timeout")
            # Thread cannot be killed -- but it's a daemon thread,
            # so it will die when the server process exits

# Game side (gymnasium.py):
while game_num <= settings.num_games:
    if stop_event.is_set():
        break
    # ... step logic ...
```

**Note on unkillable threads**: Unlike subprocesses, threads cannot be forcefully
killed (no SIGTERM/SIGKILL equivalent). The game thread is marked as `daemon=True`
so it dies when the server process exits. If the thread hangs in NLE C code, the
10-second join timeout will expire and the server will log a warning. The thread
will continue running but the server remains responsive. This is acceptable because:

1. NLE steps complete in <100ms (never hangs in practice)
2. The stop_event check happens every step
3. Daemon threads die on process exit (server restart cleans up)

**Validated in prototype**: The NLE prototype tested stopping a game running 100
games (max 10000 steps each) mid-play. The stop signal was set after ~2585 ticks.
The thread exited cleanly, `thread.join()` returned, `thread_alive` was False, and
all accumulated frame data remained accessible. The stop latency was sub-step
(the thread checks the event between NLE steps).

## Thread Safety Analysis

### What Is Already Thread-Safe

| Component | Why |
|-----------|-----|
| StepBuffer | Has `_lock` and `_listener_lock` (separate locks, deadlock-proof) |
| DataStore | Has `_lock` and `_store_lock` (separate locks) |
| DuckLakeStore | Has `_lock` serializing all DuckDB access |
| RxPY Subject | Designed for concurrent subscriptions |
| Socket.io emission | Uses `run_coroutine_threadsafe` |
| ParquetExporter | Background thread with queue (designed for cross-thread use) |

### What Needs Thread Safety Added

#### A. GraphCache (Node/Edge LRU Cache) -- CRITICAL

The LRU cache (`cachetools.LRUCache`) is NOT thread-safe. Both the game thread
(creating/modifying nodes) and the API thread (reading nodes for graph queries)
will access it concurrently.

**Required change**: Add a `threading.RLock` to GraphCache:

```python
class GraphCache[CacheKey, CacheValue](LRUCache[CacheKey, CacheValue]):
    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)
```

**Why RLock**: Node operations may trigger nested cache access (e.g., loading an
edge triggers loading its src/dst nodes). RLock allows reentrant acquisition.

**Performance impact**: Minimal. LRU operations are O(1). Lock contention is low
because the game thread writes sequentially (one step at a time) and API reads are
infrequent (user clicks in dashboard).

#### B. Node/Edge ID Counters -- CRITICAL

Global counters `next_new_node` and `next_new_edge` use read-then-decrement without
synchronization.

**Required change**: Use `threading.Lock`:

```python
_id_lock = threading.Lock()

def _get_next_new_node_id() -> NodeId:
    global next_new_node
    with _id_lock:
        nid = next_new_node
        next_new_node = cast(NodeId, next_new_node - 1)
        return nid
```

#### C. Event._step_counts -- LOW RISK

Module-level dict tracking per-bus event counts. Concurrent increments could race.

**Required change**: Use `threading.Lock` or accept approximate counts:

```python
_step_counts_lock = threading.Lock()

@classmethod
def _increment_step_count(cls, bus_name: str) -> None:
    with cls._step_counts_lock:
        cls._step_counts[bus_name] = cls._step_counts.get(bus_name, 0) + 1
```

**Alternative**: Accept that counts may be slightly off during concurrent access.
These are telemetry counters, not correctness-critical.

#### D. EventBus Name Registry -- SETUP ONLY

`eventbus_names` global set checked during bus creation. Buses are created during
Component.init() which runs once on the game thread.

**Required change**: Add lock, or document that bus creation is single-threaded:

```python
_eventbus_lock = threading.Lock()

class EventBus:
    def __init__(self, name: str, ...):
        with _eventbus_lock:
            if name in eventbus_names:
                raise ValueError(...)
            eventbus_names.add(name)
```

#### E. Component Registry -- SETUP ONLY

`component_registry`, `loaded_components`, `component_set` accessed during
`Component.init()` and `Component.reset()`. These run on the game thread only.

**Required change**: Add lock for safety, even though access is single-threaded:

```python
_component_lock = threading.RLock()

class Component:
    @classmethod
    def init(cls):
        with _component_lock:
            # ... existing loading logic ...
```

#### F. GraphDB Singleton -- REQUIRES REDESIGN

`GraphDB.singleton()` has a classic double-check-locking race. But more importantly,
the mgclient connection is not thread-safe.

**Required changes**:

1. Thread-safe singleton creation:
```python
_graphdb_lock = threading.Lock()

@classmethod
def singleton(cls) -> GraphDB:
    global graph_db_singleton
    with _graphdb_lock:
        if not graph_db_singleton:
            graph_db_singleton = GraphDB()
    return graph_db_singleton
```

2. All query methods must hold a lock (they already serialize through `raw_fetch`
   and `raw_execute`, so a single lock suffices):
```python
def raw_fetch(self, query, *, params=None):
    with self._query_lock:
        # ... existing code ...
```

**Note**: For the thread-based architecture, the server's graph API will query
GraphCache (in-memory) not Memgraph (via mgclient). The GraphDB connection is used
primarily by the game thread for persist operations. API threads access GraphCache
only. This minimizes contention on the GraphDB lock.

### Summary: Thread Safety Changes

| Component | Change | Risk if Skipped | Effort |
|-----------|--------|-----------------|--------|
| GraphCache | Add RLock to LRU operations | Data corruption, crashes | Small |
| Node/Edge ID counters | Add Lock | Duplicate IDs | Trivial |
| Event._step_counts | Add Lock | Slightly wrong telemetry | Trivial |
| EventBus names | Add Lock | Duplicate bus names (crash) | Trivial |
| Component registry | Add RLock | Component conflicts | Small |
| GraphDB singleton | Add Lock + query lock | Connection corruption | Small |

Total: ~100 lines of lock additions across 3 files. All changes are additive (adding
locks around existing code) with no algorithmic changes.

## NLE Thread Safety Analysis

### The Documented Constraint

From `roc/game/CLAUDE.md`:
> "NLE embeds game state that cannot be cleanly reset within a single Python
> interpreter. A fresh process avoids all singleton/global cleanup issues between runs."

### Empirical Validation -- RESOLVED

The NLE thread prototype (`tmp/nle_thread_experiment.py`) directly tested this
concern and found **no issues**. The experiment ran 9 total game sessions in a
single server process:

| Test | What | Result |
|------|------|--------|
| Sequential run 1 | Single game, 50 steps | Clean |
| Sequential run 2 | Single game, 50 steps (after run 1 completed) | Clean |
| Sequential run 3 | Single game, 50 steps (after run 2 completed) | Clean |
| Multi-game run | 2 games via env.reset() in one thread, 30 steps each | Clean |
| Concurrent run A | 40 steps, parallel with B | Clean |
| Concurrent run B | 40 steps, parallel with A | Clean |
| Long-running stop | 100 games, stopped mid-play at tick ~2585 | Clean |
| Final validation | Single game after all above, 30 steps | Clean |

Every run produced valid game state (HP > 0 at start, valid depth, valid
position), zero errors, and clean thread shutdown. `env.close()` + `del env`
fully releases NLE's C state.

**Conclusion**: The subprocess isolation documented in CLAUDE.md was a
conservative precaution, not a hard requirement. NLE's C extensions clean up
properly when the environment is closed. The thread-based approach is viable.

The CLAUDE.md note should be updated when this design is implemented to reflect
the new threading model.

### GIL Considerations

NLE's C extensions release the GIL during computation (standard practice for C
extensions). This means:

- NLE `env.step()` runs in C without holding the GIL
- While NLE computes, the server thread (FastAPI) can run Python code freely
- GraphCache reads from the API thread won't be blocked by NLE computation
- The GIL only serializes pure Python code, not C extension work

This is favorable for the thread model -- NLE and the server run concurrently
for most of the wall-clock time.

**Observed in prototype**: The concurrent game test (two NLE instances in parallel
threads) showed both games completing their full step counts without slowdown.
The concurrent read test (40 HTTP reads during active NLE gameplay) returned all
200 responses with zero errors, confirming the GIL does not cause contention
between NLE stepping and FastAPI request handling.

## What Gets Removed

### Removed Code

| Component | What | Lines (est.) |
|-----------|------|-------------|
| `gymnasium.py` | `_push_step_to_server()` | ~30 |
| `gymnasium.py` | HTTP callback URL handling, SSL context | ~40 |
| `gymnasium.py` | `_server_stop` flag and response parsing | ~10 |
| `api_server.py` | `POST /api/internal/step` endpoint | ~20 |
| `api_server.py` | `POST /api/internal/action-map` endpoint | ~15 |
| `game_manager.py` | Subprocess spawning (`Popen`, cmd building) | ~50 |
| `game_manager.py` | Monitor thread (filesystem polling) | ~60 |
| `game_manager.py` | Log thread (stdout pipe reading) | ~30 |
| `game_manager.py` | SIGTERM/SIGKILL escalation | ~40 |
| `server_cli.py` | Callback URL construction | ~10 |
| `script.py` | Standalone `play` CLI (replaced by server-only) | ~20 |

**Total removed**: ~325 lines

### Removed Complexity

- No more JSON serialization/deserialization per step
- No more HTTP round-trip per step
- No more SSL context management for localhost callbacks
- No more cooperative shutdown via HTTP response body
- No more filesystem polling for run directory detection
- No more stdout pipe forwarding between processes
- No more process signal escalation (SIGTERM -> SIGKILL)
- No more `_convert_numpy()` for JSON compatibility
- No more `dataclasses.asdict()` conversion for HTTP payload

### What Stays

- `uv run play` CLI still works for standalone debugging (no server)
- DuckLake archival (OTel -> ParquetExporter -> DuckLake) unchanged
- Socket.io emission pattern unchanged
- StepBuffer ring buffer unchanged
- DataStore dual-tier query (live + historical) unchanged
- All REST API endpoints unchanged (except internal ones removed)

## What Gets Added

### New Code

| Component | What | Lines (est.) |
|-----------|------|-------------|
| `game_manager.py` | Thread spawning, join, Event-based stop | ~60 |
| `game_manager.py` | `_game_main()` entry point | ~40 |
| `graphdb.py` | GraphCache thread safety (RLock) | ~30 |
| `graphdb.py` | ID counter locks | ~10 |
| `graphdb.py` | GraphDB singleton lock | ~10 |
| `event.py` | Step counts lock, eventbus names lock | ~15 |
| `component.py` | Registry lock | ~15 |
| `graph_service.py` | Live graph query methods | ~80 |
| `graph_api.py` | Live vs historical routing | ~20 |

**Total added**: ~280 lines

### Net Change

~325 lines removed, ~280 lines added. Net reduction of ~45 lines while adding
live graph access -- a strictly better architecture.

## Data Persistence After Game Thread Exits

This is the key insight from the prototype: when a thread dies, Python objects it
created remain on the heap until garbage collected. Since GraphCache holds strong
references to Node/Edge objects, they persist indefinitely.

### What Persists

| Data | Location | Lifetime |
|------|----------|----------|
| Node objects | GraphCache (LRU) | Until cache eviction or server restart |
| Edge objects | GraphCache (LRU) | Until cache eviction or server restart |
| StepBuffer contents | deque in StepBuffer | Until capacity exceeded or cleared |
| DataStore indices | _GameIndex dicts | Until new game session starts |
| Action map | DataStore._action_maps | Until server restart |
| Schema | DataStore (or file) | Until server restart |

### What Gets Cleaned Up

| Data | When | How |
|------|------|-----|
| Pipeline components | Thread exit | `Component.reset()` in finally block |
| EventBus subscriptions | Thread exit | Component shutdown disposes subscriptions |
| NLE environment | Thread exit | `env.close()` + GC |
| RxPY ThreadPoolScheduler | Persists | Shared across game runs (reusable) |

### Graph Data Lifecycle

```
Game running:     GraphCache populated by pipeline
                  API can query live nodes/edges
                  Dashboard shows real-time graph

Game stopped:     GraphCache still populated (same objects)
                  API continues to serve graph queries
                  Dashboard continues to work

New game starts:  Old GraphCache entries coexist with new ones
                  LRU eviction naturally clears old entries
                  OR: explicit cache clear on new game start

Server restart:   All in-memory data lost
                  Historical data in DuckLake/Parquet survives
                  graph.json archive written at game end survives
```

## Migration Strategy

### Phase 1: Thread Safety Foundation

Add locks to GraphCache, ID counters, EventBus names, Component registry, GraphDB.
These changes are backward-compatible -- they add safety without changing behavior.
The subprocess model continues to work.

**Test**: Run existing test suite with lock instrumentation to verify no deadlocks.

### Phase 2: GameManager Thread Mode

Modify GameManager to spawn threads instead of subprocesses. Keep the subprocess
code path available behind a config flag for fallback.

```python
class GameManager:
    def __init__(self, ..., use_thread: bool = True):
        self._use_thread = use_thread

    def start_game(self, num_games):
        if self._use_thread:
            self._start_thread(num_games)
        else:
            self._start_subprocess(num_games)  # existing code
```

**Test**: Run full game with thread mode, compare StepData output against
subprocess mode. Must be identical.

### Phase 3: Remove HTTP Callback

Once thread mode is validated, modify `_push_dashboard_data()` to push directly
to StepBuffer. Remove `_push_step_to_server()` and the `/api/internal/step`
endpoint.

**Test**: Dashboard displays same data as before. Socket.io events fire correctly.

### Phase 4: Live Graph Access

Add live graph query methods to GraphService. Modify graph API endpoints to route
to live GraphCache when the run is active.

**Test**: Dashboard graph visualization works with live data. Click-to-expand
fetches real-time neighbors.

### Phase 5: Cleanup

Remove subprocess code path, monitor thread, log thread, signal escalation.
Remove the config flag from Phase 2.

## Fallback Architecture

The NLE prototype validated that pure threading works (see "Empirical Validation"
above). The hybrid subprocess fallback described below is retained for reference
but is **not expected to be needed**.

If a future NLE or Gymnasium version introduces C state issues incompatible with
threading, a hybrid approach preserves most benefits:

```
+-------------------------------------------------------------------+
|                     Server Process                                  |
|                                                                     |
|  +--------------------+                  +--------------------+     |
|  |  Pipeline Thread    |   shared heap   |  FastAPI + Socket.io|    |
|  |                     | <=============> |                     |    |
|  |  Components         |   GraphCache    |  REST API handlers  |    |
|  |  State + OTel       |   Node/Edge     |  DataStore          |    |
|  +--------------------+                  +--------------------+    |
|         ^                                                           |
|         | pipe (raw observations + actions)                         |
|         v                                                           |
|  +--------------------+                                             |
|  |  NLE Subprocess     |                                            |
|  |  env.reset/step     |                                            |
|  |  Raw obs out        |                                            |
|  |  Action ID in       |                                            |
|  +--------------------+                                             |
+-------------------------------------------------------------------+
```

**Trade-off**: Still requires serializing raw observations across process boundary
(~50KB/step for screen + stats), but GraphCache and pipeline run in-process with
the server. This is more complex than pure threading but preserves the key benefit
(live graph access).

## Open Questions

1. ~~**NLE C state between runs**~~ **RESOLVED.** The NLE prototype ran 9 game
   sessions (sequential, concurrent, multi-game, stopped mid-play) in a single
   process with zero C state issues. `env.close()` + `del env` is sufficient.

2. **Game logging**: Currently, subprocess stdout is piped to loguru via a log
   thread. With threading, the game thread's loguru output goes directly to the
   same logger. This is simpler but means game logs interleave with server logs.
   May want a logger filter to distinguish game vs server log lines.

3. **Component.init() between runs**: Currently, each subprocess starts fresh.
   With threading, we need to ensure `Component.reset()` + `Component.init()`
   cleanly reinitializes all components. The `restore_registries` test fixture
   does this in tests -- verify it works in production.

4. **EventBus name cleanup**: EventBus names are globally unique. Between game
   runs, the bus names from the previous run must be cleared. `EventBus.clear_names()`
   exists for tests. Need to call it in the game thread cleanup path.

5. **GraphCache sizing**: With shared memory, the cache serves both game pipeline
   and API queries. The default size (2^30) is enormous and should be fine, but
   monitor memory usage with long-running servers.

6. **Error isolation**: A crash in the game thread (e.g., unhandled exception in
   NLE) does not kill the server process (daemon threads don't propagate
   exceptions). The server remains responsive. But: a segfault in NLE C code WILL
   crash the entire server process. This is the main risk of in-process NLE.
   Mitigation: NLE segfaults are extremely rare in practice, and none were
   observed across 9 game sessions in the prototype.

## Prototype Artifacts

| File | Tests | What It Proves |
|------|-------|---------------|
| `tmp/thread_shared_memory_experiment.py` | 39/39 | Generic thread shared memory with FastAPI |
| `tmp/test_thread_experiment.sh` | 39/39 | Test suite for generic prototype |
| `tmp/nle_thread_experiment.py` | 63/63 | NLE in threads with shared game state |
| `tmp/test_nle_thread_experiment.sh` | 63/63 | Test suite for NLE prototype |

Combined: **102 tests** covering shared memory, NLE compatibility, concurrent
access, data persistence, stop signaling, sequential runs, and error handling.
