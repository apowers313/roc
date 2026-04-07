# Reporting

## Why This Design

Reporting serves two purposes: scientific archival and runtime debugging. ROC is a
research system, so every run must produce a permanent, self-contained data archive
that can be analyzed months or years later. The archive format must outlive any
particular tool -- if DuckDB or Grafana disappear, the data must still be readable.

The secondary purpose is debugging. The agent's graph structures are complex and
growing more complex. Understanding what nodes and edges were created, when, and
why -- in the context of the game state at that moment -- is essential for iterating
on ROC's data structures. The dashboard is a scientific instrument, not just a dev tool.

## Key Decisions

- **Parquet as primary archive format** -- chosen for broad ecosystem support and
  long-term accessibility. Dozens of tools read Parquet today; it is an open columnar
  format likely to remain readable for decades. This is a scientific reproducibility
  requirement, not a performance choice.

- **DuckLake over raw DuckDB** -- we originally used DuckDB directly but hit
  reader/writer contention that caused segfaults. DuckLake wraps DuckDB with a
  SQLite-based catalog and Parquet storage, eliminating the contention by separating
  metadata from data files.

- **Data inlining (row_limit=500)** -- without this, each small INSERT creates a
  separate Parquet file. A single run produced ~7000 tiny files, causing 5-11s query
  times. Data inlining stores small batches in the catalog; periodic CHECKPOINT flushes
  them to merged Parquet files.

- **Background writer thread** -- ParquetExporter queues records and drains them in a
  background thread ("ducklake-writer"). The game loop must never block on database
  writes. All exporter and listener errors are silently caught -- a reporting failure
  must never crash or slow the experiment.

- **OTel as the event pipeline** -- all state emissions flow through OpenTelemetry
  LogRecords. This unifies local archival (Parquet), remote monitoring (Grafana/Loki),
  live debugging (Remote Logger MCP), and dashboard display through a single emission
  path. Pipeline components do not need to know which consumers exist.

- **state.py imports nle directly** -- this is known technical debt violating
  Architectural Invariant #2 (game-agnostic core). It renders raw screen data for the
  dashboard. Should eventually receive rendered data from the perception bus instead.
  Do not extend this pattern.

## Invariants

- **Game loop never blocks on DB writes.** ParquetExporter queues records; the
  background thread drains them. If you add synchronous DuckLake calls in the export
  path, the game loop stalls and experiments take longer. Every exception handler in
  the export/listener chain catches and discards -- never re-raise.

- **All DuckDB access goes through DuckLakeStore's lock.** DuckDB connections are not
  thread-safe. The `_lock` in DuckLakeStore serializes all reads and writes. Bypassing
  it (e.g., calling `_conn.execute` directly from another thread) causes segfaults or
  corrupted results. Use `query_df` or `query_one` for thread-safe reads.

- **StepBuffer listener notification happens outside the data lock.** StepBuffer has
  separate `_lock` (data) and `_listener_lock` (callbacks). Notifying listeners inside
  the data lock deadlocks if a listener tries to read the buffer.

- **Socket.io emits use run_coroutine_threadsafe.** The game runs on a regular thread;
  Socket.io runs on an asyncio event loop. Cross-thread communication must go through
  `asyncio.run_coroutine_threadsafe(_sio_loop)`. Calling `await sio.emit()` from the
  game thread deadlocks.

## Non-Obvious Behavior

- **Two-phase Observability initialization.** `Observability.init()` runs at module
  import time (for decorator-based tracers) but does NOT create the DuckLake store.
  The store is created later when `roc.init()` calls `Observability.init(enable_parquet
  =True)`. This prevents empty run directories from appearing when modules are merely
  imported.

- **Step counter logic in ParquetExporter.** The step counter increments on
  `roc.screen` events. If no screen event arrives (e.g., `emit_state_screen=False`),
  it falls back to incrementing on `roc.game_metrics`. The `_step_incremented` flag
  prevents double-counting when both arrive in the same step. Breaking this logic
  causes step numbers to drift from reality.

- **Live data two-tier lookup.** DataStore checks the in-memory StepBuffer first
  (instant), then falls back to RunStore/DuckLake for evicted steps. For live runs, it
  also supplements missing log data from DuckLake when the buffer hit lacks logs. This
  means live and historical queries hit different code paths with different performance
  characteristics.

- **Attention bus subscribes synchronously.** StateComponent's attention handler uses
  `subject.subscribe()` directly (no ThreadPoolScheduler) so that `states.saliency` is
  updated inline when VisionAttention emits. Other bus handlers use the normal filtered
  listen pattern.

- **CHECKPOINT interval (200 steps).** ParquetExporter runs CHECKPOINT every 200 steps
  to flush inlined data to Parquet and merge small files. This was a performance tuning
  choice and should eventually become a Config option.

## Anti-Patterns

- **Do not add synchronous I/O in any OTel exporter's export() path.** The exporter
  runs on the OTel SDK's thread. Blocking calls (HTTP without timeout, disk sync, lock
  contention) stall all log processing. Queue work for a background thread instead.

- **Do not bypass DuckLakeStore for raw DuckDB access.** Other modules must not import
  duckdb or call the connection directly. DuckLakeStore handles locking, schema
  evolution, and catalog attachment. Direct access causes the same reader/writer
  contention that prompted the DuckLake migration.

- **Do not add nle or game-specific imports to reporting modules.** state.py's nle
  import is legacy debt, not a pattern to follow. New reporting code must work with
  abstract data structures from the pipeline.

- **Do not create additional DuckLakeStore instances for the same run directory.** Two
  stores pointing at the same catalog cause write contention. The live game writer and
  dashboard reader share one store instance (passed via `set_live_session`).

## Interfaces

- **StateComponent listens on all pipeline buses** (attention, object, intrinsic,
  significance, action, transformer, predict, perception). It is a Component but NOT
  auto-loaded -- initialized explicitly by `State.init()`.

- **Game thread -> StepBuffer**: The game pushes StepData directly to the shared
  StepBuffer via function call (zero serialization). Stop signaling uses
  `threading.Event`.

- **API server -> browser**: Socket.io `new_step` pushes full StepData. REST endpoints
  serve historical and buffered data.

- **ParquetExporter routes by event name**: `roc.screen` -> screens table,
  `roc.attention.saliency` -> saliency, `roc.game_metrics` -> metrics, other named
  events -> events, unnamed (loguru) -> logs.
