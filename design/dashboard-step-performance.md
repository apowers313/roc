# Dashboard Step Performance

Date: 2026-03-20
Git HEAD: `1ac9872` (chore: delint) + uncommitted changes to reporting/query batching and prefetch tuning

## Architecture Overview

The dashboard displays historical game data by fetching step data from a
FastAPI server. The data flow for stepping through frames is:

```
User clicks Next
    |
    v
React state: step = step + 1
    |
    v
useStepData(run, step, game)          -- TanStack Query hook
    |
    +-- cache hit? -----> render immediately (~10ms)
    |
    +-- cache miss? ----> fetchStep(run, step, game)
                              |
                              v
                          GET /api/runs/{run}/step/{step}
                              |
                              v
                          api_server.py: _get_step_data()
                              |
                              v
                          RunStore.get_step_data(step)
                              |
                              v
                          DuckLakeStore.query_step_batch([step])
                              |
                              v
                          5x SELECT * FROM lake."table" WHERE step IN (...)
                              |
                              v
                          DuckDB reads parquet files
```

## Current Performance Numbers (2026-03-20)

### Browser-side (Playwright measurements)

| Scenario | Time | Notes |
|----------|------|-------|
| Step from TanStack cache | 10-11ms | Includes React render |
| Step cold fetch (standalone server) | 430-500ms | No game contention |
| Step cold fetch (game running) | 800-1100ms | DuckDB lock contention |
| 10-step batch endpoint | 500-1300ms total | 50-130ms/step |

### Server-side (Python profiling)

Measured on run `20260319092311-prideful-mathew-tremann` (510 steps, parquet files):

| Operation | Time | Notes |
|-----------|------|-------|
| `get_step_data` single step | 47-56ms | After warmup, via `query_step_batch` |
| `get_steps_data` 10-step batch | 69ms total | 6.9ms/step, `step IN (...)` |
| `step_range` query | <1ms | Cached after first call |

Measured on run `20260316232939-patchiest-arnaldo-pappas` (6214 steps, ALL data inlined in 15MB catalog, no parquet files):

| Operation | Time | Notes |
|-----------|------|-------|
| `get_step_data` single step | 700-900ms | DuckLake inline data scan |
| Per-table breakdown: screens | 132ms | 1 row |
| Per-table breakdown: saliency | 114ms | 1 row |
| Per-table breakdown: events | 181ms | 7 rows |
| Per-table breakdown: metrics | 6ms | 1 row |
| Per-table breakdown: logs | 8ms | 3 rows |

### Performance test results (pytest)

Test fixture: 500 steps with checkpoints every 50 steps (creates parquet files).

| Test | Threshold | Actual |
|------|-----------|--------|
| `TestPerformanceRealistic::test_single_step_under_limit` | <100ms worst | PASS |
| `TestPerformanceRealistic::test_batch_steps_under_limit` | <500ms for 10 steps | PASS |
| `TestPerformanceRealistic::test_batch_faster_than_sequential` | batch < 75% of sequential | PASS |
| `TestPerformanceHistorical::test_get_step_data_under_limit` | <100ms worst (15-step fixture) | PASS |
| `TestPerformanceLive::test_step_buffer_get_step_under_limit` | <100ms worst (in-memory) | PASS |

## Caching Strategy

### Browser-side: TanStack Query cache

- **Query key**: `["step", run, step, game]`
- **staleTime**: `Infinity` (step data is immutable for historical runs)
- **placeholderData**: `keepPreviousData` (prevents flicker during navigation)
- **retry**: `false` (no retry on rapid navigation cancellations)

### Prefetch window: `usePrefetchWindow` hook

Located in `dashboard-ui/src/hooks/usePrefetchWindow.ts`.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `DEFAULT_RADIUS` | 100 | Prefetch +/-100 steps from current position |
| `BATCH_SIZE` | 50 | Steps per HTTP batch request |
| `DEBOUNCE_MS` | 50 | Debounce before starting prefetch sweep |

The prefetcher:
1. Waits 50ms after the step changes (debounce)
2. Builds a list of uncached steps radiating outward from center
3. Fetches them in batches of 50 via `GET /api/runs/{run}/steps?steps=...`
4. Populates the TanStack cache with each result
5. Aborts in-flight requests if the user navigates (AbortController)

With 100 steps in each direction and 50-step batches, the cache fills in
4 HTTP requests (2 batches forward + 2 batches backward). At ~500ms per
batch, full cache population takes ~2 seconds.

### Server-side: No per-step cache

The API server does not cache step data. Each request queries DuckLake.
The `RunStore` instance is cached per run (in `_run_stores` dict), which
keeps the DuckDB connection and DuckLake catalog attachment alive.

## Known Bottlenecks

### 1. DuckLake catalog overhead (~100ms per table query)

Each `SELECT * FROM lake."table" WHERE step = ?` incurs DuckLake metadata
resolution overhead. This is the dominant cost for single-step fetches.

**Mitigation**: `DuckLakeStore.query_step_batch()` queries all 5 tables
under a single lock acquisition. `RunStore.get_steps_data()` uses
`step IN (...)` to scan each table once for multiple steps.

**Evidence**: Single-step query takes ~50ms (5 tables at ~10ms each under
one lock). Sequential 5 separate `get_step()` calls took ~500ms (5 x
~100ms due to per-call lock acquisition + catalog resolution).

### 2. Inlined data in old DuckLake catalogs

Runs created before checkpointing was implemented (or where checkpoints
failed) have all data inlined in the DuckLake catalog SQLite file instead
of separate parquet files. These catalogs are 15-26MB and queries scan
the entire inline store.

**Evidence**: Run `20260316232939-patchiest-arnaldo-pappas` (6214 steps,
15MB catalog, 0 parquet files) takes 700-900ms per step. Run
`20260319092311-prideful-mathew-tremann` (510 steps, 6MB catalog, parquet
files present) takes 47-56ms per step.

**Mitigation**: The `ParquetExporter` checkpoints every 200 steps
(`checkpoint_interval=200`), flushing inlined data to parquet files.
`data_inlining_row_limit=500` controls the threshold. Old runs could be
fixed by opening and checkpointing them, but this has not been implemented.

### 3. DuckDB lock contention with running game

When a game is running, the game's DuckLakeStore holds the DuckDB lock
for writes (parquet inserts, checkpoints). API requests must wait for the
lock, adding 300-600ms to each request.

**Evidence**: Same run fetched at 430ms standalone vs 800-1100ms with game
running.

**Mitigation**: The live run uses `StepBuffer` (in-memory ring buffer,
100K capacity) for the most recent steps, bypassing DuckDB entirely. Only
historical run playback or evicted live steps hit DuckDB.

### 4. First-step cold fetch on page load

When navigating to a historical run URL, the first step is always a cold
fetch (~500ms) because the TanStack cache is empty. The prefetcher starts
50ms later but the initial step is fetched by `useStepData` directly.

Steps 2-3 after page load may also be cold if the prefetcher hasn't
finished its first batch yet.

**Mitigation**: `keepPreviousData` in `useStepData` shows the previous
step's data while the new step loads, preventing blank/flicker. The 50ms
debounce and 50-step batches minimize the cold window.

### 5. Large response payloads (~80KB per step)

Each step response includes screen data (2D char + color arrays), saliency
maps, events, metrics, and logs. The screen alone is ~40KB. JSON
serialization and transfer add latency.

**Not yet mitigated**: Could compress (gzip), use binary format, or split
screen data into a separate endpoint fetched lazily.

## Key Files

| File | Purpose |
|------|---------|
| `dashboard-ui/src/hooks/usePrefetchWindow.ts` | Browser prefetch logic (radius, batch size, debounce) |
| `dashboard-ui/src/api/queries.ts` | TanStack Query hooks, cache config (staleTime, keepPreviousData) |
| `dashboard-ui/src/api/client.ts` | `fetchStep`, `fetchStepsBatch` HTTP client functions |
| `roc/reporting/api_server.py` | FastAPI endpoints, `_get_step_data`, batch endpoint |
| `roc/reporting/run_store.py` | `get_step_data`, `get_steps_data`, `_assemble_step` |
| `roc/reporting/ducklake_store.py` | `query_step_batch` (batched multi-table query) |
| `roc/reporting/parquet_exporter.py` | Checkpoint interval, data inlining config |
| `tests/unit/test_run_store.py` | `TestPerformanceRealistic`, `TestPerformanceHistorical` |

## Instrumentation

### Server-side timing (existing)

`api_server.py` line 326 logs per-request timing:
```
GET step {step} fetch={fetch_ms:.1f}ms serialize={serialize_ms:.1f}ms total={total_ms:.1f}ms
```

Visible in the game server's stderr logs via `servherd logs roc-game --error`.

### Browser-side (not yet instrumented)

No step-timing instrumentation in the browser. Performance was measured
via Playwright `page.evaluate()` with `performance.now()` around button
clicks.

**Recommended future instrumentation**:
- Remote Logger MCP: emit `step-start` and `step-rendered` events from
  `useStepData` and `App.tsx` render cycle
- Track cache hit/miss ratio in the prefetch window
- Track prefetch completion time (how long until the full +/-100 window
  is populated)

### Performance tests (existing)

In `tests/unit/test_run_store.py`:

- `TestPerformanceRealistic` uses a 500-step DuckLake store with
  checkpoints every 50 steps (realistic parquet file count)
- Tests: single step <100ms, 10-step batch <500ms, batch faster than
  sequential
- `large_populated_store` fixture creates the test data

These tests catch regressions in the `RunStore` -> `DuckLakeStore` query
path but do NOT test:
- Browser-side cache/prefetch behavior
- API server latency under game contention
- Response serialization time
- Network transfer time

### Grafana dashboards

The ROC Grafana dashboard (`uid: eed6q6f1c94owa`) has:
- Observations/sec rate
- CPU/memory utilization
- Pyroscope CPU flamegraphs
- Tempo trace spans for game pipeline operations

These instrument the game pipeline, NOT the dashboard API. Adding
API-specific metrics (request latency histogram, cache hit rate) would
require new Prometheus instrumentation in `api_server.py`.

## Historical Context

### Original implementation (commit `392741e`)

`RunStore` used direct `read_parquet()` via in-memory DuckDB -- no DuckLake
catalog. Fast for small files but fell apart with many small parquet files
(7000+ files = 5-11s query times).

### DuckLake migration (commit `16ec21d`)

Introduced DuckLake catalogs with data inlining and periodic checkpoints.
Solved the many-small-files problem but added per-query catalog overhead.

### React dashboard (commit `c9c0a8c`)

Replaced Panel/Bokeh with React + FastAPI + Socket.io. Added TanStack Query
caching and `usePrefetchWindow`.

### Performance test commit (`c46a486`)

Added Playwright e2e performance tests and per-step API timing logs. Changed
live run from path-based `RunStore(run_dir)` to `RunStore(DuckLakeStore(...))`.

### This session's changes (2026-03-20, uncommitted)

1. `DuckLakeStore.query_step_batch()`: queries all tables for one or more
   steps under a single lock acquisition with `step IN (...)`
2. `RunStore.get_step_data()`: uses `query_step_batch` instead of 5 separate
   `get_step()` calls
3. `RunStore.get_steps_data()`: batch method for multiple steps
4. `api_server.py`: batch endpoint uses `get_steps_data` for historical runs
5. `usePrefetchWindow.ts`: `BATCH_SIZE` 5->50, `DEBOUNCE_MS` 300->50
6. `tests/unit/test_run_store.py`: `TestPerformanceRealistic` with 500-step
   fixture, batch performance tests

## Optimization Opportunities (Not Yet Implemented)

1. **Server-side step cache**: LRU cache in `api_server.py` for historical
   runs (immutable data). Would eliminate DuckDB queries for repeated access.

2. **Gzip compression**: Step responses are ~80KB JSON. Gzip could reduce
   to ~10KB, cutting transfer time.

3. **Checkpoint old runs**: Script to open old inlined-data catalogs and run
   `CHECKPOINT` to flush to parquet files.

4. **Split screen data**: Fetch screen (40KB) and metadata (40KB) separately.
   Metadata loads first for fast UI update, screen loads lazily.

5. **DuckDB connection pooling**: One DuckDB connection per historical run is
   expensive. A shared connection pool with catalog switching could reduce
   overhead.

6. **Browser performance instrumentation**: Emit timing events via Remote
   Logger for step-start, cache-hit/miss, step-rendered, and prefetch
   completion.
