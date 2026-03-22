# Live Step Performance Fix

## Problem

When a game is running via subprocess (Game Menu), stepping through frames in
the dashboard takes 1-65 seconds instead of ~150ms. The dashboard becomes
unusable during live games.

## Root Cause

`DataStore.get_step_data()` has a log supplementation path (lines 326-335):

```python
buf_data = self._live_buffer.get_step(step)  # instant -- in memory
if buf_data is not None:
    if buf_data.logs is None:                  # always true for live steps
        store = self._get_run_store(run_name)  # opens DuckDB connection
        logs_df = store.get_step(step, "logs") # BLOCKS on file lock
```

Live-pushed StepData never includes `logs` (log messages go through a separate
OTel logger to DuckDB, not through the game loop's StepData assembly). So
`buf_data.logs is None` is **always true** for live steps.

In subprocess mode, `_get_run_store()` opens a new DuckDB connection to the
same database the game subprocess is writing to. DuckDB uses file-level
locking, so every log query blocks until the subprocess releases the lock
(~1.3 seconds per step).

The prefetcher makes this catastrophic: it sends 50-step batch requests, each
step triggers a 1.3s DuckDB wait, totaling ~65 seconds per batch. The browser
aborts and retries, but the server keeps processing abandoned requests
(FastAPI sync endpoints don't detect client disconnects). This floods the
thread pool and starves all other requests -- even for completely different
historical runs.

## Evidence

Server-side timing logs during a live game:
```
GET steps batch (50 steps) total=64497.2ms
GET steps batch (50 steps) total=64415.6ms
GET steps batch (50 steps) total=64246.7ms
```

After stopping the game and restarting the server:
```
step 1 (cold): 258ms
step 2+:       ~150ms
```

## Fix

### Skip DuckDB log supplementation in subprocess mode

In `DataStore.get_step_data()`, only attempt log supplementation when
`_live_store` is set (in-process mode with shared DuckDB connection):

```python
if buf_data.logs is None and self._live_store is not None:
    # In-process mode: shared connection, no lock contention
    try:
        store = self._get_run_store(run_name)
        logs_df = store.get_step(step, "logs")
        ...
```

In subprocess mode (`_live_store is None`), the DuckDB file is locked by
the subprocess. Skip the query entirely.

## UX Impact

| Scenario | Before fix | After fix |
|----------|-----------|-----------|
| Stepping during live subprocess game | 1-65s per step (unusable) | ~10ms per step (instant from StepBuffer) |
| Stepping during in-process game (`uv run play`) | ~150ms (shared DuckDB, no contention) | Unchanged |
| Stepping on historical run (no game) | ~150ms | Unchanged |
| Stepping on historical run while game is running | 1-65s (thread pool starved by zombie requests) | ~150ms (no DuckDB contention from log queries) |
| **Log Messages panel during subprocess game** | **Shows "No log data" (queries block/timeout, never succeed)** | **Shows "No log data" (same visible behavior, but no blocking)** |
| Log Messages panel during in-process game | Shows log messages | Unchanged |
| Log Messages panel on historical runs | Shows log messages | Unchanged |
| Browsing a historical run after game ends | Full data including logs | Unchanged |

The Log Messages panel during subprocess games shows "No log data" both before
and after this fix. The difference is that before the fix, the server spends
1.3 seconds per step **trying and failing** to get logs; after the fix, it
skips the attempt entirely.

## Data Impact

**None.** This fix only changes the read path (what the dashboard server
queries). No changes to:

- What the game subprocess writes to DuckDB/DuckLake
- What the game subprocess sends via HTTP callback
- What gets stored in parquet files
- What data is available when browsing historical runs after the game ends
- The StepBuffer contents or DataStore indices

All data continues to be written to DuckLake by the game subprocess. The only
change is that the dashboard server stops trying to read from a locked DuckDB
file during live games.

## Future: Logs in Live Subprocess Games

To get logs during live subprocess games without DuckDB, options include:

a. **Include logs in HTTP callback StepData**: The game subprocess would need
   to capture recent loguru output and attach it to each StepData POST. This
   requires changes to `gymnasium.py`'s callback assembly.

b. **Read JSONL debug log**: When `roc_debug_log=true`, the game writes a
   JSONL file. The dashboard could tail this file for log messages. But this
   is a debug-only feature, not always enabled.

c. **Forward logs via remote logger**: The game's loguru output could be sent
   to the remote logger, and the dashboard could query it. This is already
   partially wired up but not integrated with the Log Messages panel.

These are all follow-up work. The immediate fix unblocks the dashboard from
being unusable during live games.

## Files Changed

| File | Change |
|------|--------|
| `roc/reporting/data_store.py` | Add `self._live_store is not None` guard |
