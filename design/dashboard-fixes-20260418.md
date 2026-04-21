# Dashboard Reliability Fixes -- 2026-04-18

Two problems: (1) the PWA auto-plays on cold load causing blank-screen flickering,
and (2) errors are systematically swallowed at every layer of the dashboard stack,
making failures invisible to users and developers.

Part 1 is the immediate bug fix. Part 2 is a comprehensive silent-error audit with
fixes to implement in a follow-up session.

---

## Part 1: PWA Cold-Load Auto-Play Fix

### Root Cause

`dashboard-ui/src/state/context.tsx` initializes `playing` from sessionStorage:

```typescript
const [rawPlaying, setRawPlaying] = useState(readSessionPlaying); // returns true if saved
```

On a cold page load (iOS page-discard, manual reload), `playing=true` is restored but
the TanStack Query cache is empty. The playback timer in `TransportBar.tsx` starts
advancing steps immediately, causing blank screens as each step's data takes 1-3s to
fetch.

Remote logger evidence:
```
16:07:31.163  step=10  hasData=true        -- renders fine
16:07:31.599  step=11  hasData=false       -- auto-advanced, screen goes blank
16:07:31.730  step=11  stepIsLoading=true  -- still blank
16:07:31.898  step=11  stepIsLoading=true  -- still blank (3s until server responds)
```

### Fix: Remove `playing` persistence from sessionStorage

Playing is a transient UI action -- it should never auto-resume on cold load. The user
presses play when ready. Speed, autoFollow, and scroll position persistence remain
(they have no side effects on cold load).

**Changes to `dashboard-ui/src/state/context.tsx`:**

1. Delete `SESSION_PLAYING_KEY` constant
2. Delete `readSessionPlaying()` function
3. Delete `writeSessionPlaying()` function
4. Change `useState(readSessionPlaying)` to `useState(initialPlayback.playing)`
5. Remove `writeSessionPlaying(p)` call from `setPlaying` callback
6. Remove `writeSessionPlaying(playing)` from `freeze` event flush; remove `playing`
   from that effect's dependency array

**Regression tests in `dashboard-ui/src/state/context.test.tsx`:**

1. "playing is always false on mount regardless of sessionStorage" -- write
   `playing=true` to sessionStorage before mount, assert `playing === false`
2. "setPlaying does not write to sessionStorage" -- call `setPlaying(true)`, assert
   sessionStorage has no playing key

**Documentation updates:**

- Update `dashboard-ui/CLAUDE.md` Key Decisions: add entry explaining why `playing` is
  not persisted (cold-load auto-play causes blank screens because data cache is empty)

**Also fixed during investigation:**

- `dashboard-ui/src/hooks/useRemoteLogger.ts`: default port was 9080, remote logger
  runs on 9083. Already applied.

### Verification

1. `cd dashboard-ui && npx vitest run src/state/context.test.tsx` -- new + existing pass
2. `cd dashboard-ui && npx vitest run` -- full test suite passes
3. Remote logger: reload dashboard, confirm no `stepIsLoading=true` with auto-advancing
4. Playwright: navigate to dashboard, verify `playing` is false on load

---

## Part 2: Silent Error Audit

Comprehensive audit of dashboard frontend, server backend, and cross-boundary error
handling. Every finding below is a place where errors are caught and swallowed without
user notification or recovery.

### CRITICAL: Broken features with zero user feedback

**C1. MenuBar.tsx:54-76 -- Game start/stop errors swallowed**

`startGame()` and `stopGame()` have `catch { /* ignore */ }`. User clicks Start Game,
network error or server 500 occurs, button clears loading state, nothing happens. No
toast, no error message, no retry.

Fix: catch errors, show Mantine notification with error message. Log to remote logger.

**C2. api_server.py:346-351 -- Batch step query returns 200 {} on any error**

`get_steps_batch` catches any Exception and returns `batch = {}` with HTTP 200.
Client receives empty dict indistinguishable from "steps not emitted yet". DuckLake
corruption, lock timeout, or disk-full errors are invisible.

Fix: log the exception. Return HTTP 500 with error detail instead of empty 200. Let
the client distinguish "error" from "no data".

**C3. useRunSubscription.ts:129-155 -- Game status fetch swallowed**

Initial `/api/game/status` fetch has `.catch(() => {})`. Comment says "Socket.io
handler is the fallback path" but Socket.io only fires on state *transitions*, not on
cold-load state queries. If fetch fails, `gameState` stays null forever, GO LIVE
button is inoperable.

Fix: retry the fetch (3 attempts with backoff). If all fail, log to remote logger.

**C4. client.ts:16-17 -- fetchJson() discards response body on non-2xx**

`fetchJson()` throws `Error("API error: ${status} ${statusText}")` on non-OK
responses but never reads the response body. Server sends structured error details
(`{"detail": "...", "status": "error"}`) but the client discards them.

Fix: read `res.json()` or `res.text()` before throwing. Include server error detail in
the thrown Error message.

### HIGH: Stale or inconsistent state

**H1. TransportBar.tsx -- 3x step-range fetch errors have `.catch(() => {})`**

Three separate locations (lines ~240, ~360, ~391) catch step-range fetch errors with
empty handlers. Run/game selection appears to work but slider bounds are stale or
wrong.

Fix: log to console.error at minimum. Consider showing a brief warning if step range
can't be loaded for the selected run/game.

**H2. useRunSubscription.ts:38-87 -- Socket.io has zero error/disconnect listeners**

No handlers for `connect_error`, `disconnect`, `reconnect_failed`. When WebSocket
drops (which happens frequently per servherd error logs), `step_added` events stop,
the UI goes stale, and the user has no indication.

Fix: add `connect`, `disconnect`, `connect_error` listeners on the socket singleton.
Expose reactive `connected` state. On reconnect, invalidate step-range queries.

**H3. App.tsx:329 -- `connected` hardcoded to `true`**

Green dot in TransportBar always shows connected. Should reflect actual socket state
from H2 fix.

Fix: replace hardcoded `true` with reactive state from `useRunSubscription`.

**H4. useRemoteLogger.ts -- Remote logger connection failure is silent**

`RemoteLogClient` POSTs to the configured server URL. If the URL is wrong (port
mismatch, server down), the POST fails silently -- no console error, no indication
that remote logging is not working. During this session the default port was 9080 but
the server was on 9083; zero logs were captured for ~30 minutes with no indication.

Fix: add a health check on client init (fetch /status). Log a console.warn if the
remote logger is unreachable. Consider a periodic liveness check.

**H5. ducklake_store.py:209-210 -- Table query fails silently in batch**

`_query_table_for_steps` catches Exception with bare `pass`. If one table (e.g.,
`saliency`) fails while others succeed, the step appears to have partial data. User
sees missing fields without indication that a query failed.

Fix: log the exception with table name and step range. Consider adding a `warnings`
field to the batch response.

### MEDIUM: Data loss or misleading UI

**M1. useBookmarks.ts:39 -- Bookmark save is fire-and-forget**

`void saveBookmarks(run, updated)` discards the promise. If the POST fails, the
bookmark appears saved in the UI (local state updated) but is lost on reload.

Fix: await the save, catch errors, show notification on failure.

**M2. useBookmarks.ts:26-32 -- Bookmark load defaults to empty on error**

`.catch(() => setBookmarks([]))` on load. Network error or corrupt JSON looks
identical to "no bookmarks saved".

Fix: distinguish between "no bookmarks" (server returns []) and "load failed" (server
error). Show a warning on load failure.

**M3. App.tsx:397-407 -- Bookmark game navigation `.catch(() => {})`**

Cross-game bookmark click silently fails if step-range fetch errors.

Fix: log the error, show notification.

**M4. App.tsx:474-487 -- Cycle game falls back to step 1 on error**

`.catch(() => setStep(1))` on game cycle. Network error silently jumps user to step 1.

Fix: keep current step on error, show notification.

**M5. api_server.py:661-662 -- Bookmark GET returns [] on parse error**

Server catches all Exception on bookmark file read, returns empty list. Corrupt
bookmark file silently loses all bookmarks.

Fix: log the exception. Return 500 if file exists but can't be parsed.

**M6. api_server.py:623-624 -- Object.load errors all return 404**

Any exception from `Object.load()` (including DB corruption) returns 404 "Object not
found". Misclassifies the error.

Fix: catch specific exceptions. Return 404 for KeyError/ValueError, 500 for others.

**M7. run_reader.py:256-257, 272-273 -- Schema/action-map JSON errors return None**

Corrupt JSON files silently return None, indistinguishable from "file doesn't exist".

Fix: log the exception. Return the error so the API can distinguish corruption from
missing files.

**M8. GraphVisualization.tsx:2269-2271 -- Node expand error only debug-logged**

`.catch((err) => dbg("fetch:error", ...))` -- user clicks to expand a node, it fails
silently.

Fix: show brief inline error indicator near the node.

**M9. usePrefetchWindow.ts:118-121 -- Prefetch errors silently return false**

Batch prefetch fails without logging. Historical navigation is slow because cache
wasn't populated, but user sees no indication of why.

Fix: log errors to console.warn with batch details.

**M10. api_server.py:854-856, 982-983 -- Socket.io emit errors swallowed**

`except Exception: pass` around `sio.emit()`. If client is unreachable, no logging.
Client misses step/game-state notifications.

Fix: log at debug level so it shows up in server logs without breaking the writer.

### LOW: Intentional or minor

| Location | Status |
|----------|--------|
| `parquet_exporter.py:87,119,157` -- export errors | INTENTIONAL per CLAUDE.md |
| `context.tsx` sessionStorage -- `catch { /* private browsing */ }` | INTENTIONAL |
| `state.py` enrichment -- `except: pass` | INTENTIONAL, non-critical data |
| `gymnasium.py` graph export, action map | INTENTIONAL, game loop stability |
| `ErrorBoundary.tsx:26-28` -- console.error only | Minor: forward to remote logger |

### Cross-Boundary Error Flow

The step data path demonstrates systematic error loss:

```
useStepData (retry: false)
  -> fetchJson (discards response body on non-2xx)          <-- C4
    -> GET /step/{n} (returns 500 + envelope with details)
      -> RunReader.get_step (returns StepResponse status="error")
        -> DuckLakeStore.query_df (no timeout)
          -> DuckDB (lock contention blocks forever)
```

Error details are generated at each server layer but discarded at `client.ts:16-17`.

### Unvalidated Assumptions

A second class of silent failure beyond swallowed exceptions: code that assumes a
dependency is reachable, a configuration is correct, or a service is working -- without
ever verifying it. These never enter a failure path because they assume success.

#### Frontend

**U1. useRunSubscription.ts -- Socket.io connection never verified**

`getSocket()` creates the socket and returns it. No check that connection succeeded.
`socket.emit("subscribe_run", run)` is called without verifying the socket is
connected. If the server is unreachable, emits silently fail and `step_added` events
never arrive. The dashboard appears functional but receives zero live updates.

Fix: check `socket.connected` before emitting. Add `connect`, `disconnect`,
`connect_error` listeners. Expose reactive connection state. (Overlaps with H2/H3.)

**U2. useRemoteLogger.ts -- No startup health check**

`RemoteLogClient` accepts any URL and silently fails on every POST if the URL is
wrong. During this session the port was wrong for 30 minutes with zero indication.
The client library is designed to be fire-and-forget, so it never throws.

Fix: on client init, fetch the server's `/status` endpoint. If unreachable, log a
`console.warn("Remote logger unreachable at <url>")`. Consider periodic liveness.

**U3. Socket.io emit() -- No acknowledgment callbacks**

`socket.emit("subscribe_run", run)` and `socket.emit("unsubscribe_run", run)` have no
ack callback. If the server rejects or never receives the subscription, the client
has no way to know. It waits for `step_added` events that will never come.

Fix: use Socket.io's acknowledgment callback: `socket.emit("subscribe_run", run, (ack) => { ... })`.

**U4. TransportBar.tsx:284 -- Stale speed closure during data-wait**

The `advance()` function captures `speed` from the outer closure. If speed changes
while the timer is polling for data readiness (5ms loop), the recursive
`setTimeout(advance, speed)` uses the stale value until the effect restarts.

Fix: use a `speedRef` pattern (like the existing `stepRef` and `stepMaxRef`).

**U5. main.tsx:25 -- No fallback if #root element missing**

`document.getElementById("root")` returns null if the element doesn't exist. The
entire app silently fails to render with a blank page and no error.

Fix: add an `else` branch that creates a visible error message in the DOM.

#### Server

**U6. DuckLakeStore -- No connection validation after open**

`DuckLakeStore.__init__` calls `duckdb.connect()` and `INSTALL ducklake; LOAD ducklake`
but never runs a test query to verify the catalog is actually usable. If the DuckLake
plugin fails to load or the catalog is corrupted, initialization succeeds but every
subsequent `insert()` or `query()` fails silently (caught by parquet_exporter).

Fix: run `SELECT 1` after catalog attach to verify the connection is functional.

**U7. OTel exporters -- No endpoint reachability check**

`OTLPLogExporter`, `OTLPMetricExporter`, `OTLPSpanExporter`, and Pyroscope are all
initialized with endpoint URLs but never verify reachability. If the OTel collector
(`hal.ato.ms:4317`) is down, initialization succeeds and all exports fail silently
(OTel's batch processors swallow exceptions by design).

Fix: attempt a test gRPC connection on init. Log a warning if unreachable.

**U8. RemoteLoggerExporter -- No server validation (server-side twin of U2)**

`remote_logger_exporter.py` constructs a URL and parses SSL context but never checks
if the remote logger server is running. Every `export()` call POSTs via urllib and
silently returns `FAILURE` if the server is unreachable.

Fix: same pattern as U2 -- test the endpoint on init.

**U9. Data directory not validated at startup**

`RunRegistry.__init__` stores `data_dir` but never checks that it exists, is writable,
or has disk space. Failures are discovered at operation time (first bookmark save,
first step write) with unhelpful error messages.

Fix: validate `data_dir` exists and is writable in `init_data_dir()`. Create it if
missing. Log the resolved path at startup.

**U10. DuckLakeStore.is_valid_run() -- Only checks file existence, not integrity**

`is_valid_run()` returns True if `catalog.duckdb` exists, even if the file is
truncated or corrupted. The corruption is only discovered when trying to open the
catalog, and the run is marked "corrupt" after the fact.

Fix: acceptable for now (fail-at-open is reasonable). Consider adding a lightweight
magic-bytes check if corruption becomes frequent.

**U11. Game thread start not verified**

`GameManager._start_thread()` calls `thread.start()` and returns "starting" without
checking that the thread actually started or that `_run_game()` didn't immediately
raise. If the game thread crashes before reporting a run name, the game appears to be
"running" forever.

Fix: add a brief `thread.join(timeout=0.5)` after start and check `is_alive()`. If
the thread died immediately, transition to error state.

**U12. Global _game_manager not thread-safe**

`api_server.py` uses a module-level `_game_manager` global with no lock. Concurrent
calls to `set_game_manager()` and `get_game_manager()` from different threads could
race. The `game_status()` endpoint has a TOCTOU: `mgr = get_game_manager()` followed
by `mgr.get_status()` -- if another thread clears the manager between those calls,
it raises AttributeError.

Fix: protect with a threading lock, or use the existing `_game_manager._lock`.

### Implementation Order for Part 2

Recommended order (each item is independent and can be a single commit):

1. C4 -- fetchJson reads response body (unblocks all other error visibility)
2. H2+H3+U1+U3 -- Socket.io disconnect handling, reactive connected state, ack callbacks
3. U2+U8 -- Remote logger health check (both client and server sides)
4. C1 -- MenuBar game start/stop error feedback
5. C3 -- Game status fetch retry
6. C2 -- Batch query error response
7. H1 -- Step-range fetch error handling
8. H5 -- DuckLake table query logging
9. U6 -- DuckLake connection validation
10. M1+M2 -- Bookmark save/load error handling
11. U9 -- Data directory validation at startup
12. Remaining MEDIUM items (M3-M10)
13. U4 -- Speed ref in playback timer
14. U7 -- OTel endpoint reachability check
15. U11+U12 -- Game thread verification + manager thread safety
16. LOW items

### Documentation Updates for Part 2

- Update `dashboard-ui/CLAUDE.md`: add "Error Handling" section documenting the
  principle that errors must be surfaced to users, not swallowed. Add "Connectivity
  Validation" section requiring startup health checks for external dependencies.
- Update `roc/reporting/CLAUDE.md`: document which server-side `except: pass` patterns
  are intentional (game loop stability) vs. gaps (API endpoints returning 200 on error).
  Add requirement that fire-and-forget exporters must log failures at debug level.
- Add test coverage for error paths in each fixed component

### Tests for Part 2

Each fix should include a regression test:

- C1: test that startGame/stopGame show error notification on fetch failure
- C4: test that fetchJson includes response body detail in error message
- H2: test that socket disconnect/reconnect updates connected state
- C3: test that game status fetch retries on failure
- C2: test that batch endpoint returns 500 (not 200 {}) on DuckLake error
- M1: test that bookmark save failure shows notification
- U2: test that remote logger logs console.warn when server unreachable
- U6: test that DuckLakeStore raises on broken catalog (not silent success)
- U9: test that init_data_dir validates directory is writable
