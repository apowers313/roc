# Claude Code Debugging Plan for ROC

## Problem Statement

During the Dirichlet-Categorical object resolution work, debugging a real game run required 4 iterations:

1. **Monkey-patched class methods** -- didn't work because ExpMod registry stores instances, not classes
2. **Config overrides ignored** -- `Config.init()` was already called by import side-effects
3. **Game hung on GraphDB.export** -- 81K nodes at ~80 nodes/sec blocked for 15+ minutes, losing log data that was buffered for post-game write
4. **Finally worked** -- after patching the instance, calling `Config.reset()`, disabling GraphDB export, and switching to incremental file writes

Each iteration required writing a new script, running it, waiting for failure, diagnosing, and rewriting. This is slow and fragile. The root causes are:

- **No way to inspect state during a running game** without writing custom instrumentation scripts
- **No structured debug output** that Claude Code can read -- loguru goes to stderr at DEBUG level (noisy), OpenTelemetry goes to a remote server (inaccessible to CLI)
- **Game-end operations block** (GraphDB.export) with no timeout or skip mechanism
- **Config singleton + import side-effects** create initialization ordering traps
- **Breakpoint system designed for Jupyter** -- useless for CLI-based debugging

## Current Systems Audit

### 1. Loguru Logging
- **What it does**: Structured logging to stderr with per-module level filtering
- **Claude Code accessibility**: Visible in command output, but very noisy at DEBUG level (component registration, bus attachments, etc.) and critical information is buried
- **Verdict**: KEEP but needs a debug-friendly output mode

### 2. OpenTelemetry (Metrics, Traces, Events)
- **What it does**: 17 metrics + 8 spans + structured events exported via OTLP gRPC to hal.ato.ms
- **Claude Code accessibility**: ZERO. All data goes to a remote Grafana/Prometheus stack that Claude Code cannot query
- **Verdict**: KEEP for production monitoring, but ADD a local file exporter option for debugging

### 3. OpenTelemetry Events (via State)
- **What it does**: Emits screen, saliency, features, objects, focus points as OTel events
- **Claude Code accessibility**: ZERO -- same remote-only export issue
- **Overlap**: Duplicates information available via loguru and State.print()
- **Verdict**: REVIEW -- may be redundant with a good local debug output

### 4. Pyroscope Profiling
- **What it does**: CPU profiling exported to remote Pyroscope server
- **Claude Code accessibility**: ZERO
- **Verdict**: KEEP as-is. Profiling is rarely needed for debugging correctness issues

### 5. State Tracking (state.py)
- **What it does**: Tracks current screen, saliency, attention, objects in memory
- **Claude Code accessibility**: Only via `State.print()` which is called inside `node_cache_gague` callback (every 5 seconds during metrics collection) -- output goes to stdout mixed with game output
- **Verdict**: KEEP but needs on-demand dump capability

### 6. Breakpoint System (breakpoint.py)
- **What it does**: Condition-based pause/resume for Jupyter notebooks
- **Claude Code accessibility**: ZERO. Uses Lock-based blocking which requires another thread to call `resume()`. Designed for interactive Jupyter use.
- **Verdict**: KEEP for Jupyter, but NOT useful for Claude Code debugging

### 7. Event System (event.py)
- **What it does**: RxPy reactive streams for component communication
- **Claude Code accessibility**: Events are traced via `logger.trace()` but that's the noisiest log level. No event capture/replay.
- **Verdict**: KEEP, but add optional event logging to file

### 8. GraphDB Export
- **What it does**: Saves all graph nodes to Memgraph and exports to file
- **Claude Code accessibility**: Blocks game completion for 15+ minutes with large graphs
- **Verdict**: NEEDS timeout/skip mechanism and should be opt-in for debug runs

## Proposed Changes

### Phase 1: Config-Driven Debug Features (High Impact, Low Effort)

Every run should be debuggable. Instead of a separate "debug run" script, these are all
config options on the normal `roc.init()` / `roc.start()` path.

#### 1A. Local OTel Log File (Structured Decision Records)

Instead of building a separate JSONL logging system, add a **local file exporter** to the
existing OTel logging pipeline. Decision records (posteriors, candidates, features) are
emitted as OTel structured log records and routed to both the remote backend and a local
JSONL file.

**New config options**:
- `roc_debug_log: bool = False` -- enable local JSONL file exporter
- `roc_debug_log_path: str = "tmp/debug_log.jsonl"` -- output path

**Implementation**: In `roc/reporting/observability.py`, when `debug_log` is enabled,
add a second log processor to the existing OTel LoggerProvider:

```python
from opentelemetry.sdk._logs.export import ConsoleLogExporter, SimpleLogRecordProcessor

if settings.debug_log:
    file_handle = open(settings.debug_log_path, "w")
    file_exporter = ConsoleLogExporter(out=file_handle, formatter=json_formatter)
    # SimpleLogRecordProcessor = synchronous, flushes immediately (survives crashes)
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(file_exporter))
```

**What gets written**: Any OTel log record, including:
- Object resolution decisions (emitted from `object.py` via OTel logger)
- New object creations with initial state
- Warnings/anomalies (NaN, low confidence, teleportation)
- Periodic summary records and game-end summary

**Key design decisions**:
- Uses `SimpleLogRecordProcessor` (synchronous) not `BatchLogRecordProcessor` -- ensures
  every record is written and flushed immediately, surviving hangs/crashes
- Same structured data goes to both remote OTLP and local file -- no divergence
- One JSONL line per OTel log record
- Replaces both the proposed standalone `debug_log.py` AND the `ObservabilityEvent`
  subclasses in `state.py` -- all structured events go through OTel's standard API
- Replaces the need for separate `analyze_resolution_log.py` / `capture_resolution_data.py`

**Architecture**:
```
Loguru -------> stderr (human-readable, colored, filtered)
  |
  +--bridge---> OTel LoggerProvider --+--> BatchLogRecordProcessor --> OTLP (remote)
                                      |
Decision records --> OTel Logger -----+--> SimpleLogRecordProcessor --> File (local JSONL)
                                      |
OTel Metrics/Traces ------------------+--> OTLP (remote)
```

This keeps the system at **two logging systems** (loguru + OTel) rather than three.
Loguru handles human-readable stderr output. OTel handles all structured data with
multiple export destinations.

#### 1B. GraphDB Export / Flush Controls

The current always-export-on-game-end behavior is the biggest blocker for fast iteration
(81K nodes at ~80 nodes/sec = 15+ minutes). These should be config options that default
to the current behavior but can be turned off.

**New config options**:
- `roc_graphdb_export: bool = True` -- export graph to file on game end
- `roc_graphdb_flush: bool = True` -- flush cache to Memgraph on game end

Setting both to `false` is the typical choice for debugging -- the game finishes in
seconds instead of minutes.

#### 1C. debugpy Listener

Enable DAP attachment on any run.

**New config options**:
- `roc_debug_port: int = 0` -- if non-zero, start debugpy listener on this port at init

**Implementation**: In `roc.init()` (or `roc.start()`), if `debug_port > 0`:
```python
import debugpy
debugpy.listen(("127.0.0.1", settings.debug_port))
logger.info(f"debugpy listening on port {settings.debug_port}")
```

This means any `roc.start()` invocation can be DAP-attached, whether from tests,
Jupyter, or the CLI. No separate script needed.

#### 1D. Config.init() Safety

The silent-ignore behavior of `Config.init()` when already initialized is a trap that
cost us an iteration during the Dirichlet debugging. Fix options:
- Make `Config.init()` raise if called twice with different values
- Or at minimum: log a WARNING (not just debug) when init is called while already
  initialized, and include the caller's stack frame

This is a bug fix, not a feature -- it prevents silent misconfiguration.

### Phase 2: Remote Logger Integration (Medium Impact, Medium Effort)

#### 2A. Remote Logger as OTel Exporter

The Remote Logger MCP server is already running at `https://dev.ato.ms:9080/log`. Rather
than building a separate integration, implement it as a **custom OTel log exporter** that
POSTs records to the remote logger endpoint. This keeps everything flowing through OTel.

**New config options**:
- `roc_debug_remote_log: bool = False` -- enable remote log posting
- `roc_debug_remote_log_url: str = "https://dev.ato.ms:9080/log"` -- endpoint

**Implementation**: Write a small `RemoteLoggerExporter(LogExporter)` that POSTs
JSON log records to the HTTP endpoint. Add it as a third processor on the LoggerProvider:

```python
if settings.debug_remote_log:
    remote_exporter = RemoteLoggerExporter(url=settings.debug_remote_log_url,
                                            session_id=instance_id)
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(remote_exporter))
```

**Benefits**:
- Claude Code can call `logs_get_recent`, `logs_search`, `logs_get_errors` during a running game
- Same data that goes to the local JSONL file also goes to the remote logger
- No separate logging system -- just another OTel export destination
- Searchable by pattern (e.g., "find all low_confidence decisions")

**Architecture with all exporters enabled**:
```
OTel LoggerProvider --+--> BatchLogRecordProcessor ----> OTLP (remote Grafana)
                      +--> SimpleLogRecordProcessor ---> File (local JSONL)
                      +--> SimpleLogRecordProcessor ---> RemoteLoggerExporter (MCP-queryable)
```

#### 2B. Tick-Level State Snapshots

At configurable intervals (e.g., every 10 ticks), emit a state snapshot as an OTel log
record with a distinct event name (e.g., `roc.state.snapshot`):
- Current screen (rendered as text)
- Active objects and their positions
- Resolution stats since last snapshot

**New config option**:
- `roc_debug_snapshot_interval: int = 0` -- emit snapshot every N ticks (0 = disabled)

These flow through the same OTel pipeline and appear in both the local file and remote
logger. Claude Code can search for them with `logs_search "roc.state.snapshot"`.

### Phase 4: Interactive Debugging via DAP MCP (High Impact, Medium Effort)

A DAP (Debug Adapter Protocol) MCP server gives Claude Code full debugger capabilities:
set breakpoints, pause execution, inspect any variable, evaluate arbitrary expressions,
step through code, and resume. This is fundamentally more powerful than custom file-based
IPC because it requires zero pre-built serialization -- Claude Code can inspect anything
on the fly.

#### 4A. DAP MCP Setup

**Available DAP MCP servers** (community-built):
- [dap-mcp](https://github.com/kashuncheng/dap_mcp) -- multi-language DAP bridge (Python via debugpy)
- [mcp-debugpy](https://github.com/markomanninen/mcp-debugpy) -- Python-specific, AI-focused
- [mcp-debugger](https://github.com/debugmcp/mcp-debugger) -- LLM-driven step-through debugging

**Setup**: Add to Claude Code's MCP configuration (`.mcp.json`):
```json
{
  "mcpServers": {
    "debugger": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "dap_mcp"]
    }
  }
}
```

**What this gives Claude Code**:
- `set_breakpoint(file, line, condition?)` -- break at specific code locations
- `pause()` -- suspend execution at any time
- `continue()` / `step_over()` / `step_into()` -- control flow
- `evaluate(expression)` -- run arbitrary Python in the paused context
- `get_variables(scope)` -- inspect local/global variables
- `get_stack_trace()` -- see the call stack

#### 4B. DAP Workflow with Config-Driven debugpy

Since debugpy is now a config option (Phase 1C), any run can be DAP-attached:

```bash
uv run play
```

Claude Code workflow:
1. Start game (debugpy listens on port 5678 by default)
2. Attach via DAP MCP: `dap_attach(host="127.0.0.1", port=5678)`
3. Set breakpoints in interesting locations:
   - `object.py:497` -- the `_decide` method's threshold check
   - `object.py:573` -- `do_object_resolution` entry point
   - `gymnasium.py:107` -- after each env.step()
4. Game runs until breakpoint
5. Inspect state:
   ```
   evaluate("len(candidates)")
   evaluate("[(obj.id, obj.resolve_count) for obj in candidates]")
   evaluate("self._alphas[best_obj.id]")
   evaluate("{k: round(math.exp(v), 4) for k, v in log_posteriors.items()}")
   evaluate("str(context)")
   ```
6. Continue or step through the resolution logic
7. Detach when done -- game continues to completion

#### 4C. Why DAP Replaces Custom Inspection

The file-based pause/resume, conditional pause, signal-based state dump, and custom
state serialization from the previous Phase 4 design are all **subsumed by DAP**:

| Previous approach | DAP equivalent |
|-------------------|---------------|
| File-based pause (`tmp/.pause`) | `dap_pause()` |
| File-based resume (`rm tmp/.pause`) | `dap_continue()` |
| Conditional pause (`echo "tick:50"`) | `set_breakpoint(file, line, condition="tick==50")` |
| State dump to JSON | `evaluate("any_expression")` -- inspect anything |
| Signal-based snapshot | `dap_pause()` + `evaluate()` + `dap_continue()` |
| Custom serialization code | Not needed -- debugger sees all objects natively |

**Key advantage**: No need to predict what state we'll want to inspect. With file-based
dumps, we had to pre-decide what to serialize. With DAP, Claude Code evaluates whatever
expression is relevant to the current investigation.

#### 4D. Threading Considerations

ROC uses RxPy `ThreadPoolScheduler` for event handlers. The main game loop runs in the
main thread (`gymnasium.start()`), but event listeners (perception, attention, object
resolution) run on worker threads. When DAP pauses:

- Setting a breakpoint in `object.py:_decide()` will pause whichever worker thread
  hits it -- other threads may continue briefly until they also block
- Setting a breakpoint in `gymnasium.py:start()` pauses the main loop thread
- `dap_pause()` suspends all threads

For most debugging, breakpoints in the resolution code are what we want. The threading
is transparent -- DAP handles it.

#### 4E. Fallback: File-Based Pause (No MCP Required)

If DAP MCP is not configured, the existing breakpoint system can provide basic
pause/inspect via file-based IPC:

- Register a breakpoint condition that checks for `tmp/.pause` file
- On trigger, write basic state info to `tmp/state_dump.json`
- Resume when `tmp/.pause` is deleted

This is less powerful (pre-determined state dump, no arbitrary inspection) but works
with zero additional dependencies.

### Phase 5: Grafana MCP (Medium Impact, Medium Effort)

A Grafana MCP server (e.g., `grafana/mcp-grafana`) would give Claude Code direct access
to the existing Grafana dashboards and data on hal.ato.ms.

#### 5A. What Grafana MCP Provides

Community Grafana MCP servers typically expose:
- **Dashboard querying**: list dashboards, get panel data
- **Datasource querying**: run PromQL/Loki queries against Prometheus/Loki
- **Alert inspection**: check alert states and history
- **Annotation reading**: see event markers on dashboards

For ROC, this means Claude Code could:
- Query `roc.dirichlet.posterior_max` histogram over time to see confidence trends
- Query `roc.resolution.decision` counter to see match/new_object rates
- Query `roc.event` counter to see event throughput across buses
- Read span traces to identify slow operations
- Compare metrics across different runs (by instance ID)

#### 5B. When Grafana MCP Is Useful vs. Not

**Good for**:
- Historical trend analysis across multiple runs
- Comparing before/after metrics when tuning parameters
- Monitoring long-running games (100K+ ticks) where JSONL files become too large
- Aggregate views: "what's the overall match rate over 1000 ticks?"
- Alerting: "did any metric cross a threshold during this run?"

**Not ideal for**:
- Per-decision debugging (metrics are aggregated, not per-event)
- Fast iteration on short runs (local JSONL is faster to read)
- Debugging when the OTel backend is down or unreachable

#### 5C. Setup

1. Install and configure the Grafana MCP server pointing at hal.ato.ms
2. Ensure the Grafana instance has a service account / API key for MCP access
3. Add to Claude Code's MCP configuration

**Recommendation**: Set up Grafana MCP as a complement to the local debug tools, not
a replacement. Use it for "big picture" analysis and the local JSONL + remote logger
for "per-decision" debugging.

### Phase 6: Deprecation (Reduce Maintenance Burden)

Implementing the above plan makes several existing systems redundant. Removing them
reduces code to maintain and eliminates confusing overlaps.

#### 6A. Replace `ObservabilityEvent` Subclasses with Standard OTel Log Records

**What**: `State.send_events()` (`state.py:148-167`) and the 7 custom event classes:
`ScreenObsEvent`, `SaliencyObsEvent`, `FeatureObsEvent`, `ObjectObsEvent`,
`FocusObsEvent`, `AttentionObsEvent`, `IntrinsicObsEvent` (`state.py:368-437`).

**Why**: These were hand-rolled OTel event wrappers built before the OTel logging API
was mature. Phase 1A replaces them with standard OTel log records that flow through
the standard pipeline (remote OTLP + local file + remote logger). The custom
`ObservabilityEvent` base class and its subclasses become dead code.

**Migration**: Replace each `Observability.event(ScreenObsEvent(data))` call with a
standard OTel logger emit: `otel_logger.emit(LogRecord(body=data, attributes={...}))`.
The data reaches the same remote backend AND now also appears in the local JSONL file.

**Effort**: Small. Replace ~70 lines of custom event classes with standard OTel logger
calls. Net reduction in code.

#### 6B. Remove `State.print()` from `node_cache_gague()` Callback

**What**: `node_cache_gague()` (`state.py:439-447`) currently calls `State.send_events()`
and `State.print()` every time the OTel metrics reader collects gauge values (every 5
seconds). This prints the full game screen, saliency map, and current object to stdout
mixed in with game output.

**Why**: With 6A removing `send_events()`, this callback should only compute the gauge
value. `State.print()` is a Jupyter convenience (called on-demand) -- it shouldn't be
triggered by a metrics collection callback. Remove the `send_events()` and `print()`
calls from the gauge callback; keep them as standalone methods for Jupyter use.

**Effort**: Tiny. Delete 2 lines from the callback.

#### 6C. Simplify Breakpoint System

**What**: `breakpoint.py` (228 lines) provides condition-based pause/resume with
Lock-based blocking, designed for Jupyter notebooks.

**Why**: DAP MCP (Phase 4) provides strictly more powerful breakpoint capabilities:
conditional breakpoints at any code location, variable inspection, expression evaluation,
stepping. The custom breakpoint system's only remaining use case is Jupyter notebooks.

**Recommendation**: Keep `breakpoint.py` for now (Jupyter users rely on it), but don't
invest in extending it. If Jupyter usage declines, deprecate it. Do NOT build the
file-based pause fallback (Phase 4E) on top of it -- the effort isn't worth it given
that DAP MCP is the primary approach.

#### 6D. Remove OTel Local Mode (Phase Was Deprioritized)

**What**: The previously proposed Phase 6A/6B (console/file metric exporters, OTel event
decoupling) from the earlier draft.

**Why**: The JSONL debug log serves the "local structured output" need better than
redirecting OTel to local files. OTel metrics are designed for time-series aggregation,
not per-event debugging. With Grafana MCP for aggregate queries and JSONL for per-event
data, there's no gap that local OTel export would fill.

**Recommendation**: Don't implement. If needed later, it's straightforward to add.

## Logging Architecture (Two Systems)

After implementing this plan, ROC has **two logging systems** with clear separation:

### 1. Loguru -- Human-Readable Application Log

- **Format**: Colored text to stderr
- **Audience**: Developers reading terminal output, CI logs
- **Content**: Application lifecycle, errors, warnings, component events
- **Example**: `2026-03-09 | INFO | roc.gymnasium:start:127 - Game 1 completed`
- **Filter**: Per-module level filtering via `roc_log_modules`
- **Bridge**: Loguru records also flow to OTel via `loguru_to_otel()` bridge for
  remote storage in Grafana/Loki

### 2. OpenTelemetry -- All Structured Data

Single system with **multiple export destinations**:

```
                                       +--> OTLP (remote Grafana/Loki)
OTel LoggerProvider --+--> Batch ------+
                      |
                      +--> Sync -------+--> Local JSONL file (debug_log)
                      |
                      +--> Sync -------+--> Remote Logger (MCP-queryable)

OTel MeterProvider ----+--> OTLP ------+--> Remote Prometheus

OTel TracerProvider ---+--> OTLP ------+--> Remote Jaeger/Tempo
```

**What flows through OTel logs**:
- Decision records: "tick 42, object -37 matched, posterior 0.87, 5 candidates"
- State snapshots: screen, objects, resolution stats (at configurable intervals)
- Anomaly alerts: NaN detected, object teleportation, low confidence

**What flows through OTel metrics**: Counters, histograms, gauges (match rates,
posterior distributions, resolution latencies, cache sizes)

**What flows through OTel traces**: Span timings for resolution pipeline stages

**Export destinations are config-driven**:
- OTLP remote: always on (existing behavior)
- Local JSONL file: `roc_debug_log=true`
- Remote logger: `roc_debug_remote_log=true`

**Why two systems instead of one**: Loguru provides superior developer experience for
terminal output (colors, per-module filtering, exception formatting, concise API).
OTel provides superior infrastructure for structured data (typed attributes, multiple
exporters, correlation with metrics/traces). Collapsing them into one would degrade
one use case or the other. The `loguru_to_otel()` bridge ensures loguru messages also
reach the remote backend, so nothing is lost.

## Implementation Priority

| Phase | Item | Effort | Impact for Claude Code |
|-------|------|--------|----------------------|
| 1B | GraphDB export/flush config | Small | Critical -- unblocks fast iteration |
| 1A | JSONL debug log | Medium | High -- structured data for analysis |
| 1C | debugpy config option | Small | High -- enables DAP on any run |
| 4A | DAP MCP setup | Small | High -- interactive debugging |
| 1D | Config.init() safety | Small | Medium -- prevents initialization traps |
| 2A | Remote logger integration | Medium | High -- live inspection during runs |
| 2B | Tick-level state snapshots | Medium | Medium -- game visibility |
| 5A | Grafana MCP setup | Medium | Medium -- historical/aggregate analysis |
| 6A | Remove ObservabilityEvent subclasses | Small | Reduces maintenance |
| 6B | Remove State.print() from gauge callback | Tiny | Reduces noise |
| 4E | File-based pause fallback | Small | Low -- only if DAP unavailable |

## Ideal Debugging Session with Claude Code

All debugging features are config options -- no separate scripts or modes.

### Quick run (short game, post-hoc analysis):

```bash
roc_debug_log=true roc_nethack_max_turns=100 roc_num_games=1 uv run play
```

1. Game starts with debug log writing to `tmp/debug_log.jsonl`
2. During the run, Claude Code can:
   - Call `logs_get_recent` to see live resolution decisions (if remote logger enabled)
   - Call `logs_search "low_confidence"` to find anomalies in real time
3. Game finishes in seconds (no GraphDB export hang)
4. Claude Code reads the final summary line from the JSONL: match rate, object count, anomalies
5. If issues found, Claude Code reads specific entries from the JSONL for details
6. No custom scripts, no monkey-patching, no multiple iteration cycles

### Interactive debugging (DAP MCP -- preferred):

```bash
roc_debug_log=true uv run play
```

1. Game starts with debugpy listening on port 5678
2. Claude Code attaches via DAP MCP: `dap_attach(host="127.0.0.1", port=5678)`
3. Sets breakpoints at interesting locations:
   - `set_breakpoint("roc/object.py", 497)` -- threshold decision
   - `set_breakpoint("roc/object.py", 573, condition="len(candidates) > 5")`
4. Game runs until breakpoint
5. Claude Code inspects anything it needs:
   ```
   evaluate("[(obj.id, obj.resolve_count) for obj in candidates]")
   evaluate("{k: round(math.exp(v), 4) for k, v in log_posteriors.items()}")
   evaluate("self._alphas[result.id]")
   ```
6. Steps through code if needed, or continues to next breakpoint
7. No pre-built serialization, no custom scripts, no file-based IPC

### Long game monitoring (Grafana + remote logger + DAP):

```bash
roc_debug_log=true roc_nethack_max_turns=100000 uv run play
```

1. Game runs with all debug channels active
2. Claude Code attaches DAP but sets no breakpoints -- game runs freely
3. During the run, uses Grafana MCP to check aggregate metrics:
   - "What's the match rate over the last 1000 ticks?"
   - "Is the object count still growing or has it stabilized?"
   - "Are there any spans taking > 100ms?"
4. Uses remote logger to search for specific anomalies: `logs_search "NaN"`
5. If something looks wrong, uses DAP to pause and inspect live state
6. Adds targeted breakpoints based on what Grafana/remote logger revealed
7. Continues -- game resumes with new breakpoints active
