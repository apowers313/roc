# Implementation Plan for Claude Code Debugging Infrastructure

## Overview

Build config-driven debugging features so any ROC game run can produce structured debug output, skip slow game-end operations, and optionally expose state to Claude Code via remote logging and DAP. No separate scripts or modes -- everything is a config option on the normal `roc.init()` / `roc.start()` path, accessible via the `uv run play` CLI (e.g. `uv run play --debug-log --num-games 1`). All Config fields are auto-generated as CLI options.

## Phase Breakdown

### Phase 1: GraphDB Export/Flush Controls + Config.init() Safety

**What this phase accomplishes**: Removes the two biggest blockers for fast iteration: the 15-minute GraphDB export hang and the silent Config.init() re-initialization trap. After this phase, debug runs finish in seconds instead of minutes, and misconfiguration is caught immediately.

**Duration**: 1 day

**Tests to Write First**:

- `tests/unit/test_config.py`: Add tests for Config.init() safety
  ```python
  def test_init_with_different_values_warns_with_diff(self):
      """Config.init() with different values should warn and show what changed."""
      Config.init(config={"db_host": "first.host"}, force=True)
      with pytest.warns(ConfigInitWarning, match="db_host"):
          Config.init(config={"db_host": "second.host"})

  def test_init_warning_includes_caller_info(self):
      """Warning message should include the caller's file and line."""
      Config.init(config={}, force=True)
      with pytest.warns(ConfigInitWarning, match=r"test_config\.py"):
          Config.init()
  ```

- `tests/unit/test_gymnasium.py` (or modify existing): Test that GraphDB controls are respected
  ```python
  def test_graphdb_export_disabled_skips_export(self, mocker):
      """When roc_graphdb_export=False, GraphDB.export() is not called."""
      settings = Config.get()
      settings.graphdb_export = False
      mock_export = mocker.patch("roc.graphdb.GraphDB.export")
      # ... trigger game end ...
      mock_export.assert_not_called()

  def test_graphdb_flush_disabled_skips_flush(self, mocker):
      """When roc_graphdb_flush=False, GraphDB.flush() is not called."""
      settings = Config.get()
      settings.graphdb_flush = False
      mock_flush = mocker.patch("roc.graphdb.GraphDB.flush")
      # ... trigger game end ...
      mock_flush.assert_not_called()
  ```

- `tests/unit/test_config.py`: Test new config fields exist with correct defaults
  ```python
  def test_graphdb_export_default_false(self):
      assert Config.get().graphdb_export is False

  def test_graphdb_flush_default_false(self):
      assert Config.get().graphdb_flush is False
  ```

**Implementation**:

1. `roc/config.py`: Add new config fields
   ```python
   # GraphDB controls
   graphdb_export: bool = False   # export graph to file on game end
   graphdb_flush: bool = False    # flush cache to Memgraph on game end
   ```

2. `roc/config.py`: Improve `Config.init()` warning
   - When called twice without `force`, compare the new config dict against the existing singleton's values
   - Include changed keys in the warning message
   - Include caller's stack frame (file:line) via `inspect.stack()`

3. `roc/gymnasium.py` (line ~129): Gate flush/export on config
   ```python
   settings = Config.get()
   if settings.graphdb_flush:
       GraphDB.flush()
   if settings.graphdb_export:
       GraphDB.export()
   ```

**Dependencies**:
- External: None (all existing packages)
- Internal: None (first phase)

**Verification**:
1. Run: `make test` -- all existing tests pass
2. Run: `uv run play --nethack-max-turns 10 --num-games 1` -- game finishes in seconds, no export hang (graphdb_export and graphdb_flush default to False)
3. Run a script that calls `Config.init()` twice with different values -- verify warning shows the changed keys and caller location

---

### Phase 2: Local JSONL Debug Log via OTel

**What this phase accomplishes**: Adds a local file exporter to the OTel logging pipeline. When enabled, all OTel log records (including the loguru bridge) are written as JSONL to a local file. Claude Code can read this file after (or during) a run to see structured decision records, warnings, and events.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_config.py`: Test new config fields
  ```python
  def test_debug_log_default_false(self):
      assert Config.get().debug_log is False

  def test_debug_log_path_default(self):
      assert Config.get().debug_log_path == "tmp/debug_log.jsonl"
  ```

- `tests/unit/test_observability.py` (new or modify existing): Test JSONL exporter setup
  ```python
  def test_debug_log_creates_file_exporter(self, tmp_path):
      """When debug_log=True, a SimpleLogRecordProcessor with file output is added."""
      settings = Config.get()
      settings.debug_log = True
      settings.debug_log_path = str(tmp_path / "test_debug.jsonl")
      Observability.init()
      # Emit a log record through OTel logger
      otel_logger = Observability.get_logger("test")
      otel_logger.emit(LogRecord(body="test message"))
      # Verify file was written
      content = (tmp_path / "test_debug.jsonl").read_text()
      assert "test message" in content

  def test_debug_log_disabled_no_file(self, tmp_path):
      """When debug_log=False, no JSONL file is created."""
      settings = Config.get()
      settings.debug_log = False
      settings.debug_log_path = str(tmp_path / "test_debug.jsonl")
      Observability.init()
      assert not (tmp_path / "test_debug.jsonl").exists()

  def test_debug_log_survives_crash(self, tmp_path):
      """SimpleLogRecordProcessor flushes synchronously -- records survive process exit."""
      settings = Config.get()
      settings.debug_log = True
      settings.debug_log_path = str(tmp_path / "test_debug.jsonl")
      Observability.init()
      otel_logger = Observability.get_logger("test")
      otel_logger.emit(LogRecord(body="before crash"))
      # File should already contain the record (sync processor)
      content = (tmp_path / "test_debug.jsonl").read_text()
      assert "before crash" in content
  ```

- `tests/integration/test_debug_log_integration.py`: End-to-end test with a short game run
  ```python
  @pytest.mark.slow
  def test_debug_log_captures_game_events(self, tmp_path):
      """A short game run with debug_log=True produces a non-empty JSONL file."""
      # Configure and run a very short game (1-2 ticks)
      # Verify JSONL file contains structured records
      # Verify records are valid JSON (one per line)
  ```

**Implementation**:

1. `roc/config.py`: Add config fields
   ```python
   debug_log: bool = False
   debug_log_path: str = "tmp/debug_log-{timestamp}.jsonl"
   ```

2. `roc/reporting/observability.py`: Add local file exporter in `Observability.init()`
   - After the existing `BatchLogRecordProcessor` setup (line ~135)
   - When `settings.debug_log` is True:
     - Open file handle at `settings.debug_log_path` (create parent dirs if needed)
     - Create a JSONL formatter function that serializes LogRecord to single-line JSON
     - Create `ConsoleLogExporter(out=file_handle, formatter=jsonl_formatter)` or write a small custom `FileLogExporter` if ConsoleLogExporter's format is not suitable
     - Add via `SimpleLogRecordProcessor` (synchronous -- crash-safe)
   - Register file handle for cleanup in `Observability.shutdown()`

3. `roc/reporting/observability.py`: Add a `get_logger()` class method to `Observability` if not already present, so other modules can emit structured OTel log records (not just via the loguru bridge)

**Key Design Decisions**:
- Use `SimpleLogRecordProcessor` (not Batch) for crash safety
- JSONL format: one JSON object per line, containing timestamp, severity, body, and attributes
- File is opened in write mode (not append) -- each run produces a fresh file
- Parent directories are created automatically (`os.makedirs`)
- `ConsoleLogExporter` from the OTel SDK may work, but test its output format first. If it produces multi-line output, write a minimal custom exporter (~30 lines)

**Dependencies**:
- External: None new -- `ConsoleLogExporter` and `SimpleLogRecordProcessor` are in `opentelemetry-sdk` (already installed)
- Internal: Phase 1 (config fields pattern)

**Verification**:
1. Run: `make test` -- all tests pass
2. Run: `uv run play --debug-log --nethack-max-turns 50 --num-games 1`
3. Inspect `tmp/debug_log.jsonl` -- should contain one JSON object per line with timestamps, log levels, and structured data
4. Run: `wc -l tmp/debug_log.jsonl` -- should have many lines (loguru bridge sends all log records)
5. Run: `python -c "import json; [json.loads(l) for l in open('tmp/debug_log.jsonl')]"` -- all lines parse as valid JSON

---

### Phase 3: debugpy Config Option + DAP MCP Setup

**What this phase accomplishes**: Any game run can be DAP-attached by setting a single config option. Combined with a DAP MCP server, Claude Code gains full interactive debugging: breakpoints, variable inspection, expression evaluation, and stepping.

**Duration**: 1-2 days

**Tests to Write First**:

- `tests/unit/test_config.py`: Test config field
  ```python
  def test_debug_port_default_zero(self):
      assert Config.get().debug_port == 0
  ```

- `tests/unit/test_init.py` (new or modify existing): Test debugpy activation
  ```python
  def test_debugpy_not_started_when_port_zero(self, mocker):
      """debugpy should not be started when debug_port is 0."""
      mock_debugpy = mocker.patch("roc.debugpy_setup.debugpy")
      settings = Config.get()
      settings.debug_port = 0
      # Call the debugpy setup function
      mock_debugpy.listen.assert_not_called()

  def test_debugpy_started_when_port_nonzero(self, mocker):
      """debugpy should listen on the configured port."""
      mock_debugpy = mocker.patch("roc.debugpy_setup.debugpy")
      settings = Config.get()
      settings.debug_port = 5678
      # Call the debugpy setup function
      mock_debugpy.listen.assert_called_once_with(("127.0.0.1", 5678))
  ```

**Implementation**:

1. `roc/config.py`: Add config field
   ```python
   debug_port: int = 0  # if non-zero, start debugpy listener on this port
   ```

2. `roc/__init__.py`: Add debugpy setup in `roc.init()`, after `Observability.init()`
   ```python
   if settings.debug_port > 0:
       import debugpy
       debugpy.listen(("127.0.0.1", settings.debug_port))
       logger.info(f"debugpy listening on port {settings.debug_port}")
   ```

3. `pyproject.toml`: Add debugpy as optional dependency
   ```toml
   [project.optional-dependencies]
   debug = ["debugpy>=1.8.0"]
   ```

4. `.mcp.json` (or equivalent MCP config): Document DAP MCP server setup
   - Evaluate available DAP MCP servers (dap-mcp, mcp-debugpy, mcp-debugger)
   - Add configuration for the chosen one
   - Test that Claude Code can attach and inspect variables

**Dependencies**:
- External: `debugpy` (optional dependency), DAP MCP server package
- Internal: Phase 1 (config pattern)

**Verification**:
1. Run: `make test` -- all tests pass
2. Run: `uv run play --nethack-max-turns 100 --num-games 1` -- should log "debugpy listening on port 5678" (debug_port defaults to 5678)
3. From another terminal or via DAP MCP: attach to port 5678, set a breakpoint, verify it triggers
4. Verify that `uv run play --debug-port 0` does not import or start debugpy

---

### Phase 4: Remote Logger OTel Exporter

**What this phase accomplishes**: Adds the Remote Logger MCP server as an OTel export destination. During a running game, Claude Code can call `logs_get_recent`, `logs_search`, and `logs_get_errors` to inspect structured debug data in real time without reading files.

**Duration**: 2 days

**Tests to Write First**:

- `tests/unit/test_config.py`: Test config fields
  ```python
  def test_debug_remote_log_default_false(self):
      assert Config.get().debug_remote_log is False

  def test_debug_remote_log_url_default(self):
      assert Config.get().debug_remote_log_url == "https://dev.ato.ms:9080/log"
  ```

- `tests/unit/test_remote_logger_exporter.py`: Test the custom exporter
  ```python
  def test_exporter_posts_to_endpoint(self, httpserver):
      """RemoteLoggerExporter POSTs JSON to the configured URL."""
      httpserver.expect_request("/log", method="POST").respond_with_data("ok")
      exporter = RemoteLoggerExporter(
          url=httpserver.url_for("/log"),
          session_id="test-session",
      )
      record = create_test_log_record(body="test message")
      result = exporter.export([record])
      assert result == LogExportResult.SUCCESS

  def test_exporter_handles_connection_error(self):
      """Exporter should not raise on connection failure."""
      exporter = RemoteLoggerExporter(
          url="http://localhost:1/log",
          session_id="test-session",
      )
      record = create_test_log_record(body="test message")
      result = exporter.export([record])
      assert result == LogExportResult.FAILURE

  def test_exporter_formats_records_as_remote_logger_json(self, httpserver):
      """Records should match the Remote Logger's expected format."""
      # Verify: sessionId, logs[{time, level, message}] structure
  ```

**Implementation**:

1. `roc/config.py`: Add config fields
   ```python
   debug_remote_log: bool = False
   debug_remote_log_url: str = "https://dev.ato.ms:9080/log"
   ```

2. `roc/reporting/remote_logger_exporter.py` (new file): Custom OTel LogExporter
   ```python
   class RemoteLoggerExporter(LogExporter):
       """OTel log exporter that POSTs records to the Remote Logger MCP server."""
       def __init__(self, url: str, session_id: str): ...
       def export(self, batch: Sequence[LogData]) -> LogExportResult:
           # Convert OTel LogRecords to Remote Logger format:
           # {"sessionId": ..., "logs": [{"time": ..., "level": ..., "message": ...}]}
           # POST to self.url
           # Return SUCCESS or FAILURE (never raise)
       def shutdown(self) -> None: ...
   ```

3. `roc/reporting/observability.py`: Register the exporter when enabled
   ```python
   if settings.debug_remote_log:
       from roc.reporting.remote_logger_exporter import RemoteLoggerExporter
       remote_exporter = RemoteLoggerExporter(
           url=settings.debug_remote_log_url,
           session_id=instance_id,
       )
       logger_provider.add_log_record_processor(
           SimpleLogRecordProcessor(remote_exporter)
       )
   ```

**Key Design Decisions**:
- Use `SimpleLogRecordProcessor` (synchronous) so records appear immediately in `logs_get_recent`
- Exporter catches all HTTP errors silently (returns FAILURE) -- network issues must not crash the game
- Session ID ties all records from one run together for `logs_list_sessions`
- Uses `urllib.request` (stdlib) to avoid adding a dependency on `requests`/`httpx`

**Dependencies**:
- External: `pytest-httpserver` (test only, for mocking HTTP endpoints)
- Internal: Phase 2 (OTel exporter pattern in observability.py)

**Verification**:
1. Run: `make test` -- all tests pass
2. Verify Remote Logger MCP server is running: call `logs_status` via MCP
3. Run: `uv run play --nethack-max-turns 50 --num-games 1` (debug_remote_log defaults to True)
4. During or after the run, call `logs_get_recent` via MCP -- should show structured log records
5. Call `logs_search "object"` -- should find object resolution records
6. Call `logs_list_sessions` -- should show the session from this run

---

### Phase 5: Tick-Level State Snapshots

**What this phase accomplishes**: At configurable intervals, emit a comprehensive state snapshot as an OTel log record. These snapshots flow through all configured exporters (JSONL file, remote logger, OTLP) and provide periodic visibility into the game state without needing to attach a debugger.

**Duration**: 1-2 days

**Tests to Write First**:

- `tests/unit/test_config.py`: Test config field
  ```python
  def test_debug_snapshot_interval_default_zero(self):
      assert Config.get().debug_snapshot_interval == 0
  ```

- `tests/unit/test_state_snapshots.py` (new file):
  ```python
  def test_snapshot_emitted_at_interval(self, mocker):
      """Snapshot is emitted every N ticks when interval > 0."""
      settings = Config.get()
      settings.debug_snapshot_interval = 5
      mock_logger = mocker.patch("roc.reporting.state.otel_logger")
      # Simulate 10 ticks
      for i in range(10):
          State.tick(i)
      # Should have 2 snapshots (tick 5 and tick 10)
      assert mock_logger.emit.call_count == 2

  def test_snapshot_not_emitted_when_disabled(self, mocker):
      """No snapshots when interval is 0."""
      settings = Config.get()
      settings.debug_snapshot_interval = 0
      mock_logger = mocker.patch("roc.reporting.state.otel_logger")
      for i in range(10):
          State.tick(i)
      mock_logger.emit.assert_not_called()

  def test_snapshot_contains_expected_fields(self):
      """Snapshot record should include screen, objects, resolution stats."""
      # Verify the LogRecord body/attributes contain the right data
  ```

**Implementation**:

1. `roc/config.py`: Add config field
   ```python
   debug_snapshot_interval: int = 0  # emit snapshot every N ticks (0 = disabled)
   ```

2. `roc/reporting/state.py`: Add snapshot emission logic
   - Add a tick counter to `State` (or use the existing `loop` state)
   - On each tick, check if `tick_count % snapshot_interval == 0`
   - If so, emit an OTel log record with event name `roc.state.snapshot` containing:
     - Current screen (text representation)
     - Active objects and positions
     - Resolution stats since last snapshot (match count, new object count, avg confidence)
   - The record flows through all configured OTel exporters automatically

3. Identify where ticks are counted (likely `gymnasium.py` main loop or an event listener) and add the snapshot check call there

**Dependencies**:
- External: None
- Internal: Phase 2 (OTel log emission pattern), Phase 4 (remote logger for live querying)

**Verification**:
1. Run: `make test` -- all tests pass
2. Run with snapshots enabled:
   ```bash
   uv run play --debug-log --debug-snapshot-interval 10 --nethack-max-turns 50 --num-games 1
   ```
3. Grep the JSONL file: `grep "roc.state.snapshot" tmp/debug_log.jsonl | wc -l` -- should be ~5 (50 ticks / 10 interval)
4. Inspect a snapshot record -- should contain screen text and object data

---

### Phase 6: Deprecation and Cleanup

**What this phase accomplishes**: Removes redundant code made obsolete by the new OTel-based debug infrastructure. Reduces maintenance burden and eliminates confusing overlaps.

**Duration**: 1-2 days

**Tests to Write First**:

- Verify existing tests still pass after removing code (no new test files needed for removal)
- `tests/unit/test_observability.py`: Add tests verifying that the standard OTel logger emits the same data that ObservabilityEvent subclasses used to emit
  ```python
  def test_screen_data_emitted_via_otel_logger(self):
      """Screen data should be emitted as a standard OTel log record."""
      # Set up State with screen data
      # Trigger the emission
      # Verify OTel logger received a record with screen data

  def test_gauge_callback_does_not_print(self):
      """node_cache_gauge callback should not call State.print() or send_events()."""
      # Mock State.print and State.send_events
      # Trigger gauge callback
      # Verify neither was called
  ```

**Implementation**:

1. `roc/reporting/state.py`: Replace `ObservabilityEvent` subclasses
   - Replace `State.send_events()` (lines 148-167) with standard OTel logger emissions
   - Each former `ObservabilityEvent` subclass becomes a simple OTel `logger.emit()` call with appropriate attributes
   - Remove the 7 custom event classes: `ScreenObsEvent`, `SaliencyObsEvent`, `FeatureObsEvent`, `ObjectObsEvent`, `FocusObsEvent`, `AttentionObsEvent`, `IntrinsicObsEvent` (~70 lines)
   - Keep `State.print()` as a standalone method for Jupyter use (not called from gauge callback)

2. `roc/reporting/state.py`: Clean up gauge callback
   - Remove `State.send_events()` call from `node_cache_gauge()` (line ~443)
   - Remove `State.print()` call from `node_cache_gauge()` (line ~444)
   - Callback should only compute and return the gauge value

3. `roc/reporting/observability.py`: Remove `ObservabilityEvent` base class if no longer used
   - Check for any remaining callers first
   - If `Observability.event()` method exists, deprecate or remove it

**Dependencies**:
- External: None
- Internal: Phase 2 (OTel log emission is the replacement)

**Verification**:
1. Run: `make test` -- all existing tests pass
2. Run: `make lint` -- no type errors or style issues
3. Run: `uv run play --debug-log --nethack-max-turns 50 --num-games 1` -- verify that the JSONL file contains the same data that used to flow through ObservabilityEvent (screen, saliency, objects, etc.)
4. Run: `uv run play --nethack-max-turns 50 --num-games 1` -- verify stdout is cleaner (no periodic State.print() dumps from gauge callback)

---

## Common Utilities Needed

- **JSONL Formatter**: Function to serialize OTel LogRecord to single-line JSON. Used by the file exporter (Phase 2) and potentially the remote logger exporter (Phase 4). Place in `roc/reporting/formatters.py` or inline in observability.py if small enough.
- **OTel Logger Access**: A `get_logger(name)` method on `Observability` to provide named OTel loggers to other modules. Used in Phases 2, 4, 5, and 6.

## External Libraries Assessment

- **debugpy**: Required for Phase 3. Well-maintained Microsoft package, standard for Python DAP. Add as optional dependency (`debug` extra).
- **DAP MCP server**: Required for Phase 3. Evaluate `dap-mcp`, `mcp-debugpy`, and `mcp-debugger`. Choose based on: active maintenance, Python support quality, and ease of setup.
- **pytest-httpserver**: Required for Phase 4 tests. Provides a local HTTP server fixture for testing the RemoteLoggerExporter without a real Remote Logger instance.
- **Grafana MCP**: Mentioned in design Phase 5 but not included in this implementation plan. It is a pure configuration task (no ROC code changes) and can be set up independently at any time.

## Risk Mitigation

- **OTel ConsoleLogExporter format may not be JSONL**: Test the output format early in Phase 2. If it produces multi-line or non-JSON output, write a minimal custom `FileLogExporter` (~30 lines). The OTel SDK's exporter interface is simple: implement `export(batch)` and `shutdown()`.
- **debugpy import may slow startup**: Guard the import behind the `debug_port > 0` check so it is never imported in normal runs. The import only happens when debugging is explicitly requested.
- **Remote Logger HTTP calls may slow the game**: Use `SimpleLogRecordProcessor` (as designed) but monitor latency. If the remote logger is slow or down, the synchronous call will block. Mitigation: use a short HTTP timeout (2 seconds) and swallow failures. If this becomes a problem, switch to `BatchLogRecordProcessor` for the remote exporter only (trades immediacy for throughput).
- **Large JSONL files in long runs**: At DEBUG level, the loguru bridge may produce thousands of records per second. Mitigation: the JSONL exporter should respect the existing `observability_logging_level` config to filter what gets written. For long runs, users should also enable the remote logger (which supports search) rather than relying solely on file inspection.
- **Config.init() change may break existing callers**: The enhanced warning in Phase 1 is additive (more information in the warning message). It does not change the behavior (still warns and returns existing config). No existing code should break.
- **ObservabilityEvent removal (Phase 6) may break Jupyter notebooks**: Search for all callers of `Observability.event()` and the custom event classes before removing. Migrate any remaining callers to standard OTel logger calls first.
