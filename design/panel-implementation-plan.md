# Implementation Plan for Panel Debug Dashboard

## Overview

Build a per-step game state debugger using Panel (HoloViz) that displays all ROC agent
data synchronized to a single step slider. Data is stored in Parquet files (written by a
new OTel exporter) and queried via DuckDB. Replaces W&B media panels and JSONL debug log.

## Existing Code Inventory

Key files that already exist and will be extended or reused:

| File | Relevant Contents |
|------|------------------|
| `roc/reporting/observability.py` | `JsonlFileExporter`, OTel logger provider setup, `Observability` singleton |
| `roc/reporting/state.py` | `State` classes, `emit_state_logs()`, OTel record emission for screen/saliency/features/objects/focus |
| `roc/reporting/screen_renderer.py` | `render_grid_html()`, `screen_to_html_vals()`, curses color mapping |
| `roc/reporting/metrics.py` | `RocMetrics` dispatch layer |
| `roc/config.py` | `data_dir` field (already exists), `debug_log`/`debug_log_path` (to be removed) |
| `pyproject.toml` | `panel`, `pandas`, `bokeh` already in deps; needs `pyarrow`, `duckdb` |

## Phase Breakdown

### Phase 1: ParquetExporter -- Write Per-Step Data to Parquet

**What this phase accomplishes:**
Replace `JsonlFileExporter` with `ParquetExporter` that writes OTel log records to
Parquet files partitioned by event type. After this phase, running `uv run play` produces
Parquet files in `data_dir/<run-name>/` that can be inspected with Python/DuckDB.

**Duration**: 2 days

**Tests to Write First**:
- `tests/unit/test_parquet_exporter.py`:
  - `test_export_creates_run_directory` -- exporter creates `<data_dir>/<run>/` on flush
  - `test_export_routes_screen_to_screens_parquet` -- `roc.screen` events go to `screens.parquet`
  - `test_export_routes_saliency_to_saliency_parquet` -- `roc.attention.saliency` -> `saliency.parquet`
  - `test_export_routes_named_events_to_events_parquet` -- other named events -> `events.parquet`
  - `test_export_routes_unnamed_to_logs_parquet` -- no event.name -> `logs.parquet`
  - `test_step_counter_increments_on_screen_event` -- step number increments with each `roc.screen`
  - `test_game_counter_increments_on_game_start` -- game_number increments on `roc.game_start`
  - `test_flush_interval_triggers_write` -- data flushed after N steps
  - `test_shutdown_flushes_remaining` -- all buffered data written on shutdown
  - `test_append_mode` -- second flush appends to existing Parquet files, not overwrites
  - `test_parquet_has_step_and_game_columns` -- every row has `step` and `game_number`
  - `test_record_to_dict_preserves_attributes` -- OTel attributes appear as columns
  - `test_record_to_dict_preserves_body` -- log body serialized correctly
  - `test_record_to_dict_preserves_timestamp` -- nanosecond timestamp preserved

  ```python
  def test_export_routes_screen_to_screens_parquet(tmp_path: Path) -> None:
      exporter = ParquetExporter(run_dir=tmp_path, flush_interval=1)
      record = make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')
      exporter.export([record])
      assert (tmp_path / "screens.parquet").exists()
      df = pd.read_parquet(tmp_path / "screens.parquet")
      assert len(df) == 1
      assert df.iloc[0]["step"] == 1
  ```

**Implementation**:
- `roc/reporting/parquet_exporter.py`: New file (~150 lines)
  - `ParquetExporter(LogExporter)` class
    - `__init__(run_dir, flush_interval=100)`
    - `export(batch)` -- route records to named buffers, increment step/game counters
    - `shutdown()` -- flush all remaining buffers
    - `_flush_all()` -- write each buffer to its Parquet file (append if exists)
    - `_record_to_dict(record)` -- convert OTel LogRecord to flat dict
  - Buffer routing rules:
    - `roc.screen` -> `screens`
    - `roc.attention.saliency` -> `saliency`
    - named events -> `events`
    - unnamed (loguru) -> `logs`

- `roc/reporting/observability.py`: Modify
  - Import and instantiate `ParquetExporter` in `Observability._init_logging()`
  - Add it to the logger provider's processor chain
  - Use `Config.data_dir / instance_id` as the run directory
  - Remove `JsonlFileExporter` setup (or gate behind a legacy flag initially)

- `roc/config.py`: Modify
  - Keep `data_dir` as-is (already `/home/apowers/data`)
  - Mark `debug_log` and `debug_log_path` as deprecated (remove in Phase 2)

- `pyproject.toml`: Modify
  - Add `pyarrow>=15.0.0` to dependencies

**Dependencies**:
- External: `pyarrow`
- Internal: `roc/reporting/observability.py` (OTel pipeline), `roc/config.py`

**Verification**:
1. Run: `uv run play` (play for a few ticks, then quit)
2. Run: `ls /home/apowers/data/<latest-run>/`
3. Expected: `screens.parquet`, `saliency.parquet`, `events.parquet`, `logs.parquet`
4. Run: `uv run python -c "import pandas as pd; df = pd.read_parquet('/home/apowers/data/<run>/screens.parquet'); print(df.columns.tolist()); print(len(df))"`
5. Expected: columns include `step`, `game_number`, `timestamp`, `body`; row count matches ticks played

---

### Phase 2: RunStore -- DuckDB Query Layer

**What this phase accomplishes:**
Build the `RunStore` class that queries Parquet files via DuckDB. This is the data access
layer for the dashboard. After this phase, you can load a run and query per-step data
from a Python REPL.

**Duration**: 1-2 days

**Tests to Write First**:
- `tests/unit/test_run_store.py`:
  - `test_step_count_returns_max_step` -- total steps across all games
  - `test_step_count_filtered_by_game` -- step count for a specific game
  - `test_get_step_returns_matching_rows` -- query single step from a table
  - `test_get_step_returns_empty_for_missing` -- missing step returns empty DataFrame
  - `test_list_games_returns_summary` -- game_number, steps, timestamps
  - `test_list_runs_finds_directories` -- scans data_dir for run directories
  - `test_list_runs_ignores_incomplete` -- skips dirs without `screens.parquet`
  - `test_get_step_data_assembles_all_sources` -- returns populated `StepData`
  - `test_get_step_data_handles_missing_tables` -- missing Parquet files -> None fields
  - `test_step_range_for_game` -- returns min/max step for a game

  ```python
  def test_get_step_returns_matching_rows(populated_run_dir: Path) -> None:
      store = RunStore(populated_run_dir)
      df = store.get_step(step=5, table="events")
      assert all(df["step"] == 5)
  ```

  Fixture `populated_run_dir` creates Parquet files with known test data using `ParquetExporter` from Phase 1.

**Implementation**:
- `roc/reporting/run_store.py`: New file (~200 lines)
  - `RunStore` class
    - `__init__(run_dir)` -- stores path, creates DuckDB in-memory connection
    - `get_step(step, table)` -> `pd.DataFrame`
    - `step_count(game_number=None)` -> `int`
    - `step_range(game_number=None)` -> `tuple[int, int]` (min, max step)
    - `list_games()` -> `pd.DataFrame` (game_number, steps, start_ts, end_ts)
    - `list_runs(data_dir)` -> `list[str]` (sorted run names)
    - `get_step_data(step)` -> `StepData` (assembles all sources)
    - `reload()` -- no-op (DuckDB reads fresh each query)
  - `StepData` dataclass
    - Fields per design: step, game_number, timestamp, screen, saliency, features,
      object_info, focus_points, attenuation, resolution_metrics, graph_summary,
      event_summary, game_metrics, logs

- `pyproject.toml`: Modify
  - Add `duckdb>=1.0.0` to dependencies

- Remove JSONL: Modify `roc/config.py`
  - Remove `debug_log` and `debug_log_path` fields
  - Remove `JsonlFileExporter` from `observability.py`
  - Clean up any references in tests

**Dependencies**:
- External: `duckdb`
- Internal: Phase 1 (`ParquetExporter` and Parquet file format)

**Verification**:
1. Run `uv run play` to generate Parquet data (or reuse from Phase 1)
2. Run:
   ```bash
   uv run python -c "
   from pathlib import Path
   from roc.reporting.run_store import RunStore
   store = RunStore(Path('/home/apowers/data/<latest-run>'))
   print('Steps:', store.step_count())
   print('Games:', store.list_games())
   sd = store.get_step_data(1)
   print('Screen present:', sd.screen is not None)
   print('Features:', sd.features)
   "
   ```
3. Expected: step count > 0, games listed, screen data present for step 1

---

### Phase 3: Minimal Panel Dashboard -- Screen + Step Slider

**What this phase accomplishes:**
Build the MVP dashboard: a Panel app with a step slider and the game screen display.
When you move the slider, the screen updates. This proves the full pipeline from Parquet
storage through DuckDB query to browser rendering.

**Duration**: 2 days

**Tests to Write First**:
- `tests/unit/test_panel_debug.py`:
  - `test_dashboard_creates_without_error` -- PanelDashboard instantiates with a RunStore
  - `test_dashboard_has_step_slider` -- layout contains a Player widget
  - `test_dashboard_has_run_selector` -- layout contains a Select widget
  - `test_update_screen_returns_html` -- given StepData with screen, returns HTML string
  - `test_update_screen_handles_none` -- missing screen data returns placeholder text
  - `test_step_change_triggers_query` -- changing step value calls RunStore.get_step_data
  - `test_game_selector_filters_steps` -- changing game updates slider range

  ```python
  def test_update_screen_returns_html(sample_step_data: StepData) -> None:
      dashboard = PanelDashboard(mock_store)
      html = dashboard._update_screen(sample_step_data)
      assert "<span" in html
      assert "monospace" in html
  ```

**Implementation**:
- `roc/reporting/panel_debug.py`: New file (~300 lines initially)
  - `PanelDashboard` class
    - `__init__(store: RunStore)` -- creates widgets and layout
    - Widgets: `pn.widgets.Player` (step), `pn.widgets.Select` (run, game)
    - `_on_step_change(step)` -- queries RunStore, updates all panes
    - `_update_screen(data: StepData)` -> HTML string
    - `_on_run_change(run_name)` -- loads new RunStore, resets game/step
    - `_on_game_change(game_number)` -- filters step range
    - `servable()` -> `pn.Column` (the Panel app layout)
  - `main()` entry point
    - Parse `--run`, `--data-dir`, `--port` arguments
    - Create RunStore and PanelDashboard
    - Call `pn.serve()`

- `pyproject.toml`: Modify
  - Add `panel-debug = "roc.reporting.panel_debug:main"` to `[project.scripts]`

- `roc/reporting/screen_renderer.py`: Modify (minor)
  - Extract a `render_grid_pane(grid_data)` function that returns just the inner HTML
    (no full document wrapper) suitable for embedding in Panel's `pn.pane.HTML`

**Dependencies**:
- External: `panel` (already installed)
- Internal: Phase 2 (`RunStore`, `StepData`), `screen_renderer.py`

**Verification**:
1. Run: `uv run panel-debug --port 9042`
2. Open browser: `http://<server>:9042`
3. Expected: see run selector dropdown, game selector, step slider, and game screen
4. Move slider: screen updates to show different game states
5. Change game: slider range updates to that game's steps

---

### Phase 4: Full Panel Layout -- All Data Panels

**What this phase accomplishes:**
Add all remaining data panels: saliency map, feature report, object resolution, focus
points, saliency attenuation, game metrics, graph DB summary, event bus activity, and
log messages. Organize layout by pipeline stage per the design.

**Duration**: 2-3 days

**Tests to Write First**:
- `tests/unit/test_panel_debug.py` (extend):
  - `test_update_saliency_returns_html` -- saliency grid renders correctly
  - `test_update_features_returns_text` -- feature counts displayed
  - `test_update_object_returns_text` -- object resolution info displayed
  - `test_update_focus_returns_text` -- focus points displayed
  - `test_update_attenuation_returns_text` -- attenuation details displayed
  - `test_update_metrics_returns_text` -- game metrics (HP, score, etc.)
  - `test_update_graph_returns_text` -- graph DB summary
  - `test_update_events_returns_text` -- event bus activity
  - `test_update_logs_returns_filtered_text` -- log messages with level filtering
  - `test_log_level_filter_excludes_debug` -- filtering to INFO hides DEBUG
  - `test_layout_has_all_sections` -- Perception, Attention, Object Resolution, Game State, Logs

  ```python
  def test_update_saliency_returns_html(sample_step_data: StepData) -> None:
      dashboard = PanelDashboard(mock_store)
      html = dashboard._update_saliency(sample_step_data)
      assert "<span" in html
  ```

**Implementation**:
- `roc/reporting/panel_debug.py`: Extend (~200 more lines)
  - Add update methods: `_update_saliency()`, `_update_features()`,
    `_update_object()`, `_update_focus()`, `_update_attenuation()`,
    `_update_metrics()`, `_update_graph()`, `_update_events()`, `_update_logs()`
  - Add `pn.widgets.Select` for log level filtering
  - Build full layout with pipeline-stage grouping:
    - Perception section: screen + features side by side
    - Attention section: saliency + focus points, attenuation below
    - Object Resolution section: decision + metrics side by side
    - Game State section: vitals + graph DB + events
    - Log Messages section: scrollable, level-filtered

- `roc/reporting/state.py`: Extend
  - Emit new OTel log records for data not yet emitted:
    - `roc.saliency_attenuation` -- already emitted by `saliency_attenuation.py`
    - `roc.game_metrics` -- HP, score, depth, etc. from `BlstatsState`
    - `roc.graphdb.summary` -- node/edge counts from `NodeCacheState`/`EdgeCacheState`
    - `roc.event.summary` -- event counts (requires new instrumentation)
  - Add these to `emit_state_logs()` so they flow through the OTel pipeline

- `roc/reporting/parquet_exporter.py`: Extend routing
  - Route `roc.game_metrics` -> `metrics.parquet`
  - Other new events continue going to `events.parquet`

- `roc/reporting/run_store.py`: Extend
  - `get_step_data()` now populates all StepData fields from appropriate Parquet files
  - Parse JSON bodies for structured data (screen, saliency, attenuation, metrics)

**Dependencies**:
- Internal: Phases 1-3

**Verification**:
1. Run `uv run play` (generate fresh data with new OTel records)
2. Run: `uv run panel-debug --port 9042`
3. Expected: all panels visible in pipeline-stage layout
4. Move slider: all panels update simultaneously
5. Change log level filter: log panel shows only matching levels
6. Verify saliency map renders with blue-to-red heatmap colors

---

### Phase 5: Keyboard Shortcuts + Bookmarks

**What this phase accomplishes:**
Add keyboard-driven navigation and bookmark functionality. Users can step through frames
with arrow keys, play/pause, jump to bookmarks, and see a help overlay.

**Duration**: 2 days

**Tests to Write First**:
- `tests/unit/test_panel_debug.py` (extend):
  - `test_bookmark_toggle_adds_bookmark` -- pressing bookmark adds current step
  - `test_bookmark_toggle_removes_existing` -- pressing bookmark on bookmarked step removes it
  - `test_bookmark_persists_to_json` -- bookmarks saved to `bookmarks.json`
  - `test_bookmark_loads_from_json` -- bookmarks loaded when run is selected
  - `test_bookmark_navigation_next` -- next bookmark jumps to correct step
  - `test_bookmark_navigation_prev` -- previous bookmark jumps to correct step
  - `test_bookmark_with_annotation` -- annotation text stored and displayed
  - `test_help_overlay_content` -- help overlay contains all shortcut descriptions

  ```python
  def test_bookmark_persists_to_json(tmp_path: Path) -> None:
      dashboard = PanelDashboard(mock_store)
      dashboard._toggle_bookmark(step=42, annotation="spike here")
      bookmarks_file = tmp_path / "bookmarks.json"
      data = json.loads(bookmarks_file.read_text())
      assert len(data) == 1
      assert data[0]["step"] == 42
      assert data[0]["annotation"] == "spike here"
  ```

**Implementation**:
- `roc/reporting/panel_debug.py`: Extend (~150 more lines)
  - `BookmarkManager` class
    - `__init__(run_dir)` -- loads `bookmarks.json` if exists
    - `toggle(step, game, annotation=None)` -- add/remove bookmark
    - `next_bookmark(current_step)` -> `int | None`
    - `prev_bookmark(current_step)` -> `int | None`
    - `save()` -- write to `bookmarks.json`
    - `load()` -- read from `bookmarks.json`
    - `as_list()` -> `list[dict]` for display
  - Keyboard shortcuts via custom JS callback on `pn.pane.HTML`:
    - Register `document.addEventListener('keyup', ...)` in a hidden HTML pane
    - JS sends events to Python via Bokeh's `CustomJS` -> widget value changes
    - Map keys per R16: arrows, space, +/-, Home/End, g, b, n, p, ], [, ?, h
  - Help overlay: toggleable `pn.pane.HTML` with shortcut table
  - Bookmark list panel: shows bookmarked steps with annotations, clickable

**Dependencies**:
- Internal: Phases 1-4

**Verification**:
1. Run: `uv run panel-debug --port 9042`
2. Press right arrow: step advances by 1
3. Press left arrow: step goes back by 1
4. Press space: auto-play starts/stops
5. Press `b`: bookmark appears in bookmark list
6. Press `n`: jumps to next bookmark
7. Press `?`: help overlay appears showing all shortcuts
8. Restart dashboard: bookmarks still present

---

### Phase 6: Live Mode + Multi-Game Support

**What this phase accomplishes:**
Support live monitoring of in-progress runs and proper multi-game navigation. The
dashboard polls for new Parquet data and auto-advances when following the latest step.

**Duration**: 2 days

**Tests to Write First**:
- `tests/unit/test_parquet_exporter.py` (extend):
  - `test_periodic_flush_writes_partial_data` -- data available before shutdown
  - `test_game_boundary_increments_game_number` -- game_number column correct across games
  - `test_flush_during_game_preserves_game_number` -- mid-game flush has correct game_number

- `tests/unit/test_panel_debug.py` (extend):
  - `test_live_mode_detects_new_steps` -- step count increases after new data written
  - `test_live_mode_auto_advances_when_following` -- at latest step, auto-advances with new data
  - `test_live_mode_does_not_advance_when_reviewing` -- at earlier step, stays put
  - `test_live_badge_visible_when_following` -- "LIVE" indicator shown
  - `test_game_selector_shows_games` -- game dropdown populated
  - `test_game_change_updates_step_range` -- switching game adjusts slider min/max
  - `test_game_selector_shows_summary` -- step count and score per game

  ```python
  def test_live_mode_detects_new_steps(tmp_path: Path) -> None:
      exporter = ParquetExporter(run_dir=tmp_path, flush_interval=5)
      # Write 10 steps
      for i in range(10):
          exporter.export([make_screen_record()])
      store = RunStore(tmp_path)
      assert store.step_count() == 10
      # Write 5 more steps
      for i in range(5):
          exporter.export([make_screen_record()])
      store.reload()
      assert store.step_count() == 15
  ```

**Implementation**:
- `roc/reporting/parquet_exporter.py`: Extend
  - Background flush thread (timer-based, configurable interval)
  - Thread-safe buffer access (lock around `_buffers`)
  - `force_flush()` method for game-end and shutdown

- `roc/reporting/panel_debug.py`: Extend (~100 more lines)
  - `_poll_for_updates()` -- periodic callback (every 2s) checking for new data
  - Follow mode: track whether user is at latest step
  - "LIVE" badge in header when following
  - "New data available" indicator when not following
  - Game selector: `pn.widgets.Select` populated from `RunStore.list_games()`
  - Game change handler: update step slider range to game's step range

- `roc/reporting/run_store.py`: Extend
  - `has_new_data()` -> `bool` (check file modification time or step count)
  - `step_range(game_number)` -> `tuple[int, int]`

**Dependencies**:
- Internal: Phases 1-5

**Verification**:
1. Start a game: `uv run play` (in one terminal)
2. Start dashboard: `uv run panel-debug --port 9042` (in another terminal)
3. Expected: dashboard shows data, "LIVE" badge visible
4. As game progresses: slider range expands, screen auto-advances
5. Move slider back to step 1: auto-advance stops, "LIVE" badge disappears
6. Move to latest step: auto-advance resumes
7. If run has multiple games: game selector shows both, switching works

---

### Phase 7: ServHerd Integration + Polish

**What this phase accomplishes:**
Register the dashboard as a ServHerd service, add autoreload for development, and
polish the UI (styling, error handling, edge cases).

**Duration**: 1-2 days

**Tests to Write First**:
- `tests/unit/test_panel_debug.py` (extend):
  - `test_main_with_no_runs_shows_message` -- empty data_dir shows helpful message
  - `test_main_with_run_arg_loads_specific_run` -- `--run` flag selects correct run
  - `test_run_selector_updates_on_refresh` -- new runs appear in selector

**Implementation**:
- ServHerd config: Add Panel dashboard to `.servherd/` or equivalent config
  - Service name: `panel-debug`
  - Command: `panel serve roc/reporting/panel_debug.py --port 90XX --autoreload`
  - Random port in 9000-9099 range

- `roc/reporting/panel_debug.py`: Polish
  - Error handling: missing Parquet files, empty runs, malformed data
  - Loading states: show "Loading..." while querying
  - Empty states: show helpful messages when no data
  - CSS: pipeline section headers, compact text panels, scrollable logs
  - Responsive width (fill available browser width)

- Entry point: Ensure `uv run panel-debug` works end-to-end
  - `--port` defaults to random 9000-9099
  - `--autoreload` for dev mode
  - `--data-dir` override

**Dependencies**:
- Internal: All previous phases
- External: ServHerd (already available)

**Verification**:
1. Run: `servherd start panel-debug`
2. Check: `servherd info panel-debug` shows running status
3. Open browser: dashboard loads correctly
4. Edit `panel_debug.py`: dashboard auto-reloads
5. Run: `servherd stop panel-debug`
6. Run: `servherd restart panel-debug`

---

### Phase 8: Remove W&B Integration

**What this phase accomplishes:**
Remove all Weights & Biases code, config, dependencies, and tests. After Phases 1-7,
the Panel dashboard + Parquet replaces W&B media panels, and OTel/Grafana already handles
aggregate metrics. Sweeps (if needed later) will use Optuna. This phase eliminates a
significant amount of dead code and a heavy dependency.

**Duration**: 1-2 days

**Inventory of W&B code to remove:**

| File | What to Remove |
|------|---------------|
| `roc/reporting/wandb_reporter.py` | **Delete entire file** (286 lines) |
| `roc/reporting/metrics.py` | Remove W&B dispatch: `log_step()`, `log_media()`, `flush_step()`, W&B calls inside `record_histogram()` and `increment_counter()` |
| `roc/config.py` | Remove all 11 `wandb_*` config fields |
| `roc/__init__.py` | Remove `WandbReporter` import and `WandbReporter.init()` call |
| `roc/gymnasium.py` | Remove `WandbReporter` import, `start_game()`, `end_game()`, `finish()` calls; remove `RocMetrics.log_step()`, `log_media()`, `flush_step()` calls |
| `roc/attention.py` | Remove `RocMetrics.log_media("saliency_map", ...)` call |
| `roc/reporting/screen_renderer.py` | Remove W&B reference in docstring |
| `tests/unit/test_wandb_reporter.py` | **Delete entire file** (476 lines) |
| `tests/unit/test_config_wandb.py` | **Delete entire file** (72 lines) |
| `tests/unit/test_metrics.py` | Remove W&B-specific tests; keep OTel metric tests |
| `tests/integration/test_wandb_metrics.py` | **Delete entire file** (96 lines) |
| `pyproject.toml` | Remove `wandb>=0.25.1` from dependencies; remove `wandb` from deptry excludes |
| `docs/reference/reporting/wandb_reporter.md` | **Delete file** |
| `tmp/test_wandb.py`, `tmp/log_screens_wandb.py`, etc. | **Delete** W&B prototype scripts |
| `wandb/` directory | **Delete** archived W&B run data |
| `design/wandb-design.md` | Keep as historical reference (no code impact) |
| `design/wandb-implementation-plan.md` | Keep as historical reference (no code impact) |

**Tests to Write/Modify First**:
- `tests/unit/test_metrics.py`: Rewrite to test OTel-only `RocMetrics`
  - `test_record_histogram_sends_to_otel` -- OTel histogram recorded
  - `test_increment_counter_sends_to_otel` -- OTel counter incremented
  - `test_no_wandb_imports` -- verify wandb is not imported anywhere in roc/
  - Remove: `test_log_step_*`, `test_flush_step_*`, any W&B mock assertions

  ```python
  def test_record_histogram_sends_to_otel() -> None:
      RocMetrics.record_histogram("test.metric", 42.0, description="test")
      # Assert OTel histogram was recorded (no W&B side-effects)

  def test_no_wandb_references() -> None:
      """Ensure wandb is fully removed from the codebase."""
      import ast
      from pathlib import Path
      roc_dir = Path("roc")
      for py_file in roc_dir.rglob("*.py"):
          source = py_file.read_text()
          tree = ast.parse(source)
          for node in ast.walk(tree):
              if isinstance(node, ast.Import):
                  for alias in node.names:
                      assert "wandb" not in alias.name, f"wandb import in {py_file}"
              if isinstance(node, ast.ImportFrom) and node.module:
                  assert "wandb" not in node.module, f"wandb import in {py_file}"
  ```

- `tests/unit/test_config_defaults.py` (or equivalent existing config test):
  - Verify no `wandb_*` fields exist on Config class

**Implementation** (ordered to keep tests passing at each step):

1. **Simplify `RocMetrics`** (`roc/reporting/metrics.py`):
   - Remove `log_step()`, `log_media()`, `flush_step()` methods
   - Remove W&B calls from `record_histogram()` and `increment_counter()`
   - Remove `WandbReporter` import
   - What remains: OTel histogram/counter dispatch only (~40 lines)

2. **Remove call sites** in application code:
   - `roc/__init__.py`: Remove `WandbReporter` import and `WandbReporter.init()` call
   - `roc/gymnasium.py`: Remove `WandbReporter` import, `start_game()`, `end_game()`,
     `finish()` calls; remove `RocMetrics.log_step()`, `RocMetrics.log_media("screen", ...)`
     `RocMetrics.flush_step()` calls. Keep `RocMetrics.record_histogram()` and
     `RocMetrics.increment_counter()` calls (these still go to OTel)
   - `roc/attention.py`: Remove `RocMetrics.log_media("saliency_map", ...)` call.
     Keep `RocMetrics.record_histogram()` and `RocMetrics.increment_counter()` calls

3. **Remove config fields** (`roc/config.py`):
   - Delete all 11 `wandb_*` fields: `wandb_enabled`, `wandb_project`, `wandb_entity`,
     `wandb_host`, `wandb_api_key`, `wandb_tags`, `wandb_log_screens`,
     `wandb_log_saliency`, `wandb_log_interval`, `wandb_artifacts`, `wandb_mode`

4. **Delete W&B files**:
   - `roc/reporting/wandb_reporter.py`
   - `tests/unit/test_wandb_reporter.py`
   - `tests/unit/test_config_wandb.py`
   - `tests/integration/test_wandb_metrics.py`
   - `docs/reference/reporting/wandb_reporter.md`

5. **Clean up dependencies** (`pyproject.toml`):
   - Remove `wandb>=0.25.1` from `[project.dependencies]`
   - Remove `"wandb"` from deptry's `exclude` list in `[tool.deptry.per_rule_ignores]`

6. **Clean up temp files**:
   - Delete `tmp/test_wandb.py`, `tmp/log_screens_wandb.py`,
     `tmp/wandb_report_sample.py`, `tmp/test_wandb_sync.py`,
     `tmp/fetch_wandb_media.py`, `tmp/fetch_new_saliency.py`,
     `tmp/fetch_verify_render.py`
   - Delete `wandb/` directory (archived run data)

7. **Update tests**:
   - Rewrite `tests/unit/test_metrics.py` to test OTel-only behavior
   - Add `test_no_wandb_references` regression test
   - Run `make test` and `make lint` to catch any remaining references

8. **Update docs/comments**:
   - Remove W&B reference from `screen_renderer.py` docstring
   - Update panel-design.md relationship table (W&B row -> "Removed")

**Dependencies**:
- Internal: Phases 1-4 must be complete (ParquetExporter replaces W&B media logging;
  game metrics emitted via OTel). Phase 7 should be complete so Panel dashboard is the
  verified replacement.
- External: None removed beyond `wandb` itself

**Verification**:
1. Run: `make test` -- all tests pass, no wandb-related failures
2. Run: `make lint` -- no type errors or import errors
3. Run: `uv run python -c "import roc; print('ok')"` -- no import errors
4. Run: `uv run play` -- game runs without wandb, metrics still flow to OTel/Grafana
5. Run: `grep -r "wandb" roc/ --include="*.py"` -- no results (zero references)
6. Run: `grep -r "WandbReporter" roc/ tests/ --include="*.py"` -- no results
7. Run: `uv run panel-debug --port 9042` -- dashboard still works (unaffected)
8. Verify `uv.lock` no longer includes wandb or its transitive dependencies

---

## Common Utilities Needed

- **`make_log_record()` test helper**: Factory for creating OTel LogRecord objects with
  configurable event.name, body, and attributes. Used across `test_parquet_exporter.py`
  and `test_run_store.py`. Place in `tests/helpers/otel.py`.

- **`populated_run_dir` pytest fixture**: Creates a temporary directory with known
  Parquet files for testing RunStore and PanelDashboard. Uses `ParquetExporter` to write
  realistic test data. Place in `tests/conftest.py` or a test-specific conftest.

## External Libraries Assessment

| Library | Purpose | Status |
|---------|---------|--------|
| `pyarrow` (>=15.0.0) | Parquet file I/O (read/write/append) | **Add** -- standard for Parquet in Python |
| `duckdb` (>=1.0.0) | In-process SQL queries over Parquet | **Add** -- zero-ops analytical query engine |
| `panel` (>=1.6.2) | Dashboard framework | **Already installed** |
| `bokeh` (>=3.7.2) | Underlying server/rendering | **Already installed** (with panel) |
| `pandas` (>=2.2.2) | DataFrame handling | **Already installed** |
| `wandb` (>=0.25.1) | W&B experiment tracking | **Remove in Phase 8** -- replaced by Parquet + Panel + OTel/Grafana |

## Risk Mitigation

- **Parquet append performance**: Reading + concatenating + rewriting entire Parquet file
  on each flush could be slow for large runs. Mitigation: use row-group appending via
  `pyarrow.parquet.ParquetWriter` in append mode, or write separate chunk files and
  merge on shutdown. Monitor write times in Phase 1 and optimize if > 100ms.

- **Large grid HTML rendering**: 21x79 grid = 1659 `<span>` elements per grid, two grids
  per step. Panel/Bokeh must update these on each step change. Mitigation: if rendering
  is slow (> 200ms), switch to `<canvas>` rendering or pre-render grids as PNG images.
  Measure in Phase 3 before optimizing.

- **DuckDB query latency**: Each step change triggers multiple DuckDB queries (one per
  Parquet file). Mitigation: batch into a single query with joins, or prefetch adjacent
  steps. Measure in Phase 3; target < 50ms per step change.

- **OTel pipeline coupling**: Adding `ParquetExporter` to the OTel logger provider adds
  processing to every log record. Mitigation: `ParquetExporter.export()` only buffers
  in memory (dict append); actual I/O happens on flush. Memory impact is bounded by
  flush interval.

- **Missing data for new OTel records**: Game metrics, graph DB summary, and event bus
  activity are not yet emitted as OTel log records. Mitigation: Phase 4 adds these
  emissions. Until then, dashboard shows "No data" for those panels. Phases 1-3 work
  with existing data (screen, saliency, features, objects, focus points).

- **Keyboard shortcuts in Panel**: Panel's JS integration for keyboard events can be
  finicky. Mitigation: use a simple hidden HTML pane with `document.addEventListener`
  and Bokeh CustomJS callbacks. Test in Phase 5 before building complex interactions.
  Fall back to Panel's built-in widget controls if custom JS proves unreliable.

- **W&B removal breaking call sites**: `gymnasium.py` and `attention.py` call
  `RocMetrics.log_step()`, `log_media()`, and `flush_step()` which are W&B-only methods.
  Mitigation: remove these calls entirely (they have no OTel equivalent needed -- the
  data already flows via OTel log records to Parquet). Keep `record_histogram()` and
  `increment_counter()` calls which serve OTel. Ordered implementation in Phase 8
  ensures no step breaks `make test`.
