# Panel Dashboard Design: Per-Step Game State Debugger

## Problem Statement

When debugging the ROC agent, we need to review multiple types of data at the same game
step simultaneously: the game screen, saliency map, recognized objects, resolved features,
log messages, intrinsic values, and action decisions. Currently these are split across
separate W&B HTML panels with independent step sliders -- changing the step on the screen
panel does not update the saliency panel or any text-based data. W&B's "Sync slider by
key" feature only works for image/video/audio panels, not HTML or text. No experiment
tracking tool (W&B, TensorBoard, Neptune, Aim, ClearML) solves synchronized
heterogeneous-media stepping.

We need a custom debugging dashboard where a single step control synchronizes all views.

## Goals

1. Single step slider that synchronizes all data views for a given game tick
2. Display the game screen, saliency map, and text-based debugging data side by side
3. Support both live monitoring (during a running game) and post-hoc analysis (after a run)
4. Replace W&B's media panel functionality for per-step game state review
5. Keep the implementation simple and maintainable -- this is a developer debugging tool,
   not a production dashboard
6. Store per-step data durably for long-term archival and efficient querying

## Non-Goals

- Replacing Grafana for aggregate metrics, time-series graphs, or cross-run comparison
- Building a polished user-facing product -- this is a developer tool
- Real-time sub-second latency during gameplay (a few seconds of lag is acceptable)

## Requirements

### R1: Synchronized Step Navigation

A single slider or step control must update ALL visible panels simultaneously. When the
user moves to step 42, every panel must show data from step 42. This is the core
requirement that motivates the entire project.

The step control must support:
- Player widget with play/pause/speed controls for auto-advancing through steps
- Step forward/backward buttons for single-step (frame) navigation
- Slider for scrubbing through steps quickly
- Direct numeric input for jumping to a specific step
- Display of current step number and total steps available
- Game selector to filter steps to the selected game within a run

### R2: Game Screen Display

Display the NetHack game screen as a colored character grid, matching the current
rendering quality:
- 21 rows x 79 columns of ASCII characters
- Per-character foreground colors (16-color curses palette mapped to hex)
- Black background
- Monospace font rendering

Data source: `roc.screen` OTel log records. Format: `{"chars": int[][], "fg": hex[][],
"bg": hex[][]}`.

### R3: Saliency Map Display

Display the saliency heatmap as a colored character grid, side-by-side with the game
screen:
- Same 21x79 grid dimensions as the game screen
- Blue-to-red color scale indicating saliency strength
- Characters from the game screen overlaid on the heatmap colors

Data source: `roc.attention.saliency` OTel log records. Same `{"chars", "fg", "bg"}`
format as screen.

### R4: Feature Report

Display the feature extraction summary for the current step:
- Count of each feature type (Delta, Motion, Single, Color, Shape, etc.)
- Text-based display is fine

Data source: `roc.attention.features` OTel log records. Format: multi-line text with
tab-indented `name: count` pairs.

### R5: Object Resolution Details

Display what the object resolver decided at the current step:
- The resolved object identity (name/ID)
- Whether it was a match to an existing object or a new object
- Location (x, y) on the screen

Data source: `roc.attention.object` OTel log records. Format: string representation of
the resolved object.

### R6: Focus Points

Display the saliency focus points for the current step:
- List of (x, y, strength, label) tuples
- Sorted by strength descending

Data source: `roc.attention.focus_points` OTel log records. Format: string representation
of a pandas DataFrame.

### R7: Game Metrics

Display the agent's vital stats at the current step:
- HP / HP_MAX
- Energy / Energy_MAX
- Score, Depth, Gold, AC
- Hunger state
- XP level, Experience
- Player position (x, y)

Data source: per-step metrics logged via `RocMetrics.log_step()`. Stored in Parquet.

### R8: Saliency Attenuation Details

Display the saliency attenuation decision for the current step:
- Attenuation flavor (none, linear_decline, active_inference)
- Peak count after attenuation
- Whether the top peak shifted
- Pre/post peak locations
- Flavor-specific data (history size, entropy, omega, etc.)

Data source: `roc.saliency_attenuation` OTel log records. Format: JSON with structured
fields.

### R9: Log Messages

Display relevant log messages for the current step:
- Loguru messages emitted during this tick
- Filterable by log level (DEBUG, INFO, WARNING, ERROR)
- Scrollable if many messages

Data source: OTel log records without a specific `event.name` (general loguru output),
correlated by trace/span ID or timestamp proximity.

### R10: Graph Database Summary

Display per-step graph database state:
- Node count by label (e.g., Object, Feature, Allegiance)
- Edge count by type (e.g., HasFeature, TransformedTo)
- New nodes and edges created this step
- Cache utilization (node cache size, edge cache size)

Data source: `roc.graphdb.nodes`, `roc.graphdb.edges` counters; `roc.node_cache`,
`roc.edge_cache` gauges. These are currently Prometheus metrics -- for per-step display,
we will emit them as OTel log records alongside other step data.

### R11: Event Bus Activity

Display the events fired during the current step:
- Events grouped by bus/source
- Event count per bus
- Which components sent/received

Data source: `roc.event` counter (currently Prometheus). Will emit per-step event
summaries as OTel log records.

### R12: Object Resolution Metrics

Display resolution statistics for the current step alongside the R5 decision details:
- Number of candidate objects scanned
- Spatial distance to matched object
- Temporal gap (ticks since last seen)
- Posterior probability (Dirichlet) or symmetric difference score
- Confidence margin

Data source: `roc.resolution.*` and `roc.dirichlet.*` metrics. Will emit per-step as
OTel log records.

### R13: Data Storage -- Parquet + DuckDB

All per-step data must be stored in Parquet files for long-term archival and queried via
DuckDB for dashboard display. This replaces the JSONL debug log as the source of truth.

**Storage layout:**
```
<data_dir>/<run-name>/
    steps.parquet          -- one row per step (tick, timestamp, game_number)
    screens.parquet        -- screen grid data per step
    saliency.parquet       -- saliency grid data per step
    events.parquet         -- per-step event/log records (features, objects, focus, etc.)
    metrics.parquet        -- per-step game metrics (HP, score, etc.)
    logs.parquet           -- general log messages
```

**Configuration:**
- `data_dir` is configurable via `roc_data_dir` config option
- Default: `/var/data/roc`
- Each run gets its own directory: `/var/data/roc/<run-name>/`
- Run name uses the existing `instance` naming convention (e.g.,
  `20260313065209-uneasiest-ernestine-frankel`)

**Write path:**
- A custom OTel `LogExporter` called `ParquetExporter` replaces `JsonlFileExporter`
- Buffers records in memory during the run
- Flushes to Parquet files on game end or at configurable intervals
- Uses `pyarrow` for Parquet writing with snappy compression

**Read path:**
- DuckDB queries Parquet files in-process (no server)
- Returns pandas DataFrames natively -- ideal for Panel integration
- Example query:
  ```sql
  SELECT * FROM '/var/data/roc/20260313-run/events.parquet'
  WHERE event_name = 'roc.screen' AND step = 42
  ```

**Why Parquet + DuckDB:**
- Zero ops: no database server to maintain
- Excellent compression: 10-20x vs JSONL, critical for years of data
- Portable: Parquet files are self-describing, can be copied/rsync'd for backup
- Fast queries: DuckDB is columnar and optimized for analytical queries
- Native DataFrame returns: DuckDB -> pandas -> Panel with no conversion overhead
- Natural partitioning: one directory per run, trivial to archive or delete old runs

### R14: Run Selection

The dashboard must allow the user to select which run to view:
- Scan `data_dir` for run directories containing Parquet files
- Show run name (instance ID) and timestamp
- Show game count and step count per run
- Allow switching between runs without restarting the dashboard

### R15: Layout -- Pipeline-Organized

The dashboard layout follows the ROC processing pipeline, with panels grouped by stage.
All panels are synchronized to the current game and step.

```
+================================================================+
| ROC Debug Dashboard                                            |
+================================================================+
| [Run: 20260313-uneasiest-ernestine \/]                         |
| [Game: 1 \/]  [Step: |<  < [=====42=====] >  >|]  42 / 1847  |
+================================================================+
|                                                                |
|  PERCEPTION                                                    |
|  +---------------------------+------------------------------+  |
|  |                           |                               |  |
|  |    Game Screen            |    Feature Extractors         |  |
|  |    (21x79 grid)           |    Delta: 12                  |  |
|  |                           |    Motion: 8                  |  |
|  |                           |    Single: 5                  |  |
|  |                           |    Color: 3                   |  |
|  +---------------------------+------------------------------+  |
|                                                                |
|  ATTENTION                                                     |
|  +---------------------------+------------------------------+  |
|  |                           |                               |  |
|  |    Saliency Map           |    Focus Points               |  |
|  |    (21x79 grid)           |    (24, 5) str=35 label=0     |  |
|  |                           |    (12, 8) str=22 label=1     |  |
|  |                           |                               |  |
|  +---------------------------+------------------------------+  |
|  | Saliency Attenuation                                      |  |
|  | Flavor: active_inference  Peaks: 3                        |  |
|  | Top peak shifted: yes  (24,5) -> (12,8)                   |  |
|  | Entropy: 0.42  Omega: 0.15  History: 12                   |  |
|  +-----------------------------------------------------------+  |
|                                                                |
|  OBJECT RESOLUTION                                             |
|  +---------------------------+------------------------------+  |
|  | Resolution Decision       | Resolution Metrics            |  |
|  | Match: Object(uuid=...)   | Candidates: 14                |  |
|  | Location: (24, 5)         | Spatial dist: 2               |  |
|  | Decision: match           | Temporal gap: 0               |  |
|  |                           | Posterior: 0.87               |  |
|  |                           | Margin: 0.34                  |  |
|  +---------------------------+------------------------------+  |
|                                                                |
|  GAME STATE                                                    |
|  +---------------------------+------------------------------+  |
|  | Vitals                    | Graph DB                      |  |
|  | HP: 14/14                 | Nodes: Object=23 Feature=89  |  |
|  | Energy: 4/4               | Edges: HasFeature=89          |  |
|  | Score: 0  Depth: 1        | New this step: 2 nodes 4 edges|  |
|  | Gold: 0  AC: 6            | Cache: 45/1000 nodes          |  |
|  | Hunger: Not Hungry        |         12/500 edges          |  |
|  | XP: 1  Exp: 0             |                               |  |
|  | Pos: (24, 5)              | Events: 8 total               |  |
|  +---------------------------+------------------------------+  |
|                                                                |
|  LOG MESSAGES                                          [level] |
|  +-----------------------------------------------------------+  |
|  | INFO  [attention] Focus at (24, 5)                        |  |
|  | DEBUG [resolver] Matching 14 candidates                   |  |
|  | INFO  [resolver] Match: Object(uuid=abc...)               |  |
|  | DEBUG [graphdb] Created 2 nodes, 4 edges                  |  |
|  +-----------------------------------------------------------+  |
|                                                                |
+================================================================+
```

**Layout principles:**
- Grouped by pipeline stage: Perception, Attention, Object Resolution, Game State
- The two character grids (screen and saliency) are the largest elements
- Text panels are compact and scannable
- Log messages at the bottom with level filtering
- Game and step selectors always visible at the top
- All panels update simultaneously when game or step changes

### R16: Keyboard Shortcuts

The dashboard must support keyboard shortcuts for efficient navigation. A help overlay
(toggled by `?` or `h`) shows all available shortcuts.

| Key | Action |
|-----|--------|
| Right arrow / `l` | Step forward one frame |
| Left arrow / `j` | Step backward one frame |
| Space | Play / pause auto-advance |
| `+` / `-` | Increase / decrease playback speed |
| Home / `0` | Jump to first step |
| End / `$` | Jump to last step |
| `g` | Jump to step (opens numeric input) |
| `b` | Toggle bookmark on current step |
| `n` / `p` | Jump to next / previous bookmark |
| `]` / `[` | Next / previous game |
| `?` or `h` | Toggle keyboard shortcut help overlay |

Implementation: Panel supports `pn.bind` with `onkeyup` events via custom JS callbacks
registered on the document. The help overlay is a toggleable `pn.pane.HTML` panel.

### R17: Bookmarks

Users can bookmark interesting steps for later review. Bookmarks persist across dashboard
sessions.

**Features:**
- Toggle a bookmark on the current step with `b` key or a bookmark button
- Optional annotation text when creating a bookmark (e.g., "saliency spike here")
- Bookmark list panel showing all bookmarked steps with annotations
- Click a bookmark to jump to that step
- Navigate between bookmarks with `n` (next) and `p` (previous)
- Visual indicator on the step slider showing where bookmarks exist

**Storage:**
- Bookmarks are saved in a `bookmarks.json` file within the run directory:
  `/var/data/roc/<run-name>/bookmarks.json`
- Format: `[{"step": 42, "game": 1, "annotation": "saliency spike", "created": "..."}]`
- Loaded on run selection, saved on every bookmark change

### R18: Live Mode

The dashboard supports live monitoring of an in-progress run. When a run is active,
the dashboard periodically detects new data and updates.

**Design:**
- The `ParquetExporter` flushes buffers periodically (every N steps, configurable,
  default 100) in addition to flushing on game end and shutdown
- Flushing appends to existing Parquet files (using `pyarrow` append mode) rather than
  overwriting, so partial data is always available
- The dashboard polls the Parquet files at a configurable interval (default 2 seconds)
- When new steps are detected, the step slider range expands and a "new data" indicator
  appears
- If the user is on the latest step, the dashboard auto-advances to show new data
  (follow mode). Moving to an earlier step disables follow mode.
- A "LIVE" badge appears in the header when following an active run

**Crash safety:**
- Periodic flush ensures at most N steps of data are lost on crash
- `ParquetExporter.export()` remains synchronous -- the flush timer runs in a background
  thread

### R19: Multi-Game Runs

A single run typically contains multiple games. The dashboard must handle this:

- A game selector dropdown appears alongside the step selector
- Changing the game filters the step slider to only show steps from that game
- Game boundaries are derived from the step data: a new `roc.screen` record after a game
  reset (detected by the `roc.game_total` counter incrementing) starts a new game
- The `ParquetExporter` adds a `game_number` column to all records
- Summary stats per game (step count, duration, final score) appear in the game selector

### R20: Performance

- Loading a run (typically 500-2000 steps) from Parquet should complete within 1-2 seconds
- DuckDB queries for a single step should return in < 50ms
- Stepping between frames should feel instant (< 200ms total render)
- Memory: load step index eagerly, load step data lazily via DuckDB queries

### R21: Startup and Server Management

The dashboard runs as a Panel/Bokeh server managed by ServHerd:

**ServHerd integration:**
- Register as a ServHerd service with a name like `panel-debug`
- Listen on a port in the 9000-9099 range
- ServHerd handles start/stop/restart lifecycle

**Hot-reloading:**
- Panel supports `--autoreload` flag for development
- When source files change, the server restarts automatically
- ServHerd config should pass `--autoreload` in dev mode

**Entry point:**
```bash
uv run panel-debug                    # Start dashboard, auto-detect latest run
uv run panel-debug --run <run-name>   # Open specific run
```

**ServHerd start command:**
```bash
panel serve roc/reporting/panel_debug.py --port 90XX --autoreload
```

## Technical Approach

### Framework: Panel (HoloViz)

Panel is a Python dashboard framework that supports reactive programming -- a single
widget (the step slider) can drive updates to all other widgets via callbacks. It runs as
a Bokeh server accessible via web browser.

Key Panel features we will use:
- `pn.widgets.Player` -- step control with play/pause/speed and step-forward/step-back
- `pn.pane.HTML` -- render the colored character grids (screen and saliency)
- `pn.pane.Str` or `pn.pane.Markdown` -- render text panels (features, objects, logs)
- `pn.Column`, `pn.Row`, `pn.GridSpec` -- layout management
- `pn.widgets.Select` -- run and game selectors
- `param.depends` or `pn.bind` -- reactive updates when step changes
- Custom JS callbacks via `pn.pane.HTML` -- keyboard shortcut handling and help overlay

### Storage: Parquet + DuckDB

#### ParquetExporter (OTel LogExporter)

Replaces `JsonlFileExporter` in `roc/reporting/observability.py`. Approximately 100-150
lines of code.

```python
class ParquetExporter(LogExporter):
    """Write OTel log records to Parquet files, partitioned by run."""

    def __init__(self, run_dir: Path, flush_interval: int = 100) -> None:
        self.run_dir = run_dir
        self._buffers: dict[str, list[dict]] = defaultdict(list)
        self._step_counter = 0
        self._game_counter = 0
        self._flush_interval = flush_interval
        self._steps_since_flush = 0

    def export(self, batch: Sequence[LogRecord]) -> LogExportResult:
        for record in batch:
            event_name = (record.attributes or {}).get("event.name")
            entry = self._record_to_dict(record)

            if event_name == "roc.screen":
                self._step_counter += 1
                self._steps_since_flush += 1
            entry["step"] = self._step_counter
            entry["game_number"] = self._game_counter

            # Detect game boundary (game_total counter increment)
            if event_name == "roc.game_start":
                self._game_counter += 1
                entry["game_number"] = self._game_counter

            # Route to appropriate buffer based on event type
            if event_name in ("roc.screen",):
                self._buffers["screens"].append(entry)
            elif event_name in ("roc.attention.saliency",):
                self._buffers["saliency"].append(entry)
            elif event_name is not None:
                self._buffers["events"].append(entry)
            else:
                self._buffers["logs"].append(entry)

        # Periodic flush for live mode and crash safety
        if self._steps_since_flush >= self._flush_interval:
            self._flush_all()
            self._steps_since_flush = 0

        return LogExportResult.SUCCESS

    def shutdown(self) -> None:
        self._flush_all()

    def _flush_all(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name, records in self._buffers.items():
            if records:
                table = pa.Table.from_pylist(records)
                path = self.run_dir / f"{name}.parquet"
                if path.exists():
                    existing = pq.read_table(path)
                    table = pa.concat_tables([existing, table])
                pq.write_table(table, path, compression="snappy")
        self._buffers.clear()
```

#### DuckDB Query Layer

Thin wrapper for dashboard queries:

```python
import duckdb

class RunStore:
    """Query Parquet run data via DuckDB."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.conn = duckdb.connect()

    def get_step(self, step: int, table: str) -> pd.DataFrame:
        path = self.run_dir / f"{table}.parquet"
        return self.conn.execute(
            f"SELECT * FROM '{path}' WHERE step = ?", [step]
        ).df()

    def step_count(self, game_number: int | None = None) -> int:
        path = self.run_dir / "screens.parquet"
        if game_number is not None:
            return self.conn.execute(
                f"SELECT COUNT(*) FROM '{path}' WHERE game_number = ?",
                [game_number]
            ).fetchone()[0]
        return self.conn.execute(f"SELECT MAX(step) FROM '{path}'").fetchone()[0]

    def list_games(self) -> pd.DataFrame:
        """Return summary per game: game_number, step_count, first/last timestamp."""
        path = self.run_dir / "screens.parquet"
        return self.conn.execute(f"""
            SELECT game_number, COUNT(*) as steps,
                   MIN(timestamp) as start_ts, MAX(timestamp) as end_ts
            FROM '{path}' GROUP BY game_number ORDER BY game_number
        """).df()

    def list_runs(self, data_dir: Path) -> list[str]:
        return sorted(
            [d.name for d in data_dir.iterdir()
             if d.is_dir() and (d / "screens.parquet").exists()],
            reverse=True
        )

    def reload(self) -> None:
        """Re-read Parquet files to pick up new data (live mode)."""
        # DuckDB reads Parquet on each query, so no cache invalidation needed.
        # This method exists for the dashboard polling loop to call.
        pass
```

### Data Model

```python
@dataclass
class StepData:
    step: int
    game_number: int
    timestamp: int                   # nanosecond timestamp
    screen: dict | None              # parsed JSON from roc.screen
    saliency: dict | None            # parsed JSON from roc.attention.saliency
    features: str | None             # text from roc.attention.features
    object_info: str | None          # text from roc.attention.object
    focus_points: str | None         # text from roc.attention.focus_points
    attenuation: dict | None         # parsed JSON from roc.saliency_attenuation
    resolution_metrics: dict | None  # candidates, distance, posterior, etc.
    graph_summary: dict | None       # node/edge counts, cache stats
    event_summary: dict | None       # event counts by bus/source
    game_metrics: dict | None        # HP, score, depth, etc.
    logs: list[dict]                 # general log messages for this tick
```

### Grid Rendering

The screen and saliency grids use the same `{"chars", "fg", "bg"}` format. Rendered as
HTML with inline styles, matching the existing `render_grid_html()` in
`screen_renderer.py`:

```html
<div style="font-family: 'DejaVu Sans Mono', monospace; font-size: 14px;
            line-height: 1.15; background: #000; padding: 4px;">
  <span style="color: #ffffff">@</span>
  <span style="color: #646464">.</span>
  ...
  <br/>
  ...
</div>
```

### Reactive Update Flow

```
User changes run, game, or step selector
       |
       v
DuckDB query: fetch StepData for selected step from Parquet
       |
       v
All bound functions re-execute with new StepData:
  - update_screen()      -> renders screen grid HTML
  - update_saliency()    -> renders saliency grid HTML
  - update_features()    -> returns feature report text
  - update_object()      -> returns object resolution text + metrics
  - update_focus()       -> returns focus points text
  - update_attenuation() -> returns attenuation details
  - update_metrics()     -> returns game metrics
  - update_graph()       -> returns graph DB summary
  - update_events()      -> returns event bus activity
  - update_logs()        -> returns filtered log messages
       |
       v
Panel updates all panes in the browser simultaneously
```

### File Structure

```
roc/reporting/panel_debug.py       -- Main dashboard module
  - ParquetExporter class          -- OTel LogExporter writing Parquet files
  - RunStore class                 -- DuckDB query layer over Parquet
  - StepData dataclass             -- Per-step data container
  - GridRenderer                   -- HTML rendering for screen/saliency grids
  - PanelDashboard class           -- Panel app with reactive widgets
  - main() entry point             -- CLI startup
```

### Configuration

New config fields in `roc/config.py`:

```python
data_dir: str = Field(default="/var/data/roc")
```

The `debug_log` and `debug_log_path` config fields will be removed since JSONL is no
longer needed. The `ParquetExporter` is always active (data storage is not optional --
it is the source of truth).

### Entry Point

Add a script entry point in `pyproject.toml`:

```toml
[project.scripts]
panel-debug = "roc.reporting.panel_debug:main"
```

Usage:
```bash
uv run panel-debug                              # Auto-detect latest run
uv run panel-debug --run 20260313-uneasiest...  # Open specific run
uv run panel-debug --data-dir /other/path       # Override data directory
```

## Dependencies

New dependencies:
- `pyarrow` -- Parquet file I/O (write from exporter, read via DuckDB)
- `duckdb` -- In-process SQL queries over Parquet files

Already present:
- `panel` -- Dashboard framework
- `bokeh` -- Underlying server (installed with panel)
- `pandas` -- DataFrame handling (already a dependency)

## Relationship to Existing Tools

| Tool | Role After This Change |
|------|----------------------|
| **Panel dashboard** | Per-step game state debugging (new) |
| **Parquet files** | Source of truth for all per-step data (replaces JSONL) |
| **DuckDB** | Query engine for Parquet data (new) |
| **Grafana** | Aggregate metrics, time-series, cross-run comparison, live monitoring |
| **Remote Logger MCP** | Quick log checks during live runs |
| **DAP debugger** | Interactive breakpoint debugging |
| **W&B** | Removed -- sweeps move to Optuna, media panels replaced by Panel, metrics in Grafana |
| **JSONL debug log** | Removed -- replaced by Parquet |

## Resolved Decisions

1. **Player widget**: Yes. Use Panel's `Player` widget with play/pause/speed controls.
   Must also include step-forward and step-backward buttons for single-frame navigation.
   (See R1.)

2. **Live mode**: Yes, included in v1. Periodic Parquet flush + dashboard polling.
   (See R18.)

3. **Keyboard shortcuts**: Yes. Arrow keys, space, bookmarks, game navigation, and a
   help overlay toggled by `?`. (See R16.)

4. **Bookmarking**: Yes. Bookmarks persist in `bookmarks.json` within the run directory.
   (See R17.)

5. **Comparison mode**: Not needed at this time. May revisit later.

6. **Parquet flush interval**: Periodic flush (default every 100 steps) plus flush on
   game end and shutdown. (See R18.)

7. **Multi-game runs**: Yes, runs contain multiple games. Game boundaries derived from
   step data. Game selector filters the step slider. (See R19.)
