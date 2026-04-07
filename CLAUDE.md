# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROC (Reinforcement Learning of Concepts) is a Python agent system using a component-based, event-driven architecture. The agent perceives the game environment, identifies objects, tracks changes between frames, predicts future states, and decides actions. Currently tested against NetHack via Gymnasium.

### Package Layout

```
roc/
  framework/           # Infrastructure: component, config, event, expmod, logger, utils
  db/                  # Graph database: graphdb.py (Node, Edge, GraphDB, GraphCache)
  perception/          # Perception layer: base.py (Perception, FeatureNode), location.py
    feature_extractors/  # Game-specific extractors: color, delta, distance, flood, etc.
  pipeline/            # Processing pipeline stages
    attention/           # VisionAttention, saliency attenuation ExpMods
    object/              # Object identity resolution, ObjectInstance, ObjectTransform
    temporal/            # Sequencer (Frame), Transformer, Predict, Transformable
    action.py            # Action component, ActionRequest, TakeAction
    intrinsic.py         # IntrinsicNode, IntrinsicData, IntrinsicOp
    significance.py      # Significance component
  game/                # Game interface: gymnasium, game_manager, breakpoint
  cli/                 # CLI entry points: play, server, dashboard, cleanup
  reporting/           # Observability, data storage, dashboard API, state tracking
  jupyter/             # Jupyter/iPython magic commands
```

### Design Principle: Game-Agnostic Core

Everything from Attention onward (object resolution, sequencing, transforms, predictions, actions) must be generic -- not specific to NetHack or any other gym. Only the Perception layer (feature extractors) is game-specific. Object resolution, distance metrics, tracking, and all downstream logic must operate on abstract features without assuming anything about the game producing them.

### Architectural Invariants

These rules must never be violated regardless of local convenience. They are the structural load-bearing walls of the system.

1. **All inter-component communication MUST flow through named EventBuses.** Pipeline components (Attention, ObjectResolver, Sequencer, Transformer, Predict, Action, Significance) must never call methods directly on other pipeline component instances. Importing a component class to access its `.bus` class attribute is the only allowed cross-component reference.

2. **Game-specific code is confined to the perception layer.** Only `roc/game/gymnasium.py` and `roc/perception/feature_extractors/` may import from `nle` or `gymnasium`. The one permitted exception is `roc/reporting/state.py`, which imports `nle` for screen rendering in the reporting/display layer -- never in core pipeline logic.

3. **Algorithm selection goes through ExpMod, not hardcoded instantiation.** Object resolution, action selection, saliency attenuation, and prediction candidate/confidence scoring are all pluggable via `ExpMod.get(default="name")`. New algorithm implementations must register as ExpMod subclasses, not be wired in directly.

4. **EventBuses are declared as class-level attributes on Component subclasses.** No ad-hoc bus creation in functions, methods, or module scope. Each bus has a globally unique name enforced at creation time. The bus topology is fixed at class definition time.

5. **Database access is layered.** Graph operations go through `Node`/`Edge` API in `roc/db/graphdb.py` (never raw mgclient). Analytics/storage goes through `DuckLakeStore` in `ducklake_store.py` (never raw duckdb). No other modules make direct database calls.

6. **Pipeline components must not create raw threads.** All concurrency in the event pipeline flows through RxPY's `ThreadPoolScheduler`. Only `roc/reporting/` and `roc/game/game_manager.py` are permitted to use `threading.Thread` / `threading.Lock` directly.

7. **Config is a singleton accessed via `Config.get()`.** Never construct `Config()` directly. `Config.init()` initializes the singleton; `Config.get()` retrieves it. During pytest, env vars and `.env` are deliberately ignored to isolate tests.

8. **Edge connections must satisfy schema constraints.** Every `Edge` subclass defines `allowed_connections` listing valid `(src_label, dst_label)` pairs. When `db_strict_schema=True`, creating an edge that violates its schema raises an error. Do not add connections that are not in the schema -- update `allowed_connections` first.

9. **PHYSICAL vs RELATIONAL feature distinction matters.** Object identity resolution uses only PHYSICAL features (shape, color, spatial extent). RELATIONAL features (delta, motion, distance) are event-based and must not be used for identity matching. This is enforced in the resolution algorithms.

10. **One UI server, one API server.** Never start additional dashboard or preview servers. `roc-server` (FastAPI + Socket.io) and `roc-ui` (Vite dev) are managed by servherd. Games are started/stopped via REST API, not by restarting servers.

## Common Commands

```bash
# Setup
make setup              # Install deps + pre-commit hooks
make install            # Create venv (Python 3.13) and sync deps

# Testing
make test               # Run all tests (excludes slow and requires_observability)
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e           # End-to-end tests only
uv run pytest -c pyproject.toml tests/unit/pipeline/object/test_object.py                    # Run one test file
uv run pytest -c pyproject.toml tests/unit/pipeline/object/test_object.py -k test_name      # Run single test by name
uv run pytest -c pyproject.toml tests/unit/pipeline/object/test_object.py::TestClass::test   # Run specific test method

# Linting & Formatting
make lint               # Run mypy + check-codestyle + deptry
make check-codestyle    # Run ruff check + ruff format --check
make mypy               # Type checking only
make codestyle          # Auto-format with ruff format

# Coverage
make coverage           # Run tests with coverage (90% minimum threshold)

# Docs
make docs               # Build mkdocs
make edit-docs          # Serve docs locally on 0.0.0.0:9000

# Dashboard Server (managed by servherd)
make run                # Start roc-server (API) + roc-ui (Vite dev) via servherd
make stop               # Stop both servers
npx servherd list       # Show all managed servers and their status/ports/URLs
npx servherd info roc-ui  # Show details for a specific server
npx servherd logs roc-server  # View server logs
```

## Architecture

### Component System

Everything is built on `Component` (`roc/framework/component.py`) -- an abstract base class with:
- **Auto-registration**: Subclasses register themselves in a global `component_registry` via `__init_subclass__`. Each component has a unique `name`+`type` pair (as a `ComponentKey` tuple). Registration fails if either is missing or if the pair is already taken.
- **Auto-loading**: Components with `auto = True` are loaded at startup via `Component.init()`. Perception components are loaded separately from `Config.perception_components`.
- **Bus connections**: Components communicate via `connect_bus(bus)` which attaches them to typed EventBus channels and stores connections in `bus_conns` for cleanup.
- **Lifecycle**: `Component.init()` bootstraps the system. `Component.reset()` shuts down all loaded and active components. Individual components call `shutdown()` to dispose bus connections.
- **Tracking**: A `WeakSet[Component]` tracks all active instances, enabling component count monitoring and automatic cleanup on GC.

**Identification**: `ComponentId` is a `NamedTuple(type, name)`. `ComponentKey` is `tuple[ComponentName, ComponentType]` for registry lookups.

### Event System

Communication uses RxPy reactive streams (`roc/framework/event.py`):
- `EventBus[T]` -- typed communication channel (backed by `rx.Subject`). Names are globally unique (enforced at creation). Optional `cache_depth` retains recent events.
- `Event[T]` -- carries `data` payload + `src_id` (ComponentId of sender) + `bus` reference. Tracks per-bus event counts via `_step_counts` for telemetry.
- `BusConnection[T]` -- component-to-bus link. `send(data)` wraps in Event and publishes. `listen(callback, filter)` subscribes with optional filtering.
- All listeners run on a `ThreadPoolScheduler` with `cpu_count * 2` threads. Events are filtered to exclude self-sends by default (via `Component.event_filter()`).

### ExpMod System

`ExpMod` (`roc/framework/expmod.py`) provides runtime-swappable experimental modules -- a plugin architecture for algorithm selection:

- **Registration**: Like Components, uses `__init_subclass__` for auto-registration. Each ExpMod has a `modtype` (category, e.g. `"action"`) and a `name` (implementation, e.g. `"weighted"`). Abstract bases define `modtype` only; concrete implementations add `name`.
- **Registry**: `expmod_registry[modtype][name] -> instance`. Instances are auto-created at registration time.
- **Selection**: `ExpMod.get(default="name")` returns the active implementation for that modtype. `ExpMod.set(name, modtype)` activates one.
- **Config-driven**: `Config.expmods_use` lists `(modtype, name)` pairs to activate. `Config.expmods` and `Config.expmod_dirs` control which files to dynamically import.
- **External modules**: Live in `experiments/modules/` and are loaded at runtime via `ExpMod.import_file()`.

**Current modtypes and implementations**:

| modtype | Implementations | Location |
|---------|----------------|----------|
| `action` | `pass` (default in-tree), `weighted` (experiments/) | roc/pipeline/action.py, experiments/modules/actions.py |
| `object-resolution` | `symmetric-difference`, `dirichlet-categorical` | roc/pipeline/object/object.py |
| `saliency-attenuation` | `none`, `linear-decline`, `active-inference` | roc/pipeline/attention/saliency_attenuation.py |
| `prediction-candidate` | `object-based` | roc/pipeline/temporal/predict.py |
| `prediction-confidence` | `naive` | roc/pipeline/temporal/predict.py (or experiments/) |

**Pattern**: Define an abstract base with `modtype`, then concrete subclasses with `name`. Use `MyBase.get(default="impl")` at call sites.

### Data Pipeline

The agent processes each game step through an event-driven pipeline. Each stage is a Component that listens on typed EventBuses and sends to the next:

```
NethackGym                      [sends VisionData, AuditoryData, ProprioceptiveData]
  |                                      |
  |  perception bus                      |  intrinsic bus
  v                                      v
FeatureExtractors (9 total)       Intrinsic
  | [emit Feature + Settled]        |
  |  perception bus                 |---> Significance
  v                                 |        [significance bus]
VisionAttention                    |
  | [SaliencyMap -> focus points]  |
  |  attention bus                  |
  v                                 |
ObjectResolver                     |
  | [Object identity + matching]   |
  |  object bus                     |
  v                                 v
Sequencer  <--- also listens on action bus + intrinsic bus
  | [assembles Frame from objects, intrinsics, actions]
  |  sequencer bus
  v
Transformer
  | [computes Transform diffs between consecutive frames]
  |  transformer bus
  v
Predict
  | [applies transforms to predict future frames]
  |  predict bus
  v
Action  <--- also receives ActionRequest from Gymnasium
  | [selects action via ExpMod]
  |  action bus (TakeAction)
  v
NethackGym  [executes action in environment]
```

**Pipeline synchronization**: Action waits for both `ActionRequest` (from Gymnasium) and a prediction result (from Predict bus) before issuing `TakeAction`. Predict is the last step of the pipeline, so waiting for the prediction result naturally ensures the entire pipeline has completed. Do NOT add custom synchronization mechanisms (barriers, settled signals, gating) between pipeline stages and Action -- the predict bus is the synchronization point.

The Gymnasium's observation/action loop is synchronous: it sends observations, sends ActionRequest, then blocks waiting for TakeAction on the action bus cache (`cache_depth=10`). The game does not advance until Action responds.

**Auto-loaded components** (auto = True):
- `VisionAttention` (vision / attention)
- `CrossModalAttention` (cross-modal / attention) -- placeholder
- `ObjectResolver` (resolver / object)
- `Sequencer` (sequencer / sequencer)
- `Transformer` (transformer / transformer)
- `Predict` (predict / predict)
- `Significance` (significance / significance)
- `Action` (action / action)
- `Intrinsic` (intrinsic / intrinsic)

**Config-loaded perception components** (from `Config.perception_components`):
- delta, distance, flood, motion, single, line, color, shape, phoneme -- all type="perception"

**Reporting component** (not auto-loaded, initialized by `state.init()`):
- `StateComponent` (state / reporting) -- listens on all buses, emits OTel events

### Event Bus Topology

| Bus Name | Type Parameter | Owner | cache_depth |
|----------|---------------|-------|-------------|
| `perception` | `PerceptionData` | Perception | 0 |
| `attention` | `AttentionData` | Attention | 0 |
| `object` | `ObjectData` | ObjectResolver | 0 |
| `sequencer` | `Frame` | Sequencer | 0 |
| `transformer` | `TransformResult` | Transformer | 0 |
| `predict` | `PredictData` | Predict | 0 |
| `significance` | `SignificanceData` | Significance | 0 |
| `action` | `ActionData` | Action | 10 |
| `intrinsic` | `IntrinsicData` | Intrinsic | 0 |

The action bus uses `cache_depth=10` so Gymnasium can wait for the TakeAction response.

### Graph Database

`roc/db/graphdb.py` wraps Memgraph (via mgclient) with:
- **GraphDB** singleton: connection management, Cypher query execution (`raw_fetch`/`raw_execute`), export to multiple formats (gml, gexf, dot, graphml, json, cytoscape), NetworkX conversion.
- **Node** (Pydantic BaseModel): graph vertices with auto-save-on-modify. Features: auto-label generation from class hierarchy, LRU caching (`GraphCache`), CRUD via `create`/`load`/`update`/`save`/`delete`, query via `find`/`find_one`, DFS traversal via `walk(mode, edge_filter, node_filter)`, DOT/SVG rendering.
- **Edge** (Pydantic BaseModel): typed graph relationships. Features: `allowed_connections` schema validation (list of allowed src_label -> dst_label pairs), auto-type from class name, `connect(src, dst)` factory.
- **NodeList** / **EdgeList**: lazy-loading collection types. Fetch nodes/edges on iteration via cache. Support `select()` filtering by labels, type, or custom function.
- **Registries**: `node_registry`, `edge_registry`, `node_label_registry` -- auto-populated via `__init_subclass__`.
- **Schema mode**: `db_strict_schema` enforces `allowed_connections` on edge creation; `db_strict_schema_warns` controls warn vs raise.
- **IDs**: `NodeId` and `EdgeId` are `NewType("...", int)`. Negative IDs indicate unsaved objects.

CI requires a Memgraph service container. Tests that need the DB will fail without it.

### Key Abstractions

**Spatial**:
- **Point / PointCollection** (`roc/perception/location.py`): 2D grid primitives with `XLoc`/`YLoc` coordinate types. PointCollection supports iteration, visualization, and numpy array conversion.

**Perception**:
- **Perception** (`roc/perception/base.py`): Abstract base for feature extractors. Connects to perception bus.
- **FeatureExtractor[FeatureType]** (`roc/perception/base.py`): Generic base adding typed feature emission. Concrete implementations in `roc/perception/feature_extractors/`.
- **FeatureNode** (`roc/perception/base.py`): Graph node for an extracted feature. Has `kind` (PHYSICAL or RELATIONAL). PHYSICAL features (shape, color, spatial) are used for object identity; RELATIONAL features (delta, motion, distance) are event-based.
- **VisualFeature** (`roc/perception/base.py`): Base for visual features with caching. Subtypes: `PointFeature` (single location), `AreaFeature` (multiple adjacent locations).

**Object Identity**:
- **Object** (`roc/pipeline/object/object.py`): Persistent graph Node representing an identified entity. Fields: `uuid` (random 63-bit ID), `resolve_count`, `last_x`/`last_y`/`last_tick`, `annotations`. Properties: `feature_groups`, `features`, `frames`.
- **FeatureGroup** (`roc/pipeline/object/object.py`): Graph Node collecting feature nodes from one observation. Connected to features via `Detail` edges and to Object via `Features` edges.
- **ObjectResolutionExpMod** (`roc/pipeline/object/object.py`): Pluggable resolution algorithm. Implementations find candidate Objects via a reverse index (`_feature_to_objects`), then score matches.
- **ResolutionContext** (`roc/pipeline/object/object.py`): Spatial/temporal context (x, y, tick) passed to resolution algorithms.

**Temporal**:
- **Frame** (`roc/pipeline/temporal/sequencer.py`): Graph Node snapshot of game state at one tick. Connected to FeatureGroups, TakeAction, and IntrinsicNodes via `FrameAttribute` edges. Consecutive frames linked by `NextFrame` edges. `merge_transforms()` creates predicted frames.
- **Transformable** (`roc/pipeline/temporal/transformable.py`): ABC interface for change detection. Methods: `same_transform_type()`, `compatible_transform()`, `create_transform()`, `apply_transform()`.
- **Transform** (`roc/pipeline/temporal/transformable.py`): Graph Node representing the diff between two frame states. Connected to source/destination frames via `Change` edges.

**Agent State**:
- **IntrinsicNode** (`roc/pipeline/intrinsic.py`): Graph Node implementing Transformable. Tracks a single intrinsic (e.g. hp) with `raw_value` and `normalized_value`. Normalization via pluggable `IntrinsicOp` operators (IntOp, PercentOp, MapOp, BoolOp).
- **IntrinsicData** (`roc/pipeline/intrinsic.py`): Event payload carrying raw and normalized intrinsic values. `to_nodes()` converts to IntrinsicNode list.

**Actions**:
- **ActionRequest** (`roc/pipeline/action.py`): Dataclass signal that Gymnasium is waiting for an action.
- **TakeAction** (`roc/pipeline/action.py`): Graph Node carrying the selected action ID (int). Attached to Frame via FrameAttribute edges.
- **DefaultActionExpMod** (`roc/pipeline/action.py`): Pluggable action selection. In-tree default: `DefaultActionPass` (always action 19). Production: `WeightedAction` in experiments/modules/.

### Reporting and Observability

The `roc/reporting/` package provides a full observability stack:

**OpenTelemetry Integration** (observability.py):
- Singleton `Observability` with deferred initialization. Provides logging (BatchLogRecordProcessor), metrics (PeriodicExportingMetricReader), tracing (BatchSpanProcessor), and profiling (Pyroscope).
- Exports to GRPC endpoint at `observability_host` (default: hal.ato.ms:4317).
- Each ROC instance gets a unique `instance_id` via FlexiHumanHash (e.g. `20260324-gentle-alice-smith`).
- `RocMetrics` (metrics.py): static helpers for `record_histogram()` and `increment_counter()`.

**Data Storage** (ducklake_store.py, parquet_exporter.py):
- DuckDB + DuckLake catalog with Parquet backend. One catalog per run in `data_dir/instance_id/`.
- `ParquetExporter`: OTel LogExporter that routes records to DuckLake tables by event name (`screens`, `saliency`, `events`, `metrics`, `logs`). Background thread drains queue -- game loop never blocks on DB writes.
- Thread-safe via lock (DuckDB is not thread-safe). Periodic CHECKPOINT merges small files.

**Dashboard Data Flow** (state.py, data_store.py, step_buffer.py):
- `StateComponent` listens on all event buses, tracks current state, emits structured OTel events.
- `StepBuffer`: thread-safe ring buffer (~2000 capacity) for live step data.
- `DataStore`: unified query layer. Live runs use in-memory `_GameIndex` per game; historical runs query DuckLake via `RunStore`.
- Game subprocess POSTs StepData to dashboard server via HTTP callback (`/api/internal/step`).

**Remote Logging** (remote_logger_exporter.py):
- OTel LogExporter that POSTs records to remote logger MCP server for live monitoring.

### Config

`Config` (`roc/framework/config.py`) uses pydantic-settings, reads from `.env` file with `roc_` prefix. During pytest, env vars and .env file are deliberately ignored to isolate tests. Key categories:

- **Database**: `db_host`, `db_port`, `db_lazy`, `db_strict_schema`, `node_cache_size`, `edge_cache_size`
- **Logging**: `log_enable`, `log_level`, `log_modules`
- **Observability**: `observability_logging`, `observability_metrics`, `observability_tracing`, `observability_profiling`, `observability_host`
- **Dashboard**: `dashboard_enabled`, `dashboard_port`, `dashboard_callback_url`, `emit_state`, `emit_state_screen`, `emit_state_saliency`, `emit_state_features`
- **Gymnasium**: `num_games`, `nethack_max_turns`, `nethack_extra_options`
- **ExpMods**: `expmod_dirs`, `expmods`, `expmods_use` -- controls which experimental modules to load and activate
- **Perception**: `perception_components` -- list of (name, type) pairs to load
- **Intrinsics**: `intrinsics` -- discriminated union list (percent, map, int, bool types) defining agent internal state tracking
- **Saliency attenuation**: `saliency_attenuation_*` -- parameters for linear-decline and active-inference algorithms
- **Significance**: `significance_weights` -- per-intrinsic weight map
- **Debug**: `debug_port`, `debug_wait`, `debug_remote_log`, `debug_remote_log_url`, `debug_snapshot_interval`
- **Graph**: `graphdb_export`, `graphdb_flush`
- **Data**: `data_dir`, `experiment_dir`

### What NOT to Do

These target specific failure modes. When in doubt, prefer the constrained approach.

- **Do not introduce direct imports between pipeline components to work around the bus.** If you need data from another stage, add a new event type and listen for it. The bus topology exists to keep components decoupled and testable in isolation.
- **Do not add synchronous blocking calls inside RxPY observers.** Event handlers run on a shared `ThreadPoolScheduler`. A blocking call (sleep, synchronous HTTP, waiting on a lock) starves other event processing. If you need async work, use the reporting layer's threading patterns.
- **Do not bypass ExpMod for algorithm selection.** Even if there is only one implementation today, wire it through ExpMod so alternatives can be swapped without code changes. Hardcoding `SymmetricDifferenceResolution()` instead of `ObjectResolutionExpMod.get()` defeats the plugin architecture.
- **Do not modify Node fields in tight loops.** Node auto-saves on field modification. Setting `node.field = x` inside a loop causes repeated persistence overhead. Batch your changes or use `_no_save=True` temporarily if performance is critical.
- **Do not create EventBus instances outside of class-level Component definitions.** Buses must be class attributes so the topology is discoverable and static. Ad-hoc buses create invisible coupling.
- **Do not add game-specific logic (NetHack, nle) outside the perception layer.** If you need game-specific data downstream, extract it as an abstract feature in a FeatureExtractor first. The only exception is `roc/reporting/state.py` for display purposes.
- **Do not construct `Config()` directly.** Always use `Config.get()`. The singleton pattern ensures consistent configuration across the entire system.
- **Do not start additional web servers.** Use the existing `roc-server` + `roc-ui` pair managed by servherd. Adding a new server creates port conflicts, CORS issues, and confusion about which URL to use.
- **Do not add Edge connections without updating `allowed_connections`.** The schema validation catches invalid connections early. If you need a new connection pattern, add it to the Edge subclass's `allowed_connections` list first, then use it.

## Code Style

- **Line length**: 100 characters (ruff)
- **Type checking**: Strict mypy with pydantic and numpy plugins
- **Docstrings**: Google convention (enforced by ruff). Not required in tests or magic methods.
- **Python version**: 3.13 (requires-python >=3.12, <4.0)
- **Package manager**: uv (not pip, not poetry)

## CLI Entry Points

Defined in `pyproject.toml [project.scripts]`:

| Command | Entry Point | Purpose |
|---------|-------------|---------|
| `play` | `roc.cli.script:cli` | Run the ROC agent. Auto-generates click options from every Config field. |
| `server` | `roc.cli.server_cli:main` | Unified dashboard server with game lifecycle management via GameManager. |
| `dashboard` | `roc.cli.dashboard_cli:main` | Standalone historical dashboard viewer (no game management). |
| `cleanup` | `roc.cli.cleanup_cli:main` | Clean up empty/short game runs from data directory. |

The `server` command spawns games as subprocesses (`uv run play --no-dashboard-enabled --dashboard-callback-url=...`) and manages their lifecycle via `GameManager` (`roc/game/game_manager.py`). State machine: idle -> initializing -> running -> stopping. Cooperative shutdown via REST, then SIGTERM, then SIGKILL.

## Debugging Tools

ROC has multiple debugging tools available. Choose the right one based on what you need.

### 1. DAP MCP Debugger (Interactive -- Preferred for Live Debugging)

**When**: You need to inspect live state, step through code, or evaluate expressions during a running game. Most powerful option -- no pre-built serialization needed.

**How**:
```bash
uv run play
```

Then use MCP debugger tools:
- `create_debug_session` / `start_debugging` -- launch or attach to a debugpy process
- `set_breakpoint(file, line, condition?)` -- break at specific locations
- `pause_execution` / `continue_execution` -- control flow
- `step_over` / `step_into` / `step_out` -- step through code
- `evaluate_expression(expr)` -- run arbitrary Python in paused context
- `get_local_variables` / `get_variables` -- inspect variables
- `get_stack_trace` -- see call stack

**Tips**:
- Use `.venv/bin/play` as scriptPath, `justMyCode: true`, `stopOnEntry: false`
- `justMyCode: false` breaks breakpoint resolution -- always use `true`
- Step-over times out if the function takes >5s -- use continue-to-next-breakpoint instead
- ROC uses ThreadPoolScheduler: breakpoints in `roc/pipeline/object/object.py` pause worker threads, breakpoints in `roc/game/gymnasium.py` pause the main loop

### 2. Local JSONL Debug Log (Post-Hoc Analysis)

**When**: You need structured decision records after a run completes, or want to search for patterns across many ticks.

**How**:
```bash
roc_debug_log=true uv run play
```

Then read/search the debug log file (path printed at startup) with Read/Grep tools. Contains OTel log records: resolution decisions, object creations, anomalies, summaries.

### 3. Remote Logger MCP (Live Monitoring)

**When**: You need to watch logs in real-time during a running game, search for specific patterns, or filter errors -- without stopping execution.

**How**: Remote logging is on by default. Just run `uv run play` and use the MCP tools:

Then use Remote Logger MCP tools:
- `logs_get_recent` -- see latest log entries
- `logs_search "pattern"` -- find specific text (supports regex)
- `logs_get_errors` -- filter ERROR level only
- `logs_list_sessions` -- see all sessions
- `logs_status` -- check server health

### 4. Grafana MCP (Historical/Aggregate Analysis)

**When**: You need to analyze trends across runs, compare before/after metrics, or monitor long-running games. Not for per-decision debugging.

**How**: Use Grafana MCP tools to query the remote Grafana instance at hal.ato.ms:
- `query_prometheus` -- run PromQL queries (e.g., `roc.dirichlet.posterior_max`, `roc.resolution.decision`)
- `search_dashboards` / `get_dashboard_by_uid` -- find and read dashboards
- `query_loki_logs` -- search structured logs in Loki

### 5. Loguru Logging (Terminal Output)

**When**: You need quick human-readable output during development. Already active by default.

**How**: Filter per-module with `roc_log_modules` config. Output goes to stderr with colors. Very noisy at DEBUG level -- critical info gets buried. Best for quick sanity checks, not deep analysis.

### Non-Default Config Options

GraphDB export and flush are off by default. To enable them:
```bash
roc_graphdb_export=true    # Export graph to file on game end
roc_graphdb_flush=true     # Flush cache to Memgraph on game end
```

### Decision Tree

1. **Need to inspect arbitrary live state?** -> DAP MCP Debugger (#1)
2. **Need to search/analyze after a run?** -> JSONL Debug Log (#2)
3. **Need to watch logs during a live run without stopping it?** -> Remote Logger (#3)
4. **Need aggregate metrics across runs?** -> Grafana MCP (#4)
5. **Quick sanity check?** -> Loguru terminal output (#5)

## React Dashboard

The debug dashboard is a React app in `dashboard-ui/` with a FastAPI + Socket.io backend in `roc/reporting/api_server.py`. Key architecture:

- **Frontend**: React + Mantine (compact-mantine theme) + TanStack Query + Vite + Recharts + Socket.io client
- **Backend**: FastAPI REST API + Socket.io live push via StepBuffer + DuckLake historical queries
- **Data flow**: Game subprocess POSTs StepData to server via HTTP callback, server broadcasts via Socket.io. Historical data served from DuckLake/Parquet.
- **Server management**: All servers managed by servherd. Use `make run` / `make stop`.

### Server Architecture (One Server, One URL)

There must be exactly ONE UI server (`roc-ui`) and ONE API server (`roc-server`). Never start additional dashboard or preview servers. See `design/dashboard-server-redesign.md` for full design.

- `roc-server`: Unified FastAPI + Socket.io backend (`uv run server`). Always running. Serves API and manages game lifecycle.
- `roc-ui`: Vite dev server for frontend HMR. Proxies `/api` and `/socket.io` to `roc-server`.
- Games are started/stopped via REST API (`POST /api/game/start`, `POST /api/game/stop`), not by restarting the server.
- Both servers' ports are auto-assigned by servherd. Use `npx servherd info roc-ui` to get the dashboard URL.

### REST API Overview

Key endpoint groups (all under `/api`):
- **Runs**: `GET /runs`, `GET /runs/{run}/games`, `GET /runs/{run}/step-range`
- **Steps**: `GET /runs/{run}/step/{step}`, `GET /runs/{run}/steps?steps=1,2,3`
- **History** (time series for charts): `metrics-history`, `graph-history`, `event-history`, `intrinsics-history`, `action-history`, `resolution-history`
- **Schema/Objects**: `GET /runs/{run}/schema`, `GET /runs/{run}/all-objects`, `GET /runs/{run}/action-map`
- **Bookmarks**: `GET/POST /runs/{run}/bookmarks`
- **Game lifecycle**: `GET /game/status`, `POST /game/start`, `POST /game/stop`
- **Internal** (game subprocess callback): `POST /internal/step`

## Testing Notes

### Test Organization

Test directories mirror the source layout:

```
tests/
  conftest.py                  # Root fixtures (autouse: clear_cache, restore_registries, do_init)
  helpers/                     # Shared test utilities (FakeData, nethack_screens, testmods)
  unit/
    conftest.py                # unit_config_init, clean_expmod_state (autouse)
    framework/                 # test_component, test_config, test_event, test_expmod, ...
    db/                        # test_graphdb_pure
    perception/                # test_perception, test_location
      feature_extractors/      # test_color, test_phoneme
    pipeline/                  # test_action, test_intrinsic, test_significance
      attention/               # test_attention, test_saliency_attenuation, ...
      object/                  # test_object, test_object_instance, ...
      temporal/                # test_sequencer, test_transformer, test_predict, ...
    game/                      # test_gymnasium, test_game_manager, test_breakpoint
    cli/                       # test_script
    reporting/                 # test_api_server, test_state, test_observability, ...
  integration/
    db/                        # test_graphdb
    pipeline/
      attention/               # test_attention, test_saliency_attenuation_integration
      object/                  # test_object, test_dirichlet_integration, ...
      temporal/                # test_sequencer, test_transformer
  e2e/                         # Full pipeline tests (2 modules)
    conftest.py                # all_components fixture (loads full pipeline)
  test_feature_extractors/     # Legacy feature extractor tests (9 modules, *_test.py naming)
```

### Key Fixtures

- `clear_cache` (autouse): resets graphdb Node/Edge caches after each test
- `restore_registries` (autouse): saves/restores component, node, edge, and label registries
- `do_init` (autouse): Config.reset/init, Observability.init, logger init
- `clean_expmod_state` (autouse in unit/): saves/restores ExpMod registry
- `eb_reset`: clears EventBus names (needed when creating test buses)
- `test_tree`: creates a graph with root + 10 nodes and 10 edges

### Markers

- `@pytest.mark.slow` -- long-running (skipped by default)
- `@pytest.mark.requires_observability` -- needs OTel server (skipped by default)
- `@pytest.mark.requires_graphviz` -- needs graphviz binary
- `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e` -- test tier
- Pytest runs with `--doctest-modules` so docstrings with `>>>` examples are tested
- Warnings are treated as errors (except DeprecationWarning)
