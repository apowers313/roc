# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROC (Reinforcement Learning of Concepts) is a Python agent system using a component-based, event-driven architecture. The agent perceives the game environment, identifies objects, tracks changes between frames, and decides actions. Currently tested against NetHack via Gymnasium.

### Design Principle: Game-Agnostic Core

Everything from Attention onward (object resolution, sequencing, transforms, actions) must be generic -- not specific to NetHack or any other gym. Only the Perception layer (feature extractors) is game-specific. Object resolution, distance metrics, tracking, and all downstream logic must operate on abstract features without assuming anything about the game producing them.

## Common Commands

```bash
# Setup
make setup              # Install deps + pre-commit hooks
make install            # Create venv (Python 3.13) and sync deps

# Testing
make test               # Run all tests (excludes slow and requires_observability)
uv run pytest -c pyproject.toml tests/object_test.py                    # Run one test file
uv run pytest -c pyproject.toml tests/object_test.py -k test_name      # Run single test by name
uv run pytest -c pyproject.toml tests/object_test.py::TestClass::test   # Run specific test method

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
```

## Architecture

### Component System

Everything is built on `Component` (component.py) -- an abstract base class with:
- **Auto-registration**: Subclasses register themselves in a global `component_registry` via `__init_subclass__`. Each component has a unique `name`+`type` pair.
- **Auto-loading**: Components with `auto = True` are loaded at startup. Perception components are loaded from config.
- **Bus connections**: Components communicate via `connect_bus()` which attaches them to typed EventBus channels.

### Event System

Communication uses RxPy reactive streams (event.py):
- `EventBus[T]` -- typed communication channel (backed by `rx.Subject`)
- `Event[T]` -- carries data + source ComponentId
- `BusConnection[T]` -- component-to-bus link for sending/listening
- Listeners run on a ThreadPoolScheduler; events are filtered to exclude self-sends by default

### Data Pipeline

The agent processes each game step through this chain:

```
Gymnasium (NetHack env) -> Perception (feature extractors) -> Attention (saliency)
-> ObjectResolver (identify objects) -> Sequencer (build frames)
-> Transformer (detect changes) -> Action (decide action)
```

Each stage is a Component that listens on one EventBus and sends on the next.

### Graph Database

`graphdb.py` wraps Memgraph (via mgclient) with:
- `Node` and `Edge` classes built on Pydantic models with auto-save-on-modify
- LRU caching to reduce DB queries
- `NodeList` and `EdgeList` collection types with filtering/selection
- Cypher query generation for CRUD operations

CI requires a Memgraph service container. Tests that need the DB will fail without it.

### Key Abstractions

- **Transformable** (transformable.py): Interface for objects that can detect and represent changes between frames. Implemented by objects, intrinsics, etc.
- **Transform** (transformable.py): A Node subclass representing the diff between two frame states.
- **Intrinsic** (intrinsic.py): Agent internal state (HP, energy, hunger) with normalization operations, converted to graph nodes.
- **Frame** (sequencer.py): A snapshot of the game state at one timestep, containing objects, intrinsics, and actions.
- **Perception** (perception.py): Abstract base for feature extractors. Concrete implementations live in `roc/feature_extractors/`.

### Config

`Config` (config.py) uses pydantic-settings, reads from `.env` file with `roc_` prefix. During pytest, env vars and .env file are deliberately ignored to isolate tests.

## Code Style

- **Line length**: 100 characters (ruff)
- **Type checking**: Strict mypy with pydantic and numpy plugins
- **Docstrings**: Google convention (enforced by ruff). Not required in tests or magic methods.
- **Python version**: 3.13
- **Package manager**: uv (not pip, not poetry)

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
- ROC uses ThreadPoolScheduler: breakpoints in `object.py` pause worker threads, breakpoints in `gymnasium.py` pause the main loop

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

## Testing Notes

- Tests use fixtures from `conftest.py` for clearing caches, restoring registries, and creating test components
- `clear_cache` fixture resets graphdb Node/Edge caches between tests
- `restore_registries` fixture saves and restores the component registry
- Pytest runs with `--doctest-modules` so docstrings with `>>>` examples are tested
- Test markers: `@pytest.mark.slow`, `@pytest.mark.requires_observability`, `@pytest.mark.requires_graphviz`
