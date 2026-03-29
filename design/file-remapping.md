# File Remapping: roc/ Package Restructuring

## Motivation

The `roc/` package has grown to 30 `.py` files at the top level with only three
subpackages (`reporting/`, `feature_extractors/`, `jupyter/`). Several files are
oversized (`graphdb.py` at 2526 lines, `gymnasium.py` at 1165, `object.py` at 1108).
Related files are scattered (e.g., `object.py`, `object_instance.py`, `object_transform.py`
are siblings of unrelated modules). This makes the codebase harder to navigate and
reason about.

## Target Structure

```
roc/
  __init__.py
  debugpy_setup.py

  framework/                          # Infrastructure plumbing
    __init__.py
    component.py
    config.py
    event.py
    expmod.py
    logger.py
    utils.py

  db/                                 # Graph database layer
    __init__.py
    db.py                             # GraphDB class, connection, Cypher queries
    node.py                           # Node, NodeList, registries, GraphCache
    edge.py                           # Edge, EdgeList, schema validation
    cache.py                          # GraphCache, LRU logic

  perception/                         # Game-specific perception layer
    __init__.py
    perception.py                     # Perception base, FeatureExtractor, FeatureNode
    location.py                       # Point, PointCollection, XLoc, YLoc
    feature_extractors/
      __init__.py
      color.py
      delta.py
      distance.py
      flood.py
      line.py
      motion.py
      phoneme.py
      shape.py
      single.py

  pipeline/                           # Processing pipeline stages
    __init__.py
    action.py                         # Action component, ActionRequest, TakeAction
    intrinsic.py                      # IntrinsicNode, IntrinsicData, IntrinsicOp
    significance.py                   # Significance component
    attention/
      __init__.py
      attention.py                    # VisionAttention, SaliencyMap
      saliency_attenuation.py         # Attenuation ExpMods
    object/
      __init__.py
      object.py                       # Object, FeatureGroup, resolution ExpMods
      object_instance.py              # ObjectInstance rendering/display
      object_transform.py             # ObjectTransform change tracking
    temporal/
      __init__.py
      transformable.py                # Transformable ABC, Transform base
      sequencer.py                    # Sequencer, Frame
      transformer.py                  # Transformer component
      predict.py                      # Predict component, PredictData

  game/                               # Game interface layer
    __init__.py
    gymnasium.py                      # NethackGym, observation/action loop
    game_manager.py                   # GameManager subprocess lifecycle
    breakpoint.py                     # Breakpoint system for Jupyter

  cli/                                # CLI entry points
    __init__.py
    script.py                         # `play` command
    server_cli.py                     # `server` command
    dashboard_cli.py                  # `dashboard` command
    cleanup_cli.py                    # `cleanup` command

  reporting/                          # Unchanged
    __init__.py
    api_server.py
    data_store.py
    ducklake_store.py
    metrics.py
    observability.py
    parquet_exporter.py
    remote_logger_exporter.py
    run_store.py
    screen_renderer.py
    state.py
    step_buffer.py
    step_log_sink.py

  jupyter/                            # Unchanged
    __init__.py
    brk.py
    cont.py
    roc.py
    save.py
    state.py
    step.py
    utils.py
```

## File Mapping (old path -> new path)

### framework/

| Old | New | Lines |
|-----|-----|-------|
| `roc/component.py` | `roc/framework/component.py` | 238 |
| `roc/config.py` | `roc/framework/config.py` | 296 |
| `roc/event.py` | `roc/framework/event.py` | 172 |
| `roc/expmod.py` | `roc/framework/expmod.py` | 254 |
| `roc/logger.py` | `roc/framework/logger.py` | 115 |
| `roc/utils.py` | `roc/framework/utils.py` | 8 |

### db/ (split from graphdb.py)

| Old | New | Lines (approx) |
|-----|-----|----------------|
| `roc/graphdb.py` (GraphDB class) | `roc/db/db.py` | ~400 |
| `roc/graphdb.py` (Node, NodeList) | `roc/db/node.py` | ~1000 |
| `roc/graphdb.py` (Edge, EdgeList) | `roc/db/edge.py` | ~700 |
| `roc/graphdb.py` (GraphCache, LRU) | `roc/db/cache.py` | ~400 |

The `roc/db/__init__.py` re-exports the full public API so callers can write
`from roc.db import Node, Edge, ...`.

### perception/

| Old | New | Lines |
|-----|-----|-------|
| `roc/perception.py` | `roc/perception/perception.py` | 318 |
| `roc/location.py` | `roc/perception/location.py` | 404 |
| `roc/feature_extractors/` | `roc/perception/feature_extractors/` | 1105 |

### pipeline/

| Old | New | Lines |
|-----|-----|-------|
| `roc/action.py` | `roc/pipeline/action.py` | 72 |
| `roc/intrinsic.py` | `roc/pipeline/intrinsic.py` | 257 |
| `roc/significance.py` | `roc/pipeline/significance.py` | 54 |

### pipeline/attention/

| Old | New | Lines |
|-----|-----|-------|
| `roc/attention.py` | `roc/pipeline/attention/attention.py` | 540 |
| `roc/saliency_attenuation.py` | `roc/pipeline/attention/saliency_attenuation.py` | 603 |

### pipeline/object/

| Old | New | Lines |
|-----|-----|-------|
| `roc/object.py` | `roc/pipeline/object/object.py` | 1108 |
| `roc/object_instance.py` | `roc/pipeline/object/object_instance.py` | 212 |
| `roc/object_transform.py` | `roc/pipeline/object/object_transform.py` | 306 |

### pipeline/temporal/

| Old | New | Lines |
|-----|-----|-------|
| `roc/transformable.py` | `roc/pipeline/temporal/transformable.py` | 58 |
| `roc/sequencer.py` | `roc/pipeline/temporal/sequencer.py` | 230 |
| `roc/transformer.py` | `roc/pipeline/temporal/transformer.py` | 144 |
| `roc/predict.py` | `roc/pipeline/temporal/predict.py` | 142 |

### game/

| Old | New | Lines |
|-----|-----|-------|
| `roc/gymnasium.py` | `roc/game/gymnasium.py` | 1165 |
| `roc/game_manager.py` | `roc/game/game_manager.py` | 277 |
| `roc/breakpoint.py` | `roc/game/breakpoint.py` | 228 |

### cli/

| Old | New | Lines |
|-----|-----|-------|
| `roc/script.py` | `roc/cli/script.py` | 117 |
| `roc/server_cli.py` | `roc/cli/server_cli.py` | 82 |
| `roc/dashboard_cli.py` | `roc/cli/dashboard_cli.py` | 133 |
| `roc/cleanup_cli.py` | `roc/cli/cleanup_cli.py` | 139 |

### Unchanged

| File | Reason |
|------|--------|
| `roc/__init__.py` | Package root (imports will be updated) |
| `roc/debugpy_setup.py` | Standalone bootstrap, only imports config + logger |
| `roc/reporting/*` | Already well-organized subpackage |
| `roc/jupyter/*` | Already well-organized subpackage |

## Existing Import Cycles

Four mutual import pairs exist today. All work because Python resolves them at
attribute-access time (not import time), but they constrain how we split files.

| Cycle | New locations | Status |
|-------|---------------|--------|
| `attention` <-> `saliency_attenuation` | Both in `pipeline/attention/` | Safe -- same package |
| `object` <-> `sequencer` | `pipeline/object/` <-> `pipeline/temporal/` | Watch -- cross-package |
| `object_instance` <-> `object_transform` | Both in `pipeline/object/` | Safe -- same package |
| `sequencer` <-> `transformable` | Both in `pipeline/temporal/` | Safe -- same package |

The `object` <-> `sequencer` cycle crosses subpackage boundaries. Both are under
`pipeline/`, so the relative import becomes `from ..temporal.sequencer import Frame`
and `from ..object.object import Object`. This works because the cycle is already
deferred (imports are at module top level but the circular references resolve at
runtime). No action needed, but do not convert these to lazy imports or add
`TYPE_CHECKING` guards without testing.

## Fan-In Analysis (Import Impact)

Modules sorted by total importers (source code + test files). Higher fan-in means
more files must be updated when the module moves.

| Module | Source importers | Test importers | Total | New path |
|--------|-----------------|----------------|-------|----------|
| `graphdb` | 19 | 12 | 31 | `roc.db` |
| `component` | 8 | 21 | 29 | `roc.framework` |
| `location` | 12 | 21 | 33 | `roc.perception` |
| `perception` | 14 | 19 | 33 | `roc.perception` |
| `event` | 9 | 18 | 27 | `roc.framework` |
| `config` | 14 | 16 | 30 | `roc.framework` |
| `object` | 5 | 14 | 19 | `roc.pipeline.object` |
| `logger` | 9 | 2 | 11 | `roc.framework` |
| `sequencer` | 6 | 11 | 17 | `roc.pipeline.temporal` |
| `attention` | 2 | 10 | 12 | `roc.pipeline.attention` |
| `intrinsic` | 4 | 8 | 12 | `roc.pipeline` |
| `expmod` | 4 | 6 | 10 | `roc.framework` |
| `action` | 2 | 6 | 8 | `roc.pipeline` |
| `transformer` | 1 | 7 | 8 | `roc.pipeline.temporal` |
| `transformable` | 4 | 5 | 9 | `roc.pipeline.temporal` |
| `object_instance` | 5 | 4 | 9 | `roc.pipeline.object` |
| `object_transform` | 4 | 4 | 8 | `roc.pipeline.object` |
| `predict` | 1 | 4 | 5 | `roc.pipeline.temporal` |
| `significance` | 0 | 1 | 1 | `roc.pipeline` |
| `breakpoint` | 4 | 1 | 5 | `roc.game` |
| `gymnasium` | 0 | 1 | 1 | `roc.game` |
| `game_manager` | 1 | 1 | 2 | `roc.game` |
| `saliency_attenuation` | 2 | 3 | 5 | `roc.pipeline.attention` |
| `script` | 0 | 1 | 1 | `roc.cli` |
| `server_cli` | 0 | 0 | 0 | `roc.cli` |
| `dashboard_cli` | 1 | 1 | 2 | `roc.cli` |
| `cleanup_cli` | 0 | 1 | 1 | `roc.cli` |
| `debugpy_setup` | 0 | 2 | 2 | stays |

## External References (Non-Python)

These files reference `roc.*` module paths and must be updated:

| File | References | Change needed |
|------|-----------|---------------|
| `pyproject.toml:129` | `play = "roc.script:cli"` | `roc.cli.script:cli` |
| `pyproject.toml:130` | `dashboard = "roc.dashboard_cli:main"` | `roc.cli.dashboard_cli:main` |
| `pyproject.toml:131` | `server = "roc.server_cli:main"` | `roc.cli.server_cli:main` |
| `pyproject.toml:132` | `cleanup = "roc.cleanup_cli:main"` | `roc.cli.cleanup_cli:main` |
| `pyproject.toml:292` | `ignore::roc.graphdb.ErrorSavingDuringDelWarning` | `ignore::roc.db.(...).ErrorSavingDuringDelWarning` |

## graphdb.py Split Plan

`graphdb.py` is 2526 lines and the most-imported module (31 total importers). It
needs to be split into `roc/db/` but the public API must remain stable.

### Proposed split

**`roc/db/cache.py`** (~200 lines)
- `GraphCache` class
- LRU eviction logic
- `CacheStats` type

**`roc/db/db.py`** (~500 lines)
- `GraphDB` singleton class
- Connection management (mgclient)
- `raw_fetch()`, `raw_execute()`
- Export methods (gml, gexf, dot, graphml, json, cytoscape)
- NetworkX conversion

**`roc/db/node.py`** (~1000 lines)
- `Node` base class (Pydantic BaseModel)
- `NodeList` lazy-loading collection
- `NodeId` type
- `node_registry`, `node_label_registry`
- Node CRUD: `create`, `load`, `update`, `save`, `delete`, `find`, `find_one`
- `walk()` DFS traversal
- DOT/SVG rendering

**`roc/db/edge.py`** (~700 lines)
- `Edge` base class (Pydantic BaseModel)
- `EdgeList` lazy-loading collection
- `EdgeId` type
- `edge_registry`
- `allowed_connections` schema validation
- `connect()` factory
- `EdgeConnectionsList`

**`roc/db/__init__.py`**
- Re-exports everything: `Node`, `Edge`, `GraphDB`, `NodeId`, `EdgeId`,
  `NodeList`, `EdgeList`, `EdgeConnectionsList`, `GraphCache`,
  `ErrorSavingDuringDelWarning`, all registries
- This allows `from roc.db import Node` to work immediately

## Migration Strategy

### Approach: All-at-once, two commits

All files move in a single pass. No compatibility shims, no intermediate states.
The work is split into two commits to preserve git history:

**Commit 1 -- pure moves.** Only `git mv` operations and new empty `__init__.py`
files. Zero content changes. Every moved file is 100% identical to its predecessor,
so git's rename detection is guaranteed to work. `git log --follow` and `git blame`
will trace through this commit seamlessly.

**Commit 2 -- content changes.** All import rewrites, graphdb split, `__init__.py`
re-exports, pyproject.toml updates. The build is broken after commit 1 and fixed
by commit 2 -- that is expected.

This two-commit approach exists solely to preserve `git log --follow` history.
If moves and content changes were in the same commit, files with large import
rewrites might drop below git's 50% similarity threshold and lose rename detection.

**Limitation: graphdb split.** Splitting one file into four cannot be tracked as
a rename -- git only maps one-to-one. The four new files (`db/db.py`, `db/node.py`,
`db/edge.py`, `db/cache.py`) will have history starting from commit 2. The original
`graphdb.py` history is accessible via `git log -- roc/graphdb.py`, and individual
lines can be traced with `git blame -C -C roc/db/node.py`. To help rename detection,
commit 1 moves `graphdb.py` to `roc/db/graphdb.py` intact, and commit 2 splits it.

### Commit 1: Pure moves (preserves git history)

This commit contains ONLY `git mv` operations and new empty `__init__.py` files.
No file content is modified. This ensures 100% rename detection by git.

The build is intentionally broken after this commit. That is expected.

**Step 1.1: Create source directory structure**

```bash
mkdir -p roc/framework roc/db roc/perception/feature_extractors \
         roc/pipeline/attention roc/pipeline/object roc/pipeline/temporal \
         roc/game roc/cli
```

Create empty `__init__.py` in each new directory.

**Step 1.2: Create test directory structure**

```bash
mkdir -p tests/unit/{framework,db,perception/feature_extractors}
mkdir -p tests/unit/pipeline/{attention,object,temporal}
mkdir -p tests/unit/{game,cli,reporting}
mkdir -p tests/integration/{db,pipeline/{attention,object,temporal}}
```

Create empty `__init__.py` in each new test directory.

**Step 1.3: Move source files**

framework/:
```
git mv roc/component.py    roc/framework/component.py
git mv roc/config.py       roc/framework/config.py
git mv roc/event.py        roc/framework/event.py
git mv roc/expmod.py       roc/framework/expmod.py
git mv roc/logger.py       roc/framework/logger.py
git mv roc/utils.py        roc/framework/utils.py
```

db/ (move whole file intact -- split happens in commit 2):
```
git mv roc/graphdb.py      roc/db/graphdb.py
```

perception/:
```
git mv roc/perception.py   roc/perception/base.py
git mv roc/location.py     roc/perception/location.py
git mv roc/feature_extractors/*.py  roc/perception/feature_extractors/
rmdir roc/feature_extractors
```

pipeline/attention/:
```
git mv roc/attention.py              roc/pipeline/attention/attention.py
git mv roc/saliency_attenuation.py   roc/pipeline/attention/saliency_attenuation.py
```

pipeline/object/:
```
git mv roc/object.py           roc/pipeline/object/object.py
git mv roc/object_instance.py  roc/pipeline/object/object_instance.py
git mv roc/object_transform.py roc/pipeline/object/object_transform.py
```

pipeline/temporal/:
```
git mv roc/transformable.py  roc/pipeline/temporal/transformable.py
git mv roc/sequencer.py      roc/pipeline/temporal/sequencer.py
git mv roc/transformer.py    roc/pipeline/temporal/transformer.py
git mv roc/predict.py        roc/pipeline/temporal/predict.py
```

pipeline/ (flat):
```
git mv roc/action.py       roc/pipeline/action.py
git mv roc/intrinsic.py    roc/pipeline/intrinsic.py
git mv roc/significance.py roc/pipeline/significance.py
```

game/:
```
git mv roc/gymnasium.py    roc/game/gymnasium.py
git mv roc/game_manager.py roc/game/game_manager.py
git mv roc/breakpoint.py   roc/game/breakpoint.py
```

cli/:
```
git mv roc/script.py        roc/cli/script.py
git mv roc/server_cli.py    roc/cli/server_cli.py
git mv roc/dashboard_cli.py roc/cli/dashboard_cli.py
git mv roc/cleanup_cli.py   roc/cli/cleanup_cli.py
```

**Step 1.4: Move test files**

Unit tests:
```
git mv tests/unit/test_component.py          tests/unit/framework/
git mv tests/unit/test_config.py             tests/unit/framework/
git mv tests/unit/test_event.py              tests/unit/framework/
git mv tests/unit/test_expmod.py             tests/unit/framework/
git mv tests/unit/test_expmod_params.py      tests/unit/framework/
git mv tests/unit/test_logger.py             tests/unit/framework/
git mv tests/unit/test_utils.py              tests/unit/framework/
git mv tests/unit/test_graphdb_pure.py       tests/unit/db/
git mv tests/unit/test_perception.py         tests/unit/perception/
git mv tests/unit/test_location.py           tests/unit/perception/
git mv tests/unit/feature_extractors/*       tests/unit/perception/feature_extractors/
rmdir tests/unit/feature_extractors
git mv tests/unit/test_action.py             tests/unit/pipeline/
git mv tests/unit/test_intrinsic.py          tests/unit/pipeline/
git mv tests/unit/test_significance.py       tests/unit/pipeline/
git mv tests/unit/test_attention.py          tests/unit/pipeline/attention/
git mv tests/unit/test_multi_cycle_attention.py tests/unit/pipeline/attention/
git mv tests/unit/test_saliency_attenuation.py tests/unit/pipeline/attention/
git mv tests/unit/test_active_inference.py   tests/unit/pipeline/attention/
git mv tests/unit/test_object.py             tests/unit/pipeline/object/
git mv tests/unit/test_object_instance.py    tests/unit/pipeline/object/
git mv tests/unit/test_object_telemetry.py   tests/unit/pipeline/object/
git mv tests/unit/test_object_transform.py   tests/unit/pipeline/object/
git mv tests/unit/test_dirichlet_resolution.py tests/unit/pipeline/object/
git mv tests/unit/test_sequencer.py          tests/unit/pipeline/temporal/
git mv tests/unit/test_transformable.py      tests/unit/pipeline/temporal/
git mv tests/unit/test_transformer.py        tests/unit/pipeline/temporal/
git mv tests/unit/test_predict.py            tests/unit/pipeline/temporal/
git mv tests/unit/test_predict_unit.py       tests/unit/pipeline/temporal/
git mv tests/unit/test_gymnasium.py          tests/unit/game/
git mv tests/unit/test_game_manager.py       tests/unit/game/
git mv tests/unit/test_breakpoint.py         tests/unit/game/
git mv tests/unit/test_script.py             tests/unit/cli/
git mv tests/unit/test_api_server.py         tests/unit/reporting/
git mv tests/unit/test_data_store.py         tests/unit/reporting/
git mv tests/unit/test_observability.py      tests/unit/reporting/
git mv tests/unit/test_observability_extra.py tests/unit/reporting/
git mv tests/unit/test_parquet_exporter.py   tests/unit/reporting/
git mv tests/unit/test_remote_logger_exporter.py tests/unit/reporting/
git mv tests/unit/test_run_store.py          tests/unit/reporting/
git mv tests/unit/test_screen_renderer.py    tests/unit/reporting/
git mv tests/unit/test_state.py              tests/unit/reporting/
git mv tests/unit/test_state_extra.py        tests/unit/reporting/
git mv tests/unit/test_state_snapshots.py    tests/unit/reporting/
git mv tests/unit/test_step_buffer.py        tests/unit/reporting/
git mv tests/unit/test_step_log_sink.py      tests/unit/reporting/
```

Stays at unit/ root (no move):
- `tests/unit/test_debugpy_setup.py`
- `tests/unit/test_init.py`

Integration tests:
```
git mv tests/integration/test_graphdb.py     tests/integration/db/
git mv tests/integration/test_attention.py   tests/integration/pipeline/attention/
git mv tests/integration/test_saliency_attenuation_integration.py tests/integration/pipeline/attention/
git mv tests/integration/test_object.py      tests/integration/pipeline/object/
git mv tests/integration/test_dirichlet_integration.py tests/integration/pipeline/object/
git mv tests/integration/test_object_transform_pipeline.py tests/integration/pipeline/object/
git mv tests/integration/test_sequencer.py   tests/integration/pipeline/temporal/
git mv tests/integration/test_transformer.py tests/integration/pipeline/temporal/
```

**Step 1.5: Commit**

```bash
git commit -m "refactor: move files to new package structure (pure moves, no content changes)"
```

### Commit 2: Content changes (imports, graphdb split, config)

This commit contains all content modifications. The build goes from broken to green.

**Step 2.1: Split graphdb.py into roc/db/ submodules**

The internal dependency order within graphdb.py is:

```
cache.py        (standalone -- GraphCache, CacheStats)
  ^
db.py           (imports cache -- GraphDB, connection, queries, export)
  ^
node.py         (imports cache, db -- Node, NodeList, registries)
  ^
edge.py         (imports node, db -- Edge, EdgeList, EdgeConnectionsList)
```

1. Extract `GraphCache` and related types into `roc/db/cache.py`
2. Extract `GraphDB` singleton into `roc/db/db.py`
3. Extract `Node`, `NodeList`, `NodeId`, registries into `roc/db/node.py`
4. Extract `Edge`, `EdgeList`, `EdgeId`, `EdgeConnectionsList` into `roc/db/edge.py`
5. Delete `roc/db/graphdb.py`
6. Write `roc/db/__init__.py` with re-exports (see __init__.py section below)

**Step 2.2: Rewrite all source imports**

Every `from .X import ...` in source code must be updated to reflect the new
relative paths. See the Import Rewrite Map section for the full table.

**Step 2.3: Rewrite all test imports**

Every `from roc.X import ...` in test code must be updated to use the new
absolute paths. See the test import rewrite table in the Import Rewrite Map.

**Step 2.4: Write __init__.py re-exports**

See the __init__.py Re-export Strategy section below.

**Step 2.5: Update external references**

- `pyproject.toml` entry points (4 lines)
- `pyproject.toml` warning filter (1 line)
- `roc/__init__.py` (update imports to new paths)
- `roc/debugpy_setup.py` (update imports to new paths)
- `experiments/` directory (grep for old `roc.*` imports)
- `Makefile` test targets (verify `make test-unit` etc. still discover tests)

**Step 2.6: Verify**

```bash
make lint       # mypy + ruff check + ruff format --check
make test       # full test suite
make coverage   # verify no regressions
```

**Step 2.7: Commit**

```bash
git commit -m "refactor: rewrite imports and split graphdb for new package structure"
```

## Import Rewrite Map

The codebase uses relative imports in source code and absolute imports in tests.
All must be updated in the same pass.

### Source code: relative import rewrites

Within the **same new package**, relative imports stay simple:
```python
# Before (in roc/transformer.py):
from .sequencer import Frame

# After (in roc/pipeline/temporal/transformer.py):
from .sequencer import Frame          # same package, unchanged
```

**Cross-package** imports within `roc/pipeline/`:
```python
# Before (in roc/sequencer.py):
from .object import Object

# After (in roc/pipeline/temporal/sequencer.py):
from ..object.object import Object    # up to pipeline/, down to object/
```

**Importing framework/db** from pipeline or other packages:
```python
# Before (in roc/action.py):
from .component import Component
from .graphdb import Edge

# After (in roc/pipeline/action.py):
from ..framework.component import Component
from ..db import Edge
```

### Full source import rewrite table

This table covers every relative import that must change. Imports between files
that remain in the same package (e.g., within `pipeline/temporal/`) are omitted
since they stay as `from .X import ...`.

**Key**: `..fw` = `..framework`, `..p` = `..pipeline`

| Importing file (new location) | Old import | New import |
|-------------------------------|-----------|-----------|
| **roc/__init__.py** | `from .logger` | `from .framework.logger` |
| | `from .saliency_attenuation` | `from .pipeline.attention.saliency_attenuation` |
| | `from .feature_extractors` | `from .perception.feature_extractors` |
| **roc/debugpy_setup.py** | `from .config` | `from .framework.config` |
| | `from .logger` | `from .framework.logger` |
| **framework/component.py** | `from .config` | `from .config` (same package) |
| | `from .event` | `from .event` (same package) |
| | `from .logger` | `from .logger` (same package) |
| **framework/expmod.py** | `from .config` | `from .config` (same package) |
| **framework/logger.py** | `from .config` | `from .config` (same package) |
| **perception/base.py** | `from .component` | `from ..framework.component` |
| | `from .event` | `from ..framework.event` |
| | `from .graphdb` | `from ..db` |
| | `from .location` | `from .location` (same package) |
| **perception/feature_extractors/*.py** | `from ..graphdb` | `from ...db` |
| | `from ..perception` | `from ..base` |
| | `from ..location` | `from ..location` |
| | (cross-extractor refs stay `from .X`) | |
| **pipeline/action.py** | `from .component` | `from ..framework.component` |
| | `from .event` | `from ..framework.event` |
| | `from .expmod` | `from ..framework.expmod` |
| | `from .graphdb` | `from ..db` |
| **pipeline/intrinsic.py** | (update all `from .X` to `from ..framework.X` or `from ..db`) |
| **pipeline/significance.py** | `from .component` | `from ..framework.component` |
| | `from .config` | `from ..framework.config` |
| | `from .event` | `from ..framework.event` |
| | `from .intrinsic` | `from .intrinsic` (same package) |
| **pipeline/attention/attention.py** | `from .component` | `from ...framework.component` |
| | `from .config` | `from ...framework.config` |
| | `from .event` | `from ...framework.event` |
| | `from .location` | `from ...perception.location` |
| | `from .perception` | `from ...perception.base` |
| | `from .saliency_attenuation` | `from .saliency_attenuation` (same package) |
| | `from .reporting.X` | `from ...reporting.X` |
| **pipeline/attention/saliency_attenuation.py** | `from .attention` | `from .attention` (same package) |
| | `from .config` | `from ...framework.config` |
| | `from .expmod` | `from ...framework.expmod` |
| | `from .sequencer` | `from ..temporal.sequencer` |
| | `from .reporting.X` | `from ...reporting.X` |
| **pipeline/object/object.py** | `from .attention` | `from ..attention.attention` |
| | `from .component` | `from ...framework.component` |
| | `from .event` | `from ...framework.event` |
| | `from .expmod` | `from ...framework.expmod` |
| | `from .graphdb` | `from ...db` |
| | `from .location` | `from ...perception.location` |
| | `from .perception` | `from ...perception.base` |
| | `from .sequencer` | `from ..temporal.sequencer` |
| | `from .feature_extractors.X` | `from ...perception.feature_extractors.X` |
| | `from .reporting.X` | `from ...reporting.X` |
| **pipeline/object/object_instance.py** | `from .graphdb` | `from ...db` |
| | `from .location` | `from ...perception.location` |
| | `from .object` | `from .object` (same package) |
| | `from .object_transform` | `from .object_transform` (same package) |
| | `from .perception` | `from ...perception.base` |
| | `from .transformable` | `from ..temporal.transformable` |
| | `from .feature_extractors.X` | `from ...perception.feature_extractors.X` |
| **pipeline/object/object_transform.py** | `from .graphdb` | `from ...db` |
| | `from .object` | `from .object` (same package) |
| | `from .object_instance` | `from .object_instance` (same package) |
| | `from .transformable` | `from ..temporal.transformable` |
| **pipeline/temporal/sequencer.py** | `from .action` | `from ..action` |
| | `from .component` | `from ...framework.component` |
| | `from .event` | `from ...framework.event` |
| | `from .graphdb` | `from ...db` |
| | `from .intrinsic` | `from ..intrinsic` |
| | `from .object` | `from ..object.object` |
| | `from .object_instance` | `from ..object.object_instance` |
| | `from .perception` | `from ...perception.base` |
| | `from .transformable` | `from .transformable` (same package) |
| **pipeline/temporal/transformer.py** | `from .component` | `from ...framework.component` |
| | `from .event` | `from ...framework.event` |
| | `from .graphdb` | `from ...db` |
| | `from .object` | `from ..object.object` |
| | `from .object_instance` | `from ..object.object_instance` |
| | `from .object_transform` | `from ..object.object_transform` |
| | `from .sequencer` | `from .sequencer` (same package) |
| | `from .transformable` | `from .transformable` (same package) |
| **pipeline/temporal/predict.py** | `from .component` | `from ...framework.component` |
| | `from .event` | `from ...framework.event` |
| | `from .expmod` | `from ...framework.expmod` |
| | `from .intrinsic` | `from ..intrinsic` |
| | `from .sequencer` | `from .sequencer` (same package) |
| | `from .transformer` | `from .transformer` (same package) |
| **game/gymnasium.py** | `from .action` | `from ..pipeline.action` |
| | `from .breakpoint` | `from .breakpoint` (same package) |
| | `from .component` | `from ..framework.component` |
| | `from .config` | `from ..framework.config` |
| | `from .event` | `from ..framework.event` |
| | `from .graphdb` | `from ..db` |
| | `from .intrinsic` | `from ..pipeline.intrinsic` |
| | `from .logger` | `from ..framework.logger` |
| | `from .object_instance` | `from ..pipeline.object.object_instance` |
| | `from .object_transform` | `from ..pipeline.object.object_transform` |
| | `from .perception` | `from ..perception.base` |
| | `from .predict` | `from ..pipeline.temporal.predict` |
| | `from .sequencer` | `from ..pipeline.temporal.sequencer` |
| | `from .reporting.X` | `from ..reporting.X` (depth unchanged) |
| **game/game_manager.py** | (check for roc imports, update accordingly) |
| **game/breakpoint.py** | (check for roc imports, update accordingly) |
| **cli/script.py** | `from .config` | `from ..framework.config` |
| | `import roc` | `import roc` (unchanged) |
| **cli/server_cli.py** | `from .config` | `from ..framework.config` |
| | `from .dashboard_cli` | `from .dashboard_cli` (same package) |
| | `from .game_manager` | `from ..game.game_manager` |
| | `from .reporting.X` | `from ..reporting.X` |
| **cli/dashboard_cli.py** | `from .config` | `from ..framework.config` |
| | `from .reporting.X` | `from ..reporting.X` |
| **cli/cleanup_cli.py** | `from .config` | `from ..framework.config` |
| **reporting/*.py** | `from .config` | `from ..framework.config` |
| (reporting stays at same depth) | `from .logger` | `from ..framework.logger` |
| | `from .graphdb` | `from ..db` |
| | `from .object` | `from ..pipeline.object.object` |
| | `from .object_instance` | `from ..pipeline.object.object_instance` |
| | `from .object_transform` | `from ..pipeline.object.object_transform` |
| | `from .component` | `from ..framework.component` |
| | `from .event` | `from ..framework.event` |
| | `from .attention` | `from ..pipeline.attention.attention` |
| | `from .intrinsic` | `from ..pipeline.intrinsic` |
| | `from .perception` | `from ..perception.base` |
| | `from .predict` | `from ..pipeline.temporal.predict` |
| | `from .sequencer` | `from ..pipeline.temporal.sequencer` |
| | `from .significance` | `from ..pipeline.significance` |
| | `from .transformer` | `from ..pipeline.temporal.transformer` |
| | `from .action` | `from ..pipeline.action` |
| | `from .feature_extractors.X` | `from ..perception.feature_extractors.X` |
| **jupyter/*.py** | `from .breakpoint` | `from ..game.breakpoint` |
| (jupyter stays at same depth) | `from .logger` | `from ..framework.logger` |
| | `from .graphdb` | `from ..db` |
| | `from .reporting.state` | `from ..reporting.state` |

### Test code: absolute import rewrites

Tests use absolute imports. The rewrite is mechanical:

| Old pattern | New pattern |
|-------------|-----------|
| `from roc.component import ...` | `from roc.framework.component import ...` |
| `from roc.config import ...` | `from roc.framework.config import ...` |
| `from roc.event import ...` | `from roc.framework.event import ...` |
| `from roc.expmod import ...` | `from roc.framework.expmod import ...` |
| `from roc.logger import ...` | `from roc.framework.logger import ...` |
| `from roc.utils import ...` | `from roc.framework.utils import ...` |
| `from roc.graphdb import ...` | `from roc.db import ...` |
| `from roc.perception import ...` | `from roc.perception import ...` (via __init__.py re-export) |
| `from roc.location import ...` | `from roc.perception.location import ...` |
| `from roc.feature_extractors.X import ...` | `from roc.perception.feature_extractors.X import ...` |
| `from roc.attention import ...` | `from roc.pipeline.attention.attention import ...` |
| `from roc.saliency_attenuation import ...` | `from roc.pipeline.attention.saliency_attenuation import ...` |
| `from roc.object import ...` | `from roc.pipeline.object.object import ...` |
| `from roc.object_instance import ...` | `from roc.pipeline.object.object_instance import ...` |
| `from roc.object_transform import ...` | `from roc.pipeline.object.object_transform import ...` |
| `from roc.sequencer import ...` | `from roc.pipeline.temporal.sequencer import ...` |
| `from roc.transformable import ...` | `from roc.pipeline.temporal.transformable import ...` |
| `from roc.transformer import ...` | `from roc.pipeline.temporal.transformer import ...` |
| `from roc.predict import ...` | `from roc.pipeline.temporal.predict import ...` |
| `from roc.action import ...` | `from roc.pipeline.action import ...` |
| `from roc.intrinsic import ...` | `from roc.pipeline.intrinsic import ...` |
| `from roc.significance import ...` | `from roc.pipeline.significance import ...` |
| `from roc.gymnasium import ...` | `from roc.game.gymnasium import ...` |
| `from roc.game_manager import ...` | `from roc.game.game_manager import ...` |
| `from roc.breakpoint import ...` | `from roc.game.breakpoint import ...` |
| `from roc.script import ...` | `from roc.cli.script import ...` |
| `from roc.server_cli import ...` | `from roc.cli.server_cli import ...` |
| `from roc.dashboard_cli import ...` | `from roc.cli.dashboard_cli import ...` |
| `from roc.cleanup_cli import ...` | `from roc.cli.cleanup_cli import ...` |
| `from roc.debugpy_setup import ...` | `from roc.debugpy_setup import ...` (unchanged) |
| `from roc.reporting.X import ...` | `from roc.reporting.X import ...` (unchanged) |

### experiments/ directory

The `experiments/` directory contains external ExpMod implementations that import
from `roc`. These must also be updated using the same absolute import patterns
as test code above.

## Naming Collision: perception.perception

Moving `roc/perception.py` into `roc/perception/perception.py` creates a
module-inside-same-named-package pattern. Options:

1. **Keep it**: `roc.perception.perception` -- explicit, no ambiguity, but stutters
2. **Rename to `base.py`**: `roc.perception.base` -- common Python convention
3. **Inline into `__init__.py`**: Put Perception base class directly in
   `roc/perception/__init__.py` -- simplest, but makes the init file large (318 lines)

Recommendation: **Option 2** (`base.py`). The `__init__.py` re-exports public
symbols so callers still write `from roc.perception import FeatureNode`.

## __init__.py Re-export Strategy

Each new package's `__init__.py` re-exports its public API so callers can import
from the package directly rather than reaching into submodules. This is especially
important for `roc/db/` (which replaces the single `graphdb.py`) and
`roc/perception/` (where `perception.py` becomes `base.py`).

```python
# roc/db/__init__.py
from roc.db.cache import GraphCache
from roc.db.db import GraphDB
from roc.db.edge import Edge, EdgeConnectionsList, EdgeId, EdgeList, edge_registry
from roc.db.node import Node, NodeId, NodeList, node_label_registry, node_registry

# roc/framework/__init__.py
from roc.framework.component import Component, ComponentId, ComponentKey
from roc.framework.config import Config
from roc.framework.event import BusConnection, Event, EventBus
from roc.framework.expmod import ExpMod
from roc.framework.logger import logger

# roc/perception/__init__.py
from roc.perception.base import (
    Detail, FeatureExtractor, FeatureKind, FeatureNode, Perception,
    VisualFeature, VisionData, AuditoryData, ProprioceptiveData,
)
from roc.perception.location import Point, PointCollection, XLoc, YLoc

# roc/pipeline/__init__.py
# Intentionally minimal -- callers import from subpackages directly

# roc/pipeline/attention/__init__.py
# Empty -- callers import from attention.py or saliency_attenuation.py directly

# roc/pipeline/object/__init__.py
# Empty -- callers import from object.py, object_instance.py, etc. directly

# roc/pipeline/temporal/__init__.py
# Empty -- callers import from sequencer.py, transformer.py, etc. directly

# roc/game/__init__.py
from roc.game.gymnasium import NethackGym
from roc.game.game_manager import GameManager
from roc.game.breakpoint import breakpoints

# roc/cli/__init__.py
# Empty -- entry points are referenced by pyproject.toml, not imported
```

## Test Restructuring

Since we are doing everything at once, the test directory mirrors the new source
layout in the same pass. Both imports and file locations are updated together.

### Target test structure

```
tests/
  conftest.py                          # Root fixtures (unchanged)
  helpers/                             # Shared test utilities (unchanged)

  unit/
    conftest.py                        # Unit fixtures (unchanged)
    __init__.py

    framework/
      __init__.py
      test_component.py
      test_config.py
      test_event.py
      test_expmod.py
      test_expmod_params.py
      test_logger.py
      test_utils.py

    db/
      __init__.py
      test_graphdb_pure.py

    perception/
      __init__.py
      test_perception.py
      test_location.py
      feature_extractors/
        __init__.py
        test_color.py                  # from unit/feature_extractors/
        test_phoneme.py                # from unit/feature_extractors/

    pipeline/
      __init__.py
      test_action.py
      test_intrinsic.py
      test_significance.py
      attention/
        __init__.py
        test_attention.py
        test_multi_cycle_attention.py
        test_saliency_attenuation.py
        test_active_inference.py
      object/
        __init__.py
        test_object.py
        test_object_instance.py
        test_object_telemetry.py
        test_object_transform.py
        test_dirichlet_resolution.py
      temporal/
        __init__.py
        test_sequencer.py
        test_transformable.py
        test_transformer.py
        test_predict.py
        test_predict_unit.py

    game/
      __init__.py
      test_gymnasium.py
      test_game_manager.py
      test_breakpoint.py

    cli/
      __init__.py
      test_script.py

    reporting/
      __init__.py
      test_api_server.py
      test_data_store.py
      test_observability.py
      test_observability_extra.py
      test_parquet_exporter.py
      test_remote_logger_exporter.py
      test_run_store.py
      test_screen_renderer.py
      test_state.py
      test_state_extra.py
      test_state_snapshots.py
      test_step_buffer.py
      test_step_log_sink.py

    test_debugpy_setup.py              # stays at unit/ root
    test_init.py                       # stays at unit/ root

  integration/
    conftest.py
    __init__.py
    db/
      __init__.py
      test_graphdb.py
    pipeline/
      __init__.py
      attention/
        __init__.py
        test_attention.py
        test_saliency_attenuation_integration.py
      object/
        __init__.py
        test_object.py
        test_dirichlet_integration.py
        test_object_transform_pipeline.py
      temporal/
        __init__.py
        test_sequencer.py
        test_transformer.py

  e2e/                                 # unchanged
    conftest.py
    __init__.py
    test_agent_pipeline.py
    test_predict.py

  test_feature_extractors/             # see note below
    color_test.py
    delta_test.py
    ...
```

### Unit test file mapping

| Old location | New location |
|-------------|-------------|
| `unit/test_component.py` | `unit/framework/test_component.py` |
| `unit/test_config.py` | `unit/framework/test_config.py` |
| `unit/test_event.py` | `unit/framework/test_event.py` |
| `unit/test_expmod.py` | `unit/framework/test_expmod.py` |
| `unit/test_expmod_params.py` | `unit/framework/test_expmod_params.py` |
| `unit/test_logger.py` | `unit/framework/test_logger.py` |
| `unit/test_utils.py` | `unit/framework/test_utils.py` |
| `unit/test_graphdb_pure.py` | `unit/db/test_graphdb_pure.py` |
| `unit/test_perception.py` | `unit/perception/test_perception.py` |
| `unit/test_location.py` | `unit/perception/test_location.py` |
| `unit/feature_extractors/test_color.py` | `unit/perception/feature_extractors/test_color.py` |
| `unit/feature_extractors/test_phoneme.py` | `unit/perception/feature_extractors/test_phoneme.py` |
| `unit/test_action.py` | `unit/pipeline/test_action.py` |
| `unit/test_intrinsic.py` | `unit/pipeline/test_intrinsic.py` |
| `unit/test_significance.py` | `unit/pipeline/test_significance.py` |
| `unit/test_attention.py` | `unit/pipeline/attention/test_attention.py` |
| `unit/test_multi_cycle_attention.py` | `unit/pipeline/attention/test_multi_cycle_attention.py` |
| `unit/test_saliency_attenuation.py` | `unit/pipeline/attention/test_saliency_attenuation.py` |
| `unit/test_active_inference.py` | `unit/pipeline/attention/test_active_inference.py` |
| `unit/test_object.py` | `unit/pipeline/object/test_object.py` |
| `unit/test_object_instance.py` | `unit/pipeline/object/test_object_instance.py` |
| `unit/test_object_telemetry.py` | `unit/pipeline/object/test_object_telemetry.py` |
| `unit/test_object_transform.py` | `unit/pipeline/object/test_object_transform.py` |
| `unit/test_dirichlet_resolution.py` | `unit/pipeline/object/test_dirichlet_resolution.py` |
| `unit/test_sequencer.py` | `unit/pipeline/temporal/test_sequencer.py` |
| `unit/test_transformable.py` | `unit/pipeline/temporal/test_transformable.py` |
| `unit/test_transformer.py` | `unit/pipeline/temporal/test_transformer.py` |
| `unit/test_predict.py` | `unit/pipeline/temporal/test_predict.py` |
| `unit/test_predict_unit.py` | `unit/pipeline/temporal/test_predict_unit.py` |
| `unit/test_gymnasium.py` | `unit/game/test_gymnasium.py` |
| `unit/test_game_manager.py` | `unit/game/test_game_manager.py` |
| `unit/test_breakpoint.py` | `unit/game/test_breakpoint.py` |
| `unit/test_script.py` | `unit/cli/test_script.py` |
| `unit/test_api_server.py` | `unit/reporting/test_api_server.py` |
| `unit/test_data_store.py` | `unit/reporting/test_data_store.py` |
| `unit/test_observability.py` | `unit/reporting/test_observability.py` |
| `unit/test_observability_extra.py` | `unit/reporting/test_observability_extra.py` |
| `unit/test_parquet_exporter.py` | `unit/reporting/test_parquet_exporter.py` |
| `unit/test_remote_logger_exporter.py` | `unit/reporting/test_remote_logger_exporter.py` |
| `unit/test_run_store.py` | `unit/reporting/test_run_store.py` |
| `unit/test_screen_renderer.py` | `unit/reporting/test_screen_renderer.py` |
| `unit/test_state.py` | `unit/reporting/test_state.py` |
| `unit/test_state_extra.py` | `unit/reporting/test_state_extra.py` |
| `unit/test_state_snapshots.py` | `unit/reporting/test_state_snapshots.py` |
| `unit/test_step_buffer.py` | `unit/reporting/test_step_buffer.py` |
| `unit/test_step_log_sink.py` | `unit/reporting/test_step_log_sink.py` |
| `unit/test_debugpy_setup.py` | `unit/test_debugpy_setup.py` (stays) |
| `unit/test_init.py` | `unit/test_init.py` (stays) |

### Integration test file mapping

| Old location | New location |
|-------------|-------------|
| `integration/test_graphdb.py` | `integration/db/test_graphdb.py` |
| `integration/test_attention.py` | `integration/pipeline/attention/test_attention.py` |
| `integration/test_saliency_attenuation_integration.py` | `integration/pipeline/attention/test_saliency_attenuation_integration.py` |
| `integration/test_object.py` | `integration/pipeline/object/test_object.py` |
| `integration/test_dirichlet_integration.py` | `integration/pipeline/object/test_dirichlet_integration.py` |
| `integration/test_object_transform_pipeline.py` | `integration/pipeline/object/test_object_transform_pipeline.py` |
| `integration/test_sequencer.py` | `integration/pipeline/temporal/test_sequencer.py` |
| `integration/test_transformer.py` | `integration/pipeline/temporal/test_transformer.py` |

### tests/test_feature_extractors/

This top-level directory uses a different naming convention (`color_test.py` vs
`test_color.py`) and has no `__init__.py` or `conftest.py`. It contains 9 test
files (one per extractor), separate from the 2 files in `unit/feature_extractors/`.

Options:
1. **Move to `unit/perception/feature_extractors/`** and rename to `test_*.py`
   convention -- merges with the existing unit tests there
2. **Leave in place** -- it works, and moving it is a larger rename effort
3. **Move under perception/ but keep naming** -- `test_feature_extractors/` becomes
   `perception/feature_extractors/` at the test root level

Recommendation: **Option 1** for consistency, but this can be deferred since these
files have no `__init__.py` and are discovered by pytest's default collection.

### Conftest files

- `tests/conftest.py` -- stays, root-level fixtures apply to all tests
- `tests/unit/conftest.py` -- stays at `tests/unit/conftest.py`, its autouse
  fixtures apply to all unit tests regardless of subdirectory
- `tests/integration/conftest.py` -- stays at `tests/integration/conftest.py`
- `tests/e2e/conftest.py` -- stays, unchanged

No new conftest files needed in subdirectories unless fixtures need to be scoped
to a specific subpackage.

## Verification Checklist

Single-pass verification after all moves and rewrites are complete:

- [ ] All new source directories created with `__init__.py`
- [ ] All new test directories created with `__init__.py`
- [ ] All source files moved to new locations (no stale files at old paths)
- [ ] All test files moved to new locations (no stale files at old paths)
- [ ] All relative imports in source code updated
- [ ] All absolute imports in test code updated
- [ ] All absolute imports in `experiments/` updated
- [ ] `pyproject.toml` entry points updated
- [ ] `pyproject.toml` warning filter updated
- [ ] `roc/__init__.py` updated
- [ ] `roc/debugpy_setup.py` updated
- [ ] `roc/db/__init__.py` re-exports complete
- [ ] `roc/perception/__init__.py` re-exports complete
- [ ] No old source module paths remain:
  ```
  grep -rn "from roc\.\(component\|config\|event\|expmod\|logger\|utils\|graphdb\
  \|location\|attention\|saliency_attenuation\|object\b\|object_instance\
  \|object_transform\|sequencer\|transformable\|transformer\|predict\|action\
  \|intrinsic\|significance\|gymnasium\|game_manager\|breakpoint\|script\
  \|server_cli\|dashboard_cli\|cleanup_cli\) " roc/ tests/ experiments/
  ```
- [ ] No stale test files at old flat locations:
  ```
  ls tests/unit/test_component.py tests/unit/test_config.py \
     tests/unit/test_event.py tests/unit/test_graphdb_pure.py \
     tests/unit/test_object.py 2>/dev/null  # should find nothing
  ```
- [ ] `make lint` passes (mypy + ruff check + ruff format)
- [ ] `make test` passes
- [ ] `make coverage` shows no regressions
- [ ] Makefile test targets still work (`make test-unit`, `make test-integration`, `make test-e2e`)

## Open Questions

1. **Should `debugpy_setup.py` move?** It only imports `config` and `logger`. Could
   go into `framework/` but it is a bootstrap module, not framework infrastructure.
   Current plan: leave at top level.

2. **Should `reporting/` move under `pipeline/`?** It listens on pipeline buses but
   is more of a side-channel observer than a pipeline stage. Current plan: leave at
   top level.

3. **graphdb split boundaries**: The exact line counts above are estimates. The split
   needs a careful read of `graphdb.py` to identify clean cut points, especially
   around Node's dependency on GraphCache and Edge's dependency on Node.
