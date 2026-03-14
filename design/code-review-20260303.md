# Code Review Report - 2026-03-03

## Executive Summary

- Files reviewed: 48 production, 27 test, 5 configuration
- Critical issues: 4
- High priority issues: 10
- Medium priority issues: 14
- Low priority issues: 8

The ROC codebase has a solid architectural foundation with a well-designed component system, typed event buses, and a sophisticated graph database abstraction. The code demonstrates strong domain modeling for the NetHack agent problem. However, there are several correctness bugs, race conditions in the concurrent event system, and performance issues in the critical perception pipeline that should be addressed.

---

## File Inventory

### Production Code (roc/)
| Category | Files | Lines |
|----------|-------|-------|
| Core architecture | component.py, event.py, config.py, graphdb.py | ~3,000 |
| Data pipeline | object.py, sequencer.py, transformer.py, transformable.py, intrinsic.py | ~720 |
| Perception | perception.py, attention.py, feature_extractors/* (10 files) | ~1,500 |
| Environment | gymnasium.py, action.py | ~450 |
| Utilities | location.py, logger.py, breakpoint.py, predict.py, significance.py, expmod.py, utils.py | ~960 |
| Reporting | reporting/observability.py, reporting/state.py | ~770 |
| Jupyter | jupyter/* (8 files) | ~275 |

### Test Code (tests/)
| Category | Files | Lines |
|----------|-------|-------|
| Core tests | 18 test files | ~5,100 |
| Feature extractor tests | 9 test files | ~2,300 |
| Test helpers | 10 helper files | ~1,170 |

### Configuration
pyproject.toml, Makefile, .env, .pre-commit-config.yaml, setup.cfg

---

## Critical Issues (Fix Immediately)

### 1. NaN comparison bug in IntrinsicData.to_nodes()

- **File**: `roc/intrinsic.py:184`
- **Description**: `math.nan` is never equal to anything, including itself. The filter `self.normalized_intrinsics[k] != math.nan` is **always True** for all values, including actual NaN values. This means NaN intrinsics are never filtered out and will silently corrupt downstream calculations.
- **Example**:
```python
# intrinsic.py:184 -- BROKEN
node_intrinsics = [
    k
    for k, v in self.normalized_intrinsics.items()
    if self.normalized_intrinsics[k] != math.nan  # Always True! NaN != NaN is True
]
```
- **Fix**:
```python
node_intrinsics = [
    k
    for k, v in self.normalized_intrinsics.items()
    if not math.isnan(v)
]
```

### 2. Race conditions in concurrent event listeners

- **Files**: `roc/event.py:28`, `roc/attention.py:238-279`, `roc/object.py:157-183`, `roc/sequencer.py:118-155`
- **Description**: Events are dispatched on a `ThreadPoolScheduler` with `cpu_count * 2` threads. Multiple components mutate shared state from event handlers without any synchronization:
  - `VisionAttention.do_attention()` mutates `self.saliency_map` and `self.settled` set from multiple threads. If two `Settled` events arrive concurrently, the set membership check and clear can race, potentially sending duplicate attention events or missing features.
  - `Sequencer` mutates `self.current_frame` and `self.last_frame` from handlers for three different event types. An action event and an object resolution event arriving simultaneously would race on `self.current_frame`.
  - `Motion.get_feature()` appends to and iterates `self.delta_list` without locking, risking missed deltas or index errors during concurrent iteration.
- **Fix**: Either:
  - (a) Add `threading.Lock` guards around shared mutable state in each component, or
  - (b) Ensure each component's listeners are dispatched on a single-threaded scheduler (e.g. `rx.scheduler.CurrentThreadScheduler`), or
  - (c) Use `op.observe_on(serial_scheduler)` in the listen pipeline to serialize events per-component.

  Option (c) is the least invasive:
```python
# event.py - in BusConnection.listen()
from reactivex.scheduler import NewThreadScheduler

# Create a per-component serial scheduler
component_scheduler = NewThreadScheduler()

sub = self.subject.pipe(*pipe_args).subscribe(
    listener, scheduler=component_scheduler
)
```

### 3. Potential SQL/Cypher injection in graphdb queries

- **Files**: `roc/graphdb.py:641-643`, `roc/graphdb.py:685`, `roc/graphdb.py:774`, `roc/graphdb.py:1266`, `roc/graphdb.py:1571`, `roc/graphdb.py:1602`, `roc/graphdb.py:1675`, `roc/graphdb.py:1715`
- **Description**: Node and Edge IDs are integer NewTypes, so direct string interpolation of IDs into Cypher queries (e.g., `f"WHERE id(n) = {n.id}"`) is safe for the **current** code since IDs are always integers. However, `Node.find()` accepts an arbitrary `where` string parameter that is directly interpolated into the query. If any external input ever flows into `where`, this becomes a Cypher injection vector.
- **Example**:
```python
# graphdb.py:1310 -- where parameter is interpolated directly
res_iter = db.raw_fetch(
    f"""
        MATCH ({src_node_name}{src_label_str})-[{edge_name}{edge_type}*0..1]-()
        ...
        WHERE {where}
        ...
    """,
    params=params,
)
```
  The `where` parameter and `src_node_name`/`edge_name`/`edge_type` are all directly interpolated. While these are currently only called with controlled strings, this is a fragile pattern.
- **Fix**: Add a comment/assertion that `where` must not contain user input, or better yet, build the where clause from structured parameters rather than raw strings. At minimum, validate that identifier parameters like `src_node_name` match `^[a-z_]+$`.

### 4. File handle leak in gymnasium.py dump functions

- **File**: `roc/gymnasium.py:346`
- **Description**: `_dump_env_start()` opens a file but `_dump_env_end()` only runs at the end of the normal game loop. If the game loop raises an exception (e.g., NLE crash, keyboard interrupt), the file handle leaks.
- **Example**:
```python
# gymnasium.py:346
def _dump_env_start() -> None:
    ...
    global dump_env_file
    dump_env_file = open(settings.dump_file, "w")  # No context manager
```
- **Fix**: Use a context manager or wrap the game loop in try/finally:
```python
def _dump_env_start() -> None:
    ...
    global dump_env_file
    dump_env_file = open(settings.dump_file, "w")

# In Gym.start(), ensure cleanup:
def start(self) -> None:
    try:
        # ... game loop ...
    finally:
        _dump_env_end()
```

---

## High Priority Issues (Fix Soon)

### 1. Global mutable state makes testing fragile and prevents parallelism

- **Files**: `roc/component.py:23-24`, `roc/event.py:134`, `roc/graphdb.py:57-58,89,804,1965-1977`, `roc/intrinsic.py:37`, `roc/expmod.py:12-14`, `roc/config.py:20`
- **Description**: The codebase uses ~15 module-level mutable globals: `component_set`, `loaded_components`, `eventbus_names`, `next_new_node`/`next_new_edge`, `graph_db_singleton`, `node_cache`/`edge_cache`, `node_registry`/`edge_registry`/`node_label_registry`, `component_registry`, `default_components`, `intrinsic_op_registry`, `expmod_registry`, `_config_singleton`, `fe_list`, `cache_registry`, `tick`. Each requires careful cleanup in tests (as evidenced by the extensive `conftest.py` fixtures). This is the single largest source of test fragility.
- **Impact**: Prevents parallel test execution. Makes it easy to introduce subtle test ordering dependencies. The `conftest.py` `restore_registries` fixture already has to save/restore 3 different registries and the `clear_cache` fixture has to handle the node/edge caches.
- **Fix**: Consider consolidating globals into a `RuntimeContext` object that can be constructed fresh per test. Long-term, pass context objects explicitly rather than relying on module state.

### 2. `Edge.get_cache` uses `@classmethod` with `self` parameter

- **File**: `roc/graphdb.py:523-535`
- **Description**: `Edge.get_cache` is declared as `@classmethod` but the first parameter is named `self` instead of `cls`. This works in CPython because the name of the first parameter is just a convention, but it is misleading and will confuse readers and linters.
- **Example**:
```python
@classmethod
def get_cache(self) -> EdgeCache:  # should be 'cls'
    global edge_cache
    ...
```
- **Fix**:
```python
@classmethod
def get_cache(cls) -> EdgeCache:
    global edge_cache
    ...
```

### 3. Config.init() called at module import time

- **File**: `roc/config.py:233`
- **Description**: `Config.init()` is called at the bottom of `config.py`, meaning the configuration singleton is created as a side effect of importing the module. This happens before `roc.init()` is ever called, before any user-provided config overrides are applied, and before any environment is set up. While there is a `force` parameter to re-init, this leads to confusing double-initialization warnings.
- **Impact**: The pytest hack on line 31-37 (using a random env_prefix string during test imports) exists specifically to work around this. Any code that imports `roc.config` triggers config loading.
- **Fix**: Remove the `Config.init()` call on line 233. Make `Config.get()` lazily initialize with defaults instead.

### 4. `NoPrediction` class uses bare `None` as class body

- **File**: `roc/predict.py:11-12`
- **Description**: The `NoPrediction` class has `None` as its body, which evaluates to `None` (an expression statement, not an error) but is unconventional and misleading. It looks like a mistake.
- **Example**:
```python
class NoPrediction:
    None  # This evaluates None as an expression, does nothing
```
- **Fix**:
```python
class NoPrediction:
    pass
```

### 5. `ChangedPoint.__init_` has a typo -- missing underscore

- **File**: `roc/location.py:96`
- **Description**: The method is named `__init_` (single trailing underscore) instead of `__init__` (double trailing underscore). This means it is never called as a constructor -- `ChangedPoint` objects will use `Point.__init__` and the `old_val` attribute will never be set.
- **Example**:
```python
class ChangedPoint(Point):
    def __init_(self, x: XLoc, y: YLoc, val: int, old_val: int) -> None:  # __init_ not __init__
        super().__init__(x, y, val)
        self.old_val = old_val
```
- **Fix**:
```python
    def __init__(self, x: XLoc, y: YLoc, val: int, old_val: int) -> None:
```

### 6. Duplicate imports in graphdb.py shadow stdlib types

- **File**: `roc/graphdb.py:13,18-22`
- **Description**: `Collection` and `Iterable` are imported from both `collections.abc` (line 13) and `typing` (lines 18-22). The `typing` versions shadow the `collections.abc` versions. While both are equivalent in Python 3.9+, the duplicate imports are confusing and `typing.Collection`/`typing.Iterable` are deprecated in favor of `collections.abc`.
- **Example**:
```python
from collections.abc import Collection, Iterable, Iterator, MutableSet, Sequence
...
from typing import (
    ...
    Collection,   # shadows collections.abc.Collection
    ...
    Iterable,     # shadows collections.abc.Iterable
    ...
)
```
- **Fix**: Remove `Collection` and `Iterable` from the `typing` import block. Also remove the import of `_SpecialForm` which is a private API.

### 7. `ObjectResolver.event_filter` signature mismatch

- **File**: `roc/object.py:157`
- **Description**: `ObjectResolver.event_filter` expects `AttentionEvent` as its parameter type, but the base class `Component.event_filter` signature expects `Event[Any]`. Additionally, the ObjectResolver connects to both the Attention bus and its own bus -- the filter will be applied to events from both buses, but it only checks for attention event source IDs. Events on the ObjectResolver bus will always be filtered out since they come from the resolver itself.
- **Impact**: Currently this is fine because the default `Component.event_filter` already filters self-sends. But if the ObjectResolver ever needs to receive events from its own bus, this filter will silently block them.
- **Fix**: Match the base class signature:
```python
def event_filter(self, e: Event[Any]) -> bool:
    if isinstance(e, Event) and hasattr(e, 'src_id'):
        return e.src_id.name == "vision" and e.src_id.type == "attention"
    return e.src_id != self.id
```

### 8. `Sequencer.event_filter` prevents self-send filtering

- **File**: `roc/sequencer.py:129-134`
- **Description**: The Sequencer overrides `event_filter` to check data types instead of calling `super().event_filter()`. This means the default self-send filtering from `Component.event_filter` is bypassed. If the Sequencer ever sends an event on a bus it listens to, it would receive its own event.
- **Fix**:
```python
def event_filter(self, e: Event[Any]) -> bool:
    if not super().event_filter(e):
        return False
    return (
        isinstance(e.data, ResolvedObject)
        or isinstance(e.data, TakeAction)
        or isinstance(e.data, IntrinsicData)
    )
```
The same pattern applies to `Transformer.event_filter` (transformer.py:30), `Predict.event_filter` (predict.py:33), and `Significance.event_filter` (significance.py:29).

### 9. `VisionAttention.do_attention` calls `get_focus()` twice

- **File**: `roc/attention.py:268-272`
- **Description**: `get_focus()` is called twice: once to store in `focus` (line 268), and once inline in `VisionAttentionData` constructor (line 272). The `focus` variable is never used. `get_focus()` performs expensive scipy operations (dilation reconstruction, labeling, DataFrame creation).
- **Example**:
```python
focus = self.saliency_map.get_focus()  # called once, result unused

self.att_conn.send(
    VisionAttentionData(
        focus_points=self.saliency_map.get_focus(),  # called again
        saliency_map=self.saliency_map,
    )
)
```
- **Fix**:
```python
focus = self.saliency_map.get_focus()

self.att_conn.send(
    VisionAttentionData(
        focus_points=focus,
        saliency_map=self.saliency_map,
    )
)
```

### 10. `gymnasium.py` uses deprecated `.dict()` Pydantic method

- **Files**: `roc/gymnasium.py:318`, `roc/config.py:178`
- **Description**: `BottomlineStats(...).dict()` and `Config.__str__` use `.dict()` which is deprecated in Pydantic v2 in favor of `.model_dump()`.
- **Fix**: Replace `.dict()` with `.model_dump()`.

---

## Medium Priority Issues (Technical Debt)

### 1. Performance: Feature extractors iterate full grid per-frame

- **Files**: `roc/feature_extractors/single.py:65`, `roc/feature_extractors/delta.py:84`
- **Description**: The NetHack dungeon is 21x79 = 1,659 cells. The `Single` extractor iterates all cells and checks 8 neighbors for each (up to ~13,000 comparisons per frame). The `Delta` extractor also iterates all cells. With 9 perception components running per frame, this is the critical path. Currently there is no short-circuiting or batching.
- **Impact**: For real-time NetHack play, this is likely acceptable given the game's turn-based nature. But as more feature extractors are added, the O(rows * cols * extractors) cost will grow linearly.
- **Recommendation**: Consider numpy vectorized operations for extractors like Single (e.g., use `scipy.ndimage` neighbor comparison) and Delta (e.g., `np.where(prev != curr)`).

### 2. `EventBus` name uniqueness is global and never cleaned up

- **File**: `roc/event.py:134-155`
- **Description**: `eventbus_names` is a global set. EventBus names are added on construction but never removed (there is no destructor that removes names). `EventBus.clear_names()` exists but only for testing. Since EventBus instances are created as class variables (e.g., `Perception.bus = EventBus[...]("perception")`), they are module-level singletons that persist for the process lifetime. This means you cannot create a second agent instance or reinitialize the system without calling `clear_names()` first.
- **Fix**: Either add an `EventBus.close()` method that removes the name, or make EventBus instances scoped to a runtime context rather than being class-level singletons.

### 3. `CrossModalAttention` is a stub that registers as auto-load

- **File**: `roc/attention.py:295-301`
- **Description**: `CrossModalAttention` has `auto: bool = True` but is completely empty -- it has no `__init__`, no event listeners, and no logic. It registers itself as a component, taking up a slot in the component registry and being loaded at startup for no reason.
- **Fix**: Either implement it or set `auto = False` until it is implemented.

### 4. `Node.__del__` and `Edge.__del__` trigger database writes during garbage collection

- **Files**: `roc/graphdb.py:492-494`, `roc/graphdb.py:1088-1094`
- **Description**: Both `Node.__del__` and `Edge.__del__` attempt to save to the database. Python's garbage collector can run `__del__` at arbitrary times, including interpreter shutdown when the database connection may already be closed. The `Node.__del__` wraps this in a try/except that emits a warning, but `Edge.__del__` does not -- it will raise unhandled exceptions during GC.
- **Impact**: The `ErrorSavingDuringDelWarning` is already being generated, indicating this is a known issue. Save-on-GC is unpredictable and can cause cascading saves (saving a node triggers saving its edges which triggers saving their nodes).
- **Recommendation**: Consider an explicit `flush()` call instead of relying on `__del__`. The `GraphDB.flush()` method already exists and does the right thing.

### 5. `Node.find` and `Node.find_one` use mutable default arguments

- **Files**: `roc/graphdb.py:1283-1286`, `roc/graphdb.py:1386-1390`
- **Description**: The `src_labels`, `params`, and `extra_styles` parameters use mutable defaults (`set()`, `dict()`). In Python, mutable defaults are shared across all calls to the function. While these are not mutated in `find()` itself, the `params_to_str` block on line 1307 **does** mutate `params` in-place (`params[k] = str(params[k])`), which would corrupt the shared default dict if `params` was not passed.
- **Example**:
```python
def find(
    cls,
    where: str,
    ...
    params: QueryParamType = dict(),  # mutable default
    ...
) -> list[Self]:
    ...
    if params_to_str:
        for k in params.keys():
            params[k] = str(params[k])  # mutates in place!
```
- **Fix**: Use `None` as default and create new mutable objects inside the function:
```python
def find(
    cls,
    where: str,
    ...
    params: QueryParamType | None = None,
    ...
) -> list[Self]:
    params = params or {}
    ...
```

### 6. `EdgeList.select` stores `db` as instance attribute

- **File**: `roc/graphdb.py:966`
- **Description**: `self.db = db or GraphDB.singleton()` stores the database as an instance attribute during `select()`. This is a side effect of what should be a pure query method. It mutates the EdgeList and could cause confusion if the EdgeList is reused with a different database.
- **Fix**: Use a local variable instead:
```python
db = db or GraphDB.singleton()
```

### 7. `Config.__str__` and `Config.print` break encapsulation

- **File**: `roc/config.py:176-186`
- **Description**: `Config.__str__` uses the deprecated `self.dict()` and manually formats the output. `Config.print()` uses `print()` directly. Pydantic models already have good `__repr__` and serialization support.
- **Fix**: Use `self.model_dump()` and let the caller decide how to print.

### 8. `Object.__str__` creates a new `FlexiHumanHash` instance on every call

- **File**: `roc/object.py:62-64`
- **Description**: Every call to `str(obj)` for an Object creates a new `FlexiHumanHash` with a complex template string. This could be expensive in hot paths like logging or debugging.
- **Fix**: Make the `FlexiHumanHash` a class-level constant:
```python
class Object(Node):
    _fhh = FlexiHumanHash("{{adj}}-{{noun}}-named-{{firstname|lower}}-{{lastname|lower}}-{{hex(6)}}")

    def __str__(self) -> str:
        h = self._fhh.hash(self.uuid)
        ...
```

### 9. `CandidateObjects` performs explosion of graph traversals

- **File**: `roc/object.py:114-125`
- **Description**: For object resolution, the code traverses feature_nodes -> predecessors (FeatureGroups) -> predecessors (Objects). Each `.predecessors` call iterates all `dst_edges` and fetches each edge's source node. For a large graph, this is O(features * feature_groups * objects) individual cache lookups or DB queries.
- **Fix**: Consider adding a batch query method that retrieves all related objects in a single Cypher query rather than traversing the graph one hop at a time in Python.

### 10. `Component.reset()` iterates `component_set` while calling `shutdown()`

- **File**: `roc/component.py:203-209`
- **Description**: `Component.reset()` iterates `component_set` (a WeakSet) while calling `shutdown()` on each component. If `shutdown()` causes any component to be garbage collected (removing it from the WeakSet), the iteration may behave unexpectedly. WeakSet iteration is not safe when the set changes size.
- **Fix**: Copy the set before iterating:
```python
@staticmethod
def reset() -> None:
    global loaded_components
    for name in loaded_components:
        loaded_components[name].shutdown()
    loaded_components.clear()

    global component_set
    for c in list(component_set):  # copy to list before iterating
        c.shutdown()
```

### 11. `perception.py:259` uses `not in` without spacing

- **File**: `roc/perception.py:259`
- **Description**: `if not k in d:` should be `if k not in d:` per Python style conventions. The `not in` operator is more readable and is the idiomatic form.

### 12. `gymnasium.py` hardcodes observability host URLs

- **File**: `roc/config.py:106-107`
- **Description**: The default observability host is `http://hal.ato.ms:4317` and profiling host is `http://hal.ato.ms:4040`. These are hardcoded defaults that point to a specific machine. If this machine is unavailable, observability initialization could block or fail.
- **Impact**: Low -- these are config defaults that can be overridden. But developers who don't have access to `hal.ato.ms` will see connection errors.
- **Fix**: Default to `localhost` or disable by default.

### 13. `SaliencyMap.size` property shadows numpy `ndarray.size`

- **File**: `roc/attention.py:119-125`
- **Description**: `SaliencyMap` inherits from `Grid` which inherits from `np.ndarray`. The `size` property shadows `np.ndarray.size`, which normally returns the total number of elements. The custom `size` instead returns the sum of all feature list lengths. This could cause confusing behavior if numpy code internally uses `.size`.
- **Fix**: Rename to something like `feature_count` or `total_features` to avoid shadowing.

### 14. `intrinsic.py` IntrinsicPercentOp can divide by zero

- **File**: `roc/intrinsic.py:70-71`
- **Description**: `IntrinsicPercentOp.normalize` divides by `raw_intrinsics[self.base]`. If `hpmax` or `enemax` is 0 (which can happen in NetHack edge cases), this raises a `ZeroDivisionError`.
- **Fix**:
```python
def normalize(self, val: int, raw_intrinsics: dict[str, Any]) -> float:
    base_val = raw_intrinsics[self.base]
    if base_val == 0:
        return 0.0
    return float(val / base_val)
```

---

## Low Priority Issues (Nice to Have)

### 1. Commented-out code throughout the codebase
- `component.py:6` (`import traceback`), `component.py:59` (`traceback.print_stack()`), `component.py:87` (`component_set.remove`)
- `predict.py` has 10+ commented-out print statements
- `gymnasium.py:297-304` has commented-out proprioceptive printing
- `graphdb.py` has commented-out logger statements
- **Recommendation**: Remove dead code; it can be recovered from git history if needed.

### 2. `expmod.py:26` has a debug `print("bases:", cls.__bases__)` left in
- This will print to stdout every time a subclass with a missing `name` attribute is registered.

### 3. `roc/utils.py` contains only a single 2-line function
- Either inline `_timestamp_str()` where it is used (only 2 call sites) or make this a more meaningful utility module.

### 4. `ObjectCache` in `object.py:186` is defined but never used
- The class `ObjectCache(LRUCache[...])` is defined at the bottom of the file but never referenced anywhere in the codebase.

### 5. `pyproject.toml` has both `uv.lock` and `poetry.lock`
- The project uses uv, but a legacy `poetry.lock` still exists. Consider removing it to avoid confusion.

### 6. `roc/__init__.py` global `ng: NethackGym | None` leaks module state
- The `ng` variable is module-level mutable state. If `init()` is called twice, the old `NethackGym` is not cleaned up.

### 7. Type annotation `slice[Any, Any, Any]` is non-standard
- **Files**: `roc/graphdb.py:884,886,1844,1846`
- `slice[Any, Any, Any]` is not a valid generic form of `slice` in standard Python typing. This should be just `slice`.

### 8. `NodeCreationFailed` error message references `id` builtin instead of `n.id`
- **File**: `roc/graphdb.py:1605`
- `f"Couldn't create node ID: {id}"` references the Python builtin `id()` function, not the node ID variable. Should be `f"Couldn't create node ID: {n.id}"`.

---

## Positive Findings

- **Well-designed component registration system**: The `__init_subclass__` pattern for auto-registration is elegant and avoids manual registration boilerplate.
- **Typed EventBus**: Using generics (`EventBus[T]`) to type-check event data at the bus level is a strong pattern.
- **Comprehensive test coverage**: The test-to-code ratio is ~0.86:1, with thorough tests for graphdb, attention, and feature extractors.
- **Good use of Pydantic for data validation**: Config, Node, and Edge all use Pydantic for schema validation.
- **Clean separation of feature extractors**: Each extractor is its own module with a consistent interface.
- **Schema validation**: The graph database schema auto-validates edge connections against node types.
- **Observability integration**: OpenTelemetry tracing, metrics, and profiling are well-integrated throughout the pipeline.
- **Good docstring coverage**: Most public methods have Google-style docstrings with clear argument descriptions.
- **Debug visualization tools**: The DOT/Mermaid schema generation and `DebugGrid` terminal visualization are valuable development aids.

---

## Recommendations

1. **Fix the NaN comparison bug** (Critical #1) -- this is a silent data corruption issue that is trivial to fix.

2. **Address race conditions in event handlers** (Critical #2) -- add per-component serialization to the event dispatch pipeline. This is the highest-risk architectural issue.

3. **Consolidate global mutable state** (High #1) -- this is the single biggest barrier to test reliability and code maintainability. Start by creating a `RuntimeContext` that holds registries, caches, and singletons.

4. **Vectorize feature extractors** (Medium #1) -- use numpy/scipy operations instead of Python loops for the critical perception path. The `Single` extractor's neighbor check could be a single `scipy.ndimage` convolution.

5. **Explicit flush instead of __del__ saves** (Medium #4) -- remove database writes from destructors. They are unpredictable and cause cascading issues during shutdown.

6. **Fix mutable default arguments** (Medium #5) -- replace all `= dict()`, `= set()`, `= list()` parameter defaults with `= None` and construct inside the function body. This is a Python footgun that has already manifested (the `params_to_str` mutation).

7. **Remove CrossModalAttention stub** (Medium #3) -- an empty auto-loading component adds noise to the system.

8. **Clean up commented-out code** (Low #1) -- this is a significant amount of dead code that makes the codebase harder to read.
