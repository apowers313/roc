# Code Review Report - 2026-03-16

## Executive Summary

- Files reviewed: 17 production files, 15 test files, 2 config files
- Critical issues: 2
- High priority issues: 6
- Medium priority issues: 10
- Low priority issues: 8

Focus area: the Panel dashboard UI and its supporting infrastructure (reporting module), covering the last ~15 commits. The dashboard works but has accumulated architectural debt from iterative development -- duplicated logic, a monolithic update method, and fragile state management patterns.

---

## Critical Issues (Fix Immediately)

### 1. State name collision: two states share "curr-saliency" identifier

- **Files**: `roc/reporting/state.py:357`, `roc/reporting/state.py:379`
- **Description**: `CurrentSaliencyMapState` and `CurrentAttentionState` both register with the identifier `"curr-saliency"`. `State.__init__` stores instances by name in the global `StateList`, so whichever is instantiated second will shadow the first. Any code that looks up state by name will get the wrong one.
- **Example**: `roc/reporting/state.py:375-379`
```python
class CurrentAttentionState(State[VisionAttentionData]):
    """Tracks the most recent attention data with focus points."""

    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")
```
- **Fix**:
```python
class CurrentAttentionState(State[VisionAttentionData]):
    """Tracks the most recent attention data with focus points."""

    def __init__(self) -> None:
        super().__init__("curr-attention", display_name="Current Attention")
```

### 2. `dashboard_server._started` flag never reset on stop

- **Files**: `roc/reporting/dashboard_server.py:13`, `roc/reporting/dashboard_server.py:110-116`
- **Description**: `_started` is set to `True` after `pn.serve()` but `stop_dashboard()` never sets it back to `False`. If the game loop calls `stop_dashboard()` then `start_dashboard()` again (e.g. multiple runs in one process, or test suites), the dashboard silently refuses to start. The step buffer is cleared but a new one is never created.
- **Example**: `roc/reporting/dashboard_server.py:110-116`
```python
def stop_dashboard() -> None:
    """Stop the in-process dashboard and clean up the step buffer."""
    from roc.reporting.step_buffer import clear_step_buffer

    clear_step_buffer()
    _stop_event.set()
    logger.debug("Panel dashboard stopped.")
```
- **Fix**:
```python
def stop_dashboard() -> None:
    """Stop the in-process dashboard and clean up the step buffer."""
    global _started
    from roc.reporting.step_buffer import clear_step_buffer

    clear_step_buffer()
    _stop_event.set()
    _started = False
    logger.debug("Panel dashboard stopped.")
```

---

## High Priority Issues (Fix Soon)

### 1. `salency` typo propagated across state.py and gymnasium.py

- **Files**: `roc/reporting/state.py:91,169,170,280,439`, `roc/gymnasium.py:143`
- **Description**: The `StateList` field is named `salency` (missing "i") and all consumers use this misspelling. This is a consistency bug that will bite anyone who writes `states.saliency` (the correct spelling). It should be renamed to `saliency` everywhere.
- **Fix**: Rename `salency` to `saliency` in `StateList` and all references. Use ruff/mypy to catch any misses.

### 2. `GymComponent` exported in `roc/__init__.py` but never defined

- **Files**: `roc/__init__.py:29`
- **Description**: `__all__` includes `"GymComponent"` but no such class exists in the package. Any consumer doing `from roc import GymComponent` gets an `ImportError`.
- **Fix**: Remove `"GymComponent"` from `__all__`, or replace with the actual class name (e.g. `"NethackGym"`).

### 3. Monolithic `_apply_step_data()` method (97 lines, 6 concerns)

- **Files**: `roc/reporting/panel_debug.py:679-775`
- **Description**: This single method updates indicators, grid viewers, text panes, data tables, charts, and log tables. It mixes metric arithmetic (HP color bands), widget mutations, and data transformations. Difficult to test, reason about, or modify without risk of side effects.
- **Recommended refactoring**:
  - `_update_status_indicators(metrics)` -- HP, score, depth, gold, energy, hunger
  - `_update_data_tables(data)` -- metrics, graph, features, attenuation, resolution
  - `_update_text_panes(data)` -- object info, focus points
  - `_update_event_chart(data)` -- already extracted, just needs the guard logic moved
  - `_update_log_table(data)` -- log filtering

### 4. Duplicated info-list formatting logic

- **Files**: `roc/reporting/panel_debug.py:741-750` and `roc/reporting/panel_debug.py:755-766`
- **Description**: Object info and focus points use identical iteration patterns -- skip "step"/"game_number", format key-value pairs, join with newlines. This is copy-pasted.
- **Fix**: Extract to a shared helper:
```python
def _format_info_items(items: list[dict[str, Any]], excluded: set[str] | None = None) -> str:
    excluded = excluded or {"step", "game_number"}
    parts = []
    for item in items:
        if isinstance(item.get("raw"), str):
            parts.append(str(item["raw"]).strip())
        else:
            for k, v in item.items():
                if k not in excluded:
                    parts.append(f"{k}: {v}")
    return "\n".join(parts) if parts else ""
```

### 5. Duplicated bookmark indicator rendering

- **Files**: `roc/reporting/panel_debug.py:695-700` and `roc/reporting/panel_debug.py:965-974`
- **Description**: The info line with bookmark marker `[*]` is constructed identically in `_apply_step_data()` and `_update_bookmark_indicator()`. When the format changes, both must be updated.
- **Fix**: Extract to `_render_info_line(data: StepData) -> str` and call from both sites.

### 6. Duplicated speed-index lookup in `_increase_speed` / `_decrease_speed`

- **Files**: `roc/reporting/panel_debug.py:902-920`
- **Description**: Both methods build the same `options` list and do the same `index()` lookup with `ValueError` guard. Minor but exemplifies the copy-paste pattern.
- **Fix**: Extract `_get_speed_index() -> int | None`.

---

## Medium Priority Issues (Technical Debt)

### Theme: Magic numbers and fragile constants

**1. Timer-disabled sentinel `2**31 - 1` used in 4 places**
- **Files**: `roc/reporting/panel_debug.py:559,659,827,882`
- **Fix**: Define `_TIMER_DISABLED = 2**31 - 1` as a class constant.

**2. Hard-coded chart/table sizing**
- **Files**: `roc/reporting/panel_debug.py:798` (25px per bar), `roc/reporting/components/resolution_inspector.py:112` (25px per row + 30px header, max 150px)
- **Fix**: Define named constants with comments explaining the heuristic.

### Theme: State management complexity

**3. `_updating_game` flag for feedback-loop prevention is fragile**
- **Files**: `roc/reporting/panel_debug.py:650-654,839`
- **Description**: A boolean flag set in `_on_new_data()` and checked in `_on_game_change()` to prevent recursive updates. If any future code path triggers game changes, this guard must be manually replicated. Consider using a more robust pattern (e.g. a context manager or `_suppress_watchers` counter).

**4. `_dispatch_key()` uses a 16-branch if-elif chain**
- **Files**: `roc/reporting/panel_debug.py:868-900`
- **Description**: Hard to extend, hard to test individual bindings. A `dict[str, Callable]` built at init time would be cleaner.

### Theme: Event routing fragility in run_store

**5. 8-way if-elif event name routing**
- **Files**: `roc/reporting/run_store.py:201-235`
- **Description**: Adding a new event type requires careful insertion into this chain. Magic strings like `"roc.attention.features"` are repeated in multiple files. A dispatch dict or constants would reduce breakage risk.

**6. Attenuation field exclusion uses magic strings**
- **Files**: `roc/reporting/run_store.py:218-222`
- **Description**: Fields `"saliency_grid"`, `"focus_points"`, `"history"` are excluded inline. If the schema changes, this filter silently produces wrong data.

### Theme: Resource lifecycle

**7. Step buffer listeners never cleaned up on session close**
- **Files**: `roc/reporting/dashboard_server.py:81`
- **Description**: `step_buffer.add_listener(_on_push)` is called per session but `remove_listener()` is never called when a browser session disconnects. Over time, dead listeners accumulate. The try/except on line 78 prevents crashes but leaks memory.
- **Fix**: Register a Bokeh `on_session_destroyed` callback to remove the listener.

**8. `WandbReporter` class variables are not thread-safe**
- **Files**: `roc/reporting/wandb_reporter.py:39-50`
- **Description**: `_step_buffer`, `_game_table_rows`, `_game_tick` etc. are class-level mutables accessed from game loop and potentially from dashboard threads. No locking. Currently safe if only the game loop calls these methods, but the class API doesn't enforce that.

### Theme: Dead or incomplete code

**9. `BlstatsState` is defined but never instantiated**
- **Files**: `roc/reporting/state.py:427-428`
- **Description**: Empty class body, not in `StateList`. Either implement or remove.

**10. `_TABLES` constant in `RunStore` is unused**
- **Files**: `roc/reporting/run_store.py:46`
- **Description**: Defined but table names are hardcoded in method bodies. Either use it or remove it.

---

## Low Priority Issues (Nice to Have)

1. **`State.get()` raises bare `Exception`** -- `roc/reporting/state.py:56-61`. Use a more specific exception type (e.g. `RuntimeError` or custom `StateNotInitializedError`).

2. **`ParquetExporter.export()` returns FAILURE with no logging** -- `roc/reporting/parquet_exporter.py:115-116`. Silent failures in data export make debugging hard. Add `logger.exception()`.

3. **`_parse_body()` fallback to `{"raw": body}` on JSON parse failure** -- `roc/reporting/run_store.py:279-289`. Masks schema mismatches. Consider logging a warning.

4. **`RocMetrics` instrument caches grow unbounded** -- `roc/reporting/metrics.py:16-17`. `_histograms` and `_counters` dicts never shrink. Fine for the current fixed set of metric names but would be a problem if metric names become dynamic.

5. **`DuckLakeStore` uses string interpolation for table names** -- `roc/reporting/ducklake_store.py:74,122,141`. Table names come from trusted code (`_route()`), but the pattern is fragile. Consider validating table names against an allow-list.

6. **`observability.py` __init__ has duplicated setup paths** -- `roc/reporting/observability.py:125-207`. The OTLP and non-OTLP code paths repeat similar exporter configuration. Could be refactored into a common `_add_exporter()` helper.

7. **`ResolutionInspector` feature list silently truncates to 20** -- `roc/reporting/components/resolution_inspector.py:120-122`. No visual indicator that data was truncated.

8. **`WandbReporter.log_media()` interval check may have off-by-one** -- `roc/reporting/wandb_reporter.py:232`. `(cls._game_tick - 1) % interval != 0` depends on whether `_game_tick` starts at 0 or 1. Currently starts at 0 (incremented in `log_step()`), so `tick=0` passes (logs first frame). Verify this is the intended behavior.

---

## Positive Findings

- **Push-based live mode architecture**: The `StepBuffer` + `call_soon_threadsafe` + `add_next_tick_callback` pattern avoids polling entirely and keeps thread safety clean. Well-designed.

- **Factory-per-session pattern**: `dashboard_server.py` creates a fresh `PanelDashboard` per browser session, avoiding shared Bokeh document ownership issues. This is the correct Panel pattern.

- **In-place widget updates**: Widgets are created once in `__init__` and mutated in place. No flicker during playback, good performance.

- **Defensive data access**: `_apply_step_data()` and `ResolutionInspector` consistently use `.get()` with defaults rather than direct key access. Resilient to missing fields.

- **BookmarkManager**: Clean, focused class (65 lines) with file persistence and simple API. Good separation of concerns.

- **GridViewer**: Compact param-based reactive component. Follows Panel best practices with `pn.viewable.Viewer`.

- **Test coverage**: `test_panel_debug.py` has grown to ~2000 lines covering keyboard shortcuts, live mode, bookmarks, and game cycling. Strong test investment.

---

## Recommendations

1. **Extract `_apply_step_data()` into focused sub-methods** -- This is the highest-impact refactoring. It would reduce the monolith, make each concern independently testable, and eliminate the duplicated info-line and info-list patterns. Start here.

2. **Fix the `"curr-saliency"` state name collision** -- This is a correctness bug that could cause silent data loss. Quick fix, high impact.

3. **Rename `salency` -> `saliency`** -- Propagated typo creates a maintenance trap. Fix globally with a find-and-replace.

4. **Define constants for magic numbers** -- `_TIMER_DISABLED`, `_MAX_TRUNCATION_LEN`, `_BAR_HEIGHT_PX`. Makes intent clear and changes easier.

5. **Add session cleanup for step buffer listeners** -- Memory leak in long-running dashboard sessions. Register a `on_session_destroyed` callback.

6. **Convert `_dispatch_key()` to a dict mapping** -- Makes keybindings data-driven, easier to extend, and trivially testable.

7. **Reset `_started` in `stop_dashboard()`** -- Prevents the dashboard from being silently unreachable after stop/restart cycles.

8. **Remove dead code** -- `BlstatsState`, `_TABLES` constant, `GymComponent` export. Small cleanup, prevents confusion.
