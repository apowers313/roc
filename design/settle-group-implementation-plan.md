# SettleGroup Implementation Plan

## Overview

Implements the SettleGroup design from `design/settle-group-design.md`. Seven
discrete steps, each independently testable. Steps 1-3 are safe additive
changes. Steps 4-7 rewire the pipeline flow and should land together.

---

## Step 1: Add SettleGroup class to roc/event.py

**Files changed:**
- `roc/event.py`
- `tests/unit/test_event.py`

**Changes to `roc/event.py`:**

Add `import threading` and a new `SettleGroup` class after `EventBus`:

```python
class SettleGroup:
    def __init__(
        self,
        name: str,
        sources: set[str],
        on_settled: Callable[[], None],
        ignored: set[str] | None = None,
        timeout: float = 5.0,
    ) -> None: ...

    def notify(self, source: str) -> None: ...
    def reset(self) -> None: ...
    def add_source(self, source: str) -> None: ...
    def remove_source(self, source: str) -> None: ...
```

`notify(source)` behavior:
1. If source in `_ignored`: no-op.
2. If source not in `_sources`: log warning
   `"{name} saw '{source}' which is not in its settle watch list"`.
3. If source in `_sources`: remove from `_pending`. Start watchdog on first
   notify after reset. If `_pending` is empty, cancel watchdog, call
   `on_settled`, then auto-reset.

Watchdog: `threading.Timer` (daemon) fires after `timeout` seconds, logs
`"{name} has not settled in {timeout}s, still waiting on: {pending}"`.

Thread safety: `threading.Lock` guards all internal state. The `on_settled`
callback is invoked under the lock -- this is safe because bus sends dispatch
to the ThreadPoolScheduler asynchronously (Subject.on_next returns
immediately).

`remove_source`: if removing a source causes `_pending` to become empty (and
`_sources` is non-empty), fires the callback.

**Tests to add (`tests/unit/test_event.py`):**

New `TestSettleGroup` class:
- `test_fires_when_all_sources_notify`
- `test_does_not_fire_until_all_sources`
- `test_auto_resets_after_fire` (notify all twice, callback fires twice)
- `test_ignored_source_is_noop`
- `test_unknown_source_logs_warning` (caplog)
- `test_add_source`
- `test_remove_source_triggers_settle`
- `test_reset_clears_state`
- `test_thread_safety` (N threads each notify their own source)
- `test_watchdog_fires_on_timeout` (timeout=0.1, check warning logged)
- `test_watchdog_cancelled_on_settle`

**Risks:** Low -- purely additive.

---

## Step 2: Fix tick counter race condition in roc/sequencer.py

**Files changed:**
- `roc/sequencer.py`
- `tests/unit/test_sequencer.py`

**Changes:**

Replace module-level mutable `tick` with `itertools.count`:

```python
import itertools

_tick_counter = itertools.count(1)

def get_next_tick() -> int:
    return next(_tick_counter)
```

Remove `tick = 0` and the `global tick` statement. `itertools.count.__next__`
is atomic in CPython (C implementation, single bytecode).

Also keep the module-level `tick` accessible for reading the current tick
(used in `object.py:806` as `from .sequencer import tick as current_tick`).
Replace with:

```python
def get_current_tick() -> int:
    """Returns the most recently assigned tick number without incrementing."""
    return next(iter([]))  # Can't peek itertools.count
```

Actually, the simpler approach: keep a separate read-only variable updated by
`get_next_tick`:

```python
_tick_counter = itertools.count(1)
current_tick = 0

def get_next_tick() -> int:
    global current_tick
    current_tick = next(_tick_counter)
    return current_tick
```

This preserves the `from .sequencer import tick as current_tick` import
pattern (rename `tick` to `current_tick` at module level, or keep as `tick`
for backward compatibility).

**Update test fixtures** to swap `_tick_counter` with a fresh `itertools.count(1)`
and reset `tick = 0`.

**Risks:** Low. Grep for `sequencer.tick` to find all readers.

---

## Step 3: Refactor VisionAttention to use SettleGroup

**Files changed:**
- `roc/attention.py`
- `tests/unit/test_attention.py`

**Changes to `roc/attention.py`:**

Import `SettleGroup` from `roc.event`.

In `VisionAttention.__init__`:
- Remove `self.settled: set[str] = set()`.
- Add SettleGroup:
  ```python
  self._settle = SettleGroup(
      name="VisionAttention",
      sources=set(FeatureExtractor.list()),
      on_settled=self._all_settled,
      ignored={"vision-data"},
  )
  ```

Rewrite `do_attention`:
- `VisionData`: set saliency grid, call `self._settle.notify("vision-data")`.
- `Settled`: call `self._settle.notify(str(e.src_id))`.
- `VisualFeature`: accumulate into saliency map (unchanged).

New `_all_settled` method:
- Send `VisionAttentionData` on attention bus.
- Reset saliency map.
- Note: existing code calls `get_focus()` twice (lines 404 and 408). Fix by
  calling once.

**Behavioral equivalence:** The SettleGroup fires when all sources in
`FeatureExtractor.list()` have notified, which is the same condition as the
current `unsettled = set(FeatureExtractor.list()) - self.settled` check. The
SettleGroup version adds thread safety (lock) and a watchdog.

**Tests:** Current test file only tests SaliencyMap. No existing tests break.
Optionally add a test verifying `_settle` is a SettleGroup with correct
sources.

**Risks:** Medium. Verify that `FeatureExtractor.list()` returns the correct
IDs at VisionAttention construction time (feature extractors must be loaded
first). Check `Component.init()` load order.

---

## Step 4: Rewire Sequencer to settle on object + intrinsic

**Files changed:**
- `roc/sequencer.py`
- `tests/unit/test_sequencer.py`

**Changes to `roc/sequencer.py`:**

Import `SettleGroup` from `roc.event`.

In `Sequencer.__init__`:
```python
self._settle = SettleGroup(
    name="Sequencer",
    sources={"object", "intrinsic"},
    on_settled=self._emit_frame,
    ignored={"action"},
)
```

`handle_object_resolution_event`: keep FrameAttribute.connect, add
`self._settle.notify("object")`.

`handle_intrinsic_event`: keep loop, add `self._settle.notify("intrinsic")`.

`handle_action_event`: call `self._settle.notify("action")` (ignored, no-op).
On TakeAction, attach to `self.last_frame` retroactively (not
`self.current_frame`). Connect `TakeAction -> self.current_frame` for the
forward link.

New `_emit_frame`:
```python
def _emit_frame(self):
    self.last_frame = self.current_frame
    self.current_frame = Frame()
    NextFrame.connect(self.last_frame, self.current_frame)
    self.sequencer_conn.send(self.last_frame)
```

**Key behavioral change:** Frames emit when object + intrinsic settle,
BEFORE TakeAction arrives. TakeAction is attached retroactively.

**Tests:** Existing unit tests only cover Frame/edges, not the Sequencer
component. No breakage expected. Add tests for the new settling behavior.

**Risks:** High -- pipeline flow change. Steps 5-7 must also be implemented
for the pipeline to work end-to-end.

---

## Step 5: Transformer emits on first step

**Files changed:**
- `roc/transformer.py`
- `tests/unit/test_transformer.py`

**Changes:**

Currently `do_transformer` returns early with no emission when there is no
previous frame. Change to emit an empty TransformResult:

```python
if len(previous_frames) < 1:
    ret = Transform()
    Change.connect(ret, current_frame)
    self.transformer_conn.send(TransformResult(transform=ret))
    return
```

This ensures Predict always receives a TransformResult, even on the first
step. Predict will find no candidates and emit `NoPrediction`.

**Tests:** Add `test_first_step_emits_empty_transform` -- mock Frame with no
NextFrame edges, verify TransformResult is emitted.

**Risks:** Low -- adds behavior to a previously no-op path. Verify Predict
handles an empty Transform correctly (no Change edges with type "Change" that
have a src node -> candidates will be empty -> NoPrediction).

---

## Step 6: Action waits for PredictResult

**Files changed:**
- `roc/action.py`
- `tests/unit/test_action.py`

**Changes to `roc/action.py`:**

Action listens on Predict bus and waits for a prediction before emitting
TakeAction.

```python
class Action(Component):
    def __init__(self):
        super().__init__()
        self.action_bus_conn = self.connect_bus(self.bus)
        self.action_bus_conn.listen(self.handle_action_bus)

        from roc.predict import Predict  # deferred to avoid circular import
        self.predict_conn = self.connect_bus(Predict.bus)
        self.predict_conn.listen(self.handle_predict)

        self._predict_received = threading.Event()
        self._predict_result: PredictData | None = None
        self._predict_timeout = 2.0

    def handle_action_bus(self, e):
        if isinstance(e.data, ActionRequest):
            got_predict = self._predict_received.wait(timeout=self._predict_timeout)
            if not got_predict:
                logger.warning("Action: no prediction within timeout, using default")
            action = DefaultActionExpMod.get(default="pass").get_action()
            self.action_bus_conn.send(TakeAction(action=action))
            self._predict_received.clear()
            self._predict_result = None

    def handle_predict(self, e):
        self._predict_result = e.data
        self._predict_received.set()
```

The `handle_action_bus` handler blocks on the pool thread until a prediction
arrives or times out. This is fine -- the pool has `cpu_count * 2` threads.

The 2s timeout handles edge cases where Predict fails to emit. On timeout,
Action proceeds with the default action (same as current behavior).

**Circular import:** `action.py -> predict.py -> sequencer.py -> action.py`.
Fix with deferred import: `from roc.predict import Predict` inside
`__init__`, not at module level.

**Tests:**
- `test_action_waits_for_predict_and_request`
- `test_action_timeout_on_no_predict`
- `test_action_does_not_emit_without_request`

**Risks:** High. Threading behavior must be tested carefully. The 2s timeout
must be long enough for normal pipeline execution but short enough to avoid
hanging the game loop.

---

## Step 7: Gymnasium get_action() blocks until TakeAction

**Files changed:**
- `roc/gymnasium.py`

**Changes:**

Replace cache-based polling with `threading.Event`:

```python
# In Gym.__init__:
self._take_action_event = threading.Event()
self._last_action: TakeAction | None = None
self.action_bus_conn.listen(self._on_take_action)

def _on_take_action(self, e):
    if isinstance(e.data, TakeAction):
        self._last_action = e.data
        self._take_action_event.set()

def get_action(self):
    self._take_action_event.clear()
    self.action_bus_conn.send(ActionRequest())
    if not self._take_action_event.wait(timeout=10.0):
        raise TimeoutError("get_action: TakeAction not received within 10s")
    assert self._last_action is not None
    action = self._last_action.action
    self._last_action = None
    return action
```

The `threading.Event.set()` provides a happens-before guarantee, so
`_last_action` written before `set()` is visible after `wait()` returns.

Self-sends (ActionRequest from Gym) are filtered out by the default
`event_filter` (`e.src_id != self.id`), so `_on_take_action` only receives
TakeAction from Action.

**Risks:** High -- main loop timing. The 10s timeout is generous; a hung
pipeline raises `TimeoutError` rather than silently continuing with stale
data.

---

## Landing Order

```
Step 1 (SettleGroup)         -- can land independently
Step 2 (tick counter fix)    -- can land independently
Step 3 (VisionAttention)     -- depends on Step 1
Steps 4-7 (pipeline rewire)  -- must land together
```

Steps 4-7 form an interconnected change:
- Step 4 (Sequencer emits early) needs Step 5 (Transformer emits on first
  step) or Predict never fires on step 1.
- Step 4 needs Step 6 (Action waits for predict) or Action fires before
  pipeline completes.
- Step 6 needs Step 7 (Gym blocks properly) or get_action() returns stale
  data.

## Verification

After all steps, run `uv run play` for at least 10 steps and verify:
1. No SettleGroup watchdog warnings in logs
2. Frames are emitted and Transformer receives them
3. Predict runs and Action emits TakeAction
4. Game loop proceeds normally
5. OTel logging lands at correct step numbers (the original bug)
6. Dashboard shows correct resolution data per step
