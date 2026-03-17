# Playback State Machine Redesign

## Problem Statement

The Panel dashboard's playback logic is an **implicit state machine** -- 4 distinct
operating modes encoded across 5 mutable variables (`_user_paused`, `_live_mode`,
`_step_widget.direction`, `_step_widget.interval`, and the positional relationship
`value >= end`). No single place in the code names these modes. Every handler
reconstructs "what mode am I in?" from a different subset of flags.

This has caused repeated debugging sessions and makes the code hard to extend.
Adding a new feature (e.g., "follow a specific game" or "auto-pause on anomaly")
requires reasoning about all flag combinations, not just the feature itself.

### Current State Variables (panel_debug.py)

| Variable | Type | Location | Purpose |
|----------|------|----------|---------|
| `_live_mode` | `bool` | L390 | Whether a StepBuffer was provided (immutable) |
| `_user_paused` | `bool` | L391 | User explicitly paused |
| `_step_widget.direction` | `int` | Widget | 0=paused, 1=playing |
| `_step_widget.interval` | `int` | Widget | ms between ticks, or `2**31-1` = disabled |
| `_step_widget.value >= end` | derived | Widget | Whether at the live edge |
| `_updating_game` | `bool` | L387 | Reentrance guard |
| `_last_seen_step` | `int` | L389 | Deduplication for push notifications |

### Current Implicit Modes

These 4 modes exist but are never declared:

1. **HISTORICAL** -- No live data. Timer drives auto-advance. User plays/pauses freely.
2. **LIVE_FOLLOWING** -- At the live edge. Push-driven. Timer disabled via `interval=2**31-1` hack.
   `direction=1` (so pause button works) but the widget never actually ticks.
3. **LIVE_PAUSED** -- User hit pause during live. Timer restored for manual scrubbing.
4. **LIVE_CATCHUP** -- User hit play while behind the live edge. Timer drives auto-advance
   toward live edge. Push notifications that see `_is_following()` snap to LIVE_FOLLOWING.

### Problems with Implicit Encoding

1. **Distributed state checks.** The same mode is tested differently in different handlers:
   - `_on_new_data`: `self._is_following() and not self._user_paused`
   - `_on_speed_change`: `self._step_buffer is not None and not self._user_paused and self._is_following()`
   - `_dispatch_key("End")`: `self._step_buffer is not None` + 4 lines of flag-setting
2. **Distributed entry actions.** Timer, direction, and `_user_paused` are set in 4+ separate
   places that must all agree.
3. **Reentrance guard.** `_updating_game` exists solely because `_on_game_change` can't tell
   whether the selector change came from a user action or a live-mode auto-advance.
4. **`interval=2**31-1` hack.** Overloads the timer widget to mean "disabled" -- semantically
   invisible to anyone reading the code.

## Proposed Design

### Library Choice: `python-statemachine` v3

[python-statemachine](https://github.com/fgmacedo/python-statemachine) is the best fit:

- **Declarative, Pythonic API** -- states and transitions defined as class attributes
- **Guards** via `cond=` / `unless=` parameters with named methods
- **Entry/exit actions** via `on_enter_<state>` / `on_exit_<state>` naming convention
- **`allowed_events`** -- tells you which transitions are valid from the current state
  (directly useful for enabling/disabling UI buttons)
- **Diagram generation** -- exports state machine as PNG/SVG via graphviz
- **Active maintenance** -- last release Feb 2026, Python 3.13 support
- **Listener API** -- external objects observe state changes without modifying the machine

Install: `uv add python-statemachine`

### State Machine Definition

```python
# roc/reporting/playback_machine.py
"""Explicit state machine for dashboard playback control."""

from __future__ import annotations

from statemachine import StateMachine, State


class PlaybackMachine(StateMachine):
    """Manages the finite playback mode for the ROC debug dashboard.

    Finite states control *how* the dashboard advances through steps.
    Extended state (current step, speed, game number) lives on the
    dashboard itself -- the machine only governs mode transitions.

    States:
        historical: No live data. Timer-driven playback. This is the
            initial state and the only state used when no StepBuffer
            is provided.
        live_following: At the live edge, receiving push updates.
            Timer is disabled; advancement is push-driven.
        live_paused: User paused during a live session. Timer is
            restored so the user can scrub through history.
        live_catchup: User resumed playback while behind the live
            edge. Timer drives auto-advance; a push notification
            at the edge transitions to live_following.

    The dashboard interacts with this machine in two ways:
        1. Calls sm.send("event_name") when a user action or system
           event occurs.
        2. Implements on_enter_* callbacks (via listener) to apply
           side effects to widgets.
    """

    # -- States --
    historical = State(initial=True)
    live_following = State()
    live_paused = State()
    live_catchup = State()

    # -- Transitions --

    # Initial setup: when a StepBuffer is provided, go live immediately.
    go_live = historical.to(live_following)

    # User actions
    pause = (
        live_following.to(live_paused)
        | live_catchup.to(live_paused)
    )
    resume = live_paused.to(live_catchup)
    jump_to_end = (
        live_paused.to(live_following)
        | live_catchup.to(live_following)
    )

    # System events
    push_arrived = (
        live_following.to.itself(internal=True)  # snap to latest, no re-enter
        | live_catchup.to(live_following, cond="at_live_edge")
        | live_catchup.to.itself(internal=True)  # update end, stay catching up
        | live_paused.to.itself(internal=True)    # update end, show badge
    )

    # Historical mode: play/pause toggle (no live data)
    play = historical.to.itself(internal=True)
    # (Direction toggling is a widget concern, not a mode change)

    # -- Guards --

    def at_live_edge(self) -> bool:
        """True when the step widget is at or past the slider end."""
        db = self._dashboard
        return bool(db._step_widget.value >= db._step_widget.end)

    # -- Setup --

    def __init__(self, dashboard: object, **kwargs: object) -> None:
        self._dashboard = dashboard
        super().__init__(**kwargs)
```

### Dashboard Listener (Side Effects)

The state machine is pure logic -- it decides *which mode to enter*. A listener
on the dashboard applies the *side effects* (widget mutations). This keeps the
machine testable in isolation.

```python
# In panel_debug.py or a separate file

_TIMER_DISABLED = 2**31 - 1


class PlaybackListener:
    """Applies widget side effects when the playback machine changes state.

    Registered as a listener on the PlaybackMachine. Callback methods use
    the naming convention on_enter_<state_id> which python-statemachine
    auto-discovers.
    """

    def __init__(self, dashboard: PanelDashboard) -> None:
        self._db = dashboard

    def _get_interval(self) -> int:
        """Current speed as a timer interval in milliseconds."""
        label = self._db._speed_selector.value
        return self._db._speed_to_interval.get(
            label, self._db._speed_to_interval[self._db._DEFAULT_SPEED]
        )

    def on_enter_live_following(self) -> None:
        w = self._db._step_widget
        w.interval = _TIMER_DISABLED
        w.direction = 1
        self._db._live_badge.visible = True
        self._db._new_data_badge.visible = False

    def on_exit_live_following(self) -> None:
        self._db._live_badge.visible = False

    def on_enter_live_paused(self) -> None:
        w = self._db._step_widget
        w.direction = 0
        w.interval = self._get_interval()
        # Badge visibility is set by _apply_step_data based on position

    def on_enter_live_catchup(self) -> None:
        w = self._db._step_widget
        w.direction = 1
        w.interval = self._get_interval()

    def on_enter_historical(self) -> None:
        w = self._db._step_widget
        w.interval = self._get_interval()
```

### Integration with PanelDashboard

The dashboard creates the machine and sends events. Flag checks are replaced with
mode checks.

```python
class PanelDashboard(Viewer):
    # ... (SPEEDS, params, etc. unchanged) ...

    def __init__(self, store, data_dir=None, step_buffer=None, **params):
        super().__init__(**params)
        # ... (store, widgets, etc. unchanged) ...

        # State machine replaces: _user_paused, _updating_game, interval hack
        self._listener = PlaybackListener(self)
        self._playback = PlaybackMachine(
            dashboard=self, listeners=[self._listener]
        )
        if step_buffer is not None:
            self._playback.send("go_live")

        # ... (wire widgets, initial render) ...

    # -- Keyboard / UI handlers become simple --

    def _dispatch_key(self, key: str) -> None:
        if key == "ArrowRight":
            self._step_widget.value = min(
                self._step_widget.value + 1, self._step_widget.end
            )
        elif key == "ArrowLeft":
            self._step_widget.value = max(
                self._step_widget.value - 1, self._step_widget.start
            )
        elif key == "Home":
            self._step_widget.value = self._step_widget.start
            if self._playback.live_following.is_active or self._playback.live_catchup.is_active:
                self._playback.send("pause")
        elif key == "End":
            self._step_widget.value = self._step_widget.end
            if self._playback.live_paused.is_active or self._playback.live_catchup.is_active:
                self._playback.send("jump_to_end")
        elif key == " ":
            self._toggle_play()
        elif key in ("+", "="):
            self._increase_speed()
        elif key == "-":
            self._decrease_speed()
        elif key == "g":
            self._cycle_game()
        elif key == "b":
            self._toggle_bookmark()
        elif key in ("n", "]"):
            self._jump_next_bookmark()
        elif key in ("p", "["):
            self._jump_prev_bookmark()
        elif key in ("?", "h"):
            self._toggle_help()

    def _toggle_play(self) -> None:
        if self._playback.historical.is_active:
            # Historical: just toggle the widget direction
            if self._step_widget.direction == 0:
                self._step_widget.direction = 1
            else:
                self._step_widget.direction = 0
        elif self._playback.live_following.is_active:
            self._playback.send("pause")
        elif self._playback.live_paused.is_active:
            self._playback.send("resume")
        elif self._playback.live_catchup.is_active:
            self._playback.send("pause")

    # -- Live mode push handler --

    def _on_new_data(self) -> None:
        if self._step_buffer is None:
            return
        latest = self._step_buffer.get_latest()
        if latest is None or latest.step <= self._last_seen_step:
            return
        self._last_seen_step = latest.step

        # Always update slider end and game options
        if latest.step > self._step_widget.end:
            self._step_widget.end = latest.step

        for g in self._step_buffer.game_numbers:
            game_str = str(g)
            if game_str not in (self._game_selector.options or []):
                opts = list(self._game_selector.options or [])
                opts.append(game_str)
                self._game_selector.options = opts

        # Let the state machine decide what to do
        self._playback.send("push_arrived")

        if self._playback.live_following.is_active:
            # Snap to latest
            self._game_selector.value = str(latest.game_number)
            old_value = self._step_widget.value
            self._step_widget.value = latest.step
            if old_value == latest.step:
                self._on_step_change(latest.step)
        else:
            # Not following -- show badge so user knows new data exists
            self._new_data_badge.visible = True

    # -- Speed change --

    def _on_speed_change(self, event) -> None:
        if self._playback.live_following.is_active:
            return  # push-driven, timer is irrelevant
        label = getattr(event, "new", self._DEFAULT_SPEED)
        self._step_widget.interval = self._speed_to_interval.get(
            label, self._speed_to_interval[self._DEFAULT_SPEED]
        )

    # -- Game change --

    def _on_game_change(self, game_number: int) -> None:
        # No reentrance guard needed -- live_following auto-switches
        # game via _on_new_data, which doesn't trigger _on_game_change
        if self._playback.live_following.is_active:
            return
        try:
            min_step, max_step = self._store.step_range(game_number=game_number)
            if min_step > 0:
                if min_step < self._step_widget.start:
                    self._step_widget.start = min_step
                if max_step > self._step_widget.end:
                    self._step_widget.end = max_step
                self._step_widget.value = min_step
        except Exception:
            pass
```

### What Gets Deleted

The state machine replaces these variables and the scattered logic around them:

| Deleted | Replacement |
|---------|-------------|
| `_user_paused` (bool) | `live_paused.is_active` |
| `_updating_game` (bool) | `live_following.is_active` check in `_on_game_change` |
| `_live_mode` (bool) | `not historical.is_active` |
| `interval = 2**31-1` in 4 places | `on_enter_live_following` sets it once |
| `direction` manipulation in 3 places | `on_enter_*` callbacks set it once per mode |
| `_is_following()` derived check | `live_following.is_active` (explicit, not derived) |
| `_handle_direction_widget` dual-write of `_user_paused` | Removed entirely -- direction is set by entry actions, not watched for mode changes |

### What Stays the Same

- All widget creation (`__init__` widget block) -- unchanged
- `_apply_step_data()` -- still the central data render method (separate refactoring)
- `BookmarkManager` -- unchanged
- `KeyboardShortcuts` React component -- unchanged
- `RunStore` / `StepBuffer` integration -- unchanged
- `__panel__` layout -- unchanged

## State Transition Diagram

```
              HISTORICAL MODE (no StepBuffer)
         +--------------------------------------+
         |            HISTORICAL                |
         |  Timer-driven play/pause.            |
         |  play/pause toggle direction only.   |
         +--------------------------------------+
                        |
                        | go_live (StepBuffer provided)
                        v
         +--------------------------------------+
         |          LIVE_FOLLOWING               |<-----+-----+
         |  Push-driven. Timer disabled.        |      |     |
         |  Auto-snaps to latest step.          |      |     |
         +--------------------------------------+      |     |
              |                                        |     |
              | pause (Space / Pause btn)              |     |
              v                                        |     |
         +--------------------------------------+      |     |
         |           LIVE_PAUSED                |      |     |
         |  User paused. Timer restored for     |      |     |
         |  manual scrubbing. Badge shows       |      |     |
         |  when new data arrives.              |      |     |
         +--------------------------------------+      |     |
              |                |                       |     |
              | resume         | jump_to_end           |     |
              | (Space)        | (End key)             |     |
              v                +-------->--------------+     |
         +--------------------------------------+            |
         |          LIVE_CATCHUP                |            |
         |  Timer-driven toward live edge.      |            |
         |  push_arrived + at_live_edge         |            |
         |  transitions to LIVE_FOLLOWING.      |------------+
         +--------------------------------------+
              |                |
              | pause          | jump_to_end (End key)
              | (Space)        +-------->---------------+
              v                                         |
         (back to LIVE_PAUSED)              (back to LIVE_FOLLOWING)
```

Push notifications (`push_arrived`) are handled in every live state:
- **LIVE_FOLLOWING**: internal transition -- snap slider to latest step
- **LIVE_CATCHUP**: if `at_live_edge`, transition to LIVE_FOLLOWING; otherwise internal -- update slider end
- **LIVE_PAUSED**: internal transition -- update slider end, show "new data" badge

## Testing Strategy

### Unit Tests for the State Machine (Isolated)

The `PlaybackMachine` can be tested without any Panel widgets by providing
a mock dashboard object:

```python
class MockDashboard:
    """Minimal mock for PlaybackMachine tests."""
    class _step_widget:
        value = 1
        end = 100
        start = 1
        interval = 200
        direction = 0

    class _speed_selector:
        value = "5x"

    _speed_to_interval = {"5x": 200}
    _DEFAULT_SPEED = "5x"
    _live_badge = type("Badge", (), {"visible": False})()
    _new_data_badge = type("Badge", (), {"visible": False})()


def test_initial_state_is_historical():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    assert sm.historical.is_active


def test_go_live_transitions_to_following():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    sm.send("go_live")
    assert sm.live_following.is_active
    assert db._step_widget.interval == _TIMER_DISABLED
    assert db._step_widget.direction == 1


def test_pause_from_following():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    sm.send("go_live")
    sm.send("pause")
    assert sm.live_paused.is_active
    assert db._step_widget.direction == 0
    assert db._step_widget.interval == 200  # restored to speed


def test_resume_enters_catchup():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    sm.send("go_live")
    sm.send("pause")
    sm.send("resume")
    assert sm.live_catchup.is_active
    assert db._step_widget.direction == 1
    assert db._step_widget.interval == 200  # timer-driven


def test_push_at_edge_transitions_catchup_to_following():
    db = MockDashboard()
    db._step_widget.value = 100  # at end
    db._step_widget.end = 100
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    sm.send("go_live")
    sm.send("pause")
    sm.send("resume")  # now in catchup
    sm.send("push_arrived")  # at edge -> following
    assert sm.live_following.is_active


def test_push_behind_edge_stays_in_catchup():
    db = MockDashboard()
    db._step_widget.value = 50  # behind end
    db._step_widget.end = 100
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    sm.send("go_live")
    sm.send("pause")
    sm.send("resume")  # now in catchup
    sm.send("push_arrived")  # not at edge -> stay
    assert sm.live_catchup.is_active


def test_jump_to_end_from_paused():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    sm.send("go_live")
    sm.send("pause")
    sm.send("jump_to_end")
    assert sm.live_following.is_active


def test_invalid_transition_raises():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    # Can't pause from historical
    with pytest.raises(Exception):
        sm.send("pause")


def test_allowed_events():
    db = MockDashboard()
    sm = PlaybackMachine(dashboard=db, listeners=[PlaybackListener(db)])
    assert "go_live" in sm.allowed_events
    assert "pause" not in sm.allowed_events
    sm.send("go_live")
    assert "pause" in sm.allowed_events
    assert "go_live" not in sm.allowed_events
```

### Integration Tests

Existing `test_panel_debug.py` tests that exercise playback behavior (pause button,
End key, live mode push) should continue to pass. The test interface changes slightly:

```python
# Before: assert dashboard._user_paused is True
# After:  assert dashboard._playback.live_paused.is_active
```

### Diagram Generation

Add a dev script or test that generates the state diagram:

```python
from statemachine.contrib.diagram import DotGraphMachine
from roc.reporting.playback_machine import PlaybackMachine

graph = DotGraphMachine(PlaybackMachine)
graph().write_png("design/playback-states.png")
```

## Implementation Plan

### Phase 1: Add state machine (no behavior change)

1. `uv add python-statemachine`
2. Create `roc/reporting/playback_machine.py` with `PlaybackMachine` and `PlaybackListener`
3. Create `tests/unit/test_playback_machine.py` with isolated tests (above)
4. Verify: `make test` passes, new tests pass

### Phase 2: Integrate with dashboard

1. Add `_playback` and `_listener` to `PanelDashboard.__init__`
2. Replace `_user_paused` checks with `_playback.<state>.is_active`
3. Replace `_updating_game` check with `_playback.live_following.is_active`
4. Replace inline `interval`/`direction` setting in `_dispatch_key` with `_playback.send()`
5. Remove `_handle_direction_widget` (direction set by entry actions, not watched)
6. Remove `_user_paused`, `_updating_game`, `_live_mode` fields
7. Define `_TIMER_DISABLED = 2**31 - 1` as a named constant
8. Verify: existing `test_panel_debug.py` tests pass (update assertions where needed)

### Phase 3: Cleanup

1. Update test assertions from flag checks to state checks
2. Generate and commit the state diagram PNG
3. Remove any dead code (old flag-checking helpers)
4. Run `make lint` and `make test`

### Estimated Scope

- New code: ~100 lines (`playback_machine.py` + listener)
- New tests: ~80 lines (isolated state machine tests)
- Modified code: ~60 lines changed in `panel_debug.py` (replacing flag logic with `send()` calls)
- Deleted code: ~30 lines (flag declarations, `_handle_direction_widget`, inline flag-setting)
- Net change: roughly neutral line count, but the logic moves from implicit to explicit

## Alternatives Considered

### Plain Python Enum + `_transition_to()` Method

The simplest approach: a `PlaybackMode` enum and a `_transition_to(new_mode)` method
that applies side effects. No library dependency.

**Pros**: Zero dependencies, easy to understand.
**Cons**: No guard validation, no transition legality checking, no `allowed_events`,
no diagram generation. You have to manually enforce which transitions are valid,
which is error-prone and was the original problem.

### `transitions` (pytransitions)

The most popular Python state machine library (~6,500 stars).

**Pros**: Mature, large community, well-documented.
**Cons**: HSM support is a bolt-on extension, not native. No parallel or history
states. Guards are callables, not declarative expressions. The `Machine(model=obj)`
pattern mutates the model object with trigger methods, which feels invasive.

### `sismic`

Academic-grade statechart interpreter with YAML definitions.

**Pros**: Full UML 2 compliance, Design by Contract, BDD testing.
**Cons**: YAML-based (less Pythonic), LGPL license, smallest community,
heavier runtime (interpreter model).

### No Library -- Keep Current Approach with Better Documentation

Just add comments naming the modes and a docstring with the state diagram.

**Pros**: No new dependency, no refactoring.
**Cons**: Doesn't solve the core problem. Comments drift from code. Every handler
still independently reconstructs the current mode from flags. New features still
require reasoning about all flag combinations.

## Decision

**`python-statemachine` v3** is the recommended choice. It has the cleanest API for
our use case, native statechart support if we need it later, active maintenance,
and the `allowed_events` / diagram generation features directly address our pain points.
The library is lightweight (pure Python, no C extensions) and the declarative class-based
API matches ROC's existing style.
