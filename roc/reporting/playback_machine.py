"""Explicit state machine for dashboard playback control.

Replaces the implicit flag-based state management in PanelDashboard with
a declarative state machine using python-statemachine. See
design/playback-state-machine.md for the full design rationale.

The machine manages *which mode* the dashboard is in. A separate
PlaybackListener applies widget side effects (timer interval, direction,
badge visibility) on state entry/exit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from statemachine import State, StateMachine

if TYPE_CHECKING:
    from roc.reporting.panel_debug import PanelDashboard

#: Sentinel value for "timer disabled" -- larger than any real interval.
TIMER_DISABLED = 2**31 - 1


class PlaybackMachine(StateMachine):
    """Finite state machine for dashboard playback modes.

    States:
        historical: No live data. Timer-driven playback.
        live_following: At the live edge, push-driven. Timer disabled.
        live_paused: User paused during live session. Timer restored.
        live_catchup: Resuming toward live edge. Timer-driven.
    """

    # -- States --
    historical = State(initial=True)
    live_following = State()
    live_paused = State()
    live_catchup = State()

    # -- Transitions --

    # Setup: go live when a StepBuffer is provided.
    go_live = historical.to(live_following)

    # User actions
    pause = live_following.to(live_paused) | live_catchup.to(live_paused)
    resume = live_paused.to(live_catchup)
    jump_to_end = (
        live_paused.to(live_following)
        | live_catchup.to(live_following)
        | live_following.to.itself(internal=True)
        | historical.to.itself(internal=True)
    )

    # User navigated away from the live edge (clicked prev, slider, game change).
    # No-op in historical/paused (already not following).
    user_navigate = (
        live_following.to(live_paused)
        | live_catchup.to(live_paused)
        | live_paused.to.itself(internal=True)
        | historical.to.itself(internal=True)
    )

    # System event: new data pushed from game loop.
    # Evaluated in order -- guarded catchup->following first, then fallbacks.
    push_arrived = (
        live_catchup.to(live_following, cond="at_live_edge")
        | live_following.to.itself(internal=True)
        | live_catchup.to.itself(internal=True)
        | live_paused.to.itself(internal=True)
    )

    # Historical play/pause (no mode change, just a widget toggle).
    toggle_play = historical.to.itself(internal=True)

    # -- Guards --

    def at_live_edge(self) -> bool:
        """True when the step slider is at or past the end."""
        db: PanelDashboard = self.dashboard
        return bool(db._step_widget.value >= db._step_widget.end)

    # -- Init --

    def __init__(self, dashboard: PanelDashboard, **kwargs: Any) -> None:
        self.dashboard: PanelDashboard = dashboard
        super().__init__(**kwargs)


class PlaybackListener:
    """Applies widget side effects on playback state transitions.

    Registered as a listener on PlaybackMachine. Method names follow the
    ``on_enter_<state_id>`` / ``on_exit_<state_id>`` convention that
    python-statemachine auto-discovers.
    """

    def __init__(self, dashboard: PanelDashboard) -> None:
        self._db = dashboard

    def _get_interval(self) -> int:
        """Current speed setting as a timer interval in milliseconds."""
        label = self._db._speed_selector.value
        return self._db._speed_to_interval.get(
            label, self._db._speed_to_interval[self._db._DEFAULT_SPEED]
        )

    # -- Entry actions --
    # All direction/interval changes are wrapped in _sm_updating to prevent
    # _handle_direction_widget from re-entering the state machine.

    def on_enter_live_following(self) -> None:
        self._db._sm_updating = True
        w = self._db._step_widget
        w.interval = TIMER_DISABLED
        w.direction = 1
        # Sync slider to end once on entry so the position reflects the
        # live edge.  We do NOT update value on every push to avoid racing
        # with user clicks.
        w.value = w.end
        self._db._live_badge.visible = True
        self._db._new_data_badge.visible = False
        self._db._sm_updating = False

    def on_exit_live_following(self) -> None:
        self._db._live_badge.visible = False

    def on_enter_live_paused(self) -> None:
        self._db._sm_updating = True
        w = self._db._step_widget
        w.direction = 0
        w.interval = self._get_interval()
        self._db._sm_updating = False

    def on_enter_live_catchup(self) -> None:
        self._db._sm_updating = True
        w = self._db._step_widget
        w.direction = 1
        w.interval = self._get_interval()
        self._db._sm_updating = False

    def on_enter_historical(self) -> None:
        self._db._sm_updating = True
        w = self._db._step_widget
        w.interval = self._get_interval()
        self._db._sm_updating = False
