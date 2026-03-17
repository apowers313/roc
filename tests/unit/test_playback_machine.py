"""Unit tests for PlaybackMachine -- isolated from Panel widgets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from statemachine.exceptions import TransitionNotAllowed

from roc.reporting.playback_machine import (
    TIMER_DISABLED,
    PlaybackListener,
    PlaybackMachine,
)


def _make_dashboard(**overrides: Any) -> SimpleNamespace:
    """Build a minimal mock dashboard for state machine tests."""
    step_widget = SimpleNamespace(
        value=1,
        end=100,
        start=1,
        interval=200,
        direction=0,
    )
    speed_selector = SimpleNamespace(value="5x")
    live_badge = SimpleNamespace(visible=False)
    new_data_badge = SimpleNamespace(visible=False)

    db = SimpleNamespace(
        _step_widget=step_widget,
        _speed_selector=speed_selector,
        _speed_to_interval={"0.5x": 2000, "1x": 1000, "2x": 500, "5x": 200, "10x": 100},
        _DEFAULT_SPEED="5x",
        _live_badge=live_badge,
        _new_data_badge=new_data_badge,
    )
    for k, v in overrides.items():
        setattr(db, k, v)
    return db


def _make_sm(
    db: SimpleNamespace | None = None, **overrides: Any
) -> tuple[PlaybackMachine, SimpleNamespace]:
    """Create a PlaybackMachine + listener wired to a mock dashboard."""
    if db is None:
        db = _make_dashboard(**overrides)
    listener = PlaybackListener(db)  # type: ignore[arg-type]
    sm = PlaybackMachine(dashboard=db, listeners=[listener])  # type: ignore[arg-type]
    return sm, db


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_in_historical(self) -> None:
        sm, _ = _make_sm()
        assert sm.historical.is_active

    def test_go_live_transitions_to_following(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        assert sm.live_following.is_active
        assert db._step_widget.interval == TIMER_DISABLED
        assert db._step_widget.direction == 1
        assert db._live_badge.visible is True

    def test_cannot_pause_from_historical(self) -> None:
        sm, _ = _make_sm()
        with pytest.raises(TransitionNotAllowed):
            sm.send("pause")

    def test_cannot_resume_from_historical(self) -> None:
        sm, _ = _make_sm()
        with pytest.raises(TransitionNotAllowed):
            sm.send("resume")


# ---------------------------------------------------------------------------
# Pause / resume cycle
# ---------------------------------------------------------------------------


class TestPauseResume:
    def test_pause_from_following(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        assert sm.live_paused.is_active
        assert db._step_widget.direction == 0
        assert db._step_widget.interval == 200
        assert db._live_badge.visible is False

    def test_resume_from_paused_enters_catchup(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")
        assert sm.live_catchup.is_active
        assert db._step_widget.direction == 1
        assert db._step_widget.interval == 200  # timer-driven, not disabled

    def test_pause_from_catchup(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")
        sm.send("pause")
        assert sm.live_paused.is_active
        assert db._step_widget.direction == 0

    def test_cannot_resume_from_following(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        with pytest.raises(TransitionNotAllowed):
            sm.send("resume")

    def test_cannot_resume_from_catchup(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")
        with pytest.raises(TransitionNotAllowed):
            sm.send("resume")


# ---------------------------------------------------------------------------
# Jump to end
# ---------------------------------------------------------------------------


class TestJumpToEnd:
    def test_jump_to_end_from_paused(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("jump_to_end")
        assert sm.live_following.is_active
        assert db._step_widget.interval == TIMER_DISABLED
        assert db._step_widget.direction == 1
        assert db._live_badge.visible is True

    def test_jump_to_end_from_catchup(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")
        sm.send("jump_to_end")
        assert sm.live_following.is_active


# ---------------------------------------------------------------------------
# Push notifications
# ---------------------------------------------------------------------------


class TestPushArrived:
    def test_push_while_following_stays_following(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("push_arrived")
        assert sm.live_following.is_active

    def test_push_at_edge_transitions_catchup_to_following(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")  # catchup
        # Simulate being at the edge
        db._step_widget.value = 100
        db._step_widget.end = 100
        sm.send("push_arrived")
        assert sm.live_following.is_active

    def test_push_behind_edge_stays_in_catchup(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")  # catchup
        db._step_widget.value = 50
        db._step_widget.end = 100
        sm.send("push_arrived")
        assert sm.live_catchup.is_active

    def test_push_while_paused_stays_paused(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("push_arrived")
        assert sm.live_paused.is_active

    def test_push_not_valid_in_historical(self) -> None:
        sm, _ = _make_sm()
        with pytest.raises(TransitionNotAllowed):
            sm.send("push_arrived")


# ---------------------------------------------------------------------------
# Historical mode
# ---------------------------------------------------------------------------


class TestHistorical:
    def test_toggle_play_stays_historical(self) -> None:
        sm, _ = _make_sm()
        sm.send("toggle_play")
        assert sm.historical.is_active

    def test_toggle_play_not_valid_in_live(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        with pytest.raises(TransitionNotAllowed):
            sm.send("toggle_play")


# ---------------------------------------------------------------------------
# Allowed events
# ---------------------------------------------------------------------------


class TestAllowedEvents:
    def test_historical_allowed(self) -> None:
        sm, _ = _make_sm()
        allowed = sm.allowed_events
        assert "go_live" in [e.id for e in allowed]
        assert "toggle_play" in [e.id for e in allowed]

    def test_following_allowed(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        allowed = [e.id for e in sm.allowed_events]
        assert "pause" in allowed
        assert "push_arrived" in allowed
        assert "resume" not in allowed
        assert "go_live" not in allowed

    def test_paused_allowed(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        allowed = [e.id for e in sm.allowed_events]
        assert "resume" in allowed
        assert "jump_to_end" in allowed
        assert "push_arrived" in allowed
        assert "pause" not in allowed

    def test_catchup_allowed(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")
        allowed = [e.id for e in sm.allowed_events]
        assert "pause" in allowed
        assert "jump_to_end" in allowed
        assert "push_arrived" in allowed
        assert "resume" not in allowed


# ---------------------------------------------------------------------------
# User navigation (leaves live edge)
# ---------------------------------------------------------------------------


class TestUserNavigate:
    def test_navigate_from_following_pauses(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("user_navigate")
        assert sm.live_paused.is_active
        assert db._step_widget.direction == 0
        assert db._live_badge.visible is False

    def test_navigate_from_catchup_pauses(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("resume")  # catchup
        sm.send("user_navigate")
        assert sm.live_paused.is_active

    def test_navigate_from_paused_is_noop(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        sm.send("user_navigate")
        assert sm.live_paused.is_active  # stays paused

    def test_navigate_from_historical_is_noop(self) -> None:
        sm, _ = _make_sm()
        sm.send("user_navigate")
        assert sm.historical.is_active  # stays historical

    def test_navigate_then_jump_to_end_follows(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("user_navigate")
        assert sm.live_paused.is_active
        sm.send("jump_to_end")
        assert sm.live_following.is_active
        assert db._live_badge.visible is True

    def test_navigate_in_allowed_events(self) -> None:
        sm, _ = _make_sm()
        sm.send("go_live")
        allowed = [e.id for e in sm.allowed_events]
        assert "user_navigate" in allowed


# ---------------------------------------------------------------------------
# Speed changes via listener
# ---------------------------------------------------------------------------


class TestSpeedInteraction:
    def test_pause_restores_current_speed(self) -> None:
        sm, db = _make_sm()
        db._speed_selector.value = "10x"
        sm.send("go_live")
        sm.send("pause")
        assert db._step_widget.interval == 100  # 10x speed

    def test_resume_uses_current_speed(self) -> None:
        sm, db = _make_sm()
        sm.send("go_live")
        sm.send("pause")
        db._speed_selector.value = "2x"
        sm.send("resume")
        assert db._step_widget.interval == 500  # 2x speed
