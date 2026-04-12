# mypy: disable-error-code="no-untyped-def"

"""Tests for _game_main() and back-to-back game run cleanup.

Verifies that _game_main properly initializes, runs, and cleans up game state
so that consecutive game runs do not fail due to stale EventBus names,
Component registry conflicts, or State initialization guards.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from roc.framework.component import (
    Component,
    loaded_components,
)
from roc.framework.event import EventBus, eventbus_names
from roc.game.game_manager import GameManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_eventbus_names():
    """Save and restore eventbus names around each test."""
    saved = eventbus_names.copy()
    yield
    eventbus_names.clear()
    eventbus_names.update(saved)


# ---------------------------------------------------------------------------
# Unit tests for cleanup logic
# ---------------------------------------------------------------------------


class TestEventBusClearNames:
    """Verify that EventBus.clear_names() enables bus re-creation."""

    def test_duplicate_bus_name_raises(self):
        """Creating two buses with the same name raises ValueError."""
        EventBus[int]("test-cleanup-bus-1")
        with pytest.raises(ValueError, match="Duplicate"):
            EventBus[int]("test-cleanup-bus-1")

    def test_clear_names_allows_recreation(self):
        """After clear_names(), the same bus name can be reused."""
        EventBus[int]("test-cleanup-bus-2")
        EventBus.clear_names()
        # Should not raise
        EventBus[int]("test-cleanup-bus-2")

    def test_clear_names_is_idempotent(self):
        """Calling clear_names() twice is safe."""
        EventBus[int]("test-cleanup-bus-3")
        EventBus.clear_names()
        EventBus.clear_names()
        EventBus[int]("test-cleanup-bus-3")


class TestStateReset:
    """Verify that State.reset() allows re-initialization."""

    def test_reset_init_clears_init_guard(self):
        """After reset_init(), init() should re-run."""
        from roc.reporting import state as state_mod

        original = state_mod._state_init_done
        state_mod._state_init_done = True

        from roc.reporting.state import State

        State.reset_init()
        assert state_mod._state_init_done is False

        # Restore original state
        state_mod._state_init_done = original


# ---------------------------------------------------------------------------
# GameManager back-to-back integration tests
# ---------------------------------------------------------------------------


class TestBackToBackGameRuns:
    """Integration tests: multiple consecutive game runs via GameManager."""

    def test_two_consecutive_games_complete(self, tmp_path: Path) -> None:
        """Start a game, let it finish, start another. Both must complete cleanly."""
        run_counter = {"count": 0}

        def fake_game_entry(*, num_games, stop_event, on_run_name):
            run_counter["count"] += 1
            on_run_name(f"run-{run_counter['count']}")
            # Simulate brief game
            time.sleep(0.05)

        callback = MagicMock()
        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=callback,
            game_entry=fake_game_entry,
        )

        # First game
        gm.start_game(num_games=1)
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=5)
        assert gm.state == "idle"
        assert gm._error_message is None
        assert gm._current_run_name == "run-1"

        # Second game -- must not fail
        gm.start_game(num_games=1)
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=5)
        assert gm.state == "idle"
        assert gm._error_message is None
        assert gm._current_run_name == "run-2"

    def test_three_consecutive_games(self, tmp_path: Path) -> None:
        """Three sequential games to stress the cleanup path."""
        run_counter = {"count": 0}

        def fake_game_entry(*, num_games, stop_event, on_run_name):
            run_counter["count"] += 1
            on_run_name(f"run-{run_counter['count']}")

        gm = GameManager(
            data_dir=tmp_path,
            game_entry=fake_game_entry,
        )

        for i in range(1, 4):
            gm.start_game(num_games=1)
            assert gm._game_thread is not None
            gm._game_thread.join(timeout=5)
            assert gm.state == "idle", f"Game {i} did not return to idle"
            assert gm._error_message is None, f"Game {i} had error: {gm._error_message}"
            assert gm._current_run_name == f"run-{i}"

    def test_stop_then_restart(self, tmp_path: Path) -> None:
        """Stop a running game, then start a new one."""
        game_started = threading.Event()
        run_counter = {"count": 0}

        def fake_game_entry(*, num_games, stop_event, on_run_name):
            run_counter["count"] += 1
            on_run_name(f"run-{run_counter['count']}")
            game_started.set()
            # Block until stopped
            stop_event.wait(timeout=10)

        gm = GameManager(
            data_dir=tmp_path,
            game_entry=fake_game_entry,
        )

        # Start and stop first game
        gm.start_game(num_games=1)
        assert game_started.wait(timeout=2)
        game_started.clear()

        gm.stop_game()
        assert gm.state == "idle"

        # Start second game
        gm.start_game(num_games=1)
        assert game_started.wait(timeout=2)
        game_started.clear()

        gm.stop_game()
        assert gm.state == "idle"
        assert run_counter["count"] == 2

    def test_error_then_restart(self, tmp_path: Path) -> None:
        """A game that crashes should not prevent the next game from starting."""
        call_count = {"n": 0}

        def sometimes_crashes(*, num_games, stop_event, on_run_name):
            call_count["n"] += 1
            on_run_name(f"run-{call_count['n']}")
            if call_count["n"] == 1:
                raise RuntimeError("First run crashes")
            # Second run succeeds

        gm = GameManager(
            data_dir=tmp_path,
            game_entry=sometimes_crashes,
        )

        # First game crashes
        gm.start_game(num_games=1)
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=5)
        assert gm.state == "idle"
        assert gm._error_message == "First run crashes"

        # Second game should work fine
        gm.start_game(num_games=1)
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=5)
        assert gm.state == "idle"
        assert gm._error_message is None

    def test_run_name_callback_triggers_state_change(self, tmp_path: Path) -> None:
        """When on_run_name is called, the state change callback fires with the run name."""
        state_changes: list[dict[str, object]] = []

        def track_state(status):
            state_changes.append(dict(status))

        def fake_game_entry(*, num_games, stop_event, on_run_name):
            on_run_name("my-run-123")

        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=track_state,
            game_entry=fake_game_entry,
        )

        gm.start_game(num_games=1)
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=5)

        # Find the state change notification that includes the run name
        run_name_notifications = [s for s in state_changes if s.get("run_name") == "my-run-123"]
        assert len(run_name_notifications) > 0, (
            f"Expected a state change with run_name='my-run-123', got: {state_changes}"
        )


class TestGameMainCleanup:
    """Test that _game_main's cleanup path works for back-to-back runs.

    These tests use EventBus name re-creation as the canary -- if cleanup
    didn't clear names, the second run's Component.init() would fail when
    pipeline components try to create their buses.
    """

    def test_eventbus_cleanup_enables_reuse(self):
        """Simulates the cleanup sequence from _game_main's finally block."""
        # First "run": create some buses
        EventBus[int]("sim-perception")
        EventBus[int]("sim-action")
        assert "sim-perception" in eventbus_names
        assert "sim-action" in eventbus_names

        # Cleanup (as _game_main does)
        EventBus.clear_names()

        # Second "run": same names should work
        EventBus[int]("sim-perception")
        EventBus[int]("sim-action")

    def test_component_reset_and_reinit(self):
        """Component.reset() + Component.init() sequence works twice."""
        # This test verifies the pattern used in _game_main's cleanup
        # We just verify reset doesn't crash when there are no loaded components
        loaded_before = dict(loaded_components)
        Component.reset()
        assert len(loaded_components) == 0
        # Restore
        loaded_components.update(loaded_before)


class TestGameMainCallsCleanup:
    """Regression: _game_main must call cleanup in its finally block.

    Without EventBus.clear_names(), the second game run crashes with
    'Duplicate EventBus name'. Without State.reset_init(), the second
    run's State.init() silently no-ops, leaving state tracking disconnected.
    Without Component.reset(), pipeline components from the first run
    linger and cause 'component already exists' errors.

    These tests use patch.object on the real classes (not module-level patches)
    because _game_main uses local imports inside its function body.
    """

    def test_game_main_calls_eventbus_clear_names(self):
        """Verify _game_main's finally block calls EventBus.clear_names()."""
        from roc.framework.config import Config
        from roc.framework.event import EventBus
        from roc.framework.expmod import ExpMod
        from roc.reporting.state import State
        from roc.game.gymnasium import _game_main

        with (
            patch("roc.game.gymnasium.NethackGym") as mock_gym_cls,
            patch("roc.game.gymnasium.roc_logger"),
            patch("roc.reporting.api_server.start_dashboard"),
            patch.object(Config, "init"),
            patch.object(Component, "init"),
            patch.object(Component, "reset"),
            patch.object(ExpMod, "init"),
            patch.object(State, "init"),
            patch.object(State, "reset_init"),
            patch.object(EventBus, "clear_names") as mock_clear,
            patch("roc.reporting.observability.Observability.init"),
        ):
            mock_gym_cls.return_value = MagicMock()
            _game_main(num_games=1, stop_event=threading.Event(), on_run_name=MagicMock())
            mock_clear.assert_called_once()

    def test_game_main_calls_state_reset_init(self):
        """Verify _game_main's finally block calls State.reset_init()."""
        from roc.framework.config import Config
        from roc.framework.event import EventBus
        from roc.framework.expmod import ExpMod
        from roc.reporting.state import State
        from roc.game.gymnasium import _game_main

        with (
            patch("roc.game.gymnasium.NethackGym") as mock_gym_cls,
            patch("roc.game.gymnasium.roc_logger"),
            patch("roc.reporting.api_server.start_dashboard"),
            patch.object(Config, "init"),
            patch.object(Component, "init"),
            patch.object(Component, "reset"),
            patch.object(ExpMod, "init"),
            patch.object(State, "init"),
            patch.object(State, "reset_init") as mock_reset_init,
            patch.object(EventBus, "clear_names"),
            patch("roc.reporting.observability.Observability.init"),
        ):
            mock_gym_cls.return_value = MagicMock()
            _game_main(num_games=1, stop_event=threading.Event(), on_run_name=MagicMock())
            mock_reset_init.assert_called_once()

    def test_game_main_calls_component_reset(self):
        """Verify _game_main's finally block calls Component.reset()."""
        from roc.framework.config import Config
        from roc.framework.event import EventBus
        from roc.framework.expmod import ExpMod
        from roc.reporting.state import State
        from roc.game.gymnasium import _game_main

        with (
            patch("roc.game.gymnasium.NethackGym") as mock_gym_cls,
            patch("roc.game.gymnasium.roc_logger"),
            patch("roc.reporting.api_server.start_dashboard"),
            patch.object(Config, "init"),
            patch.object(Component, "init"),
            patch.object(Component, "reset") as mock_reset,
            patch.object(ExpMod, "init"),
            patch.object(State, "init"),
            patch.object(State, "reset_init"),
            patch.object(EventBus, "clear_names"),
            patch("roc.reporting.observability.Observability.init"),
        ):
            mock_gym_cls.return_value = MagicMock()
            _game_main(num_games=1, stop_event=threading.Event(), on_run_name=MagicMock())
            mock_reset.assert_called_once()

    def test_game_main_cleanup_runs_on_exception(self):
        """Verify cleanup still runs when the game crashes."""
        from roc.framework.config import Config
        from roc.framework.event import EventBus
        from roc.framework.expmod import ExpMod
        from roc.reporting.state import State
        from roc.game.gymnasium import _game_main

        with (
            patch("roc.game.gymnasium.NethackGym") as mock_gym_cls,
            patch("roc.game.gymnasium.roc_logger"),
            patch("roc.reporting.api_server.start_dashboard"),
            patch.object(Config, "init"),
            patch.object(Component, "init"),
            patch.object(Component, "reset") as mock_comp_reset,
            patch.object(ExpMod, "init"),
            patch.object(State, "init"),
            patch.object(State, "reset_init") as mock_state_reset,
            patch.object(EventBus, "clear_names") as mock_clear,
            patch("roc.reporting.observability.Observability.init"),
        ):
            mock_gym = MagicMock()
            mock_gym.start.side_effect = RuntimeError("NLE crashed")
            mock_gym_cls.return_value = mock_gym

            with pytest.raises(RuntimeError, match="NLE crashed"):
                _game_main(num_games=1, stop_event=threading.Event(), on_run_name=MagicMock())

            # Cleanup must still run even after a crash
            mock_comp_reset.assert_called_once()
            mock_clear.assert_called_once()
            mock_state_reset.assert_called_once()

    def test_game_main_reports_run_name(self):
        """Verify _game_main calls on_run_name with the instance_id."""
        from roc.framework.config import Config
        from roc.framework.event import EventBus
        from roc.framework.expmod import ExpMod
        from roc.reporting.state import State
        from roc.game.gymnasium import _game_main

        with (
            patch("roc.game.gymnasium.NethackGym") as mock_gym_cls,
            patch("roc.game.gymnasium.roc_logger"),
            patch("roc.reporting.api_server.start_dashboard"),
            patch.object(Config, "init"),
            patch.object(Component, "init"),
            patch.object(Component, "reset"),
            patch.object(ExpMod, "init"),
            patch.object(State, "init"),
            patch.object(State, "reset_init"),
            patch.object(EventBus, "clear_names"),
            patch("roc.reporting.observability.Observability.init"),
            patch("roc.reporting.observability.instance_id", "test-instance-42"),
        ):
            mock_gym_cls.return_value = MagicMock()
            on_run_name = MagicMock()

            _game_main(num_games=1, stop_event=threading.Event(), on_run_name=on_run_name)

            on_run_name.assert_called_once_with("test-instance-42")


class TestGameMainRealGymCrashPropagation:
    """Regression: when the REAL ``NethackGym.start`` raises, the exception
    must propagate through ``_game_main`` so ``GameManager`` can capture it
    as ``_error_message``.

    Until 2026-04-09, ``NethackGym.start`` was decorated with the default
    ``@logger.catch`` (no ``reraise=True``), which logs and SUPPRESSES the
    exception. A real crash inside the pipeline (e.g., a broken Memgraph
    connection raising ``mgclient.DatabaseError``) would be swallowed,
    ``gym.start`` would return ``None``, ``_game_main`` would clean up and
    return normally, and ``GameManager._run_game`` would set the state
    back to ``idle`` with ``error=None`` -- indistinguishable from a
    clean game completion.

    The existing ``test_game_main_cleanup_runs_on_exception`` doesn't
    catch this regression because it patches ``NethackGym`` with a
    ``MagicMock``, bypassing the real decorator. This test class uses a
    real ``Gym`` subclass whose ``send_obs`` raises the exception, so
    ``@logger.catch`` is actually exercised.
    """

    @staticmethod
    def _enter_game_main_patches(stack, crashing_gym_cls):
        """Enter all patches needed to drive _game_main in isolation.

        Uses contextlib.ExitStack to avoid Python's "too many statically
        nested blocks" limit on ``with`` chains.
        """
        from roc.framework.config import Config
        from roc.framework.event import EventBus
        from roc.framework.expmod import ExpMod
        from roc.reporting.state import State

        stack.enter_context(patch("roc.game.gymnasium.NethackGym", crashing_gym_cls))
        stack.enter_context(patch("roc.game.gymnasium.roc_logger"))
        stack.enter_context(patch("roc.reporting.api_server.start_dashboard"))
        stack.enter_context(patch("roc.reporting.api_server.stop_dashboard"))
        stack.enter_context(patch.object(Config, "init"))
        stack.enter_context(patch.object(Component, "init"))
        stack.enter_context(patch.object(Component, "reset"))
        stack.enter_context(patch.object(ExpMod, "init"))
        stack.enter_context(patch.object(State, "init"))
        stack.enter_context(patch.object(State, "reset_init"))
        stack.enter_context(patch.object(EventBus, "clear_names"))
        stack.enter_context(patch("roc.reporting.observability.Observability.init"))
        stack.enter_context(patch("roc.reporting.observability.Observability.shutdown"))
        stack.enter_context(patch("roc.reporting.observability.Observability.reset"))
        stack.enter_context(patch("roc.game.gymnasium._dump_env_start"))
        stack.enter_context(patch("roc.game.gymnasium._dump_env_record"))
        stack.enter_context(patch("roc.game.gymnasium._dump_env_end"))
        stack.enter_context(patch("roc.game.gymnasium._publish_action_map"))
        stack.enter_context(patch("roc.game.gymnasium.breakpoints"))
        stack.enter_context(patch("roc.game.gymnasium.State"))
        stack.enter_context(patch("roc.game.gymnasium._export_graph_archive"))
        stack.enter_context(patch("roc.game.gymnasium.GraphDB.flush"))
        stack.enter_context(patch("roc.game.gymnasium.GraphDB.export"))

    def test_real_gym_start_propagates_through_game_main(self):
        """A Gym subclass that raises during the pipeline must propagate out."""
        import contextlib

        import numpy as np

        from roc.framework.config import Config
        from roc.game.gymnasium import Gym, _game_main

        # Unique name so Component registration does not collide with
        # sibling tests in the same process.
        class CrashingGymPropagate(Gym):
            name: str = "crash-gym-propagate"
            type: str = "game"

            def __init__(self) -> None:
                obs = {
                    "tty_chars": np.full((24, 80), ord(" "), dtype=np.uint8),
                    "tty_colors": np.zeros((24, 80), dtype=np.int8),
                    "tty_cursor": np.array([0, 0]),
                    "blstats": np.zeros(27, dtype=np.int64),
                }
                self.env = MagicMock()
                self.env.reset.return_value = (obs, {})
                self.env.step.return_value = (obs, 0, True, False, {})

            def send_obs(self, obs: object) -> None:
                raise RuntimeError("pipeline crash: broken graphdb connection")

            def config(self, env: object) -> None:
                """No-op; required by abstract base class."""

            def get_action(self) -> int:
                return 0

        with contextlib.ExitStack() as stack:
            self._enter_game_main_patches(stack, CrashingGymPropagate)

            settings = Config.get()
            settings.num_games = 1

            with pytest.raises(RuntimeError, match="pipeline crash"):
                _game_main(
                    num_games=1,
                    stop_event=threading.Event(),
                    on_run_name=MagicMock(),
                )

    def test_game_manager_captures_real_gym_crash(self, tmp_path: Path) -> None:
        """End-to-end: GameManager must store the error from a real gym crash.

        This is the direct regression for the 2026-04-09 "silent idle"
        bug. Before the fix, ``GameManager._error_message`` was ``None``
        after a crash because ``@logger.catch`` on ``Gym.start``
        swallowed the exception. After the fix (reraise=True), the
        error propagates to ``_run_game``'s ``except`` block and is
        captured as ``_error_message``.
        """
        import contextlib

        import numpy as np

        from roc.framework.config import Config
        from roc.game.game_manager import GameManager
        from roc.game.gymnasium import Gym, _game_main

        class CrashingGymE2E(Gym):
            name: str = "crash-gym-e2e"
            type: str = "game"

            def __init__(self) -> None:
                obs = {
                    "tty_chars": np.full((24, 80), ord(" "), dtype=np.uint8),
                    "tty_colors": np.zeros((24, 80), dtype=np.int8),
                    "tty_cursor": np.array([0, 0]),
                    "blstats": np.zeros(27, dtype=np.int64),
                }
                self.env = MagicMock()
                self.env.reset.return_value = (obs, {})
                self.env.step.return_value = (obs, 0, True, False, {})

            def send_obs(self, obs: object) -> None:
                raise RuntimeError("e2e crash marker")

            def config(self, env: object) -> None:
                """No-op; required by abstract base class."""

            def get_action(self) -> int:
                return 0

        with contextlib.ExitStack() as stack:
            self._enter_game_main_patches(stack, CrashingGymE2E)

            settings = Config.get()
            settings.num_games = 1

            gm = GameManager(data_dir=tmp_path, game_entry=_game_main)
            gm.start_game(num_games=1)
            assert gm._game_thread is not None
            gm._game_thread.join(timeout=5)

            assert gm.state == "idle"
            assert gm._error_message is not None, (
                "Expected _error_message to be set after a crash, "
                "but it was None -- the 'silent idle' regression is back"
            )
            assert "e2e crash marker" in gm._error_message
