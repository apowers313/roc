# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/gymnasium.py -- GraphDB flush/export gating."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from roc.config import Config


def _make_fake_obs() -> dict[str, Any]:
    """Create a minimal observation dict matching what Gym.start() expects."""
    return {
        "tty_chars": np.full((24, 80), ord(" "), dtype=np.uint8),
        "tty_colors": np.zeros((24, 80), dtype=np.int8),
        "tty_cursor": np.array([0, 0]),
    }


class TestGraphDBControls:
    def test_graphdb_export_disabled_skips_export(self, mocker):
        """When roc_graphdb_export=False, GraphDB.export() is not called at game end."""
        settings = Config.get()
        settings.graphdb_export = False
        settings.graphdb_flush = True
        settings.num_games = 1

        obs = _make_fake_obs()

        mock_flush = mocker.patch("roc.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.gymnasium.GraphDB.export")

        # Patch Gym.__init__ to bypass real gym setup, then call start() directly
        with patch("roc.gymnasium.Gym.__init__", return_value=None):
            from roc.gymnasium import Gym

            # Create a concrete subclass for testing
            class FakeGym(Gym):
                name: str = "fakegym"
                type: str = "game"

                def send_obs(self, obs: Any) -> None:
                    pass

                def config(self, env: Any) -> None:
                    pass

                def get_action(self) -> Any:
                    return 0

            gym_instance = FakeGym.__new__(FakeGym)
            # Set up minimal env mock: first step returns done=True
            gym_instance.env = MagicMock()
            gym_instance.env.reset.return_value = (obs, {})
            gym_instance.env.step.return_value = (obs, 0, True, False, {})

            # Patch _dump_env_start/_dump_env_record/_dump_env_end and State/breakpoints
            mocker.patch("roc.gymnasium._dump_env_start")
            mocker.patch("roc.gymnasium._dump_env_record")
            mocker.patch("roc.gymnasium._dump_env_end")
            mocker.patch("roc.gymnasium.breakpoints")
            mocker.patch("roc.gymnasium.State")

            gym_instance.start()

        mock_flush.assert_called()
        mock_export.assert_not_called()

    def test_graphdb_flush_disabled_skips_flush(self, mocker):
        """When roc_graphdb_flush=False, GraphDB.flush() is not called at game end."""
        settings = Config.get()
        settings.graphdb_flush = False
        settings.graphdb_export = True
        settings.num_games = 1

        obs = _make_fake_obs()

        mock_flush = mocker.patch("roc.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.gymnasium.GraphDB.export")

        with patch("roc.gymnasium.Gym.__init__", return_value=None):
            from roc.gymnasium import Gym

            class FakeGym(Gym):
                name: str = "fakegym2"
                type: str = "game"

                def send_obs(self, obs: Any) -> None:
                    pass

                def config(self, env: Any) -> None:
                    pass

                def get_action(self) -> Any:
                    return 0

            gym_instance = FakeGym.__new__(FakeGym)
            gym_instance.env = MagicMock()
            gym_instance.env.reset.return_value = (obs, {})
            gym_instance.env.step.return_value = (obs, 0, True, False, {})

            mocker.patch("roc.gymnasium._dump_env_start")
            mocker.patch("roc.gymnasium._dump_env_record")
            mocker.patch("roc.gymnasium._dump_env_end")
            mocker.patch("roc.gymnasium.breakpoints")
            mocker.patch("roc.gymnasium.State")

            gym_instance.start()

        mock_flush.assert_not_called()
        mock_export.assert_called()

    def test_graphdb_both_disabled_skips_both(self, mocker):
        """When both are False, neither flush nor export is called."""
        settings = Config.get()
        settings.graphdb_flush = False
        settings.graphdb_export = False
        settings.num_games = 1

        obs = _make_fake_obs()

        mock_flush = mocker.patch("roc.gymnasium.GraphDB.flush")
        mock_export = mocker.patch("roc.gymnasium.GraphDB.export")

        with patch("roc.gymnasium.Gym.__init__", return_value=None):
            from roc.gymnasium import Gym

            class FakeGym(Gym):
                name: str = "fakegym3"
                type: str = "game"

                def send_obs(self, obs: Any) -> None:
                    pass

                def config(self, env: Any) -> None:
                    pass

                def get_action(self) -> Any:
                    return 0

            gym_instance = FakeGym.__new__(FakeGym)
            gym_instance.env = MagicMock()
            gym_instance.env.reset.return_value = (obs, {})
            gym_instance.env.step.return_value = (obs, 0, True, False, {})

            mocker.patch("roc.gymnasium._dump_env_start")
            mocker.patch("roc.gymnasium._dump_env_record")
            mocker.patch("roc.gymnasium._dump_env_end")
            mocker.patch("roc.gymnasium.breakpoints")
            mocker.patch("roc.gymnasium.State")

            gym_instance.start()

        mock_flush.assert_not_called()
        mock_export.assert_not_called()
