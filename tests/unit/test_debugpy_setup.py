"""Unit tests for debugpy setup -- verifies graceful handling of port conflicts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from roc.debugpy_setup import maybe_start_debugpy


class TestDebugpyPortConflict:
    """Regression: debugpy.listen() should not crash the process if the port is busy."""

    @patch("roc.debugpy_setup.Config")
    def test_port_in_use_does_not_crash(self, mock_config_cls: MagicMock) -> None:
        """When the debug port is already bound, maybe_start_debugpy logs a warning
        and returns instead of raising RuntimeError."""
        mock_config_cls.get.return_value = MagicMock(debug_port=5678, debug_wait=False)

        fake_debugpy = MagicMock()
        fake_debugpy.listen.side_effect = RuntimeError(
            "Can't listen for client connections: [Errno 98] Address already in use"
        )

        with patch.dict("sys.modules", {"debugpy": fake_debugpy, "pydevd": None}):
            # Remove pydevd so the early-return branch is not taken
            import sys

            sys.modules.pop("pydevd", None)
            # Should NOT raise
            maybe_start_debugpy()

        fake_debugpy.listen.assert_called_once_with(("127.0.0.1", 5678))
        # wait_for_client should NOT be called since listen failed
        fake_debugpy.wait_for_client.assert_not_called()

    @patch("roc.debugpy_setup.Config")
    def test_port_zero_skips_debugpy(self, mock_config_cls: MagicMock) -> None:
        """When debug_port is 0, debugpy is not imported or started."""
        mock_config_cls.get.return_value = MagicMock(debug_port=0)
        # Should return immediately without importing debugpy
        maybe_start_debugpy()
