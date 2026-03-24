# mypy: disable-error-code="no-untyped-def"

"""Unit tests for debugpy setup in roc/__init__.py."""

from roc.config import Config


class TestDebugPortConfig:
    def test_debug_port_default(self):
        assert Config.get().debug_port == 5678


class TestDebugpySetup:
    def test_debugpy_not_started_when_port_zero(self, mocker):
        """debugpy should not be started when debug_port is 0."""
        mocker.patch.dict("sys.modules", {"debugpy": mocker.MagicMock()})
        settings = Config.get()
        settings.debug_port = 0

        import sys

        from roc.debugpy_setup import maybe_start_debugpy

        maybe_start_debugpy()
        sys.modules["debugpy"].listen.assert_not_called()

    def test_debugpy_started_when_port_nonzero(self, mocker):
        """debugpy should listen on the configured port."""
        mock_debugpy = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"debugpy": mock_debugpy})
        settings = Config.get()
        settings.debug_port = 5678

        from roc.debugpy_setup import maybe_start_debugpy

        maybe_start_debugpy()
        mock_debugpy.listen.assert_called_once_with(("127.0.0.1", 5678))

    def test_debugpy_not_imported_when_port_zero(self, mocker):
        """debugpy module should not be imported when port is 0."""
        settings = Config.get()
        settings.debug_port = 0
        mock_import = mocker.patch("builtins.__import__", wraps=__import__)

        from roc.debugpy_setup import maybe_start_debugpy

        maybe_start_debugpy()

        # Check that debugpy was never imported
        debugpy_calls = [call for call in mock_import.call_args_list if call[0][0] == "debugpy"]
        assert len(debugpy_calls) == 0

    def test_debugpy_logs_when_started(self, mocker):
        """debugpy should log the port when started."""
        mock_debugpy = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"debugpy": mock_debugpy})
        mock_logger = mocker.patch("roc.debugpy_setup.logger")
        settings = Config.get()
        settings.debug_port = 9042

        from roc.debugpy_setup import maybe_start_debugpy

        maybe_start_debugpy()
        mock_logger.info.assert_called_once()
        assert "9042" in str(mock_logger.info.call_args)

    def test_debug_wait_calls_wait_for_client(self, mocker):
        """debugpy should wait for client when debug_wait is True."""
        mock_debugpy = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"debugpy": mock_debugpy})
        mocker.patch("roc.debugpy_setup.logger")
        settings = Config.get()
        settings.debug_port = 5678
        settings.debug_wait = True

        from roc.debugpy_setup import maybe_start_debugpy

        maybe_start_debugpy()
        mock_debugpy.wait_for_client.assert_called_once()

    def test_debug_wait_false_skips_wait(self, mocker):
        """debugpy should not wait when debug_wait is False."""
        mock_debugpy = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"debugpy": mock_debugpy})
        mocker.patch("roc.debugpy_setup.logger")
        settings = Config.get()
        settings.debug_port = 5678
        settings.debug_wait = False

        from roc.debugpy_setup import maybe_start_debugpy

        maybe_start_debugpy()
        mock_debugpy.wait_for_client.assert_not_called()
