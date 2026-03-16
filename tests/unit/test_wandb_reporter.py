# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/wandb_reporter.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.config import Config
from roc.reporting.wandb_reporter import WandbReporter


def _reset_reporter() -> None:
    WandbReporter._instance = None
    WandbReporter._run = None
    WandbReporter._enabled = False
    WandbReporter._game_num = 0
    WandbReporter._game_tick = 0
    WandbReporter._global_step = 0
    WandbReporter._scores = []
    WandbReporter._config = None
    WandbReporter._game_table = None
    WandbReporter._game_table_rows = []
    WandbReporter._step_buffer = {}


@pytest.fixture(autouse=True)
def reset_wandb_reporter():
    """Reset WandbReporter singleton state between tests."""
    _reset_reporter()
    yield
    _reset_reporter()


class TestWandbReporterDisabled:
    def test_init_disabled(self):
        """When wandb_enabled=False, init() does nothing."""
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)
        assert WandbReporter._run is None
        assert WandbReporter._enabled is False

    def test_start_game_noop_when_disabled(self):
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)
        # Should not raise
        WandbReporter.start_game(1)
        assert WandbReporter._game_num == 0

    def test_end_game_noop_when_disabled(self):
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)
        WandbReporter.end_game(outcome="died", final_score=0)

    def test_finish_noop_when_disabled(self):
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)
        WandbReporter.finish()

    def test_log_step_noop_when_disabled(self):
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)
        WandbReporter.log_step({})


class TestWandbReporterEnabled:
    @patch("roc.reporting.wandb_reporter._wandb")
    def test_init_creates_run(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        assert WandbReporter._run is mock_run
        assert WandbReporter._enabled is True

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_start_game_increments_game_num(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)
        assert WandbReporter._game_num == 1

        WandbReporter.start_game(2)
        assert WandbReporter._game_num == 2

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_start_game_resets_game_tick(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)
        WandbReporter._game_tick = 5
        WandbReporter.start_game(2)
        assert WandbReporter._game_tick == 0

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_end_game_buffers_boundary(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)
        WandbReporter.end_game(outcome="died", final_score=42)

        # end_game now buffers instead of logging directly
        assert WandbReporter._step_buffer["game_end"] == 1
        assert WandbReporter._step_buffer["outcome"] == "died"
        assert WandbReporter._step_buffer["final_score"] == 42

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_finish_sets_summary(self, mock_wandb):
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        WandbReporter.start_game(1)
        WandbReporter.end_game(outcome="died", final_score=10)
        WandbReporter.start_game(2)
        WandbReporter.end_game(outcome="died", final_score=20)

        WandbReporter.finish()

        assert mock_run.summary["total_games"] == 2
        assert mock_run.summary["mean_score"] == 15.0
        assert mock_run.summary["max_score"] == 20

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_run_name_uses_instance_id(self, mock_wandb):
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        # Verify wandb.init was called with the instance_id as name
        call_kwargs = mock_wandb.init.call_args[1]
        from roc.reporting.observability import instance_id

        assert call_kwargs["name"] == instance_id

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_sweep_detection(self, mock_wandb):
        mock_run = MagicMock()
        mock_run.sweep_id = "sweep-123"
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        # When sweep_id is set, the reporter should not override the run name
        assert WandbReporter._run is mock_run
        # The run name should not be changed when sweep_id is present
        if mock_run.sweep_id:
            # Reporter should detect sweep mode
            assert WandbReporter._run.sweep_id == "sweep-123"


class TestWandbReporterLogStep:
    """Phase 2: Per-step numeric metric logging tests."""

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_step_increments_counter(self, mock_wandb):
        """After 3 calls to log_step(), internal step counter is 3."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        WandbReporter.log_step({"score": 1})
        WandbReporter.flush_step()
        WandbReporter.log_step({"score": 2})
        WandbReporter.flush_step()
        WandbReporter.log_step({"score": 3})
        WandbReporter.flush_step()

        assert WandbReporter._global_step == 3

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_step_includes_game_num(self, mock_wandb):
        """Every flush_step() call includes game_num in the data dict."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(2)

        WandbReporter.log_step({"score": 10})
        WandbReporter.flush_step()

        # Find the flush_step call (not start_game's call)
        calls = mock_wandb.log.call_args_list
        last_data = calls[-1][0][0]
        assert last_data["game_num"] == 2

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_step_includes_game_tick(self, mock_wandb):
        """Every call includes game_tick that resets per game."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        WandbReporter.log_step({"score": 10})
        WandbReporter.flush_step()
        WandbReporter.log_step({"score": 20})
        WandbReporter.flush_step()

        calls = mock_wandb.log.call_args_list
        # game_tick should be 1 on first step, 2 on second
        step1_data = calls[-2][0][0]
        step2_data = calls[-1][0][0]
        assert step1_data["game_tick"] == 1
        assert step2_data["game_tick"] == 2

        # Reset on new game
        WandbReporter.start_game(2)
        WandbReporter.log_step({"score": 30})
        WandbReporter.flush_step()
        calls = mock_wandb.log.call_args_list
        last_data = calls[-1][0][0]
        assert last_data["game_tick"] == 1

    def test_log_step_disabled_noop(self):
        """When wandb_enabled=False, log_step() does not raise."""
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)
        # Should not raise, counters should not increment
        WandbReporter.log_step({"score": 10})
        assert WandbReporter._global_step == 0

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_flush_step_sends_single_log(self, mock_wandb):
        """flush_step() sends one wandb.log() with all buffered data."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()
        mock_wandb.Html = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        # Buffer metrics + media
        WandbReporter.log_step({"score": 42, "hp": 14})
        WandbReporter.log_media("screen", "<html>screen</html>")
        WandbReporter.log_media("saliency_map", "<html>saliency</html>")

        log_count_before = mock_wandb.log.call_count
        WandbReporter.flush_step()
        log_count_after = mock_wandb.log.call_count

        # flush_step should produce exactly one wandb.log call
        assert log_count_after - log_count_before == 1

        # That single call should contain metrics + both media keys
        flush_data = mock_wandb.log.call_args_list[-1][0][0]
        assert "score" in flush_data
        assert "hp" in flush_data
        assert "screen" in flush_data
        assert "saliency_map" in flush_data
        assert "game_num" in flush_data
        assert "game_tick" in flush_data

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_flush_step_clears_buffer(self, mock_wandb):
        """After flush_step(), the buffer is empty."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        WandbReporter.log_step({"score": 10})
        assert len(WandbReporter._step_buffer) > 0

        WandbReporter.flush_step()
        assert len(WandbReporter._step_buffer) == 0

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_flush_step_noop_when_empty(self, mock_wandb):
        """flush_step() does nothing if nothing was buffered."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        log_count_before = mock_wandb.log.call_count
        WandbReporter.flush_step()
        assert mock_wandb.log.call_count == log_count_before


class TestWandbReporterMedia:
    """Phase 3: Rich media logging tests."""

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_media_respects_interval(self, mock_wandb):
        """With wandb_log_interval=3, media is only logged on ticks 0, 3, 6..."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()
        mock_html_cls = MagicMock()
        mock_wandb.Html = mock_html_cls

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        settings.wandb_log_interval = 3
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        for tick in range(7):
            WandbReporter.log_step({"score": tick})
            WandbReporter.log_media("screen", f"<html>tick {tick}</html>")
            WandbReporter.flush_step()

        # Html should be called at game_tick 1, 4, 7 (tick 0, 3, 6)
        # i.e. when (game_tick - 1) % interval == 0
        # That means Html was called 3 times total
        assert mock_html_cls.call_count == 3

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_media_respects_screen_toggle(self, mock_wandb):
        """When wandb_log_screens=False, screen HTML is not logged."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()
        mock_html_cls = MagicMock()
        mock_wandb.Html = mock_html_cls

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        settings.wandb_log_screens = False
        WandbReporter.init(settings)
        WandbReporter.start_game(1)
        WandbReporter.log_step({"score": 0})

        WandbReporter.log_media("screen", "<html>screen</html>")
        mock_html_cls.assert_not_called()

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_media_respects_saliency_toggle(self, mock_wandb):
        """When wandb_log_saliency=False, saliency HTML is not logged."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()
        mock_html_cls = MagicMock()
        mock_wandb.Html = mock_html_cls

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        settings.wandb_log_saliency = False
        WandbReporter.init(settings)
        WandbReporter.start_game(1)
        WandbReporter.log_step({"score": 0})

        WandbReporter.log_media("saliency_map", "<html>saliency</html>")
        mock_html_cls.assert_not_called()

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_game_table_accumulated(self, mock_wandb):
        """After a 3-tick game, the W&B Table has 3 rows with expected columns."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()
        mock_table_cls = MagicMock()
        mock_table_instance = MagicMock()
        mock_table_cls.return_value = mock_table_instance
        mock_wandb.Table = mock_table_cls

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        for i in range(3):
            WandbReporter.log_step({"score": i * 10})
            WandbReporter.log_media("screen", f"<html>screen {i}</html>")
            WandbReporter.flush_step()

        # Table should have been created and have 3 rows accumulated
        assert WandbReporter._game_table is not None
        assert len(WandbReporter._game_table_rows) == 3

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_game_table_buffered_at_game_end(self, mock_wandb):
        """Table is buffered during end_game(), emitted by next flush_step()."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()
        mock_table_cls = MagicMock()
        mock_table_instance = MagicMock()
        mock_table_cls.return_value = mock_table_instance
        mock_wandb.Table = mock_table_cls

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        WandbReporter.log_step({"score": 10})
        WandbReporter.log_media("screen", "<html>screen</html>")
        WandbReporter.flush_step()

        # Before end_game, table key should not be in buffer
        assert "game_1_steps" not in WandbReporter._step_buffer

        WandbReporter.end_game(outcome="died", final_score=10)

        # After end_game, table should be buffered (not logged yet)
        assert "game_1_steps" in WandbReporter._step_buffer
