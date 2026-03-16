# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/metrics.py -- RocMetrics unified abstraction."""

from unittest.mock import MagicMock, patch

import pytest

from roc.config import Config
from roc.reporting.wandb_reporter import WandbReporter


@pytest.fixture(autouse=True)
def reset_wandb_reporter():
    """Reset WandbReporter singleton state between tests."""
    WandbReporter._instance = None
    WandbReporter._run = None
    WandbReporter._enabled = False
    WandbReporter._game_num = 0
    WandbReporter._game_tick = 0
    WandbReporter._global_step = 0
    WandbReporter._scores = []
    WandbReporter._step_buffer = {}
    yield
    WandbReporter._instance = None
    WandbReporter._run = None
    WandbReporter._enabled = False
    WandbReporter._game_num = 0
    WandbReporter._game_tick = 0
    WandbReporter._global_step = 0
    WandbReporter._scores = []
    WandbReporter._step_buffer = {}


class TestRocMetricsHistogram:
    def test_record_histogram_calls_otel(self):
        """record_histogram() calls OTel histogram."""
        from roc.reporting.metrics import RocMetrics

        mock_histogram = MagicMock()
        with patch.object(RocMetrics, "_get_histogram", return_value=mock_histogram):
            RocMetrics.record_histogram("roc.test.metric", 42.0)
            mock_histogram.record.assert_called_once_with(42.0, attributes=None)

    @patch("roc.reporting.wandb_reporter._wandb")
    def test_record_histogram_calls_wandb(self, mock_wandb):
        """record_histogram() calls WandbReporter.log_step()."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        from roc.reporting.metrics import RocMetrics

        mock_histogram = MagicMock()
        with patch.object(RocMetrics, "_get_histogram", return_value=mock_histogram):
            RocMetrics.record_histogram("roc.test.metric", 42.0)
            RocMetrics.flush_step()

        # Verify wandb.log was called (via flush_step)
        mock_wandb.log.assert_called()
        last_data = mock_wandb.log.call_args[0][0]
        assert "roc.test.metric" in last_data
        assert last_data["roc.test.metric"] == 42.0


class TestRocMetricsCounter:
    def test_increment_counter_calls_both(self):
        """increment_counter() calls both OTel counter and W&B."""
        from roc.reporting.metrics import RocMetrics

        mock_counter = MagicMock()
        with patch.object(RocMetrics, "_get_counter", return_value=mock_counter):
            RocMetrics.increment_counter("roc.test.counter", 1)
            mock_counter.add.assert_called_once_with(1, attributes=None)

    def test_wandb_disabled_otel_still_works(self):
        """When W&B is off, OTel calls still fire."""
        settings = Config.get()
        settings.wandb_enabled = False
        WandbReporter.init(settings)

        from roc.reporting.metrics import RocMetrics

        mock_histogram = MagicMock()
        with patch.object(RocMetrics, "_get_histogram", return_value=mock_histogram):
            RocMetrics.record_histogram("roc.test.metric", 10.0)
            mock_histogram.record.assert_called_once_with(10.0, attributes=None)


class TestRocMetricsLogStep:
    @patch("roc.reporting.wandb_reporter._wandb")
    def test_log_step_wandb_only(self, mock_wandb):
        """log_step() only goes to W&B, not OTel."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)

        from roc.reporting.metrics import RocMetrics

        RocMetrics.log_step({"custom_metric": 99})
        RocMetrics.flush_step()

        mock_wandb.log.assert_called()
        last_data = mock_wandb.log.call_args[0][0]
        assert last_data["custom_metric"] == 99
