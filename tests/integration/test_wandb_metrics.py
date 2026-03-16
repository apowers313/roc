# mypy: disable-error-code="no-untyped-def"

"""Integration test: verify WandbReporter.log_step() receives expected metric keys."""

from unittest.mock import MagicMock, patch

import pytest

from roc.config import Config
from roc.reporting.metrics import RocMetrics
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


class TestFullTickMetrics:
    @patch("roc.reporting.wandb_reporter._wandb")
    def test_full_tick_metrics_logged(self, mock_wandb):
        """Simulate a single tick's worth of metric emissions and verify
        WandbReporter.log_step() received expected metric keys.
        """
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Settings.return_value = MagicMock()

        settings = Config.get()
        settings.wandb_enabled = True
        settings.wandb_mode = "disabled"
        WandbReporter.init(settings)
        WandbReporter.start_game(1)

        # Simulate metrics that would be emitted during a tick
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        with (
            patch.object(RocMetrics, "_get_histogram", return_value=mock_histogram),
            patch.object(RocMetrics, "_get_counter", return_value=mock_counter),
        ):
            # Saliency attenuation metrics
            RocMetrics.record_histogram("roc.saliency_attenuation.peak_count", 5.0)
            RocMetrics.record_histogram("roc.saliency_attenuation.top_peak_strength", 0.85)
            RocMetrics.increment_counter("roc.saliency_attenuation.top_peak_shifted", 1)

            # Core game state logged via log_step
            RocMetrics.log_step(
                {
                    "score": 100,
                    "hp": 15,
                    "depth": 1,
                    "x": 40,
                    "y": 10,
                }
            )

            # Flush buffered data as a single wandb.log call
            RocMetrics.flush_step()

        # Verify wandb.log was called: 1 flush_step (start_game only buffers)
        assert mock_wandb.log.call_count >= 1

        # Collect all logged data keys
        all_keys: set[str] = set()
        for call in mock_wandb.log.call_args_list:
            data = call[0][0]
            all_keys.update(data.keys())

        # Verify expected keys are present
        assert "score" in all_keys
        assert "hp" in all_keys
        assert "depth" in all_keys
        assert "roc.saliency_attenuation.peak_count" in all_keys
        assert "roc.saliency_attenuation.top_peak_strength" in all_keys
        assert "roc.saliency_attenuation.top_peak_shifted" in all_keys
        assert "game_num" in all_keys
        assert "game_tick" in all_keys
