"""Integration tests for saliency attenuation over multi-tick sequences."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
from strictly_typed_pandas import DataSet

from roc.pipeline.attention.attention import SaliencyMap, VisionAttentionSchema
from roc.pipeline.attention.saliency_attenuation import (
    ActiveInferenceAttenuation,
    LinearDeclineAttenuation,
)


def make_focus_points(
    points: list[tuple[int, int, float, int]],
) -> DataSet[VisionAttentionSchema]:
    """Create a typed DataFrame of focus points from (x, y, strength, label) tuples."""
    df = pd.DataFrame(points, columns=["x", "y", "strength", "label"]).astype(
        {"x": int, "y": int, "strength": float, "label": int}
    )
    return DataSet[VisionAttentionSchema](df)


def make_three_peak_image() -> npt.NDArray[np.floating[Any]]:
    """Create a 20x10 image with 3 well-separated peaks."""
    img = np.zeros((20, 10))
    img[3, 2] = 0.9  # peak A
    img[10, 5] = 0.85  # peak B
    img[17, 8] = 0.8  # peak C
    return img


class TestLinearDeclineIntegration:
    def test_five_tick_sequence(self) -> None:
        """Run 5 ticks through linear-decline attenuation.

        Expected: peaks rotate rather than persevering on the same location.
        """
        att = LinearDeclineAttenuation()
        att.capacity = 3
        att._history = att._make_history()
        att.radius = 2
        sm = MagicMock(spec=SaliencyMap)

        img = make_three_peak_image()
        top_peaks: list[tuple[int, int]] = []

        for _ in range(5):
            result = att.attenuate(img, sm)
            # Find the top peak in the attenuated image
            peak_idx = np.unravel_index(np.argmax(result), result.shape)
            top_peaks.append((int(peak_idx[0]), int(peak_idx[1])))
            # Notify the attended peak
            fp = make_focus_points(
                [(int(peak_idx[0]), int(peak_idx[1]), float(result[peak_idx]), 1)]
            )
            att.notify_focus(fp)

        # Not all peaks should be the same -- rotation should occur
        unique_peaks = set(top_peaks)
        assert len(unique_peaks) > 1, f"Expected rotation but got: {top_peaks}"

    def test_returns_to_location_after_decay(self) -> None:
        """After capacity ticks, the oldest location is evicted and can re-emerge."""
        att = LinearDeclineAttenuation()
        att.capacity = 2
        att._history = att._make_history()
        att.radius = 2
        sm = MagicMock(spec=SaliencyMap)

        img = make_three_peak_image()

        # Attend peak A and B to fill history
        att.notify_focus(make_focus_points([(3, 2, 0.9, 1)]))
        att.notify_focus(make_focus_points([(10, 5, 0.85, 1)]))

        # Now history is full with A and B. A should be least recent.
        # Attend peak C to evict A from history
        att.notify_focus(make_focus_points([(17, 8, 0.8, 1)]))

        # Now A should no longer be attenuated
        result = att.attenuate(img, sm)
        # Peak A (3,2) should be at or near its original value
        assert result[3, 2] == img[3, 2]


class TestActiveInferenceIntegration:
    def test_explores_all_peaks(self) -> None:
        """Over 15 ticks with 3 equal saliency peaks, all 3 emerge as
        top-ranked peak at least once."""
        att = ActiveInferenceAttenuation()
        att.saliency_weight = 0.0  # pure epistemic drive
        att.max_attenuation = 0.95
        sm = MagicMock(spec=SaliencyMap)

        # Each location has unique features
        feature_map = {
            (3, 2): "feature_A",
            (10, 5): "feature_B",
            (17, 8): "feature_C",
        }
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s, k=(x, y): feature_map.get(k, "f"))]

        # Use equal-strength peaks so epistemic value is the deciding factor
        img = np.zeros((20, 10))
        img[3, 2] = 0.9
        img[10, 5] = 0.9
        img[17, 8] = 0.9

        top_peaks: set[tuple[int, int]] = set()

        for _ in range(15):
            result = att.attenuate(img, sm)
            peak_idx = np.unravel_index(np.argmax(result), result.shape)
            peak = (int(peak_idx[0]), int(peak_idx[1]))
            top_peaks.add(peak)

            # Notify the attended peak
            fp = make_focus_points([(peak[0], peak[1], float(result[peak_idx]), 1)])
            att._last_saliency_map = sm
            att.notify_focus(fp)

            # Propagate uncertainty for all beliefs to simulate time passing
            for belief in att._beliefs.values():
                belief.propagate(omega=0.3, ticks_elapsed=1)

        # All 3 peaks should have been explored
        expected_peaks = {(3, 2), (10, 5), (17, 8)}
        assert top_peaks == expected_peaks, f"Expected all 3 peaks but got: {top_peaks}"

    def test_volatile_environment_faster_return(self) -> None:
        """When features change at a location, agent revisits sooner."""
        att = ActiveInferenceAttenuation()
        att.saliency_weight = 0.0  # pure epistemic

        sm = MagicMock(spec=SaliencyMap)
        call_count = [0]

        def get_val_changing(x: int, y: int) -> list[Any]:
            call_count[0] += 1
            return [MagicMock(__str__=lambda s, c=call_count[0]: f"f_{c}")]

        sm.get_val = get_val_changing

        img = np.zeros((10, 5))
        img[3, 2] = 0.9

        # Observe once
        fp = make_focus_points([(3, 2, 0.9, 1)])
        att._last_saliency_map = sm
        att.notify_focus(fp)

        # With changing features, each observation creates prediction error
        # which increases volatility, speeding up entropy recovery.
        # After propagation, the cell should be less attenuated.
        for belief in att._beliefs.values():
            belief.propagate(omega=2.0, ticks_elapsed=3)

        result = att.attenuate(img, sm)
        # Should have recovered significantly
        assert result[3, 2] > img[3, 2] * 0.7

    def test_stable_environment_longer_inhibition(self) -> None:
        """When features are stable, agent takes longer to revisit."""
        att = ActiveInferenceAttenuation()
        att.saliency_weight = 0.0  # pure epistemic
        att.max_attenuation = 0.95

        sm = MagicMock(spec=SaliencyMap)
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: "stable_feature")]

        img = np.zeros((10, 5))
        img[3, 2] = 0.9

        # Observe same features multiple times to create very peaked belief
        fp = make_focus_points([(3, 2, 0.9, 1)])
        att._last_saliency_map = sm
        for _ in range(5):
            att.notify_focus(fp)

        # With very low omega, entropy recovery is minimal
        for belief in att._beliefs.values():
            belief.propagate(omega=0.001, ticks_elapsed=1)

        result = att.attenuate(img, sm)
        # Should still be significantly attenuated (below 80% of original)
        assert result[3, 2] < img[3, 2] * 0.8
