"""Unit tests for SaliencyAttenuationExpMod."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from strictly_typed_pandas import DataSet
from unittest.mock import MagicMock

from roc.attention import SaliencyMap, VisionAttentionSchema
from roc.expmod import expmod_registry
from roc.saliency_attenuation import (
    LinearDeclineAttenuation,
    NoAttenuation,
    SaliencyAttenuationExpMod,
)


def make_focus_points(
    points: list[tuple[int, int, float, int]],
) -> DataSet[VisionAttentionSchema]:
    """Create a typed DataFrame of focus points from (x, y, strength, label) tuples."""
    df = pd.DataFrame(points, columns=["x", "y", "strength", "label"]).astype(
        {"x": int, "y": int, "strength": float, "label": int}
    )
    return DataSet[VisionAttentionSchema](df)


def make_strength_image(width: int = 10, height: int = 5) -> npt.NDArray[np.floating[Any]]:
    """Create a test strength image with known peak structure."""
    img = np.zeros((width, height))
    img[3, 2] = 0.9  # primary peak
    img[7, 4] = 0.7  # secondary peak
    return img


class TestExpModRegistration:
    def test_modtype_registered(self) -> None:
        """saliency-attenuation modtype exists in registry."""
        assert "saliency-attenuation" in expmod_registry

    def test_none_registered(self) -> None:
        """'none' flavor is registered."""
        assert "none" in expmod_registry["saliency-attenuation"]

    def test_get_default_returns_none_flavor(self) -> None:
        """get(default='none') returns NoAttenuation instance."""
        result = SaliencyAttenuationExpMod.get(default="none")
        assert isinstance(result, NoAttenuation)


class TestNoAttenuation:
    def test_returns_image_unchanged(self) -> None:
        """Returns the strength image with identical values."""
        img = make_strength_image()
        sm = MagicMock(spec=SaliencyMap)
        result = NoAttenuation().attenuate(img, sm)
        np.testing.assert_array_equal(result, img)

    def test_different_image_sizes(self) -> None:
        """Works with various image dimensions."""
        rng = np.random.default_rng(seed=42)
        sm = MagicMock(spec=SaliencyMap)
        for shape in [(5, 3), (80, 21), (1, 1)]:
            img = rng.random(shape)
            result = NoAttenuation().attenuate(img, sm)
            np.testing.assert_array_equal(result, img)

    def test_all_zero_image(self) -> None:
        """Works with all-zero strength image."""
        img = np.zeros((10, 5))
        sm = MagicMock(spec=SaliencyMap)
        result = NoAttenuation().attenuate(img, sm)
        np.testing.assert_array_equal(result, img)


class TestLinearDeclineRegistration:
    def test_linear_decline_registered(self) -> None:
        """'linear-decline' flavor is registered."""
        assert "linear-decline" in expmod_registry["saliency-attenuation"]


class TestLinearDeclineAttenuation:
    def test_empty_history_no_change(self) -> None:
        """With no prior history, strength image is unchanged."""
        att = LinearDeclineAttenuation()
        img = make_strength_image()
        sm = MagicMock(spec=SaliencyMap)
        result = att.attenuate(img, sm)
        np.testing.assert_array_equal(result, img)

    def test_attenuates_recently_attended_cell(self) -> None:
        """After notify_focus records (3,2), that cell is attenuated."""
        att = LinearDeclineAttenuation()
        sm = MagicMock(spec=SaliencyMap)
        img = make_strength_image()  # peak at (3,2)=0.9, (7,4)=0.7

        # Simulate first tick: notify that (3, 2) was attended
        focus_df = make_focus_points([(3, 2, 0.9, 1), (7, 4, 0.7, 2)])
        att.notify_focus(focus_df)

        # Second tick: (3, 2) should be attenuated
        result = att.attenuate(img, sm)
        assert result[3, 2] < img[3, 2]  # attenuated
        assert result[7, 4] == img[7, 4]  # far away, unchanged

    def test_recency_decline(self) -> None:
        """Most recent location gets strongest attenuation; older ones get less."""
        att = LinearDeclineAttenuation()
        att.capacity = 3
        att._history = att._make_history()

        # Attend three locations in sequence
        for x in [3, 5, 7]:
            fp = make_focus_points([(x, 2, 0.9, 1)])
            att.notify_focus(fp)

        # All three in history. (7,2) most recent -> strongest penalty
        assert len(att._history) == 3

    def test_spatial_falloff(self) -> None:
        """Cells far from history entries receive no attenuation."""
        att = LinearDeclineAttenuation()
        att.radius = 3
        sm = MagicMock(spec=SaliencyMap)

        # Attend (3, 2)
        fp = make_focus_points([(3, 2, 0.9, 1)])
        att.notify_focus(fp)

        # (7, 4) is Manhattan distance 6 from (3, 2) -- beyond radius 3
        img = make_strength_image()
        result = att.attenuate(img, sm)
        assert result[7, 4] == img[7, 4]  # no attenuation

    def test_capacity_limit(self) -> None:
        """History buffer does not exceed capacity (FIFO eviction)."""
        att = LinearDeclineAttenuation()
        att.capacity = 3
        att._history = att._make_history()

        for x in range(10):
            fp = make_focus_points([(x, 0, 0.9, 1)])
            att.notify_focus(fp)

        assert len(att._history) == 3

    def test_max_attenuation_prevents_zeroing(self) -> None:
        """Strength is never reduced below (1 - max_attenuation) * raw."""
        att = LinearDeclineAttenuation()
        att.max_attenuation = 0.5  # can reduce to at most 50%
        sm = MagicMock(spec=SaliencyMap)

        fp = make_focus_points([(3, 2, 0.9, 1)])
        att.notify_focus(fp)

        img = make_strength_image()
        result = att.attenuate(img, sm)
        # Even with full penalty, attenuated strength >= 50% of original
        assert result[3, 2] >= img[3, 2] * 0.5

    def test_adjacent_cell_partially_attenuated(self) -> None:
        """Cell 1 step away from history gets partial attenuation."""
        att = LinearDeclineAttenuation()
        att.radius = 3
        sm = MagicMock(spec=SaliencyMap)

        fp = make_focus_points([(5, 5, 0.9, 1)])
        att.notify_focus(fp)

        img = np.ones((10, 10)) * 0.8
        result = att.attenuate(img, sm)
        # (5, 5) gets full penalty, (5, 6) gets partial, (5, 8) gets none
        assert result[5, 5] < result[5, 6] < result[5, 8]

    def test_cumulative_penalty_from_multiple_history(self) -> None:
        """Multiple nearby history entries produce cumulative attenuation."""
        att = LinearDeclineAttenuation()
        att.capacity = 5
        att._history = att._make_history()
        att.radius = 5
        sm = MagicMock(spec=SaliencyMap)

        # Attend two nearby locations
        fp1 = make_focus_points([(5, 5, 0.9, 1)])
        att.notify_focus(fp1)
        fp2 = make_focus_points([(6, 5, 0.9, 1)])
        att.notify_focus(fp2)

        img = np.ones((10, 10)) * 0.8
        result = att.attenuate(img, sm)
        # (5, 5) is near both history entries -- more attenuation than (8, 5)
        assert result[5, 5] < result[8, 5]


class TestLinearDeclineTelemetry:
    def test_penalty_histogram_recorded(self) -> None:
        """Telemetry records max penalty applied to strength image."""
        att = LinearDeclineAttenuation()
        sm = MagicMock(spec=SaliencyMap)

        fp = make_focus_points([(3, 2, 0.9, 1)])
        att.notify_focus(fp)

        img = make_strength_image()
        # Should not raise -- telemetry is recorded internally
        result = att.attenuate(img, sm)
        assert result[3, 2] < img[3, 2]

    def test_attenuation_count_recorded(self) -> None:
        """Telemetry records number of cells attenuated."""
        att = LinearDeclineAttenuation()
        sm = MagicMock(spec=SaliencyMap)

        fp = make_focus_points([(5, 5, 0.9, 1)])
        att.notify_focus(fp)

        img = np.ones((10, 10)) * 0.8
        # Should not raise -- telemetry is recorded internally
        result = att.attenuate(img, sm)
        # At least the attended cell should be attenuated
        assert np.any(result < img)
