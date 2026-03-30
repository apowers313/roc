"""Unit tests for active inference saliency attenuation internals."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from strictly_typed_pandas import DataSet
from unittest.mock import MagicMock

from roc.pipeline.attention.attention import SaliencyMap, VisionAttentionSchema
from roc.framework.expmod import expmod_registry
from roc.pipeline.attention.saliency_attenuation import (
    ActiveInferenceAttenuation,
    LocationBelief,
    StateVocabulary,
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


class TestStateVocabulary:
    def test_encode_new_features(self) -> None:
        """New feature set gets a unique state ID."""
        vocab = StateVocabulary(max_states=64)
        features = ["SingleNode(a)", "ColorNode(red)"]
        sid = vocab.encode(features)
        assert isinstance(sid, int)
        assert 0 <= sid < 64

    def test_encode_same_features_same_id(self) -> None:
        """Same feature set always maps to same state ID."""
        vocab = StateVocabulary(max_states=64)
        features = ["SingleNode(a)", "ColorNode(red)"]
        sid1 = vocab.encode(features)
        sid2 = vocab.encode(features)
        assert sid1 == sid2

    def test_encode_different_features_different_id(self) -> None:
        """Different feature sets get different state IDs."""
        vocab = StateVocabulary(max_states=64)
        sid1 = vocab.encode(["SingleNode(a)"])
        sid2 = vocab.encode(["SingleNode(b)"])
        assert sid1 != sid2

    def test_encode_order_independent(self) -> None:
        """Feature set order does not affect state ID."""
        vocab = StateVocabulary(max_states=64)
        sid1 = vocab.encode(["SingleNode(a)", "ColorNode(red)"])
        sid2 = vocab.encode(["ColorNode(red)", "SingleNode(a)"])
        assert sid1 == sid2

    def test_capacity_limit(self) -> None:
        """State IDs stay within [0, max_states)."""
        vocab = StateVocabulary(max_states=4)
        ids = set()
        for i in range(10):
            sid = vocab.encode([f"feature_{i}"])
            ids.add(sid)
        assert all(0 <= sid < 4 for sid in ids)

    def test_vocab_size(self) -> None:
        """size property tracks unique entries."""
        vocab = StateVocabulary(max_states=64)
        assert vocab.size == 0
        vocab.encode(["a"])
        assert vocab.size == 1
        vocab.encode(["b"])
        assert vocab.size == 2
        vocab.encode(["a"])  # duplicate
        assert vocab.size == 2


class TestLocationBelief:
    def test_initial_belief_is_uniform(self) -> None:
        """New LocationBelief has uniform distribution over states."""
        belief = LocationBelief.uniform(n_states=8)
        assert belief.q_s.shape == (8,)
        assert np.allclose(belief.q_s, 1.0 / 8)

    def test_entropy_of_uniform(self) -> None:
        """Uniform distribution has maximum entropy."""
        belief = LocationBelief.uniform(n_states=8)
        expected = math.log(8)
        assert belief.entropy() == pytest.approx(expected, rel=1e-6)

    def test_entropy_of_peaked(self) -> None:
        """Near-deterministic distribution has near-zero entropy."""
        belief = LocationBelief.uniform(n_states=8)
        belief.q_s = np.zeros(8)
        belief.q_s[0] = 1.0
        assert belief.entropy() == pytest.approx(0.0, abs=1e-9)

    def test_observe_reduces_entropy(self) -> None:
        """Observing a state reduces entropy at that location."""
        belief = LocationBelief.uniform(n_states=8)
        initial_entropy = belief.entropy()
        belief.observe(state_id=3, zeta=2.0)
        assert belief.entropy() < initial_entropy

    def test_observe_concentrates_on_state(self) -> None:
        """After observation, belief is concentrated on observed state."""
        belief = LocationBelief.uniform(n_states=8)
        belief.observe(state_id=3, zeta=5.0)
        assert belief.q_s[3] > belief.q_s[0]
        assert np.argmax(belief.q_s) == 3

    def test_propagate_increases_entropy(self) -> None:
        """Uncertainty propagation increases entropy toward uniform."""
        belief = LocationBelief.uniform(n_states=8)
        belief.observe(state_id=3, zeta=5.0)
        peaked_entropy = belief.entropy()

        belief.propagate(omega=0.5, ticks_elapsed=5)
        assert belief.entropy() > peaked_entropy

    def test_propagate_rate_depends_on_omega(self) -> None:
        """Higher omega (more volatility) -> faster entropy increase."""
        b_low = LocationBelief.uniform(n_states=8)
        b_high = LocationBelief.uniform(n_states=8)
        # Same peaked state
        b_low.observe(state_id=3, zeta=5.0)
        b_high.observe(state_id=3, zeta=5.0)

        b_low.propagate(omega=0.1, ticks_elapsed=5)
        b_high.propagate(omega=1.0, ticks_elapsed=5)

        assert b_high.entropy() > b_low.entropy()

    def test_propagate_rate_depends_on_ticks(self) -> None:
        """More elapsed ticks -> more entropy increase."""
        b_short = LocationBelief.uniform(n_states=8)
        b_long = LocationBelief.uniform(n_states=8)
        b_short.observe(state_id=3, zeta=5.0)
        b_long.observe(state_id=3, zeta=5.0)

        b_short.propagate(omega=0.5, ticks_elapsed=1)
        b_long.propagate(omega=0.5, ticks_elapsed=10)

        assert b_long.entropy() > b_short.entropy()

    def test_propagate_bounded_by_uniform(self) -> None:
        """Propagation never exceeds uniform entropy."""
        belief = LocationBelief.uniform(n_states=8)
        belief.observe(state_id=3, zeta=5.0)
        belief.propagate(omega=10.0, ticks_elapsed=1000)

        max_entropy = math.log(8)
        assert belief.entropy() <= max_entropy + 1e-9


class TestActiveInferenceAttenuation:
    def test_registered(self) -> None:
        """'active-inference' is registered in the ExpMod registry."""
        assert "active-inference" in expmod_registry["saliency-attenuation"]

    def test_no_prior_observations_no_change(self) -> None:
        """With no observation history, strength image is unchanged."""
        att = ActiveInferenceAttenuation()
        img = make_strength_image()
        sm = MagicMock(spec=SaliencyMap)
        result = att.attenuate(img, sm)
        np.testing.assert_array_equal(result, img)

    def test_ior_effect(self) -> None:
        """After observing location (3,2), that cell is attenuated."""
        att = ActiveInferenceAttenuation()
        att.saliency_weight = 0.3  # bias toward epistemic
        sm = MagicMock(spec=SaliencyMap)
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: f"f_{x}_{y}")]

        # Tick 1: notify that (3, 2) was attended
        fp = make_focus_points([(3, 2, 0.9, 1), (7, 4, 0.7, 2)])
        att._last_saliency_map = sm
        att.notify_focus(fp)

        # Tick 2: (3, 2) entropy is low -> attenuated
        img = make_strength_image()
        result = att.attenuate(img, sm)
        assert result[3, 2] < img[3, 2]

    def test_entropy_recovery_reduces_attenuation(self) -> None:
        """After enough ticks, previously attended location is less attenuated."""
        att = ActiveInferenceAttenuation()
        att.saliency_weight = 0.0  # pure epistemic
        sm = MagicMock(spec=SaliencyMap)
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: "f_a")]

        # Observe (3, 2)
        fp = make_focus_points([(3, 2, 0.9, 1)])
        att._last_saliency_map = sm
        att.notify_focus(fp)

        # Simulate many ticks passing by propagating uncertainty
        for belief in att._beliefs.values():
            belief.propagate(omega=1.0, ticks_elapsed=100)

        # Now (3, 2) should have high entropy -> minimal attenuation
        img = make_strength_image()
        result = att.attenuate(img, sm)
        assert result[3, 2] > img[3, 2] * 0.8  # nearly no attenuation

    def test_saliency_weight_1_no_attenuation(self) -> None:
        """With saliency_weight=1.0, no epistemic attenuation occurs."""
        att = ActiveInferenceAttenuation()
        att.saliency_weight = 1.0
        sm = MagicMock(spec=SaliencyMap)
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: f"f_{x}")]

        # Observe (3, 2)
        fp = make_focus_points([(3, 2, 0.9, 1)])
        att._last_saliency_map = sm
        att.notify_focus(fp)

        # Even after observation, no attenuation with weight=1.0
        img = make_strength_image()
        result = att.attenuate(img, sm)
        np.testing.assert_array_equal(result, img)

    def test_high_volatility_short_ior(self) -> None:
        """With high omega, entropy recovers quickly -- less attenuation."""
        att_low = ActiveInferenceAttenuation()
        att_low.saliency_weight = 0.0
        att_high = ActiveInferenceAttenuation()
        att_high.saliency_weight = 0.0

        sm = MagicMock(spec=SaliencyMap)
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: "f_a")]

        # Both observe same location
        fp = make_focus_points([(3, 2, 0.9, 1)])
        att_low._last_saliency_map = sm
        att_low.notify_focus(fp)
        att_high._last_saliency_map = sm
        att_high.notify_focus(fp)

        # Propagate with different omega values
        for belief in att_low._beliefs.values():
            belief.propagate(omega=0.01, ticks_elapsed=5)
        for belief in att_high._beliefs.values():
            belief.propagate(omega=10.0, ticks_elapsed=5)

        img = make_strength_image()
        result_low = att_low.attenuate(img, sm)
        result_high = att_high.attenuate(img, sm)

        # High-omega agent should have less attenuation (higher entropy recovery)
        assert result_high[3, 2] > result_low[3, 2]

    def test_max_locations_eviction(self) -> None:
        """Belief dict respects max_locations via LRU eviction."""
        att = ActiveInferenceAttenuation()
        att.max_locations = 3
        sm = MagicMock(spec=SaliencyMap)
        sm.get_val = lambda x, y: [MagicMock(__str__=lambda s: "f")]

        for i in range(10):
            fp = make_focus_points([(i * 10, 0, 0.9, 1)])
            att._last_saliency_map = sm
            att.notify_focus(fp)

        assert len(att._beliefs) <= 3
