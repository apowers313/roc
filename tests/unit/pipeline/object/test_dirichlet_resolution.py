# mypy: disable-error-code="no-untyped-def"

"""Unit tests for DirichletCategoricalResolution."""

import math
from unittest.mock import MagicMock, patch

import pytest

from roc.db.graphdb import NodeId
from roc.perception.location import XLoc, YLoc
from roc.expmods.object_resolution.dirichlet_categorical import DirichletCategoricalResolution
from roc.pipeline.object.object import (
    Object,
    ResolutionContext,
    _feature_to_objects,
)
from roc.perception.base import FeatureKind


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


# A realistic global vocabulary -- in a real system there are many more feature
# strings than any single observation contains.  This dilutes the new-object
# uniform likelihood so that objects with strong alpha profiles can win.
REALISTIC_VOCAB = {f"Feat({i})" for i in range(30)}


def make_feature_node(label: str, str_repr: str) -> MagicMock:
    """Create a mock FeatureNode with label and string representation."""
    fn = MagicMock()
    fn.labels = {label, "FeatureNode"}
    fn.kind = FeatureKind.PHYSICAL
    fn.configure_mock(**{"__str__": MagicMock(return_value=str_repr)})
    return fn


def make_object_with_position(x: int | None, y: int | None, tick: int) -> Object:
    """Create an Object with last-seen position and tick."""
    o = Object()
    o.last_x = XLoc(x) if x is not None else None
    o.last_y = YLoc(y) if y is not None else None
    o.last_tick = tick
    return o


def setup_candidate_return(feature_nodes: list[MagicMock], obj: Object):
    """Populate the reverse index so _find_candidates returns obj."""
    for fn in feature_nodes:
        _feature_to_objects[fn.id].add(obj.id)


def setup_multi_candidate_return(feature_nodes: list[MagicMock], objs: list[Object]):
    """Populate the reverse index so _find_candidates returns multiple objects."""
    for fn in feature_nodes:
        for obj in objs:
            _feature_to_objects[fn.id].add(obj.id)


def setup_no_candidates(feature_nodes: list[MagicMock]):
    """Ensure _find_candidates returns nothing for these feature nodes."""
    for fn in feature_nodes:
        _feature_to_objects.pop(fn.id, None)


class TestColdStart:
    def test_no_existing_objects_creates_new(self):
        """First observation ever -- 'new object' hypothesis wins."""
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")
        fn2 = make_feature_node("ColorNode", "ColorNode(red)")
        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=1)

        setup_no_candidates([fn1, fn2])

        result = resolution.resolve([fn1, fn2], fg, ctx)
        assert result is None  # None means "create new object"


class TestExactMatch:
    def test_exact_feature_match_returns_object(self):
        """Observation exactly matches a known Object's features.

        With a realistic vocab, the object's strong alphas dominate the
        uniform new-object likelihood.
        """
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")
        fn2 = make_feature_node("ColorNode", "ColorNode(red)")

        obj = make_object_with_position(5, 5, tick=10)
        resolution._alphas[obj.id] = {
            "SingleNode(a)": 11.0,
            "ColorNode(red)": 11.0,
        }
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)", "ColorNode(red)"}

        setup_candidate_return([fn1, fn2], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn1, fn2], fg, ctx)
        assert result is obj


class TestPartialMatch:
    def test_two_of_three_features_still_matches(self):
        """2 of 3 features match -- should match with moderate posterior."""
        resolution = DirichletCategoricalResolution()
        fn_a = make_feature_node("SingleNode", "SingleNode(a)")
        fn_b = make_feature_node("ColorNode", "ColorNode(red)")
        fn_d = make_feature_node("ShapeNode", "ShapeNode(square)")

        obj = make_object_with_position(5, 5, tick=10)
        resolution._alphas[obj.id] = {
            "SingleNode(a)": 11.0,
            "ColorNode(red)": 11.0,
            "ShapeNode(circle)": 11.0,
        }
        resolution._global_vocab = REALISTIC_VOCAB | {
            "SingleNode(a)",
            "ColorNode(red)",
            "ShapeNode(circle)",
            "ShapeNode(square)",
        }

        setup_candidate_return([fn_a, fn_b, fn_d], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn_a, fn_b, fn_d], fg, ctx)
        assert result is obj


class TestNoMatch:
    def test_entirely_different_features_creates_new(self):
        """Features are entirely unlike any known Object -- new wins."""
        resolution = DirichletCategoricalResolution()
        fn_x = make_feature_node("SingleNode", "SingleNode(x)")
        fn_y = make_feature_node("ColorNode", "ColorNode(blue)")
        fn_z = make_feature_node("ShapeNode", "ShapeNode(triangle)")

        obj = make_object_with_position(5, 5, tick=10)
        # Strong alphas for completely different features
        resolution._alphas[obj.id] = {
            "SingleNode(a)": 21.0,
            "ColorNode(red)": 21.0,
            "ShapeNode(circle)": 21.0,
        }
        resolution._global_vocab = {
            "SingleNode(a)",
            "ColorNode(red)",
            "ShapeNode(circle)",
            "SingleNode(x)",
            "ColorNode(blue)",
            "ShapeNode(triangle)",
        }

        setup_candidate_return([fn_x, fn_y, fn_z], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn_x, fn_y, fn_z], fg, ctx)
        assert result is None  # new object wins


class TestUniformPriors:
    def test_identical_candidates_match_one(self):
        """Two objects with identical features and uniform priors -- one should win.

        With uniform priors (no spatial/temporal), both candidates have the
        same posterior. The system should still pick one (the MAP winner) or
        return None if the posterior is split below threshold.
        """
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        obj_a = make_object_with_position(5, 5, tick=10)
        obj_b = make_object_with_position(50, 50, tick=10)

        alphas = {"SingleNode(a)": 11.0}
        resolution._alphas[obj_a.id] = alphas.copy()
        resolution._alphas[obj_b.id] = alphas.copy()
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)"}

        setup_multi_candidate_return([fn1], [obj_a, obj_b])

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(6), y=YLoc(6), tick=11)
        result = resolution.resolve([fn1], fg, ctx)
        # With 2 identical candidates + new, each gets ~1/3 posterior.
        # That's below 0.5 threshold, so result is None (low confidence).
        assert result is None

    def test_stronger_alphas_win(self):
        """Object with stronger alpha profile wins over weaker one."""
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        obj_strong = make_object_with_position(5, 5, tick=10)
        obj_weak = make_object_with_position(50, 50, tick=10)

        resolution._alphas[obj_strong.id] = {"SingleNode(a)": 50.0}
        resolution._alphas[obj_weak.id] = {"SingleNode(a)": 2.0}
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)"}

        setup_multi_candidate_return([fn1], [obj_strong, obj_weak])

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(6), y=YLoc(6), tick=11)
        result = resolution.resolve([fn1], fg, ctx)
        assert result is obj_strong


class TestAlphaAccumulation:
    def test_well_characterized_object_resists_mismatch(self):
        """After 20 observations, object should resist different features."""
        resolution = DirichletCategoricalResolution()
        fn_d = make_feature_node("SingleNode", "SingleNode(d)")
        fn_e = make_feature_node("ColorNode", "ColorNode(green)")
        fn_f = make_feature_node("ShapeNode", "ShapeNode(star)")

        obj = make_object_with_position(5, 5, tick=99)
        resolution._alphas[obj.id] = {
            "SingleNode(a)": 21.0,
            "ColorNode(red)": 21.0,
            "ShapeNode(circle)": 21.0,
        }
        resolution._global_vocab = {
            "SingleNode(a)",
            "ColorNode(red)",
            "ShapeNode(circle)",
            "SingleNode(d)",
            "ColorNode(green)",
            "ShapeNode(star)",
        }

        setup_candidate_return([fn_d, fn_e, fn_f], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=100)
        result = resolution.resolve([fn_d, fn_e, fn_f], fg, ctx)
        assert result is None  # mismatch too strong


class TestConfidenceThreshold:
    def test_ambiguous_posterior_creates_new(self):
        """Best posterior below threshold creates new with low_confidence."""
        resolution = DirichletCategoricalResolution()
        resolution.confidence_threshold = 0.99  # very high threshold

        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        obj = make_object_with_position(5, 5, tick=10)
        # Weak alphas -- only slightly above prior
        resolution._alphas[obj.id] = {"SingleNode(a)": 2.0}
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)"}

        setup_candidate_return([fn1], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn1], fg, ctx)
        assert result is None


class TestAlphaUpdate:
    def test_alpha_updated_after_match(self):
        """After resolve() matches an object, its alphas are incremented."""
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")
        fn2 = make_feature_node("ColorNode", "ColorNode(red)")

        obj = make_object_with_position(5, 5, tick=10)
        resolution._alphas[obj.id] = {
            "SingleNode(a)": 11.0,
            "ColorNode(red)": 11.0,
        }
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)", "ColorNode(red)"}

        setup_candidate_return([fn1, fn2], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn1, fn2], fg, ctx)
        assert result is obj
        # Alphas should have been incremented by 1
        assert resolution._alphas[obj.id]["SingleNode(a)"] == pytest.approx(12.0)
        assert resolution._alphas[obj.id]["ColorNode(red)"] == pytest.approx(12.0)

    def test_new_object_gets_initial_alphas(self):
        """initialize_alphas sets up prior alphas for a new object."""
        resolution = DirichletCategoricalResolution()
        obj_id = NodeId(42)
        resolution.initialize_alphas(obj_id, ["SingleNode(a)", "ColorNode(red)"])
        assert resolution._alphas[obj_id]["SingleNode(a)"] == pytest.approx(
            resolution.prior_alpha + 1.0
        )
        assert resolution._alphas[obj_id]["ColorNode(red)"] == pytest.approx(
            resolution.prior_alpha + 1.0
        )


class TestFeatureExclusion:
    def test_excluded_features_ignored_in_likelihood(self):
        """Features with excluded labels don't affect likelihood."""
        resolution = DirichletCategoricalResolution()
        resolution.excluded_feature_labels = {"PositionNode"}

        fn1 = make_feature_node("SingleNode", "SingleNode(a)")
        fn_pos = make_feature_node("PositionNode", "PositionNode(5,5)")

        obj = make_object_with_position(5, 5, tick=10)
        resolution._alphas[obj.id] = {"SingleNode(a)": 11.0}
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)"}

        setup_candidate_return([fn1, fn_pos], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn1, fn_pos], fg, ctx)
        assert result is obj  # PositionNode excluded, only SingleNode(a) considered


class TestNumericalStability:
    def test_no_nan_with_many_features(self):
        """Log-space arithmetic stays stable with many features."""
        resolution = DirichletCategoricalResolution()
        features = [make_feature_node("SingleNode", f"SingleNode({i})") for i in range(20)]

        obj = make_object_with_position(5, 5, tick=10)
        resolution._alphas[obj.id] = {f"SingleNode({i})": 5.0 for i in range(20)}
        resolution._global_vocab = {f"SingleNode({i})" for i in range(25)}

        setup_candidate_return(features, obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve(features, fg, ctx)
        # Should not raise and should return a valid result (not None due to NaN)
        assert result is obj

    def test_no_nan_with_zero_candidates(self):
        """Edge case: zero candidates should return None cleanly."""
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")
        setup_no_candidates([fn1])

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=1)
        result = resolution.resolve([fn1], fg, ctx)
        assert result is None

    def test_no_nan_with_single_candidate(self):
        """Edge case: exactly one candidate, posteriors still valid."""
        resolution = DirichletCategoricalResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        obj = make_object_with_position(5, 5, tick=10)
        resolution._alphas[obj.id] = {"SingleNode(a)": 11.0}
        resolution._global_vocab = REALISTIC_VOCAB | {"SingleNode(a)"}

        setup_candidate_return([fn1], obj)

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
        result = resolution.resolve([fn1], fg, ctx)
        assert result is obj


class TestComputePriors:
    def test_spatial_weight_decay(self):
        """Spatial weight decreases with manhattan distance."""
        resolution = DirichletCategoricalResolution()
        resolution.spatial_scale = 3.0
        obj = make_object_with_position(0, 0, tick=0)
        ctx = ResolutionContext(x=XLoc(3), y=YLoc(0), tick=1)
        weight = resolution._spatial_weight(obj, ctx)
        expected = math.exp(-3.0 / 3.0)
        assert abs(weight - expected) < 1e-6

    def test_temporal_weight_decay(self):
        """Temporal weight decreases with tick gap."""
        resolution = DirichletCategoricalResolution()
        resolution.temporal_scale = 50.0
        obj = make_object_with_position(0, 0, tick=50)
        ctx = ResolutionContext(x=XLoc(0), y=YLoc(0), tick=100)
        weight = resolution._temporal_weight(obj, ctx)
        expected = math.exp(-50.0 / 50.0)
        assert abs(weight - expected) < 1e-6

    def test_no_position_gets_uniform_spatial(self):
        """Object with no last position gets spatial weight 1.0."""
        resolution = DirichletCategoricalResolution()
        obj = make_object_with_position(None, None, tick=0)
        ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=1)
        weight = resolution._spatial_weight(obj, ctx)
        assert weight == pytest.approx(1.0)


class TestComputeLikelihoods:
    def test_seen_features_higher_likelihood(self):
        """Features with high alpha get higher likelihood."""
        resolution = DirichletCategoricalResolution()
        resolution._global_vocab = {"A", "B", "C"}

        obj_id = NodeId(1)
        resolution._alphas[obj_id] = {"A": 10.0, "B": 10.0, "C": 1.0}

        ll_ab = resolution._log_likelihood_for_object(obj_id, ["A", "B"])
        ll_ac = resolution._log_likelihood_for_object(obj_id, ["A", "C"])
        # A+B should have higher likelihood than A+C since B has alpha=10 vs C=1
        assert ll_ab > ll_ac

    def test_new_object_likelihood_uses_uniform(self):
        """New object model assigns uniform probability across vocab."""
        resolution = DirichletCategoricalResolution()
        resolution.prior_alpha = 1.0
        resolution._global_vocab = {"A", "B", "C", "D", "E"}

        ll = resolution._log_likelihood_new(["A", "B"])
        # Each symbol has uniform probability 1/5
        expected = math.log(1.0 / 5.0) + math.log(1.0 / 5.0)
        assert abs(ll - expected) < 1e-6
