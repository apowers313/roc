# mypy: disable-error-code="no-untyped-def"

"""Integration tests for Dirichlet-Categorical resolution pipeline.

These tests exercise the full resolution pipeline through multi-observation
sequences without a live game, verifying that alpha accumulation, spatial
disambiguation, and telemetry all work correctly end-to-end.
"""

import math
from typing import Any
from unittest.mock import MagicMock, patch

from roc.location import XLoc, YLoc
from roc.object import (
    DirichletCategoricalResolution,
    Object,
    ResolutionContext,
    _feature_to_objects,
)


def make_feature_node(label: str, str_repr: str) -> MagicMock:
    """Create a mock FeatureNode with label and string representation."""
    fn = MagicMock()
    fn.labels = {label, "FeatureNode"}
    fn.configure_mock(**{"__str__": MagicMock(return_value=str_repr)})
    return fn


def make_object_with_position(x: int | None, y: int | None, tick: int) -> Object:
    """Create an Object with last-seen position and tick."""
    o = Object()
    o.last_x = XLoc(x) if x is not None else None
    o.last_y = YLoc(y) if y is not None else None
    o.last_tick = tick
    return o


def setup_no_candidates(feature_nodes: list[MagicMock]):
    """Ensure _find_candidates returns nothing for these feature nodes."""
    for fn in feature_nodes:
        _feature_to_objects.pop(fn.id, None)


def setup_candidate_return(feature_nodes: list[MagicMock], objs: list[Object]):
    """Populate the reverse index so _find_candidates returns the given objects."""
    for fn in feature_nodes:
        for obj in objs:
            _feature_to_objects[fn.id].add(obj.id)


# A realistic global vocabulary to dilute new-object uniform likelihood.
REALISTIC_VOCAB = {f"Feat({i})" for i in range(30)}


class TestDirichletIntegrationFlow:
    """End-to-end tests exercising multi-observation resolution sequences."""

    @patch("roc.graphdb.GraphDB.singleton")
    def test_full_resolution_cycle(self, mock_db_cls):
        """End-to-end: create objects, resolve matches, verify alphas grow."""
        mock_db = MagicMock()
        mock_db.strict_schema = False
        mock_db.strict_schema_warns = False
        mock_db_cls.return_value = mock_db

        resolution = DirichletCategoricalResolution()
        fg = MagicMock()

        # Entity A features
        fn_a1 = make_feature_node("SingleNode", "SingleNode(a)")
        fn_a2 = make_feature_node("ColorNode", "ColorNode(red)")

        # Entity B features (completely different)
        fn_b1 = make_feature_node("SingleNode", "SingleNode(b)")
        fn_b2 = make_feature_node("ColorNode", "ColorNode(blue)")

        # 1. First observation: no candidates -> new object
        setup_no_candidates([fn_a1, fn_a2])
        ctx1 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=1)
        result1 = resolution.resolve([fn_a1, fn_a2], fg, ctx1)
        assert result1 is None  # create new object

        # Simulate what ObjectResolver does: initialize alphas for "obj_a"
        obj_a = make_object_with_position(5, 5, tick=1)
        resolution.initialize_alphas(obj_a.id, ["SingleNode(a)", "ColorNode(red)"])
        # Seed vocab to be realistic
        resolution._global_vocab.update(REALISTIC_VOCAB)

        # 2. Second observation: same features, same position -> should match obj_a
        setup_candidate_return([fn_a1, fn_a2], [obj_a])
        ctx2 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=2)
        result2 = resolution.resolve([fn_a1, fn_a2], fg, ctx2)
        assert result2 is obj_a

        # 3. Third observation: different features -> new object
        setup_no_candidates([fn_b1, fn_b2])
        ctx3 = ResolutionContext(x=XLoc(20), y=YLoc(20), tick=3)
        result3 = resolution.resolve([fn_b1, fn_b2], fg, ctx3)
        assert result3 is None  # create new object

        obj_b = make_object_with_position(20, 20, tick=3)
        resolution.initialize_alphas(obj_b.id, ["SingleNode(b)", "ColorNode(blue)"])

        # 4. Fourth observation: same features as first -> should match obj_a
        setup_candidate_return([fn_a1, fn_a2], [obj_a])
        obj_a.last_x = XLoc(5)
        obj_a.last_y = YLoc(5)
        obj_a.last_tick = 2
        ctx4 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=4)
        result4 = resolution.resolve([fn_a1, fn_a2], fg, ctx4)
        assert result4 is obj_a

        # Verify alpha vectors reflect observation counts
        # initial (prior+1=2.0) + 2 match updates = 4.0
        assert resolution._alphas[obj_a.id]["SingleNode(a)"] == 4.0
        assert resolution._alphas[obj_a.id]["ColorNode(red)"] == 4.0

    @patch("roc.graphdb.GraphDB.singleton")
    def test_warmup_then_stable(self, mock_db_cls):
        """Simulate 50+ observations and verify match rate stabilizes."""
        mock_db = MagicMock()
        mock_db.strict_schema = False
        mock_db.strict_schema_warns = False
        mock_db_cls.return_value = mock_db

        resolution = DirichletCategoricalResolution()
        resolution._global_vocab = REALISTIC_VOCAB.copy()
        fg = MagicMock()

        # Create 5 entities with distinct feature profiles
        entities: list[dict[str, Any]] = []
        for i in range(5):
            fn1 = make_feature_node("SingleNode", f"SingleNode(ent{i})")
            fn2 = make_feature_node("ColorNode", f"ColorNode(color{i})")
            feature_strs = [f"SingleNode(ent{i})", f"ColorNode(color{i})"]
            entities.append({"fns": [fn1, fn2], "strs": feature_strs, "obj": None})

        resolution._global_vocab.update(f for e in entities for f in e["strs"])

        matches_after_warmup = 0
        total_after_warmup = 0

        # Simulate 100 observations cycling through entities
        for tick in range(1, 101):
            ent = entities[tick % 5]
            fns = ent["fns"]
            ctx = ResolutionContext(x=XLoc(10 * (tick % 5)), y=YLoc(10), tick=tick)

            if ent["obj"] is None:
                # First time seeing this entity -- set up no candidates
                setup_no_candidates(fns)
                result = resolution.resolve(fns, fg, ctx)
                assert result is None
                obj = make_object_with_position(int(ctx.x), int(ctx.y), tick)
                resolution.initialize_alphas(obj.id, ent["strs"])
                ent["obj"] = obj
            else:
                obj = ent["obj"]
                obj.last_tick = tick - 5  # last seen 5 ticks ago
                obj.last_x = ctx.x
                obj.last_y = ctx.y
                setup_candidate_return(fns, [obj])
                result = resolution.resolve(fns, fg, ctx)

                if tick > 10:  # after warmup
                    total_after_warmup += 1
                    if result is obj:
                        matches_after_warmup += 1

        # After warmup, match rate should be > 70%
        match_rate = matches_after_warmup / total_after_warmup
        assert match_rate > 0.70, f"Match rate {match_rate:.2%} <= 70%"

    @patch("roc.graphdb.GraphDB.singleton")
    def test_object_count_does_not_explode(self, mock_db_cls):
        """Verify object count stays bounded for repeated observations."""
        mock_db = MagicMock()
        mock_db.strict_schema = False
        mock_db.strict_schema_warns = False
        mock_db_cls.return_value = mock_db

        resolution = DirichletCategoricalResolution()
        resolution._global_vocab = REALISTIC_VOCAB.copy()
        fg = MagicMock()

        # 3 entities, each observed 50 times
        entity_features = [[f"SingleNode(e{i})", f"ColorNode(c{i})"] for i in range(3)]
        resolution._global_vocab.update(f for feats in entity_features for f in feats)

        objects_created: list[Object] = []

        for round_num in range(50):
            for ent_idx in range(3):
                tick = round_num * 3 + ent_idx + 1
                feat_strs = entity_features[ent_idx]
                fns = [
                    make_feature_node("SingleNode", feat_strs[0]),
                    make_feature_node("ColorNode", feat_strs[1]),
                ]
                ctx = ResolutionContext(x=XLoc(10 * ent_idx), y=YLoc(10), tick=tick)

                # Find existing objects for this entity
                existing = [
                    o
                    for o in objects_created
                    if resolution._alphas.get(o.id, {}).get(feat_strs[0], 0) > 0
                ]

                if existing:
                    obj = existing[0]
                    obj.last_tick = tick - 3
                    obj.last_x = ctx.x
                    obj.last_y = ctx.y
                    setup_candidate_return(fns, [obj])
                else:
                    setup_no_candidates(fns)

                result = resolution.resolve(fns, fg, ctx)
                if result is None:
                    new_obj = make_object_with_position(int(ctx.x), int(ctx.y), tick)
                    resolution.initialize_alphas(new_obj.id, feat_strs)
                    objects_created.append(new_obj)

        # Should produce approximately 3 objects, not 150
        assert len(objects_created) <= 6, f"Created {len(objects_created)} objects, expected ~3"

    @patch("roc.graphdb.GraphDB.singleton")
    def test_spatial_helps_disambiguate_identical_features(self, mock_db_cls):
        """Two entities with same features at different positions."""
        mock_db = MagicMock()
        mock_db.strict_schema = False
        mock_db.strict_schema_warns = False
        mock_db_cls.return_value = mock_db

        resolution = DirichletCategoricalResolution()
        resolution._global_vocab = REALISTIC_VOCAB.copy()
        fg = MagicMock()

        feat_strs = ["SingleNode(x)", "ColorNode(y)"]
        resolution._global_vocab.update(feat_strs)

        # Create two objects with identical features at different positions
        obj_a = make_object_with_position(5, 5, tick=1)
        obj_b = make_object_with_position(50, 50, tick=1)
        resolution.initialize_alphas(obj_a.id, feat_strs)
        resolution.initialize_alphas(obj_b.id, feat_strs)

        # Warm up both objects with 10 observations each at their positions
        for tick in range(2, 12):
            fn1 = make_feature_node("SingleNode", "SingleNode(x)")
            fn2 = make_feature_node("ColorNode", "ColorNode(y)")

            # Observation at obj_a's position
            setup_candidate_return([fn1, fn2], [obj_a, obj_b])
            obj_a.last_tick = tick - 1
            obj_b.last_tick = tick - 1
            ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=tick)
            result = resolution.resolve([fn1, fn2], fg, ctx)
            # Update last seen for whichever matched
            if result is not None:
                result.last_tick = tick
                result.last_x = ctx.x
                result.last_y = ctx.y

        # After warmup, observation near obj_a should match obj_a
        fn1 = make_feature_node("SingleNode", "SingleNode(x)")
        fn2 = make_feature_node("ColorNode", "ColorNode(y)")
        setup_candidate_return([fn1, fn2], [obj_a, obj_b])
        ctx_near_a = ResolutionContext(x=XLoc(6), y=YLoc(6), tick=20)
        result_a = resolution.resolve([fn1, fn2], fg, ctx_near_a)
        assert result_a is obj_a, "Observation near obj_a should match obj_a"

        # Observation near obj_b should match obj_b
        fn1 = make_feature_node("SingleNode", "SingleNode(x)")
        fn2 = make_feature_node("ColorNode", "ColorNode(y)")
        setup_candidate_return([fn1, fn2], [obj_a, obj_b])
        obj_b.last_tick = 19  # recent
        ctx_near_b = ResolutionContext(x=XLoc(50), y=YLoc(50), tick=21)
        result_b = resolution.resolve([fn1, fn2], fg, ctx_near_b)
        assert result_b is obj_b, "Observation near obj_b should match obj_b"


class TestDirichletTelemetry:
    """Verify all Dirichlet-specific metrics are recorded during resolution."""

    @patch("roc.graphdb.GraphDB.singleton")
    def test_all_metrics_emitted(self, mock_db_cls):
        """Verify all Dirichlet-specific metrics are recorded."""
        mock_db = MagicMock()
        mock_db.strict_schema = False
        mock_db.strict_schema_warns = False
        mock_db_cls.return_value = mock_db

        resolution = DirichletCategoricalResolution()
        resolution._global_vocab = REALISTIC_VOCAB.copy()
        fg = MagicMock()

        # Patch all histogram/counter record/add methods
        with (
            patch.object(resolution.posterior_max_histogram, "record") as mock_max,
            patch.object(resolution.posterior_margin_histogram, "record") as mock_margin,
            patch.object(resolution.new_object_posterior_histogram, "record") as mock_new_post,
            patch.object(resolution.alpha_sum_histogram, "record") as mock_alpha_sum,
            patch.object(resolution.dirichlet_decision_counter, "add") as mock_decision,
        ):
            # Set up a matchable object
            fn1 = make_feature_node("SingleNode", "SingleNode(a)")
            obj = make_object_with_position(5, 5, tick=10)
            resolution._alphas[obj.id] = {"SingleNode(a)": 11.0}
            resolution._global_vocab.add("SingleNode(a)")

            setup_candidate_return([fn1], [obj])
            ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=11)
            result = resolution.resolve([fn1], fg, ctx)

            assert result is obj
            mock_max.assert_called_once()
            mock_margin.assert_called_once()
            mock_new_post.assert_called_once()
            mock_alpha_sum.assert_called_once()
            mock_decision.assert_called_once_with(1, attributes={"outcome": "match"})

            # Verify recorded values are valid probabilities
            max_val = mock_max.call_args[0][0]
            assert 0.0 <= max_val <= 1.0
            assert not math.isnan(max_val)

    @patch("roc.graphdb.GraphDB.singleton")
    def test_decision_counter_outcomes(self, mock_db_cls):
        """Verify decision counter tracks match, new_object, low_confidence."""
        mock_db = MagicMock()
        mock_db.strict_schema = False
        mock_db.strict_schema_warns = False
        mock_db_cls.return_value = mock_db

        resolution = DirichletCategoricalResolution()
        resolution._global_vocab = REALISTIC_VOCAB.copy()
        fg = MagicMock()

        outcomes: list[str] = []
        original_add = resolution.dirichlet_decision_counter.add

        def capture_add(val, attributes=None):
            if attributes:
                outcomes.append(attributes["outcome"])
            original_add(val, attributes=attributes)

        with patch.object(resolution.dirichlet_decision_counter, "add", side_effect=capture_add):
            # 1. No candidates -> new_object
            fn1 = make_feature_node("SingleNode", "SingleNode(a)")
            setup_no_candidates([fn1])
            ctx = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=1)
            resolution.resolve([fn1], fg, ctx)

            # 2. Match
            obj = make_object_with_position(5, 5, tick=1)
            resolution._alphas[obj.id] = {"SingleNode(a)": 11.0}
            resolution._global_vocab.add("SingleNode(a)")
            setup_candidate_return([fn1], [obj])
            ctx2 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=2)
            resolution.resolve([fn1], fg, ctx2)

            # 3. Low confidence (very high threshold)
            resolution.confidence_threshold = 0.999
            fn2 = make_feature_node("SingleNode", "SingleNode(z)")
            obj2 = make_object_with_position(5, 5, tick=2)
            resolution._alphas[obj2.id] = {"SingleNode(z)": 2.0}
            resolution._global_vocab.add("SingleNode(z)")
            setup_candidate_return([fn2], [obj2])
            ctx3 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=3)
            resolution.resolve([fn2], fg, ctx3)

        assert "new_object" in outcomes
        assert "match" in outcomes
        assert "low_confidence" in outcomes
