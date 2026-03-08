# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/predict.py -- do_predict early return and scoring logic."""

from unittest.mock import MagicMock, patch

import pytest

from roc.predict import (
    NoPrediction,
    Predict,
    PredictionCandidateFramesExpMod,
    PredictionConfidenceExpMod,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


@pytest.fixture
def fresh_predict():
    """Create a Predict instance, clearing component_set to avoid duplicates."""
    import roc.component as comp_mod

    saved = comp_mod.component_set.copy()
    # Remove any existing predict component
    to_remove = [c for c in comp_mod.component_set if c.name == "predict" and c.type == "predict"]
    for c in to_remove:
        comp_mod.component_set.discard(c)

    p = Predict()
    yield p

    # Restore
    comp_mod.component_set.discard(p)
    for c in saved:
        comp_mod.component_set.add(c)


class TestDoPredict:
    def test_returns_early_when_no_change_edges(self, fresh_predict):
        """When transform_src is empty (no Change edges), do_predict returns early."""
        predict = fresh_predict

        mock_event = MagicMock()
        mock_transform = MagicMock()
        mock_edge_list = MagicMock()
        mock_edge_list.select.return_value = []
        mock_edge_list.__len__ = MagicMock(return_value=0)
        mock_transform.dst_edges = mock_edge_list
        mock_event.data.transform = mock_transform

        # Mock the send method on predict_conn
        predict.predict_conn = MagicMock()

        result = predict.do_predict(mock_event)
        assert result is None
        predict.predict_conn.send.assert_not_called()

    def test_sends_no_prediction_when_no_candidates(self, fresh_predict):
        """When candidates list is empty, sends NoPrediction."""
        predict = fresh_predict
        predict.predict_conn = MagicMock()

        mock_event = MagicMock()
        mock_transform = MagicMock()
        mock_edge_list = MagicMock()
        mock_change_edge = MagicMock()

        from roc.sequencer import Frame

        mock_frame = MagicMock(spec=Frame)
        mock_change_edge.src = mock_frame
        mock_edge_list.select.return_value = [mock_change_edge]
        mock_edge_list.__len__ = MagicMock(return_value=1)
        mock_transform.dst_edges = mock_edge_list
        mock_event.data.transform = mock_transform

        mock_mod = MagicMock()
        mock_mod.get_candidates.return_value = []

        with patch.object(PredictionCandidateFramesExpMod, "get", return_value=mock_mod):
            predict.do_predict(mock_event)

        assert predict.predict_conn.send.call_count == 1
        sent = predict.predict_conn.send.call_args[0][0]
        assert isinstance(sent, NoPrediction)

    def test_sends_best_prediction(self, fresh_predict):
        """When candidates exist, scores them and sends the best one."""
        predict = fresh_predict
        predict.predict_conn = MagicMock()

        mock_event = MagicMock()
        mock_transform = MagicMock()
        mock_edge_list = MagicMock()
        mock_change_edge = MagicMock()

        from roc.sequencer import Frame

        mock_current = MagicMock(spec=Frame)
        mock_change_edge.src = mock_current
        mock_edge_list.select.return_value = [mock_change_edge]
        mock_edge_list.__len__ = MagicMock(return_value=1)
        mock_transform.dst_edges = mock_edge_list
        mock_event.data.transform = mock_transform

        candidate1 = MagicMock(spec=Frame)
        candidate2 = MagicMock(spec=Frame)

        mock_candidate_mod = MagicMock()
        mock_candidate_mod.get_candidates.return_value = [candidate1, candidate2]

        merged1 = MagicMock(spec=Frame)
        merged2 = MagicMock(spec=Frame)

        mock_scoring = MagicMock()
        mock_scoring.calculate_confidence.side_effect = [0.3, 0.9]

        with (
            patch.object(PredictionCandidateFramesExpMod, "get", return_value=mock_candidate_mod),
            patch.object(Frame, "merge_transforms", side_effect=[merged1, merged2]),
            patch.object(PredictionConfidenceExpMod, "get", return_value=mock_scoring),
        ):
            predict.do_predict(mock_event)

        assert predict.predict_conn.send.call_count == 1
        sent = predict.predict_conn.send.call_args[0][0]
        assert sent is merged2


class TestPredictionConfidenceExpMod:
    def test_is_abstract(self):
        assert getattr(
            PredictionConfidenceExpMod.calculate_confidence, "__isabstractmethod__", False
        )

    def test_modtype(self):
        assert PredictionConfidenceExpMod.modtype == "prediction-confidence"
