# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/predict.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.expmods.prediction_candidate.object_based import ObjectBasedPrediction
from roc.pipeline.temporal.predict import (
    NoPrediction,
    PredictionCandidateFramesExpMod,
    PredictionConfidenceExpMod,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestNoPrediction:
    def test_instantiate(self):
        np_obj = NoPrediction()
        assert isinstance(np_obj, NoPrediction)


class TestPredictionCandidateFramesExpMod:
    def test_modtype(self):
        assert PredictionCandidateFramesExpMod.modtype == "prediction-candidate"

    def test_has_abstract_get_candidates(self):
        assert hasattr(PredictionCandidateFramesExpMod, "get_candidates")
        assert getattr(
            PredictionCandidateFramesExpMod.get_candidates, "__isabstractmethod__", False
        )


class TestPredictionConfidenceExpMod:
    def test_modtype(self):
        assert PredictionConfidenceExpMod.modtype == "prediction-confidence"

    def test_has_abstract_calculate_confidence(self):
        assert hasattr(PredictionConfidenceExpMod, "calculate_confidence")
        assert getattr(
            PredictionConfidenceExpMod.calculate_confidence, "__isabstractmethod__", False
        )


class TestObjectBasedPrediction:
    def test_get_candidates(self):
        from roc.pipeline.temporal.sequencer import Frame

        obp = ObjectBasedPrediction()

        mock_frame = MagicMock(spec=Frame)
        mock_obj1 = MagicMock()
        mock_obj2 = MagicMock()
        mock_frame1 = MagicMock(spec=Frame)
        mock_frame2 = MagicMock(spec=Frame)
        mock_obj1.frames = [mock_frame1]
        mock_obj2.frames = [mock_frame2]
        mock_frame.objects = [mock_obj1, mock_obj2]

        result = obp.get_candidates(mock_frame)
        assert len(result) == 2
        assert mock_frame1 in result
        assert mock_frame2 in result

    def test_get_candidates_empty(self):
        from roc.pipeline.temporal.sequencer import Frame

        obp = ObjectBasedPrediction()

        mock_frame = MagicMock(spec=Frame)
        mock_frame.objects = []

        result = obp.get_candidates(mock_frame)
        assert result == []
