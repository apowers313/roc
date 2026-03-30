# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/transformer.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.pipeline.temporal.transformer import Change, TransformResult
from roc.pipeline.temporal.transformable import Transform


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestTransformResult:
    def test_constructor(self):
        t = Transform()
        tr = TransformResult(transform=t)
        assert tr.transform is t

    def test_transform_field(self):
        t = Transform()
        tr = TransformResult(transform=t)
        assert isinstance(tr.transform, Transform)


class TestChange:
    def test_allowed_connections(self):
        expected = [
            ("Transform", "Transform"),
            ("Frame", "Transform"),
            ("Transform", "Frame"),
        ]
        assert Change.model_fields["allowed_connections"].default == expected
