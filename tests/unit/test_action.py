# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/action.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.action import ActionRequest, DefaultActionPass, TakeAction


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestActionRequest:
    def test_instantiate(self):
        ar = ActionRequest()
        assert isinstance(ar, ActionRequest)


class TestTakeAction:
    def test_constructor(self):
        ta = TakeAction(action=42)
        assert ta.action == 42

    def test_action_field(self):
        ta = TakeAction(action="move_left")
        assert ta.action == "move_left"


class TestDefaultActionPass:
    def test_get_action_returns_19(self):
        dap = DefaultActionPass()
        assert dap.get_action() == 19
